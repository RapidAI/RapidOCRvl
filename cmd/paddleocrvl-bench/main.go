package main

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	"paddleocrvl-go/internal/backend"
	"paddleocrvl-go/internal/model"
	"paddleocrvl-go/internal/tokenizer"
)

type requestBody struct {
	Prompt              string  `json:"prompt,omitempty"`
	Task                string  `json:"task,omitempty"`
	ImagePath           string  `json:"image_path,omitempty"`
	ImageBase64         string  `json:"image_base64,omitempty"`
	Tokens              []int   `json:"tokens,omitempty"`
	MaxNewTokens        int     `json:"max_new_tokens"`
	Temperature         float64 `json:"temperature,omitempty"`
	TopK                int     `json:"top_k,omitempty"`
	Decode              bool    `json:"decode"`
	DecodeGeneratedOnly bool    `json:"decode_generated_only"`
	SkipSpecial         bool    `json:"skip_special"`
}

const maxBenchImageBytes = 128 << 20
const maxBenchResponseBytes = 8 << 20

type generateResponse struct {
	Tokens          []int `json:"tokens"`
	PromptTokens    int   `json:"prompt_tokens"`
	GeneratedTokens int   `json:"generated_tokens"`
}

type batchRequest struct {
	Requests []requestBody `json:"requests"`
}

type batchResponse struct {
	Responses       []generateResponse `json:"responses"`
	Items           int                `json:"items"`
	GeneratedTokens int                `json:"generated_tokens"`
}

type benchReport struct {
	Mode         string             `json:"mode,omitempty"`
	URL          string             `json:"url,omitempty"`
	Backend      string             `json:"backend,omitempty"`
	Quantization string             `json:"quantization,omitempty"`
	WeightPath   string             `json:"weight_path,omitempty"`
	WeightSource string             `json:"weight_source,omitempty"`
	LoadStats    model.LoadStats    `json:"load_stats,omitempty"`
	Requests     int                `json:"requests"`
	OK           int64              `json:"ok"`
	Errors       int64              `json:"errors"`
	Items        int64              `json:"items"`
	BatchSize    int                `json:"batch_size"`
	Tokens       int64              `json:"tokens"`
	ElapsedMS    int64              `json:"elapsed_ms"`
	QPS          float64            `json:"qps"`
	ItemsPerS    float64            `json:"items_per_s"`
	TokensPerS   float64            `json:"tokens_per_s"`
	LastError    string             `json:"last_error,omitempty"`
	LatencyMinMS int64              `json:"latency_min_ms"`
	LatencyP50MS int64              `json:"latency_p50_ms"`
	LatencyP95MS int64              `json:"latency_p95_ms"`
	LatencyP99MS int64              `json:"latency_p99_ms"`
	LatencyMaxMS int64              `json:"latency_max_ms"`
	LatencyAvgMS int64              `json:"latency_avg_ms"`
	CPU          backend.CPUInfo    `json:"cpu"`
	Memory       backend.MemoryInfo `json:"memory"`
}

type benchContext struct {
	Mode         string
	URL          string
	Backend      string
	Quantization string
	WeightPath   string
	WeightSource string
	LoadStats    model.LoadStats
}

func main() {
	mode := flag.String("mode", "http", "benchmark mode: http or local")
	url := flag.String("url", "http://127.0.0.1:8080/v1/generate", "server generate URL")
	modelDir := flag.String("model-dir", ".", "model dir for local mode")
	quant := flag.String("quant", "f32", "local mode weight quantization: f32, q8, q6, q4, auto, auto-fast, or auto-quality")
	backendName := flag.String("backend", "cpu", "local mode compute backend: cpu, vulkan, or auto")
	gomaxprocs := flag.Int("gomaxprocs", 0, "set Go GOMAXPROCS; 0 keeps current value")
	gcPercent := flag.Int("gc-percent", 0, "set Go GC percent; 0 keeps current value, -1 disables GC")
	requests := flag.Int("n", 10, "number of requests")
	concurrency := flag.Int("c", 1, "concurrent workers")
	batchSize := flag.Int("batch-size", 1, "HTTP requests per /v1/batch call; 1 uses /v1/generate payload shape")
	prompt := flag.String("prompt", "<|begin_of_sentence|>hello", "prompt for text benchmark")
	task := flag.String("task", "", "task for image benchmark")
	imagePath := flag.String("image-path", "", "server-side image path")
	imageBase64 := flag.String("image-base64", "", "base64 image data or data URL")
	maxNew := flag.Int("max-new-tokens", 1, "max new tokens")
	temperature := flag.Float64("temperature", 0, "sampling temperature; 0 means greedy")
	topK := flag.Int("top-k", 0, "sample from top K tokens when temperature > 0; 0 means full vocab")
	timeout := flag.Duration("timeout", 30*time.Minute, "HTTP client timeout")
	outputJSON := flag.Bool("json", false, "emit benchmark result as JSON")
	flag.Parse()
	if _, err := backend.SetGOMAXPROCS(*gomaxprocs); err != nil {
		log.Fatal(err)
	}
	backend.SetGCPercent(*gcPercent)

	body := requestBody{
		Prompt:              *prompt,
		Task:                *task,
		ImagePath:           *imagePath,
		ImageBase64:         *imageBase64,
		MaxNewTokens:        *maxNew,
		Temperature:         *temperature,
		TopK:                *topK,
		Decode:              false,
		DecodeGeneratedOnly: true,
		SkipSpecial:         true,
	}
	if body.Task != "" {
		body.Prompt = ""
	}
	if err := validateBenchArgs(*requests, *concurrency, *batchSize, *timeout, body); err != nil {
		log.Fatal(err)
	}
	if *mode == "local" {
		runLocal(*modelDir, *quant, *backendName, body, *requests, *concurrency, *outputJSON)
		return
	}
	if *mode != "http" {
		log.Fatal("-mode must be http or local")
	}
	targetURL := normalizeBenchURL(*url, *batchSize)
	if err := validateBenchURL(targetURL); err != nil {
		log.Fatal(err)
	}
	payload, err := benchPayload(body, *batchSize)
	if err != nil {
		log.Fatal(err)
	}
	client := &http.Client{Timeout: *timeout}
	latencies := make([]time.Duration, *requests)
	var next atomic.Int64
	var errors atomic.Int64
	var tokens atomic.Int64
	var lastErr atomic.Value
	start := time.Now()
	var wg sync.WaitGroup
	for worker := 0; worker < *concurrency; worker++ {
		wg.Add(1)
		go func() {
			for {
				i := int(next.Add(1)) - 1
				if i >= *requests {
					wg.Done()
					return
				}
				t0 := time.Now()
				n, err := post(client, targetURL, payload, *batchSize)
				if err != nil {
					errors.Add(1)
					lastErr.Store(err.Error())
				} else {
					tokens.Add(int64(n))
				}
				latencies[i] = time.Since(t0)
			}
		}()
	}
	wg.Wait()
	elapsed := time.Since(start)
	report(latencies, elapsed, errors.Load(), tokens.Load(), *batchSize, loadLastError(&lastErr), *outputJSON, benchContext{Mode: "http", URL: targetURL})
}

func normalizeBenchURL(url string, batchSize int) string {
	if batchSize <= 1 || !strings.HasSuffix(url, "/v1/generate") {
		return url
	}
	return strings.TrimSuffix(url, "/v1/generate") + "/v1/batch"
}

func validateBenchURL(raw string) error {
	u, err := url.Parse(strings.TrimSpace(raw))
	if err != nil || u.Host == "" {
		return fmt.Errorf("invalid URL")
	}
	if u.Scheme != "http" && u.Scheme != "https" {
		return fmt.Errorf("URL must use http or https")
	}
	return nil
}

func validateBenchArgs(requests, concurrency, batchSize int, timeout time.Duration, body requestBody) error {
	if requests < 1 {
		return fmt.Errorf("-n must be >= 1")
	}
	if concurrency < 1 {
		return fmt.Errorf("-c must be >= 1")
	}
	if batchSize < 1 {
		return fmt.Errorf("-batch-size must be >= 1")
	}
	if timeout < 0 {
		return fmt.Errorf("-timeout must be >= 0")
	}
	opts := model.GenerateOptions{MaxNewTokens: body.MaxNewTokens, Temperature: body.Temperature, TopK: body.TopK}
	if err := model.ValidateGenerateOptions(opts); err != nil {
		return err
	}
	return nil
}

func benchPayload(body requestBody, batchSize int) ([]byte, error) {
	if batchSize <= 1 {
		return json.Marshal(body)
	}
	reqs := make([]requestBody, batchSize)
	for i := range reqs {
		reqs[i] = body
	}
	return json.Marshal(batchRequest{Requests: reqs})
}

func runLocal(modelDir, quant, backendName string, body requestBody, requests, concurrency int, outputJSON bool) {
	backendSel, err := backend.Select(backendName)
	if err != nil {
		log.Fatal(err)
	}
	rt, err := model.LoadWithOptions(modelDir, model.LoadOptions{Quantization: quant, Backend: backendSel.Active, Progress: model.ProgressLogger("loader")})
	if err != nil {
		log.Fatal(err)
	}
	defer rt.Close()
	tok, err := tokenizer.Load(modelDir)
	if err != nil {
		log.Fatal(err)
	}
	ids, err := inputIDs(tok, body)
	if err != nil {
		log.Fatal(err)
	}
	imageData, err := bodyImageBytes(body)
	if err != nil {
		log.Fatal(err)
	}
	opts := model.GenerateOptions{MaxNewTokens: body.MaxNewTokens, Temperature: body.Temperature, TopK: body.TopK}
	if err := model.ValidateGenerateOptions(opts); err != nil {
		log.Fatal(err)
	}
	latencies := make([]time.Duration, requests)
	var next atomic.Int64
	var errors atomic.Int64
	var tokens atomic.Int64
	var lastErr atomic.Value
	start := time.Now()
	var wg sync.WaitGroup
	for worker := 0; worker < concurrency; worker++ {
		wg.Add(1)
		go func() {
			for {
				i := int(next.Add(1)) - 1
				if i >= requests {
					wg.Done()
					return
				}
				t0 := time.Now()
				var err error
				var res model.GenerateResult
				if len(imageData) > 0 {
					res, err = rt.GenerateWithImageBytesOptions(context.Background(), ids, imageData, opts)
				} else if body.ImagePath != "" {
					res, err = rt.GenerateWithImageOptions(context.Background(), ids, body.ImagePath, opts)
				} else {
					res, err = rt.GenerateWithOptions(context.Background(), ids, opts)
				}
				if err != nil {
					errors.Add(1)
					lastErr.Store(err.Error())
				} else {
					tokens.Add(int64(max(0, len(res.Tokens)-res.PromptTokens)))
				}
				latencies[i] = time.Since(t0)
			}
		}()
	}
	wg.Wait()
	report(latencies, time.Since(start), errors.Load(), tokens.Load(), 1, loadLastError(&lastErr), outputJSON, benchContext{Mode: "local", Backend: rt.Backend(), Quantization: rt.Quantization(), WeightPath: rt.WeightPath(), WeightSource: rt.WeightSource(), LoadStats: rt.LoadStats()})
}

func post(client *http.Client, url string, payload []byte, batchSize int) (int, error) {
	resp, err := client.Post(url, "application/json", bytes.NewReader(payload))
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		_, _ = io.CopyN(io.Discard, resp.Body, maxBenchResponseBytes)
		return 0, fmt.Errorf("status %d", resp.StatusCode)
	}
	if batchSize <= 1 {
		var out generateResponse
		if err := decodeLimitedJSON(resp.Body, maxBenchResponseBytes, &out); err != nil {
			return 0, err
		}
		return generatedTokenCount(out), nil
	}
	var out batchResponse
	if err := decodeLimitedJSON(resp.Body, maxBenchResponseBytes, &out); err != nil {
		return 0, err
	}
	if out.GeneratedTokens > 0 {
		return out.GeneratedTokens, nil
	}
	total := 0
	for _, item := range out.Responses {
		total += generatedTokenCount(item)
	}
	return total, nil
}

func decodeLimitedJSON(r io.Reader, maxBytes int64, v any) error {
	data, err := io.ReadAll(io.LimitReader(r, maxBytes+1))
	if err != nil {
		return err
	}
	if int64(len(data)) > maxBytes {
		return fmt.Errorf("response too large: exceeds %d bytes", maxBytes)
	}
	return json.Unmarshal(data, v)
}

func generatedTokenCount(out generateResponse) int {
	if out.GeneratedTokens > 0 {
		return out.GeneratedTokens
	}
	return max(0, len(out.Tokens)-out.PromptTokens)
}

func inputIDs(tok *tokenizer.Tokenizer, body requestBody) ([]int, error) {
	if len(body.Tokens) > 0 {
		ids := append([]int(nil), body.Tokens...)
		for _, id := range ids {
			if id < 0 {
				return nil, fmt.Errorf("token ids must not be negative: %d", id)
			}
		}
		return ids, nil
	}
	if body.Task != "" {
		return tok.EncodeReadOnly(chatPrompt(taskPrompt(body.Task), body.ImagePath != "" || body.ImageBase64 != "")), nil
	}
	return tok.EncodeReadOnly(body.Prompt), nil
}

func bodyImageBytes(body requestBody) ([]byte, error) {
	if body.ImageBase64 == "" {
		return nil, nil
	}
	data := body.ImageBase64
	if i := strings.IndexByte(data, ','); i >= 0 {
		data = data[i+1:]
	}
	decodedLen := decodedBase64Len(data)
	if decodedLen > maxBenchImageBytes {
		decodedLen = decodedBase64LenIgnoringCRLF(data)
	}
	if decodedLen > maxBenchImageBytes {
		return nil, fmt.Errorf("image_base64 exceeds %d decoded bytes", maxBenchImageBytes)
	}
	raw := make([]byte, decodedLen)
	src := unsafe.Slice(unsafe.StringData(data), len(data))
	n, err := base64.StdEncoding.Decode(raw, src)
	if err != nil {
		return nil, err
	}
	if n > maxBenchImageBytes {
		return nil, fmt.Errorf("image_base64 exceeds %d decoded bytes", maxBenchImageBytes)
	}
	return raw[:n], nil
}

func decodedBase64Len(data string) int {
	n := base64.StdEncoding.DecodedLen(len(data))
	for len(data) > 0 && data[len(data)-1] == '=' {
		n--
		data = data[:len(data)-1]
	}
	if n < 0 {
		return 0
	}
	return n
}

func decodedBase64LenIgnoringCRLF(data string) int {
	encoded := 0
	padding := 0
	for i := 0; i < len(data); i++ {
		switch data[i] {
		case '\r', '\n':
			continue
		case '=':
			encoded++
			padding++
		default:
			encoded++
			padding = 0
		}
	}
	n := base64.StdEncoding.DecodedLen(encoded) - padding
	if n < 0 {
		return 0
	}
	return n
}

func taskPrompt(task string) string {
	switch task {
	case "ocr":
		return "OCR:"
	case "table":
		return "Table Recognition:"
	case "formula":
		return "Formula Recognition:"
	case "chart":
		return "Chart Recognition:"
	default:
		return task
	}
}

func chatPrompt(userText string, withImage bool) string {
	const prefix = "<|begin_of_sentence|>User: "
	const image = "<|IMAGE_START|><|IMAGE_PLACEHOLDER|><|IMAGE_END|>"
	const suffix = "\nAssistant: "
	if withImage {
		switch userText {
		case "OCR:":
			return prefix + image + "OCR:" + suffix
		case "Table Recognition:":
			return prefix + image + "Table Recognition:" + suffix
		case "Formula Recognition:":
			return prefix + image + "Formula Recognition:" + suffix
		case "Chart Recognition:":
			return prefix + image + "Chart Recognition:" + suffix
		}
		return prefix + image + userText + suffix
	}
	switch userText {
	case "OCR:":
		return prefix + "OCR:" + suffix
	case "Table Recognition:":
		return prefix + "Table Recognition:" + suffix
	case "Formula Recognition:":
		return prefix + "Formula Recognition:" + suffix
	case "Chart Recognition:":
		return prefix + "Chart Recognition:" + suffix
	}
	return prefix + userText + suffix
}

func report(latencies []time.Duration, elapsed time.Duration, errors int64, tokens int64, batchSize int, lastError string, outputJSON bool, ctx benchContext) {
	sort.Slice(latencies, func(i, j int) bool { return latencies[i] < latencies[j] })
	r := buildReport(latencies, elapsed, errors, tokens, batchSize, lastError, ctx)
	if outputJSON {
		_ = json.NewEncoder(io.Writer(os.Stdout)).Encode(r)
		return
	}
	fmt.Printf("requests=%d ok=%d errors=%d items=%d batch_size=%d tokens=%d elapsed=%s qps=%.3f items_per_s=%.3f tokens_per_s=%.3f\n", r.Requests, r.OK, r.Errors, r.Items, r.BatchSize, r.Tokens, elapsed.Round(time.Millisecond), r.QPS, r.ItemsPerS, r.TokensPerS)
	if r.LastError != "" {
		fmt.Printf("last_error=%s\n", r.LastError)
	}
	fmt.Printf("latency_ms min=%d p50=%d p95=%d p99=%d max=%d avg=%d\n",
		r.LatencyMinMS,
		r.LatencyP50MS,
		r.LatencyP95MS,
		r.LatencyP99MS,
		r.LatencyMaxMS,
		r.LatencyAvgMS,
	)
}

func buildReport(latencies []time.Duration, elapsed time.Duration, errors int64, tokens int64, batchSize int, lastError string, ctx benchContext) benchReport {
	n := len(latencies)
	ok := int64(n) - errors
	items := ok * int64(max(1, batchSize))
	qps := float64(ok) / elapsed.Seconds()
	ips := float64(items) / elapsed.Seconds()
	tps := float64(tokens) / elapsed.Seconds()
	if n == 0 {
		return benchReport{}
	}
	return benchReport{
		Mode:         ctx.Mode,
		URL:          ctx.URL,
		Backend:      ctx.Backend,
		Quantization: ctx.Quantization,
		WeightPath:   ctx.WeightPath,
		WeightSource: ctx.WeightSource,
		LoadStats:    ctx.LoadStats,
		Requests:     n,
		OK:           ok,
		Errors:       errors,
		Items:        items,
		BatchSize:    max(1, batchSize),
		Tokens:       tokens,
		ElapsedMS:    elapsed.Milliseconds(),
		QPS:          qps,
		ItemsPerS:    ips,
		TokensPerS:   tps,
		LastError:    lastError,
		LatencyMinMS: ms(latencies[0]),
		LatencyP50MS: ms(percentile(latencies, 0.50)),
		LatencyP95MS: ms(percentile(latencies, 0.95)),
		LatencyP99MS: ms(percentile(latencies, 0.99)),
		LatencyMaxMS: ms(latencies[n-1]),
		LatencyAvgMS: ms(avg(latencies)),
		CPU:          backend.CPU(),
		Memory:       backend.Memory(),
	}
}

func loadLastError(v *atomic.Value) string {
	x := v.Load()
	if x == nil {
		return ""
	}
	return x.(string)
}

func percentile(xs []time.Duration, p float64) time.Duration {
	if len(xs) == 0 {
		return 0
	}
	i := int(float64(len(xs)-1) * p)
	return xs[i]
}

func avg(xs []time.Duration) time.Duration {
	if len(xs) == 0 {
		return 0
	}
	var sum time.Duration
	for _, x := range xs {
		sum += x
	}
	return sum / time.Duration(len(xs))
}

func ms(d time.Duration) int64 {
	return d.Milliseconds()
}
