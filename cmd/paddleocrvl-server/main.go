package main

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
	"unsafe"

	"paddleocrvl-go/internal/backend"
	"paddleocrvl-go/internal/fileutil"
	"paddleocrvl-go/internal/model"
	"paddleocrvl-go/internal/tokenizer"
)

type server struct {
	rt            *model.Runtime
	tok           *tokenizer.Tokenizer
	taskIDs       map[string][]int
	admin         *adminState
	modelDir      string
	runSlots      chan struct{}
	requestLimit  int64
	multipartMem  int64
	timeout       time.Duration
	maxNewLimit   int
	maxInputLimit int
	maxBatchSize  int
	started       time.Time
	concurrency   int
	metrics       metrics
	weightSHA256  string
	backendSel    backend.Selection
	cpuInfo       backend.CPUInfo
}

type metrics struct {
	queued          atomic.Int64
	started         atomic.Int64
	succeeded       atomic.Int64
	failed          atomic.Int64
	canceled        atomic.Int64
	batches         atomic.Int64
	batchItems      atomic.Int64
	generatedTokens atomic.Int64
	latencyNanos    atomic.Int64
	queueWaitNanos  atomic.Int64
	lastError       atomic.Value
}

var contentTypeJSONHeader = []string{"application/json"}
var okJSONBytes = []byte("{\"ok\":true}\n")
var adminSessionJSON = [4][]byte{
	[]byte("{\"initialized\":false,\"authenticated\":false}\n"),
	[]byte("{\"initialized\":true,\"authenticated\":false}\n"),
	[]byte("{\"initialized\":false,\"authenticated\":true}\n"),
	[]byte("{\"initialized\":true,\"authenticated\":true}\n"),
}
var errorBufferPool = sync.Pool{New: func() any {
	buf := make([]byte, 0, 512)
	return &buf
}}

type generateRequest struct {
	Prompt              string `json:"prompt"`
	Task                string `json:"task"`
	ImagePath           string `json:"image_path"`
	ImageBase64         string `json:"image_base64"`
	imageData           []byte
	Tokens              []int   `json:"tokens"`
	MaxNewTokens        int     `json:"max_new_tokens"`
	Temperature         float64 `json:"temperature"`
	TopK                int     `json:"top_k"`
	Seed                int64   `json:"seed"`
	EOSTokenIDs         []int   `json:"eos_token_ids"`
	Decode              bool    `json:"decode"`
	DecodeGeneratedOnly bool    `json:"decode_generated_only"`
	SkipSpecial         bool    `json:"skip_special"`
}

type generateResponse struct {
	Tokens          []int  `json:"tokens"`
	PromptTokens    int    `json:"prompt_tokens"`
	GeneratedTokens int    `json:"generated_tokens"`
	Text            string `json:"text,omitempty"`
}

type batchRequest struct {
	Requests []generateRequest `json:"requests"`
}

type batchResponse struct {
	Responses       []generateResponse `json:"responses"`
	Items           int                `json:"items"`
	GeneratedTokens int                `json:"generated_tokens"`
}

type healthResponse struct {
	Status       string `json:"status"`
	Quantization string `json:"quantization"`
	WeightPath   string `json:"weight_path"`
	WeightSource string `json:"weight_source"`
	Backend      string `json:"backend"`
	VisionLoaded bool   `json:"vision_loaded"`
}

type readyResponse struct {
	Status         string `json:"status"`
	Reason         string `json:"reason,omitempty"`
	Quantization   string `json:"quantization,omitempty"`
	WeightPath     string `json:"weight_path,omitempty"`
	WeightSource   string `json:"weight_source,omitempty"`
	Backend        string `json:"backend,omitempty"`
	VisionLoaded   bool   `json:"vision_loaded,omitempty"`
	Concurrency    int    `json:"concurrency,omitempty"`
	InFlight       int    `json:"in_flight,omitempty"`
	AvailableSlots int    `json:"available_slots,omitempty"`
}

type statsResponse struct {
	Status               string                       `json:"status"`
	UptimeSeconds        int64                        `json:"uptime_seconds"`
	Concurrency          int                          `json:"concurrency"`
	InFlight             int                          `json:"in_flight"`
	AvailableSlots       int                          `json:"available_slots"`
	RequestLimit         int64                        `json:"request_limit"`
	MultipartMemory      int64                        `json:"multipart_memory"`
	MaxNewLimit          int                          `json:"max_new_limit"`
	MaxInputTokens       int                          `json:"max_input_tokens"`
	MaxBatchSize         int                          `json:"max_batch_size"`
	TimeoutSeconds       int64                        `json:"timeout_seconds"`
	Quantization         string                       `json:"quantization"`
	RequestedQuant       string                       `json:"requested_quant"`
	WeightPath           string                       `json:"weight_path"`
	WeightSource         string                       `json:"weight_source"`
	WeightSHA256         string                       `json:"weight_sha256"`
	Weights              model.WeightStats            `json:"weights"`
	LoadStats            model.LoadStats              `json:"load_stats"`
	Backend              string                       `json:"backend"`
	VisionLoaded         bool                         `json:"vision_loaded"`
	Backends             backend.Selection            `json:"backends"`
	VulkanPlans          []backend.VulkanPlan         `json:"vulkan_plans,omitempty"`
	VulkanPlanSummary    backend.VulkanPlanSummary    `json:"vulkan_plan_summary,omitempty"`
	VulkanExecutionGraph backend.VulkanExecutionGraph `json:"vulkan_execution_graph,omitempty"`
	VulkanPipelinePlan   []backend.VulkanPipelinePlan `json:"vulkan_pipeline_plan,omitempty"`
	VulkanCommandPlan    backend.VulkanCommandPlan    `json:"vulkan_command_plan,omitempty"`
	VulkanCommandPlanOK  bool                         `json:"vulkan_command_plan_valid"`
	VulkanCommandPlanErr string                       `json:"vulkan_command_plan_error,omitempty"`
	CPU                  backend.CPUInfo              `json:"cpu"`
	Memory               backend.MemoryInfo           `json:"memory"`
	Requests             statsRequests                `json:"requests"`
	Model                statsModel                   `json:"model"`
	Cache                statsCache                   `json:"cache"`
}

type statsRequests struct {
	Queued          int64  `json:"queued"`
	Started         int64  `json:"started"`
	Succeeded       int64  `json:"succeeded"`
	Failed          int64  `json:"failed"`
	Canceled        int64  `json:"canceled"`
	Batches         int64  `json:"batches"`
	BatchItems      int64  `json:"batch_items"`
	GeneratedTokens int64  `json:"generated_tokens"`
	AvgLatencyMS    int64  `json:"avg_latency_ms"`
	AvgQueueWaitMS  int64  `json:"avg_queue_wait_ms"`
	LastError       string `json:"last_error"`
}

type statsModel struct {
	VocabSize int              `json:"vocab_size"`
	Text      statsTextModel   `json:"text"`
	Vision    statsVisionModel `json:"vision"`
}

type statsTextModel struct {
	Layers  int `json:"layers"`
	Hidden  int `json:"hidden"`
	Heads   int `json:"heads"`
	KVHeads int `json:"kv_heads"`
	HeadDim int `json:"head_dim"`
}

type statsVisionModel struct {
	Layers int `json:"layers"`
	Hidden int `json:"hidden"`
	Heads  int `json:"heads"`
	Patch  int `json:"patch"`
}

type statsCache struct {
	TaskPrompts int                  `json:"task_prompts"`
	Runtime     model.CacheStats     `json:"runtime"`
	Tokenizer   tokenizer.CacheStats `json:"tokenizer"`
}

func main() {
	if err := runMain(os.Args[1:]); err != nil {
		log.Fatal(err)
	}
}

func runMain(args []string) error {
	if handled, err := handleServiceCommand(args); handled {
		return err
	}
	return runServer(args, nil)
}

func runServer(args []string, stop <-chan struct{}) error {
	fs := flag.NewFlagSet("paddleocrvl-server", flag.ContinueOnError)
	modelDir := fs.String("model-dir", ".", "directory containing model files")
	adminConfigPath := fs.String("admin-config", "paddleocrvl-admin.json", "admin console config path")
	quant := fs.String("quant", "f32", "weight quantization: f32, q8, q6, q4, auto, auto-fast, or auto-quality")
	backendName := fs.String("backend", "cpu", "compute backend: cpu, vulkan, or auto")
	addr := fs.String("addr", "127.0.0.1:8080", "HTTP listen address")
	timeout := fs.Duration("timeout", 0, "per-request timeout; 0 disables")
	shutdownTimeout := fs.Duration("shutdown-timeout", 30*time.Second, "graceful shutdown timeout")
	requestLimit := fs.Int64("request-limit", 128<<20, "max request body bytes")
	multipartMem := fs.Int64("multipart-memory", 32<<20, "max memory used while parsing multipart forms")
	maxNewLimit := fs.Int("max-new-limit", 4096, "maximum max_new_tokens per request; 0 disables")
	maxInputLimit := fs.Int("max-input-tokens", 0, "maximum prompt/input tokens per request; 0 disables")
	maxBatchSize := fs.Int("max-batch-size", 0, "maximum /v1/batch request count; 0 disables")
	concurrency := fs.Int("concurrency", 1, "max concurrent inference requests")
	gomaxprocs := fs.Int("gomaxprocs", 0, "set Go GOMAXPROCS; 0 keeps current value")
	gcPercent := fs.Int("gc-percent", 0, "set Go GC percent; 0 keeps current value, -1 disables GC")
	preloadVision := fs.Bool("preload-vision", false, "load vision weights at startup")
	warmup := fs.Bool("warmup", false, "run one text-token warmup during startup")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if err := validateServerLimits(*concurrency, *maxBatchSize, *maxInputLimit, *requestLimit, *multipartMem); err != nil {
		return err
	}
	if _, err := backend.SetGOMAXPROCS(*gomaxprocs); err != nil {
		return err
	}
	backend.SetGCPercent(*gcPercent)

	backendSel, err := backend.Select(*backendName)
	if err != nil {
		return err
	}
	rt, err := model.LoadWithOptions(*modelDir, model.LoadOptions{Quantization: *quant, Backend: backendSel.Active, Progress: model.ProgressLogger("loader")})
	if err != nil {
		return err
	}
	defer rt.Close()
	tok, err := tokenizer.Load(*modelDir)
	if err != nil {
		return err
	}
	if *preloadVision {
		log.Printf("preloading vision weights")
		if err := rt.PreloadVision(); err != nil {
			return err
		}
	}
	if *warmup {
		log.Printf("warming up text decoder")
		if _, err := rt.GenerateWithOptions(context.Background(), []int{0}, model.GenerateOptions{MaxNewTokens: 1}); err != nil {
			return err
		}
	}
	s := &server{
		rt:            rt,
		tok:           tok,
		taskIDs:       buildTaskIDs(tok),
		admin:         loadAdminState(*adminConfigPath),
		modelDir:      *modelDir,
		timeout:       *timeout,
		requestLimit:  *requestLimit,
		multipartMem:  *multipartMem,
		maxNewLimit:   *maxNewLimit,
		maxInputLimit: *maxInputLimit,
		maxBatchSize:  *maxBatchSize,
		runSlots:      make(chan struct{}, *concurrency),
		started:       time.Now(),
		concurrency:   *concurrency,
		weightSHA256:  fileutil.SHA256(rt.WeightPath()),
		backendSel:    backendSel,
		cpuInfo:       backend.CPU(),
	}

	mux := http.NewServeMux()
	mux.HandleFunc("GET /doc", s.docPage)
	mux.HandleFunc("GET /doc/", s.docPage)
	mux.HandleFunc("GET /doc/openapi.json", s.openapiJSON)
	mux.HandleFunc("GET /doc/llms.txt", s.llmsTXT)
	mux.HandleFunc("GET /admin", s.adminPage)
	mux.HandleFunc("GET /admin/", s.adminPage)
	mux.HandleFunc("POST /admin/api/init", s.adminInit)
	mux.HandleFunc("POST /admin/api/login", s.adminLogin)
	mux.HandleFunc("POST /admin/api/logout", s.adminLogout)
	mux.HandleFunc("GET /admin/api/session", s.adminSession)
	mux.HandleFunc("GET /admin/api/overview", s.requireAdmin(s.adminOverview))
	mux.HandleFunc("GET /admin/api/config", s.requireAdmin(s.adminConfigGet))
	mux.HandleFunc("POST /admin/api/config", s.requireAdmin(s.adminConfigSave))
	mux.HandleFunc("GET /admin/api/config/backup", s.requireAdmin(s.adminConfigBackup))
	mux.HandleFunc("POST /admin/api/config/restore", s.requireAdmin(s.adminConfigRestore))
	mux.HandleFunc("GET /admin/api/audit", s.requireAdmin(s.adminAuditList))
	mux.HandleFunc("GET /admin/api/audit.csv", s.requireAdmin(s.adminAuditCSV))
	mux.HandleFunc("POST /admin/api/audit/clear", s.requireAdmin(s.adminAuditClear))
	mux.HandleFunc("GET /admin/api/keys", s.requireAdmin(s.adminKeysList))
	mux.HandleFunc("POST /admin/api/keys", s.requireAdmin(s.adminKeysCreate))
	mux.HandleFunc("POST /admin/api/keys/update", s.requireAdmin(s.adminKeysUpdate))
	mux.HandleFunc("POST /admin/api/keys/rotate", s.requireAdmin(s.adminKeysRotate))
	mux.HandleFunc("POST /admin/api/keys/reset", s.requireAdmin(s.adminKeysReset))
	mux.HandleFunc("POST /admin/api/keys/delete", s.requireAdmin(s.adminKeysDelete))
	mux.HandleFunc("POST /admin/api/validate-model-dir", s.requireAdmin(s.adminValidateModelDir))
	mux.HandleFunc("GET /health", s.health)
	mux.HandleFunc("GET /ready", s.ready)
	mux.HandleFunc("GET /stats", s.stats)
	mux.HandleFunc("POST /v1/generate", s.requireAPIKey(s.generateJSON))
	mux.HandleFunc("POST /v1/batch", s.requireAPIKey(s.batchJSON))
	mux.HandleFunc("POST /v1/ocr", s.requireAPIKey(s.ocrMultipart))

	log.Printf("paddleocrvl-go serving on http://%s quant=%s backend=%s weight_path=%s weight_source=%s load_stats=%+v weights=%+v", *addr, rt.Quantization(), rt.Backend(), rt.WeightPath(), rt.WeightSource(), rt.LoadStats(), rt.WeightStats())
	srv := &http.Server{
		Addr:              *addr,
		Handler:           mux,
		ReadHeaderTimeout: 10 * time.Second,
	}
	if err := serveWithShutdownStop(srv, *shutdownTimeout, stop); err != nil {
		return err
	}
	return nil
}

func validateServerLimits(concurrency, maxBatchSize, maxInputLimit int, requestLimit, multipartMem int64) error {
	if concurrency < 1 {
		return fmt.Errorf("-concurrency must be >= 1")
	}
	if maxBatchSize < 0 {
		return fmt.Errorf("-max-batch-size must be >= 0")
	}
	if maxInputLimit < 0 {
		return fmt.Errorf("-max-input-tokens must be >= 0")
	}
	if requestLimit < 1 {
		return fmt.Errorf("-request-limit must be >= 1")
	}
	if multipartMem < 0 {
		return fmt.Errorf("-multipart-memory must be >= 0")
	}
	return nil
}

func serveWithShutdown(srv *http.Server, timeout time.Duration) error {
	return serveWithShutdownStop(srv, timeout, nil)
}

func serveWithShutdownStop(srv *http.Server, timeout time.Duration, stop <-chan struct{}) error {
	errCh := make(chan error, 1)
	go func() {
		err := srv.ListenAndServe()
		if err == http.ErrServerClosed {
			err = nil
		}
		errCh <- err
	}()
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, os.Interrupt, syscall.SIGTERM)
	defer signal.Stop(sigCh)
	select {
	case err := <-errCh:
		return err
	case sig := <-sigCh:
		log.Printf("received %s; shutting down", sig)
		return shutdownHTTPServer(srv, timeout, errCh)
	case <-stop:
		log.Printf("received service stop; shutting down")
		return shutdownHTTPServer(srv, timeout, errCh)
	}
}

func shutdownHTTPServer(srv *http.Server, timeout time.Duration, errCh <-chan error) error {
	ctx := context.Background()
	var cancel context.CancelFunc
	if timeout > 0 {
		ctx, cancel = context.WithTimeout(ctx, timeout)
	} else {
		ctx, cancel = context.WithCancel(ctx)
	}
	defer cancel()
	if err := srv.Shutdown(ctx); err != nil {
		return err
	}
	return <-errCh
}

func (s *server) health(w http.ResponseWriter, r *http.Request) {
	writeHealthJSON(w, healthResponse{
		Status:       "ok",
		Quantization: s.rt.Quantization(),
		WeightPath:   s.rt.WeightPath(),
		WeightSource: s.rt.WeightSource(),
		Backend:      s.rt.Backend(),
		VisionLoaded: s.rt.VisionLoaded(),
	})
}

func (s *server) ready(w http.ResponseWriter, r *http.Request) {
	status, body := s.readyState()
	writeReadyJSON(w, status, body)
}

func (s *server) readyState() (int, readyResponse) {
	if s.rt == nil || s.tok == nil || s.runSlots == nil || s.concurrency < 1 {
		return http.StatusServiceUnavailable, readyResponse{
			Status: "not_ready",
			Reason: "model, tokenizer, or inference slots not initialized",
		}
	}
	inFlight := len(s.runSlots)
	body := readyResponse{
		Status:         "ready",
		Quantization:   s.rt.Quantization(),
		WeightPath:     s.rt.WeightPath(),
		WeightSource:   s.rt.WeightSource(),
		Backend:        s.rt.Backend(),
		VisionLoaded:   s.rt.VisionLoaded(),
		Concurrency:    s.concurrency,
		InFlight:       inFlight,
		AvailableSlots: s.concurrency - inFlight,
	}
	return http.StatusOK, body
}

func (s *server) stats(w http.ResponseWriter, r *http.Request) {
	cfg := s.rt.Config()
	inFlight := len(s.runSlots)
	cmdPlan := s.rt.VulkanCommandPlan()
	cmdPlanErr := backend.ValidateVulkanCommandPlan(cmdPlan)
	writeJSON(w, http.StatusOK, statsResponse{
		Status:               "ok",
		UptimeSeconds:        int64(time.Since(s.started).Seconds()),
		Concurrency:          s.concurrency,
		InFlight:             inFlight,
		AvailableSlots:       s.concurrency - inFlight,
		RequestLimit:         s.requestLimit,
		MultipartMemory:      s.multipartMem,
		MaxNewLimit:          s.maxNewLimit,
		MaxInputTokens:       s.maxInputLimit,
		MaxBatchSize:         s.maxBatchSize,
		TimeoutSeconds:       int64(s.timeout.Seconds()),
		Quantization:         s.rt.Quantization(),
		RequestedQuant:       s.rt.RequestedQuantization(),
		WeightPath:           s.rt.WeightPath(),
		WeightSource:         s.rt.WeightSource(),
		WeightSHA256:         s.weightSHA256,
		Weights:              s.rt.WeightStats(),
		LoadStats:            s.rt.LoadStats(),
		Backend:              s.rt.Backend(),
		VisionLoaded:         s.rt.VisionLoaded(),
		Backends:             s.backendSel,
		VulkanPlans:          s.rt.VulkanPlans(),
		VulkanPlanSummary:    s.rt.VulkanPlanSummary(),
		VulkanExecutionGraph: s.rt.VulkanExecutionGraph(),
		VulkanPipelinePlan:   s.rt.VulkanPipelinePlan(),
		VulkanCommandPlan:    cmdPlan,
		VulkanCommandPlanOK:  cmdPlanErr == "",
		VulkanCommandPlanErr: cmdPlanErr,
		CPU:                  s.cpuInfo,
		Memory:               backend.Memory(),
		Requests: statsRequests{
			Queued:          s.metrics.queued.Load(),
			Started:         s.metrics.started.Load(),
			Succeeded:       s.metrics.succeeded.Load(),
			Failed:          s.metrics.failed.Load(),
			Canceled:        s.metrics.canceled.Load(),
			Batches:         s.metrics.batches.Load(),
			BatchItems:      s.metrics.batchItems.Load(),
			GeneratedTokens: s.metrics.generatedTokens.Load(),
			AvgLatencyMS:    s.avgLatencyMillis(),
			AvgQueueWaitMS:  s.avgQueueWaitMillis(),
			LastError:       s.lastError(),
		},
		Model: statsModel{
			VocabSize: cfg.VocabSize,
			Text: statsTextModel{
				Layers:  cfg.NumHiddenLayers,
				Hidden:  cfg.HiddenSize,
				Heads:   cfg.NumAttentionHeads,
				KVHeads: cfg.NumKeyValueHeads,
				HeadDim: cfg.HeadDim,
			},
			Vision: statsVisionModel{
				Layers: cfg.VisionConfig.NumHiddenLayers,
				Hidden: cfg.VisionConfig.HiddenSize,
				Heads:  cfg.VisionConfig.NumAttentionHeads,
				Patch:  cfg.VisionConfig.PatchSize,
			},
		},
		Cache: statsCache{
			TaskPrompts: len(s.taskIDs),
			Runtime:     s.rt.CacheStats(),
			Tokenizer:   s.tok.CacheStats(),
		},
	})
}

func (s *server) generateJSON(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, s.requestLimit)
	defer r.Body.Close()
	var req generateRequest
	if err := decodeJSON(r.Body, &req); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	ctx, cancel := s.requestContext(r)
	defer cancel()
	res, err := s.run(ctx, req)
	if err != nil {
		writeRunError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, res)
}

func (s *server) batchJSON(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, s.requestLimit)
	defer r.Body.Close()
	var req batchRequest
	if err := decodeJSON(r.Body, &req); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	if len(req.Requests) == 0 {
		writeError(w, http.StatusBadRequest, fmt.Errorf("requests must not be empty"))
		return
	}
	if s.maxBatchSize > 0 && len(req.Requests) > s.maxBatchSize {
		writeError(w, http.StatusBadRequest, fmt.Errorf("batch size %d exceeds limit %d", len(req.Requests), s.maxBatchSize))
		return
	}
	s.metrics.batches.Add(1)
	s.metrics.batchItems.Add(int64(len(req.Requests)))
	ctx, cancel := s.requestContext(r)
	defer cancel()
	ctx, cancelBatch := context.WithCancel(ctx)
	defer cancelBatch()
	s.metrics.queued.Add(int64(len(req.Requests)))
	if len(req.Requests) == 1 {
		if err := s.acquire(ctx); err != nil {
			s.recordFailure(err)
			writeRunError(w, fmt.Errorf("request 0: %w", err))
			return
		}
		res, err := s.runAcquired(ctx, req.Requests[0])
		s.release()
		if err != nil {
			writeRunError(w, fmt.Errorf("request 0: %w", err))
			return
		}
		writeJSON(w, http.StatusOK, batchResponse{Responses: []generateResponse{res}, Items: 1, GeneratedTokens: res.GeneratedTokens})
		return
	}
	responses := make([]generateResponse, len(req.Requests))
	errs := make([]error, len(req.Requests))
	jobs := make(chan int)
	var wg sync.WaitGroup
	workers := min(len(req.Requests), s.concurrency)
	if workers < 1 {
		workers = 1
	}
	for worker := 0; worker < workers; worker++ {
		wg.Add(1)
		go func() {
			for i := range jobs {
				item := req.Requests[i]
				if err := s.acquire(ctx); err != nil {
					s.recordFailure(err)
					errs[i] = err
					continue
				}
				res, err := s.runAcquired(ctx, item)
				s.release()
				if err != nil {
					errs[i] = err
					cancelBatch()
					continue
				}
				responses[i] = res
			}
			wg.Done()
		}()
	}
sendJobs:
	for i := range req.Requests {
		select {
		case jobs <- i:
		case <-ctx.Done():
			err := ctx.Err()
			for j := i; j < len(req.Requests); j++ {
				errs[j] = err
			}
			break sendJobs
		}
	}
	close(jobs)
	wg.Wait()
	generatedTokens := 0
	if i, err := firstBatchError(errs); err != nil {
		writeRunError(w, fmt.Errorf("request %d: %w", i, err))
		return
	}
	for i := range responses {
		generatedTokens += responses[i].GeneratedTokens
	}
	writeJSON(w, http.StatusOK, batchResponse{Responses: responses, Items: len(responses), GeneratedTokens: generatedTokens})
}

func firstBatchError(errs []error) (int, error) {
	first := -1
	var firstErr error
	for i, err := range errs {
		if err == nil {
			continue
		}
		if firstErr == nil {
			first, firstErr = i, err
		}
		if !errors.Is(err, context.Canceled) && !errors.Is(err, context.DeadlineExceeded) {
			return i, err
		}
	}
	return first, firstErr
}

func (s *server) ocrMultipart(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, s.requestLimit)
	if err := r.ParseMultipartForm(s.multipartMem); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	if r.MultipartForm != nil {
		defer r.MultipartForm.RemoveAll()
	}
	file, _, err := r.FormFile("image")
	if err != nil {
		writeError(w, http.StatusBadRequest, fmt.Errorf("missing multipart field image: %w", err))
		return
	}
	defer file.Close()
	data, err := io.ReadAll(file)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	task := formDefault(r, "task", "ocr")
	maxNew := formIntDefault(r, "max_new_tokens", 1024)
	req := generateRequest{
		Task:                task,
		imageData:           data,
		MaxNewTokens:        maxNew,
		Decode:              true,
		DecodeGeneratedOnly: true,
		SkipSpecial:         true,
	}
	ctx, cancel := s.requestContext(r)
	defer cancel()
	res, err := s.run(ctx, req)
	if err != nil {
		writeRunError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, res)
}

func (s *server) requestContext(r *http.Request) (context.Context, context.CancelFunc) {
	if s.timeout <= 0 {
		return context.WithCancel(r.Context())
	}
	return context.WithTimeout(r.Context(), s.timeout)
}

func (s *server) run(ctx context.Context, req generateRequest) (generateResponse, error) {
	s.metrics.queued.Add(1)
	if err := s.acquire(ctx); err != nil {
		s.recordFailure(err)
		return generateResponse{}, err
	}
	res, err := s.runAcquired(ctx, req)
	s.release()
	return res, err
}

func (s *server) runAcquired(ctx context.Context, req generateRequest) (generateResponse, error) {
	s.metrics.started.Add(1)
	started := time.Now()
	hasImage := requestHasImage(&req)
	imageData, err := imageBytes(&req)
	if err != nil {
		s.recordFailure(err)
		return generateResponse{}, err
	}
	ids, err := s.inputIDs(&req, hasImage)
	if err != nil {
		s.recordFailure(err)
		return generateResponse{}, err
	}
	if s.maxInputLimit > 0 && len(ids) > s.maxInputLimit {
		err := fmt.Errorf("input tokens %d exceeds limit %d", len(ids), s.maxInputLimit)
		s.recordFailure(err)
		return generateResponse{}, err
	}
	maxNew := req.MaxNewTokens
	if maxNew == 0 {
		maxNew = 16
	}
	if s.maxNewLimit > 0 && maxNew > s.maxNewLimit {
		err := fmt.Errorf("max_new_tokens %d exceeds limit %d", maxNew, s.maxNewLimit)
		s.recordFailure(err)
		return generateResponse{}, err
	}
	seed := req.Seed
	if seed == 0 && needsSamplingSeed(maxNew, req.Temperature, req.TopK) {
		seed = time.Now().UnixNano()
	}
	opts := model.GenerateOptions{
		MaxNewTokens: maxNew,
		Temperature:  req.Temperature,
		TopK:         req.TopK,
		Seed:         seed,
		EOSTokenIDs:  req.EOSTokenIDs,
	}
	if err := model.ValidateGenerateOptions(opts); err != nil {
		s.recordFailure(err)
		return generateResponse{}, err
	}
	var res model.GenerateResult
	if len(imageData) > 0 {
		res, err = s.rt.GenerateWithImageBytesOptions(ctx, ids, imageData, opts)
	} else if req.ImagePath != "" {
		res, err = s.rt.GenerateWithImageOptions(ctx, ids, req.ImagePath, opts)
	} else {
		res, err = s.rt.GenerateWithOptions(ctx, ids, opts)
	}
	if err != nil {
		s.recordFailure(err)
		return generateResponse{}, err
	}
	resp := generateResponse{Tokens: res.Tokens, PromptTokens: res.PromptTokens, GeneratedTokens: max(0, len(res.Tokens)-res.PromptTokens)}
	if req.Decode {
		toDecode := decodeTokenRange(res.Tokens, res.PromptTokens, req.DecodeGeneratedOnly)
		resp.Text = s.tok.Decode(toDecode, req.SkipSpecial)
	}
	s.metrics.succeeded.Add(1)
	s.metrics.generatedTokens.Add(int64(resp.GeneratedTokens))
	s.metrics.latencyNanos.Add(time.Since(started).Nanoseconds())
	return resp, nil
}

func decodeTokenRange(tokens []int, promptTokens int, generatedOnly bool) []int {
	if !generatedOnly {
		return tokens
	}
	if promptTokens <= 0 {
		return tokens
	}
	if promptTokens >= len(tokens) {
		return nil
	}
	return tokens[promptTokens:]
}

func needsSamplingSeed(maxNew int, temperature float64, topK int) bool {
	return maxNew > 0 && temperature > 0 && topK != 1
}

func (s *server) recordFailure(err error) {
	s.metrics.failed.Add(1)
	if err != nil {
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			s.metrics.canceled.Add(1)
		}
		s.metrics.lastError.Store(err.Error())
	}
}

func (s *server) avgLatencyMillis() int64 {
	ok := s.metrics.succeeded.Load()
	if ok == 0 {
		return 0
	}
	return s.metrics.latencyNanos.Load() / ok / int64(time.Millisecond)
}

func (s *server) avgQueueWaitMillis() int64 {
	started := s.metrics.started.Load()
	if started == 0 {
		return 0
	}
	return s.metrics.queueWaitNanos.Load() / started / int64(time.Millisecond)
}

func (s *server) lastError() string {
	v := s.metrics.lastError.Load()
	if v == nil {
		return ""
	}
	return v.(string)
}

func (s *server) acquire(ctx context.Context) error {
	select {
	case s.runSlots <- struct{}{}:
		return nil
	default:
	}
	started := time.Now()
	select {
	case s.runSlots <- struct{}{}:
		s.metrics.queueWaitNanos.Add(time.Since(started).Nanoseconds())
		return nil
	case <-ctx.Done():
		s.metrics.queueWaitNanos.Add(time.Since(started).Nanoseconds())
		return ctx.Err()
	}
}

func (s *server) release() {
	<-s.runSlots
}

func requestHasImage(req *generateRequest) bool {
	return req.ImagePath != "" || req.ImageBase64 != "" || len(req.imageData) > 0
}

func imageBytes(req *generateRequest) ([]byte, error) {
	if len(req.imageData) > 0 {
		if req.ImagePath != "" || req.ImageBase64 != "" {
			return nil, fmt.Errorf("provide only one image source")
		}
		return req.imageData, nil
	}
	if req.ImageBase64 == "" {
		return nil, nil
	}
	if req.ImagePath != "" {
		return nil, fmt.Errorf("provide only one of image_path or image_base64")
	}
	data := req.ImageBase64
	if strings.HasPrefix(data, "data:") {
		if i := strings.IndexByte(data, ','); i >= 0 {
			data = data[i+1:]
		}
	}
	raw := make([]byte, base64.StdEncoding.DecodedLen(len(data)))
	src := unsafe.Slice(unsafe.StringData(data), len(data))
	n, err := base64.StdEncoding.Decode(raw, src)
	if err != nil {
		return nil, fmt.Errorf("decode image_base64: %w", err)
	}
	return raw[:n], nil
}

func (s *server) inputIDs(req *generateRequest, hasImage bool) ([]int, error) {
	if len(req.Tokens) > 0 {
		return append([]int(nil), req.Tokens...), nil
	}
	if req.Task != "" {
		key, err := taskKey(req.Task, hasImage)
		if err != nil {
			return nil, err
		}
		if ids, ok := s.taskIDs[key]; ok {
			return ids, nil
		}
		p, err := taskPrompt(req.Task)
		if err != nil {
			return nil, err
		}
		return s.tok.EncodeReadOnly(chatPrompt(p, hasImage)), nil
	}
	if req.Prompt != "" {
		return s.tok.EncodeReadOnly(req.Prompt), nil
	}
	return nil, fmt.Errorf("provide tokens, prompt, or task")
}

func buildTaskIDs(tok *tokenizer.Tokenizer) map[string][]int {
	out := make(map[string][]int, 8)
	for _, task := range []string{"ocr", "table", "formula", "chart"} {
		p, err := taskPrompt(task)
		if err != nil {
			continue
		}
		for _, withImage := range []bool{false, true} {
			key, _ := taskKey(task, withImage)
			out[key] = tok.EncodeReadOnly(chatPrompt(p, withImage))
		}
	}
	return out
}

func taskKey(task string, withImage bool) (string, error) {
	t, _, ok := canonicalTask(task)
	if !ok {
		return "", fmt.Errorf("unknown task %q", task)
	}
	return canonicalTaskKey(t, withImage), nil
}

func taskPrompt(task string) (string, error) {
	_, p, ok := canonicalTask(task)
	if ok {
		return p, nil
	}
	return "", fmt.Errorf("unknown task %q", task)
}

func canonicalTask(task string) (key, prompt string, ok bool) {
	task = trimASCIIForm(task)
	switch len(task) {
	case 3:
		if asciiEqualFold(task, "ocr") {
			return "ocr", "OCR:", true
		}
	case 5:
		if asciiEqualFold(task, "table") {
			return "table", "Table Recognition:", true
		}
		if asciiEqualFold(task, "chart") {
			return "chart", "Chart Recognition:", true
		}
	case 7:
		if asciiEqualFold(task, "formula") {
			return "formula", "Formula Recognition:", true
		}
	}
	return "", "", false
}

func canonicalTaskKey(key string, withImage bool) string {
	switch key {
	case "ocr":
		if withImage {
			return "ocr:image"
		}
		return "ocr:text"
	case "table":
		if withImage {
			return "table:image"
		}
		return "table:text"
	case "formula":
		if withImage {
			return "formula:image"
		}
		return "formula:text"
	case "chart":
		if withImage {
			return "chart:image"
		}
		return "chart:text"
	default:
		if withImage {
			return key + ":image"
		}
		return key + ":text"
	}
}

func asciiEqualFold(s, lower string) bool {
	if len(s) != len(lower) {
		return false
	}
	for i := 0; i < len(s); i++ {
		c := s[i]
		if 'A' <= c && c <= 'Z' {
			c += 'a' - 'A'
		}
		if c != lower[i] {
			return false
		}
	}
	return true
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

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header()["Content-Type"] = contentTypeJSONHeader
	w.WriteHeader(status)
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false)
	_ = enc.Encode(v)
}

func writeOKJSON(w http.ResponseWriter) {
	w.Header()["Content-Type"] = contentTypeJSONHeader
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(okJSONBytes)
}

func writeOKErrorJSON(w http.ResponseWriter, msg string) {
	w.Header()["Content-Type"] = contentTypeJSONHeader
	w.WriteHeader(http.StatusOK)
	p := errorBufferPool.Get().(*[]byte)
	buf := (*p)[:0]
	if len(msg)+24 > cap(buf) {
		buf = make([]byte, 0, len(msg)+24)
	}
	buf = append(buf, `{"ok":false,"error":`...)
	buf = strconv.AppendQuote(buf, msg)
	buf = append(buf, "}\n"...)
	_, _ = w.Write(buf)
	*p = buf
	errorBufferPool.Put(p)
}

func writeAdminSessionJSON(w http.ResponseWriter, initialized, authenticated bool) {
	w.Header()["Content-Type"] = contentTypeJSONHeader
	w.WriteHeader(http.StatusOK)
	idx := 0
	if initialized {
		idx = 1
	}
	if authenticated {
		idx += 2
	}
	_, _ = w.Write(adminSessionJSON[idx])
}

func writeHealthJSON(w http.ResponseWriter, res healthResponse) {
	w.Header()["Content-Type"] = contentTypeJSONHeader
	w.WriteHeader(http.StatusOK)
	p := errorBufferPool.Get().(*[]byte)
	buf := (*p)[:0]
	need := len(res.Status) + len(res.Quantization) + len(res.WeightPath) + len(res.WeightSource) + len(res.Backend) + 118
	if cap(buf) < need {
		buf = make([]byte, 0, need)
	}
	buf = append(buf, `{"status":`...)
	buf = strconv.AppendQuote(buf, res.Status)
	buf = append(buf, `,"quantization":`...)
	buf = strconv.AppendQuote(buf, res.Quantization)
	buf = append(buf, `,"weight_path":`...)
	buf = strconv.AppendQuote(buf, res.WeightPath)
	buf = append(buf, `,"weight_source":`...)
	buf = strconv.AppendQuote(buf, res.WeightSource)
	buf = append(buf, `,"backend":`...)
	buf = strconv.AppendQuote(buf, res.Backend)
	buf = append(buf, `,"vision_loaded":`...)
	if res.VisionLoaded {
		buf = append(buf, "true}\n"...)
	} else {
		buf = append(buf, "false}\n"...)
	}
	_, _ = w.Write(buf)
	*p = buf
	errorBufferPool.Put(p)
}

func writeReadyJSON(w http.ResponseWriter, status int, res readyResponse) {
	w.Header()["Content-Type"] = contentTypeJSONHeader
	w.WriteHeader(status)
	p := errorBufferPool.Get().(*[]byte)
	buf := (*p)[:0]
	if cap(buf) < 256+len(res.Reason)+len(res.Quantization)+len(res.WeightPath)+len(res.WeightSource)+len(res.Backend) {
		buf = make([]byte, 0, 256+len(res.Reason)+len(res.Quantization)+len(res.WeightPath)+len(res.WeightSource)+len(res.Backend))
	}
	buf = append(buf, `{"status":`...)
	buf = strconv.AppendQuote(buf, res.Status)
	if res.Reason != "" {
		buf = append(buf, `,"reason":`...)
		buf = strconv.AppendQuote(buf, res.Reason)
	}
	if res.Quantization != "" {
		buf = append(buf, `,"quantization":`...)
		buf = strconv.AppendQuote(buf, res.Quantization)
	}
	if res.WeightPath != "" {
		buf = append(buf, `,"weight_path":`...)
		buf = strconv.AppendQuote(buf, res.WeightPath)
	}
	if res.WeightSource != "" {
		buf = append(buf, `,"weight_source":`...)
		buf = strconv.AppendQuote(buf, res.WeightSource)
	}
	if res.Backend != "" {
		buf = append(buf, `,"backend":`...)
		buf = strconv.AppendQuote(buf, res.Backend)
	}
	if res.VisionLoaded {
		buf = append(buf, `,"vision_loaded":true`...)
	}
	if res.Concurrency != 0 {
		buf = append(buf, `,"concurrency":`...)
		buf = strconv.AppendInt(buf, int64(res.Concurrency), 10)
	}
	if res.InFlight != 0 {
		buf = append(buf, `,"in_flight":`...)
		buf = strconv.AppendInt(buf, int64(res.InFlight), 10)
	}
	if res.AvailableSlots != 0 {
		buf = append(buf, `,"available_slots":`...)
		buf = strconv.AppendInt(buf, int64(res.AvailableSlots), 10)
	}
	buf = append(buf, "}\n"...)
	_, _ = w.Write(buf)
	*p = buf
	errorBufferPool.Put(p)
}

func decodeJSON(r io.Reader, v any) error {
	dec := json.NewDecoder(r)
	if err := dec.Decode(v); err != nil {
		return err
	}
	var extra any
	if err := dec.Decode(&extra); err != io.EOF {
		if err == nil {
			return fmt.Errorf("request body must contain only one JSON value")
		}
		return err
	}
	return nil
}

func writeError(w http.ResponseWriter, status int, err error) {
	w.Header()["Content-Type"] = contentTypeJSONHeader
	w.WriteHeader(status)
	msg := err.Error()
	p := errorBufferPool.Get().(*[]byte)
	buf := (*p)[:0]
	if len(msg)+14 > cap(buf) {
		buf = make([]byte, 0, len(msg)+14)
	}
	buf = append(buf, `{"error":`...)
	buf = strconv.AppendQuote(buf, msg)
	buf = append(buf, "}\n"...)
	_, _ = w.Write(buf)
	if cap(buf) <= 512 {
		*p = buf[:0]
		errorBufferPool.Put(p)
	}
}

func writeRunError(w http.ResponseWriter, err error) {
	writeError(w, statusForRunError(err), err)
}

func statusForRunError(err error) int {
	if errors.Is(err, context.DeadlineExceeded) {
		return http.StatusGatewayTimeout
	}
	if errors.Is(err, context.Canceled) {
		return http.StatusRequestTimeout
	}
	return http.StatusBadRequest
}

func formDefault(r *http.Request, key, fallback string) string {
	v := trimASCIIForm(r.FormValue(key))
	if v == "" {
		return fallback
	}
	return v
}

func formIntDefault(r *http.Request, key string, fallback int) int {
	s := r.FormValue(key)
	i, n := 0, len(s)
	for i < n && isASCIISpace(s[i]) {
		i++
	}
	if i == n {
		return fallback
	}
	v := 0
	for ; i < n; i++ {
		c := s[i]
		if c < '0' || c > '9' {
			break
		}
		v = v*10 + int(c-'0')
		if v <= 0 {
			return fallback
		}
	}
	for i < n && isASCIISpace(s[i]) {
		i++
	}
	if i != n || v <= 0 {
		return fallback
	}
	return v
}

func isASCIISpace(c byte) bool {
	return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f'
}

func trimASCIIForm(s string) string {
	if len(s) == 0 || (!isASCIISpace(s[0]) && !isASCIISpace(s[len(s)-1])) {
		return s
	}
	start, end := 0, len(s)
	for start < end && isASCIISpace(s[start]) {
		start++
	}
	for end > start && isASCIISpace(s[end-1]) {
		end--
	}
	return s[start:end]
}
