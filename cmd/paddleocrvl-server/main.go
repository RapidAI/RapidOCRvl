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
	"runtime"
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
	autoBatchSize int
	autoBatchWait time.Duration
	started       time.Time
	concurrency   int
	metrics       metrics
	weightSHA256  string
	backendSel    backend.Selection
	cpuInfo       backend.CPUInfo
	infer         inferenceFunc
	batcher       *requestBatcher
}

type inferenceFunc func(context.Context, generateRequest, []int, []byte, model.GenerateOptions) (model.GenerateResult, error)

type batchItem struct {
	ctx  context.Context
	req  generateRequest
	done chan batchItemResult
}

type batchItemResult struct {
	res generateResponse
	err error
}

type requestBatcher struct {
	s       *server
	maxSize int
	wait    time.Duration
	ch      chan batchItem
	stop    chan struct{}
	once    sync.Once
	closed  atomic.Bool
	done    chan struct{}
	procWG  sync.WaitGroup
	procSem chan struct{}
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

const maxImagePayloadBytes = 128 << 20

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

const maxResponseBufferPoolCap = 4 << 10

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
	AutoBatchSize        int                          `json:"auto_batch_size"`
	AutoBatchWaitMS      int64                        `json:"auto_batch_wait_ms"`
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
	readTimeout := fs.Duration("read-timeout", 60*time.Second, "maximum time to read the full request, including body; 0 disables")
	idleTimeout := fs.Duration("idle-timeout", 120*time.Second, "maximum keep-alive idle time; 0 disables")
	shutdownTimeout := fs.Duration("shutdown-timeout", 30*time.Second, "graceful shutdown timeout")
	requestLimit := fs.Int64("request-limit", 128<<20, "max request body bytes")
	multipartMem := fs.Int64("multipart-memory", 32<<20, "max memory used while parsing multipart forms")
	maxNewLimit := fs.Int("max-new-limit", 4096, "maximum max_new_tokens per request; 0 disables")
	maxInputLimit := fs.Int("max-input-tokens", 0, "maximum prompt/input tokens per request; 0 disables")
	maxBatchSize := fs.Int("max-batch-size", 0, "maximum /v1/batch request count; 0 disables")
	autoBatchSize := fs.Int("auto-batch-size", 2, "max concurrent /v1/generate requests to coalesce; 0 or 1 disables")
	autoBatchWait := fs.Duration("auto-batch-wait", 2*time.Millisecond, "maximum wait to collect /v1/generate auto-batch requests")
	concurrency := fs.Int("concurrency", defaultInferenceConcurrency(), "max concurrent inference requests")
	gomaxprocs := fs.Int("gomaxprocs", 0, "set Go GOMAXPROCS; 0 keeps current value")
	gcPercent := fs.Int("gc-percent", 0, "set Go GC percent; 0 keeps current value, -1 disables GC")
	preloadVision := fs.Bool("preload-vision", false, "load vision weights at startup")
	warmup := fs.Bool("warmup", false, "run one text-token warmup during startup")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if err := validateServerLimits(*concurrency, *maxBatchSize, *autoBatchSize, *maxInputLimit, *requestLimit, *multipartMem, *timeout, *readTimeout, *idleTimeout, *shutdownTimeout, *autoBatchWait); err != nil {
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
	admin := loadAdminState(*adminConfigPath)
	if admin.loadErr != nil {
		return admin.loadErr
	}
	effectiveAutoBatchSize := autoBatchSizeForConcurrency(*autoBatchSize, *concurrency)
	s := &server{
		rt:            rt,
		tok:           tok,
		taskIDs:       buildTaskIDs(tok),
		admin:         admin,
		modelDir:      *modelDir,
		timeout:       *timeout,
		requestLimit:  *requestLimit,
		multipartMem:  *multipartMem,
		maxNewLimit:   *maxNewLimit,
		maxInputLimit: *maxInputLimit,
		maxBatchSize:  *maxBatchSize,
		autoBatchSize: effectiveAutoBatchSize,
		autoBatchWait: *autoBatchWait,
		runSlots:      make(chan struct{}, *concurrency),
		started:       time.Now(),
		concurrency:   *concurrency,
		weightSHA256:  fileutil.SHA256(rt.WeightPath()),
		backendSel:    backendSel,
		cpuInfo:       backend.CPU(),
	}
	if effectiveAutoBatchSize > 1 {
		s.batcher = newRequestBatcher(s, effectiveAutoBatchSize, *autoBatchWait)
		defer s.batcher.close()
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
		ReadTimeout:       *readTimeout,
		IdleTimeout:       *idleTimeout,
	}
	if err := serveWithShutdownStop(srv, *shutdownTimeout, stop); err != nil {
		return err
	}
	return nil
}

func validateServerLimits(concurrency, maxBatchSize, autoBatchSize, maxInputLimit int, requestLimit, multipartMem int64, timeout, readTimeout, idleTimeout, shutdownTimeout, autoBatchWait time.Duration) error {
	if concurrency < 1 {
		return fmt.Errorf("-concurrency must be >= 1")
	}
	if maxBatchSize < 0 {
		return fmt.Errorf("-max-batch-size must be >= 0")
	}
	if autoBatchSize < 0 {
		return fmt.Errorf("-auto-batch-size must be >= 0")
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
	if timeout < 0 {
		return fmt.Errorf("-timeout must be >= 0")
	}
	if readTimeout < 0 {
		return fmt.Errorf("-read-timeout must be >= 0")
	}
	if idleTimeout < 0 {
		return fmt.Errorf("-idle-timeout must be >= 0")
	}
	if shutdownTimeout < 0 {
		return fmt.Errorf("-shutdown-timeout must be >= 0")
	}
	if autoBatchWait < 0 {
		return fmt.Errorf("-auto-batch-wait must be >= 0")
	}
	return nil
}

func defaultInferenceConcurrency() int {
	if runtime.NumCPU() < 2 {
		return 1
	}
	return 2
}

func autoBatchSizeForConcurrency(autoBatchSize, concurrency int) int {
	if autoBatchSize <= 1 || concurrency <= 1 {
		return 0
	}
	return autoBatchSize
}

func newRequestBatcher(s *server, maxSize int, wait time.Duration) *requestBatcher {
	b := &requestBatcher{
		s:       s,
		maxSize: maxSize,
		wait:    wait,
		ch:      make(chan batchItem, maxSize*4),
		stop:    make(chan struct{}),
		done:    make(chan struct{}),
		procSem: make(chan struct{}, batchProcessLimit(s, maxSize)),
	}
	go b.loop()
	return b
}

func batchProcessLimit(s *server, batchSize int) int {
	if batchSize < 1 {
		batchSize = 1
	}
	concurrency := s.effectiveConcurrency()
	limit := (concurrency + batchSize - 1) / batchSize
	if limit < 1 {
		return 1
	}
	return limit
}

func (b *requestBatcher) submit(ctx context.Context, req generateRequest) (generateResponse, error) {
	if b.closed.Load() {
		return generateResponse{}, context.Canceled
	}
	item := batchItem{ctx: ctx, req: req, done: make(chan batchItemResult, 1)}
	select {
	case b.ch <- item:
	case <-ctx.Done():
		return generateResponse{}, ctx.Err()
	case <-b.stop:
		return generateResponse{}, context.Canceled
	}
	select {
	case res := <-item.done:
		return res.res, res.err
	case <-ctx.Done():
		return generateResponse{}, ctx.Err()
	case <-b.stop:
		return generateResponse{}, context.Canceled
	}
}

func (b *requestBatcher) close() {
	b.once.Do(func() {
		b.closed.Store(true)
		close(b.stop)
	})
	<-b.done
}

func (b *requestBatcher) loop() {
	defer func() {
		b.procWG.Wait()
		close(b.done)
	}()
	for {
		select {
		case first := <-b.ch:
			batch := b.collect(first)
			if !b.acquireProcessSlot() {
				b.completeBatch(batch, context.Canceled)
				return
			}
			b.procWG.Add(1)
			go func() {
				defer b.procWG.Done()
				defer b.releaseProcessSlot()
				b.process(batch)
			}()
		case <-b.stop:
			return
		}
	}
}

func (b *requestBatcher) acquireProcessSlot() bool {
	select {
	case b.procSem <- struct{}{}:
		return true
	case <-b.stop:
		return false
	}
}

func (b *requestBatcher) releaseProcessSlot() {
	<-b.procSem
}

func (b *requestBatcher) collect(first batchItem) []batchItem {
	batch := make([]batchItem, 0, b.maxSize)
	batch = append(batch, first)
	if b.maxSize <= 1 {
		return batch
	}
	if b.wait <= 0 {
		return b.collectAvailable(batch)
	}
	timer := time.NewTimer(b.wait)
	defer timer.Stop()
	for len(batch) < b.maxSize {
		select {
		case item := <-b.ch:
			batch = append(batch, item)
		case <-timer.C:
			return batch
		case <-b.stop:
			return batch
		}
	}
	return batch
}

func (b *requestBatcher) collectAvailable(batch []batchItem) []batchItem {
	for len(batch) < b.maxSize {
		select {
		case item := <-b.ch:
			batch = append(batch, item)
		default:
			return batch
		}
	}
	return batch
}

func (b *requestBatcher) process(batch []batchItem) {
	if len(batch) == 0 {
		return
	}
	select {
	case <-b.stop:
		b.completeBatch(batch, context.Canceled)
		return
	default:
	}
	batch = b.cancelInactive(batch)
	if len(batch) == 0 {
		return
	}
	b.s.metrics.batches.Add(1)
	b.s.metrics.batchItems.Add(int64(len(batch)))
	b.s.metrics.queued.Add(int64(len(batch)))
	var wg sync.WaitGroup
	workers := min(len(batch), b.s.effectiveConcurrency())
	jobs := make(chan int)
	for worker := 0; worker < workers; worker++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range jobs {
				item := batch[i]
				select {
				case <-b.stop:
					item.done <- batchItemResult{err: context.Canceled}
					continue
				case <-item.ctx.Done():
					item.done <- batchItemResult{err: item.ctx.Err()}
					continue
				default:
				}
				res, err := b.s.runBatchItem(item.ctx, item.req)
				item.done <- batchItemResult{res: res, err: err}
			}
		}()
	}
	for i := range batch {
		jobs <- i
	}
	close(jobs)
	wg.Wait()
}

func (b *requestBatcher) cancelInactive(batch []batchItem) []batchItem {
	active := batch[:0]
	for _, item := range batch {
		select {
		case <-item.ctx.Done():
			item.done <- batchItemResult{err: item.ctx.Err()}
		default:
			active = append(active, item)
		}
	}
	return active
}

func (b *requestBatcher) completeBatch(batch []batchItem, err error) {
	for _, item := range batch {
		item.done <- batchItemResult{err: err}
	}
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
	if s.rt == nil || s.tok == nil || s.runSlots == nil {
		return http.StatusServiceUnavailable, readyResponse{
			Status: "not_ready",
			Reason: "model, tokenizer, or inference slots not initialized",
		}
	}
	inFlight := len(s.runSlots)
	concurrency := s.effectiveConcurrency()
	body := readyResponse{
		Status:         "ready",
		Quantization:   s.rt.Quantization(),
		WeightPath:     s.rt.WeightPath(),
		WeightSource:   s.rt.WeightSource(),
		Backend:        s.rt.Backend(),
		VisionLoaded:   s.rt.VisionLoaded(),
		Concurrency:    concurrency,
		InFlight:       inFlight,
		AvailableSlots: concurrency - inFlight,
	}
	return http.StatusOK, body
}

func (s *server) stats(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, s.statsSnapshot())
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
	res, err := s.runGenerate(ctx, req)
	if err != nil {
		writeRunError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, res)
}

func (s *server) runGenerate(ctx context.Context, req generateRequest) (generateResponse, error) {
	if s.batcher != nil {
		return s.batcher.submit(ctx, req)
	}
	return s.run(ctx, req)
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
		res, err := s.runBatchItem(ctx, req.Requests[0])
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
	workers := min(len(req.Requests), s.effectiveConcurrency())
	for worker := 0; worker < workers; worker++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range jobs {
				item := req.Requests[i]
				res, err := s.runBatchItem(ctx, item)
				if err != nil {
					errs[i] = err
					cancelBatch()
					continue
				}
				responses[i] = res
			}
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
	data, err := readAllLimited(file, maxImagePayloadBytes, "image")
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
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
	res, err := s.runGenerate(ctx, req)
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
	return s.runBatchItem(ctx, req)
}

func (s *server) runBatchItem(ctx context.Context, req generateRequest) (generateResponse, error) {
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
	if err := validateRequestTokenIDs(ids); err != nil {
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
	res, err := s.inferModel(ctx, req, ids, imageData, opts)
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

func (s *server) inferModel(ctx context.Context, req generateRequest, ids []int, imageData []byte, opts model.GenerateOptions) (model.GenerateResult, error) {
	if s.infer != nil {
		return s.infer(ctx, req, ids, imageData, opts)
	}
	if len(imageData) > 0 {
		return s.rt.GenerateWithImageBytesOptions(ctx, ids, imageData, opts)
	}
	if req.ImagePath != "" {
		return s.rt.GenerateWithImageOptions(ctx, ids, req.ImagePath, opts)
	}
	return s.rt.GenerateWithOptions(ctx, ids, opts)
}

func validateRequestTokenIDs(ids []int) error {
	for _, id := range ids {
		if id < 0 {
			return fmt.Errorf("input tokens must not contain negative ids")
		}
	}
	return nil
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
	return avgMillis(s.metrics.latencyNanos.Load(), s.metrics.succeeded.Load())
}

func (s *server) avgQueueWaitMillis() int64 {
	return avgMillis(s.metrics.queueWaitNanos.Load(), s.metrics.started.Load())
}

func avgMillis(totalNanos, count int64) int64 {
	if count == 0 {
		return 0
	}
	return totalNanos / count / int64(time.Millisecond)
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

func (s *server) effectiveConcurrency() int {
	if s.concurrency > 0 {
		return s.concurrency
	}
	if s.runSlots != nil && cap(s.runSlots) > 0 {
		return cap(s.runSlots)
	}
	return 1
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
	decodedLen := base64DecodedLen(data)
	if decodedLen > maxImagePayloadBytes {
		decodedLen = base64DecodedLenIgnoringCRLF(data)
	}
	if decodedLen > maxImagePayloadBytes {
		return nil, fmt.Errorf("image_base64 too large: decoded size exceeds %d bytes", maxImagePayloadBytes)
	}
	raw := make([]byte, decodedLen)
	src := unsafe.Slice(unsafe.StringData(data), len(data))
	n, err := base64.StdEncoding.Decode(raw, src)
	if err != nil {
		return nil, fmt.Errorf("decode image_base64: %w", err)
	}
	if n > maxImagePayloadBytes {
		return nil, fmt.Errorf("image_base64 too large: decoded size exceeds %d bytes", maxImagePayloadBytes)
	}
	return raw[:n], nil
}

func base64DecodedLen(s string) int {
	n := base64.StdEncoding.DecodedLen(len(s))
	for len(s) > 0 && s[len(s)-1] == '=' {
		n--
		s = s[:len(s)-1]
	}
	if n < 0 {
		return 0
	}
	return n
}

func base64DecodedLenIgnoringCRLF(s string) int {
	encoded := 0
	padding := 0
	for i := 0; i < len(s); i++ {
		switch s[i] {
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

func readAllLimited(r io.Reader, maxBytes int64, label string) ([]byte, error) {
	data, err := io.ReadAll(io.LimitReader(r, maxBytes+1))
	if err != nil {
		return nil, err
	}
	if int64(len(data)) > maxBytes {
		return nil, fmt.Errorf("%s too large: exceeds %d bytes", label, maxBytes)
	}
	return data, nil
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
	putResponseBuffer(p, buf)
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
	putResponseBuffer(p, buf)
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
	putResponseBuffer(p, buf)
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
	putResponseBuffer(p, buf)
}

func putResponseBuffer(p *[]byte, buf []byte) {
	if cap(buf) > maxResponseBufferPoolCap {
		return
	}
	*p = buf[:0]
	errorBufferPool.Put(p)
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
