package main

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"
	"unsafe"

	"github.com/wailsapp/wails/v2/pkg/runtime"
)

type App struct {
	ctx           context.Context
	client        *http.Client
	cancelMu      sync.Mutex
	cancelSeq     int64
	currentCancel context.CancelFunc
	currentSeq    int64
}

type OCRRequest struct {
	APIURL       string `json:"apiUrl"`
	APIKey       string `json:"apiKey"`
	ImagePath    string `json:"imagePath"`
	Task         string `json:"task"`
	MaxNewTokens int    `json:"maxNewTokens"`
	TimeoutSecs  int    `json:"timeoutSecs"`
}

type OCRResult struct {
	Text            string `json:"text"`
	PromptTokens    int    `json:"promptTokens"`
	GeneratedTokens int    `json:"generatedTokens"`
	Raw             string `json:"raw"`
}

type ReadyResult struct {
	Status         string `json:"status"`
	Backend        string `json:"backend"`
	Quantization   string `json:"quantization"`
	WeightSource   string `json:"weightSource"`
	VisionLoaded   bool   `json:"visionLoaded"`
	Concurrency    int    `json:"concurrency"`
	AvailableSlots int    `json:"availableSlots"`
	Raw            string `json:"raw"`
}

type generateResponse struct {
	Tokens          []int  `json:"tokens"`
	PromptTokens    int    `json:"prompt_tokens"`
	GeneratedTokens int    `json:"generated_tokens"`
	Text            string `json:"text,omitempty"`
}

type readyResponse struct {
	Status         string `json:"status"`
	Reason         string `json:"reason,omitempty"`
	Quantization   string `json:"quantization,omitempty"`
	WeightSource   string `json:"weight_source,omitempty"`
	Backend        string `json:"backend,omitempty"`
	VisionLoaded   bool   `json:"vision_loaded,omitempty"`
	Concurrency    int    `json:"concurrency,omitempty"`
	AvailableSlots int    `json:"available_slots,omitempty"`
}

type errorResponse struct {
	Error string `json:"error"`
}

const (
	maxPreviewBytes   = 20 << 20
	maxUploadBytes    = 64 << 20
	maxOpenTextBytes  = 2 << 20
	maxResponseBytes  = 8 << 20
	maxErrorTextBytes = 4096
	defaultMaxNew     = 1024
	maxClientMaxNew   = 4096
	defaultTimeout    = 600
	maxTimeoutSecs    = 3600
)

func NewApp() *App {
	return &App{client: &http.Client{}}
}

func (a *App) startup(ctx context.Context) {
	a.ctx = ctx
}

func (a *App) SelectImage() (string, error) {
	return runtime.OpenFileDialog(a.ctx, runtime.OpenDialogOptions{
		Title: "Select image",
		Filters: []runtime.FileFilter{
			{DisplayName: "Images (*.png;*.jpg;*.jpeg;*.webp;*.bmp)", Pattern: "*.png;*.jpg;*.jpeg;*.webp;*.bmp"},
			{DisplayName: "All Files (*.*)", Pattern: "*.*"},
		},
	})
}

func (a *App) SelectImages() ([]string, error) {
	return runtime.OpenMultipleFilesDialog(a.ctx, runtime.OpenDialogOptions{
		Title: "Select images",
		Filters: []runtime.FileFilter{
			{DisplayName: "Images (*.png;*.jpg;*.jpeg;*.webp;*.bmp)", Pattern: "*.png;*.jpg;*.jpeg;*.webp;*.bmp"},
			{DisplayName: "All Files (*.*)", Pattern: "*.*"},
		},
	})
}

func (a *App) ImageDataURL(imagePath string) (string, error) {
	imagePath = strings.TrimSpace(imagePath)
	if imagePath == "" {
		return "", fmt.Errorf("select an image")
	}
	if err := checkFileSize(imagePath, maxPreviewBytes, "preview image"); err != nil {
		return "", err
	}
	data, err := os.ReadFile(imagePath)
	if err != nil {
		return "", fmt.Errorf("read image: %w", err)
	}
	mimeType := http.DetectContentType(data)
	return imageDataURLFromBytes(data, mimeType), nil
}

func imageDataURLFromBytes(data []byte, mimeType string) string {
	prefix := "data:" + mimeType + ";base64,"
	n := len(prefix) + base64.StdEncoding.EncodedLen(len(data))
	buf := make([]byte, 0, n)
	buf = append(buf, prefix...)
	buf = base64.StdEncoding.AppendEncode(buf, data)
	return unsafe.String(unsafe.SliceData(buf), len(buf))
}

func (a *App) SaveText(defaultName, content string) (string, error) {
	if strings.TrimSpace(content) == "" {
		return "", fmt.Errorf("nothing to save")
	}
	path, err := runtime.SaveFileDialog(a.ctx, runtime.SaveDialogOptions{
		Title:           "Save result",
		DefaultFilename: safeDefaultFilename(defaultName, "paddleocrvl-result.txt"),
		Filters: []runtime.FileFilter{
			{DisplayName: "Text (*.txt)", Pattern: "*.txt"},
			{DisplayName: "JSON (*.json)", Pattern: "*.json"},
			{DisplayName: "All Files (*.*)", Pattern: "*.*"},
		},
	})
	if err != nil {
		return "", err
	}
	if path == "" {
		return "", nil
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		return "", fmt.Errorf("save result: %w", err)
	}
	return path, nil
}

func (a *App) OpenText(title string) (string, error) {
	path, err := runtime.OpenFileDialog(a.ctx, runtime.OpenDialogOptions{
		Title: defaultString(title, "Open file"),
		Filters: []runtime.FileFilter{
			{DisplayName: "JSON (*.json)", Pattern: "*.json"},
			{DisplayName: "Text (*.txt)", Pattern: "*.txt"},
			{DisplayName: "All Files (*.*)", Pattern: "*.*"},
		},
	})
	if err != nil {
		return "", err
	}
	if path == "" {
		return "", nil
	}
	if err := checkFileSize(path, maxOpenTextBytes, "text file"); err != nil {
		return "", err
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("read file: %w", err)
	}
	return string(data), nil
}

func (a *App) OpenDocs(apiURL string) error {
	docURL, err := docsURL(apiURL)
	if err != nil {
		return err
	}
	runtime.BrowserOpenURL(a.ctx, docURL)
	return nil
}

func (a *App) CancelRequest() {
	a.cancelMu.Lock()
	cancel := a.currentCancel
	a.cancelMu.Unlock()
	if cancel != nil {
		cancel()
	}
}

func (a *App) Recognize(req OCRRequest) (OCRResult, error) {
	endpoint, err := normalizeAPIURL(req.APIURL)
	if err != nil {
		return OCRResult{}, err
	}
	imagePath := strings.TrimSpace(req.ImagePath)
	if imagePath == "" {
		return OCRResult{}, fmt.Errorf("select an image")
	}
	if err := checkFileSize(imagePath, maxUploadBytes, "upload image"); err != nil {
		return OCRResult{}, err
	}
	file, err := os.Open(imagePath)
	if err != nil {
		return OCRResult{}, fmt.Errorf("open image: %w", err)
	}
	defer file.Close()

	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	if err := writer.WriteField("task", defaultString(req.Task, "ocr")); err != nil {
		return OCRResult{}, err
	}
	maxNew := normalizeMaxNewTokens(req.MaxNewTokens)
	if err := writer.WriteField("max_new_tokens", strconv.Itoa(maxNew)); err != nil {
		return OCRResult{}, err
	}
	part, err := writer.CreateFormFile("image", filepath.Base(imagePath))
	if err != nil {
		return OCRResult{}, err
	}
	if _, err := io.Copy(part, file); err != nil {
		return OCRResult{}, fmt.Errorf("read image: %w", err)
	}
	if err := writer.Close(); err != nil {
		return OCRResult{}, err
	}

	ctx, cancel := requestContext(req.TimeoutSecs)
	seq := a.setCurrentCancel(cancel)
	defer a.clearCurrentCancel(seq)
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, &body)
	if err != nil {
		return OCRResult{}, err
	}
	httpReq.Header.Set("Content-Type", writer.FormDataContentType())
	addAPIKey(httpReq, req.APIKey)

	resp, err := a.client.Do(httpReq)
	if err != nil {
		return OCRResult{}, requestError(err)
	}
	defer resp.Body.Close()
	raw, err := readLimited(resp.Body, maxResponseBytes, "response")
	if err != nil {
		return OCRResult{}, fmt.Errorf("read response: %w", err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return OCRResult{}, responseError(resp.StatusCode, raw)
	}
	var out generateResponse
	if err := json.Unmarshal(raw, &out); err != nil {
		return OCRResult{}, fmt.Errorf("parse response: %w", err)
	}
	return OCRResult{
		Text:            out.Text,
		PromptTokens:    out.PromptTokens,
		GeneratedTokens: out.GeneratedTokens,
		Raw:             string(raw),
	}, nil
}

func (a *App) CheckReady(apiURL, apiKey string) (ReadyResult, error) {
	endpoint, err := readyURL(apiURL)
	if err != nil {
		return ReadyResult{}, err
	}
	ctx, cancel := requestContext(10)
	seq := a.setCurrentCancel(cancel)
	defer a.clearCurrentCancel(seq)
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return ReadyResult{}, err
	}
	addAPIKey(httpReq, apiKey)
	resp, err := a.client.Do(httpReq)
	if err != nil {
		return ReadyResult{}, requestError(err)
	}
	defer resp.Body.Close()
	raw, err := readLimited(resp.Body, maxResponseBytes, "response")
	if err != nil {
		return ReadyResult{}, fmt.Errorf("read response: %w", err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return ReadyResult{}, responseError(resp.StatusCode, raw)
	}
	var out readyResponse
	if err := json.Unmarshal(raw, &out); err != nil {
		return ReadyResult{}, fmt.Errorf("parse response: %w", err)
	}
	return ReadyResult{
		Status:         out.Status,
		Backend:        out.Backend,
		Quantization:   out.Quantization,
		WeightSource:   out.WeightSource,
		VisionLoaded:   out.VisionLoaded,
		Concurrency:    out.Concurrency,
		AvailableSlots: out.AvailableSlots,
		Raw:            string(raw),
	}, nil
}

func requestContext(timeoutSecs int) (context.Context, context.CancelFunc) {
	timeoutSecs = normalizeTimeoutSecs(timeoutSecs)
	return context.WithTimeout(context.Background(), time.Duration(timeoutSecs)*time.Second)
}

func normalizeMaxNewTokens(maxNew int) int {
	if maxNew <= 0 {
		return defaultMaxNew
	}
	if maxNew > maxClientMaxNew {
		return maxClientMaxNew
	}
	return maxNew
}

func normalizeTimeoutSecs(timeoutSecs int) int {
	if timeoutSecs <= 0 {
		return defaultTimeout
	}
	if timeoutSecs > maxTimeoutSecs {
		return maxTimeoutSecs
	}
	return timeoutSecs
}

func (a *App) setCurrentCancel(cancel context.CancelFunc) int64 {
	a.cancelMu.Lock()
	a.cancelSeq++
	a.currentSeq = a.cancelSeq
	a.currentCancel = cancel
	seq := a.currentSeq
	a.cancelMu.Unlock()
	return seq
}

func (a *App) clearCurrentCancel(seq int64) {
	a.cancelMu.Lock()
	cancel := a.currentCancel
	if a.currentSeq == seq {
		a.currentCancel = nil
		a.currentSeq = 0
	} else {
		cancel = nil
	}
	a.cancelMu.Unlock()
	if cancel != nil {
		cancel()
	}
}

func requestError(err error) error {
	if errors.Is(err, context.Canceled) {
		return fmt.Errorf("request canceled")
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return fmt.Errorf("request timed out")
	}
	return fmt.Errorf("request service: %w", err)
}

func normalizeAPIURL(s string) (string, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return "", fmt.Errorf("enter API URL")
	}
	if !strings.Contains(s, "://") {
		s = "http://" + s
	}
	u, err := url.Parse(s)
	if err != nil || u.Host == "" {
		return "", fmt.Errorf("invalid API URL")
	}
	if u.Scheme != "http" && u.Scheme != "https" {
		return "", fmt.Errorf("API URL must use http or https")
	}
	if u.Path == "" || u.Path == "/" {
		u.Path = "/v1/ocr"
	}
	return u.String(), nil
}

func readyURL(s string) (string, error) {
	endpoint, err := normalizeAPIURL(s)
	if err != nil {
		return "", err
	}
	u, err := url.Parse(endpoint)
	if err != nil {
		return "", err
	}
	u.Path = "/ready"
	u.RawQuery = ""
	u.Fragment = ""
	return u.String(), nil
}

func docsURL(s string) (string, error) {
	endpoint, err := normalizeAPIURL(s)
	if err != nil {
		return "", err
	}
	u, err := url.Parse(endpoint)
	if err != nil {
		return "", err
	}
	u.Path = "/doc"
	u.RawQuery = ""
	u.Fragment = ""
	return u.String(), nil
}

func responseError(status int, raw []byte) error {
	var er errorResponse
	if json.Unmarshal(raw, &er) == nil && er.Error != "" {
		return fmt.Errorf("service returned %d: %s", status, er.Error)
	}
	msg := strings.TrimSpace(string(raw))
	if msg == "" {
		msg = http.StatusText(status)
	}
	msg = truncateErrorText(msg)
	return fmt.Errorf("service returned %d: %s", status, msg)
}

func defaultString(v, fallback string) string {
	v = strings.TrimSpace(v)
	if v == "" {
		return fallback
	}
	return v
}

func safeDefaultFilename(v, fallback string) string {
	v = filepath.Base(strings.TrimSpace(v))
	if v == "." || v == string(filepath.Separator) || v == "" {
		return fallback
	}
	return v
}

func truncateErrorText(s string) string {
	if len(s) <= maxErrorTextBytes {
		return s
	}
	return s[:maxErrorTextBytes] + "...[truncated]"
}

func checkFileSize(path string, maxBytes int64, label string) error {
	st, err := os.Stat(path)
	if err != nil {
		return fmt.Errorf("stat %s: %w", label, err)
	}
	if !st.Mode().IsRegular() {
		return fmt.Errorf("%s is not a regular file", label)
	}
	if st.Size() > maxBytes {
		return fmt.Errorf("%s too large: %d bytes exceeds %d", label, st.Size(), maxBytes)
	}
	return nil
}

func readLimited(r io.Reader, maxBytes int64, label string) ([]byte, error) {
	data, err := io.ReadAll(io.LimitReader(r, maxBytes+1))
	if err != nil {
		return nil, err
	}
	if int64(len(data)) > maxBytes {
		return nil, fmt.Errorf("%s too large: exceeds %d bytes", label, maxBytes)
	}
	return data, nil
}

func addAPIKey(req *http.Request, apiKey string) {
	if key := strings.TrimSpace(apiKey); key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
		req.Header.Set("X-API-Key", key)
	}
}
