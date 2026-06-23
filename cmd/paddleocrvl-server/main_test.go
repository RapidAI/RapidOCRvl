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
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"paddleocrvl-go/internal/model"
)

func TestImageBytesBase64(t *testing.T) {
	raw := []byte{0x89, 'P', 'N', 'G'}
	req := generateRequest{
		ImageBase64: "data:image/png;base64," + base64.StdEncoding.EncodeToString(raw),
	}
	got, err := imageBytes(&req)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != string(raw) {
		t.Fatalf("imageBytes=%v want %v", got, raw)
	}
}

func TestBase64DecodedLenAccountsForPadding(t *testing.T) {
	for _, raw := range [][]byte{
		{},
		{1},
		{1, 2},
		{1, 2, 3},
		{1, 2, 3, 4},
	} {
		encoded := base64.StdEncoding.EncodeToString(raw)
		if got := base64DecodedLen(encoded); got != len(raw) {
			t.Fatalf("base64DecodedLen(%q)=%d want %d", encoded, got, len(raw))
		}
		wrapped := encoded
		if len(wrapped) > 2 {
			wrapped = wrapped[:2] + "\r\n" + wrapped[2:]
		}
		if got := base64DecodedLenIgnoringCRLF(wrapped); got != len(raw) {
			t.Fatalf("base64DecodedLenIgnoringCRLF(%q)=%d want %d", wrapped, got, len(raw))
		}
	}
}

func BenchmarkImageBytesBase64(b *testing.B) {
	raw := make([]byte, 1<<20)
	for i := range raw {
		raw[i] = byte(i)
	}
	req := generateRequest{ImageBase64: "data:image/png;base64," + base64.StdEncoding.EncodeToString(raw)}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		got, err := imageBytes(&req)
		if err != nil {
			b.Fatal(err)
		}
		if len(got) != len(raw) {
			b.Fatalf("len=%d want %d", len(got), len(raw))
		}
	}
}

func BenchmarkImageBytesBase64Raw(b *testing.B) {
	raw := make([]byte, 1<<20)
	for i := range raw {
		raw[i] = byte(i)
	}
	req := generateRequest{ImageBase64: base64.StdEncoding.EncodeToString(raw)}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		got, err := imageBytes(&req)
		if err != nil {
			b.Fatal(err)
		}
		if len(got) != len(raw) {
			b.Fatalf("len=%d want %d", len(got), len(raw))
		}
	}
}

func TestImageBytesRejectsPathAndBase64(t *testing.T) {
	req := generateRequest{
		ImagePath:   "x.png",
		ImageBase64: base64.StdEncoding.EncodeToString([]byte("x")),
	}
	_, err := imageBytes(&req)
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestReadAllLimitedRejectsOversize(t *testing.T) {
	got, err := readAllLimited(strings.NewReader("abcd"), 4, "test")
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != "abcd" {
		t.Fatalf("got %q", got)
	}
	if _, err := readAllLimited(strings.NewReader("abcde"), 4, "test"); err == nil || !strings.Contains(err.Error(), "too large") {
		t.Fatalf("err=%v", err)
	}
}

func TestRequestHasImage(t *testing.T) {
	if requestHasImage(&generateRequest{}) {
		t.Fatal("empty request should not have image")
	}
	if !requestHasImage(&generateRequest{ImagePath: "x.png"}) {
		t.Fatal("image_path should count as image")
	}
	if !requestHasImage(&generateRequest{ImageBase64: "eA=="}) {
		t.Fatal("image_base64 should count as image")
	}
	if !requestHasImage(&generateRequest{imageData: []byte("x")}) {
		t.Fatal("multipart image data should count as image")
	}
}

func TestTaskPrompt(t *testing.T) {
	cases := map[string]string{
		"ocr":     "OCR:",
		" OCR ":   "OCR:",
		"TaBlE":   "Table Recognition:",
		"table":   "Table Recognition:",
		"formula": "Formula Recognition:",
		"chart":   "Chart Recognition:",
	}
	for in, want := range cases {
		got, err := taskPrompt(in)
		if err != nil {
			t.Fatal(err)
		}
		if got != want {
			t.Fatalf("%s -> %q want %q", in, got, want)
		}
	}
}

func TestTaskKey(t *testing.T) {
	got, err := taskKey(" OCR ", true)
	if err != nil {
		t.Fatal(err)
	}
	if got != "ocr:image" {
		t.Fatalf("taskKey=%q", got)
	}
	if _, err := taskKey("bad", false); err == nil {
		t.Fatal("expected error")
	}
}

func BenchmarkTaskPrompt(b *testing.B) {
	for i := 0; i < b.N; i++ {
		got, err := taskPrompt(" OCR ")
		if err != nil {
			b.Fatal(err)
		}
		if got != "OCR:" {
			b.Fatalf("got %q", got)
		}
	}
}

func BenchmarkTaskKey(b *testing.B) {
	for i := 0; i < b.N; i++ {
		got, err := taskKey(" OCR ", true)
		if err != nil {
			b.Fatal(err)
		}
		if got != "ocr:image" {
			b.Fatalf("got %q", got)
		}
	}
}

func TestInputIDsCopiesTokenSlice(t *testing.T) {
	src := []int{1, 2, 3}
	s := &server{}
	req := generateRequest{Tokens: src}
	got, err := s.inputIDs(&req, requestHasImage(&req))
	if err != nil {
		t.Fatal(err)
	}
	src[0] = 9
	if got[0] != 1 {
		t.Fatalf("inputIDs aliased source slice: %v", got)
	}
}

func TestMetricsHelpers(t *testing.T) {
	s := &server{}
	if got := s.avgLatencyMillis(); got != 0 {
		t.Fatalf("avgLatencyMillis=%d want 0", got)
	}
	if got := s.avgQueueWaitMillis(); got != 0 {
		t.Fatalf("avgQueueWaitMillis=%d want 0", got)
	}
	s.metrics.succeeded.Add(2)
	s.metrics.latencyNanos.Add(int64(300 * time.Millisecond))
	if got := s.avgLatencyMillis(); got != 150 {
		t.Fatalf("avgLatencyMillis=%d want 150", got)
	}
	s.metrics.started.Add(2)
	s.metrics.queueWaitNanos.Add(int64(40 * time.Millisecond))
	if got := s.avgQueueWaitMillis(); got != 20 {
		t.Fatalf("avgQueueWaitMillis=%d want 20", got)
	}
	s.recordFailure(errors.New("boom"))
	if got := s.lastError(); got != "boom" {
		t.Fatalf("lastError=%q want boom", got)
	}
	if got := s.metrics.failed.Load(); got != 1 {
		t.Fatalf("failed=%d want 1", got)
	}
	s.recordFailure(context.Canceled)
	if got := s.metrics.canceled.Load(); got != 1 {
		t.Fatalf("canceled=%d want 1", got)
	}
}

func BenchmarkAvgMillis(b *testing.B) {
	for i := 0; i < b.N; i++ {
		if got := avgMillis(int64(300*time.Millisecond), 2); got != 150 {
			b.Fatalf("got %d", got)
		}
	}
}

func TestBatchJSONRejectsEmptyRequests(t *testing.T) {
	s := &server{requestLimit: 1 << 20}
	req := httptest.NewRequest(http.MethodPost, "/v1/batch", strings.NewReader(`{"requests":[]}`))
	rec := httptest.NewRecorder()
	s.batchJSON(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status=%d want %d", rec.Code, http.StatusBadRequest)
	}
	if !strings.Contains(rec.Body.String(), "requests must not be empty") {
		t.Fatalf("body=%q", rec.Body.String())
	}
}

func TestBatchJSONRejectsTooManyRequests(t *testing.T) {
	s := &server{requestLimit: 1 << 20, maxBatchSize: 1}
	req := httptest.NewRequest(http.MethodPost, "/v1/batch", strings.NewReader(`{"requests":[{"tokens":[1]},{"tokens":[2]}]}`))
	rec := httptest.NewRecorder()
	s.batchJSON(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status=%d want %d", rec.Code, http.StatusBadRequest)
	}
	if !strings.Contains(rec.Body.String(), "exceeds limit") {
		t.Fatalf("body=%q", rec.Body.String())
	}
}

func TestDecodeJSONRejectsTrailingValue(t *testing.T) {
	var req generateRequest
	err := decodeJSON(strings.NewReader(`{"tokens":[1]} {"tokens":[2]}`), &req)
	if err == nil || !strings.Contains(err.Error(), "only one JSON value") {
		t.Fatalf("err=%v", err)
	}
}

func TestWriteJSONDoesNotEscapeHTML(t *testing.T) {
	rec := httptest.NewRecorder()
	writeJSON(rec, http.StatusOK, map[string]string{"text": "<table>&ocr"})
	if body := rec.Body.String(); !strings.Contains(body, `"<table>&ocr"`) {
		t.Fatalf("body=%q", body)
	}
}

func TestBatchJSONReleasesSlotsOnItemError(t *testing.T) {
	s := &server{requestLimit: 1 << 20, runSlots: make(chan struct{}, 1)}
	req := httptest.NewRequest(http.MethodPost, "/v1/batch", strings.NewReader(`{"requests":[{},{}]}`))
	rec := httptest.NewRecorder()
	s.batchJSON(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status=%d want %d", rec.Code, http.StatusBadRequest)
	}
	if got := len(s.runSlots); got != 0 {
		t.Fatalf("runSlots len=%d want 0", got)
	}
	if got := s.metrics.queued.Load(); got != 2 {
		t.Fatalf("queued=%d want 2", got)
	}
}

func TestBatchJSONQueuesAllItemsWithMultipleSlots(t *testing.T) {
	s := &server{requestLimit: 1 << 20, runSlots: make(chan struct{}, 2)}
	req := httptest.NewRequest(http.MethodPost, "/v1/batch", strings.NewReader(`{"requests":[{},{}]}`))
	rec := httptest.NewRecorder()
	s.batchJSON(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status=%d want %d", rec.Code, http.StatusBadRequest)
	}
	if got := len(s.runSlots); got != 0 {
		t.Fatalf("runSlots len=%d want 0", got)
	}
	if got := s.metrics.queued.Load(); got != 2 {
		t.Fatalf("queued=%d want 2", got)
	}
	if got := s.metrics.failed.Load(); got == 0 {
		t.Fatalf("failed=%d want >0", got)
	}
}

func TestBatchJSONRunsItemsConcurrentlyWhenSlotsAvailable(t *testing.T) {
	entered := make(chan int, 2)
	release := make(chan struct{})
	s := &server{
		requestLimit: 1 << 20,
		runSlots:     make(chan struct{}, 2),
		infer: func(ctx context.Context, req generateRequest, ids []int, imageData []byte, opts model.GenerateOptions) (model.GenerateResult, error) {
			entered <- ids[0]
			select {
			case <-release:
			case <-ctx.Done():
				return model.GenerateResult{}, ctx.Err()
			}
			return model.GenerateResult{Tokens: []int{ids[0], ids[0] + 10}, PromptTokens: len(ids)}, nil
		},
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/batch", strings.NewReader(`{"requests":[{"tokens":[1],"max_new_tokens":1},{"tokens":[2],"max_new_tokens":1}]}`))
	rec := httptest.NewRecorder()
	done := make(chan struct{})
	go func() {
		s.batchJSON(rec, req)
		close(done)
	}()

	for i := 0; i < 2; i++ {
		select {
		case <-entered:
		case <-time.After(time.Second):
			close(release)
			<-done
			t.Fatal("batch items did not enter inference concurrently")
		}
	}
	close(release)
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("batch request did not complete")
	}
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%q", rec.Code, rec.Body.String())
	}
	var body batchResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if body.Items != 2 || body.GeneratedTokens != 2 {
		t.Fatalf("body=%+v", body)
	}
}

func TestGenerateAutoBatcherCoalescesConcurrentRequests(t *testing.T) {
	entered := make(chan int, 2)
	release := make(chan struct{})
	s := &server{
		runSlots: make(chan struct{}, 2),
		infer: func(ctx context.Context, req generateRequest, ids []int, imageData []byte, opts model.GenerateOptions) (model.GenerateResult, error) {
			entered <- ids[0]
			select {
			case <-release:
			case <-ctx.Done():
				return model.GenerateResult{}, ctx.Err()
			}
			return model.GenerateResult{Tokens: []int{ids[0], ids[0] + 10}, PromptTokens: len(ids)}, nil
		},
	}
	s.batcher = newRequestBatcher(s, 2, time.Second)
	defer s.batcher.close()

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	resCh := make(chan batchItemResult, 2)
	for _, id := range []int{1, 2} {
		id := id
		go func() {
			res, err := s.runGenerate(ctx, generateRequest{Tokens: []int{id}, MaxNewTokens: 1})
			resCh <- batchItemResult{res: res, err: err}
		}()
	}
	for i := 0; i < 2; i++ {
		select {
		case <-entered:
		case <-time.After(time.Second):
			close(release)
			t.Fatal("auto-batched requests did not enter inference together")
		}
	}
	close(release)
	for i := 0; i < 2; i++ {
		select {
		case got := <-resCh:
			if got.err != nil {
				t.Fatalf("runGenerate err=%v", got.err)
			}
			if got.res.GeneratedTokens != 1 {
				t.Fatalf("response=%+v", got.res)
			}
		case <-time.After(time.Second):
			t.Fatal("timed out waiting for auto-batched response")
		}
	}
	if got := s.metrics.batches.Load(); got != 1 {
		t.Fatalf("batches=%d want 1", got)
	}
	if got := s.metrics.batchItems.Load(); got != 2 {
		t.Fatalf("batchItems=%d want 2", got)
	}
}

func TestGenerateAutoBatcherProcessesMultipleBatchesConcurrently(t *testing.T) {
	entered := make(chan int, 4)
	release := make(chan struct{})
	s := &server{
		runSlots:    make(chan struct{}, 4),
		concurrency: 4,
		infer: func(ctx context.Context, req generateRequest, ids []int, imageData []byte, opts model.GenerateOptions) (model.GenerateResult, error) {
			entered <- ids[0]
			select {
			case <-release:
			case <-ctx.Done():
				return model.GenerateResult{}, ctx.Err()
			}
			return model.GenerateResult{Tokens: []int{ids[0], ids[0] + 10}, PromptTokens: len(ids)}, nil
		},
	}
	s.batcher = newRequestBatcher(s, 2, time.Millisecond)
	defer s.batcher.close()
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	resCh := make(chan error, 4)
	for id := 1; id <= 4; id++ {
		id := id
		go func() {
			_, err := s.runGenerate(ctx, generateRequest{Tokens: []int{id}, MaxNewTokens: 1})
			resCh <- err
		}()
	}
	for i := 0; i < 4; i++ {
		select {
		case <-entered:
		case <-time.After(time.Second):
			close(release)
			t.Fatal("auto-batched requests did not use available concurrency")
		}
	}
	close(release)
	for i := 0; i < 4; i++ {
		select {
		case err := <-resCh:
			if err != nil {
				t.Fatalf("runGenerate err=%v", err)
			}
		case <-time.After(time.Second):
			t.Fatal("timed out waiting for auto-batched response")
		}
	}
	if got := s.metrics.batchItems.Load(); got != 4 {
		t.Fatalf("batchItems=%d want 4", got)
	}
}

func TestRequestBatcherCloseIsIdempotent(t *testing.T) {
	s := &server{runSlots: make(chan struct{}, 1)}
	b := newRequestBatcher(s, 2, time.Millisecond)
	b.close()
	b.close()
}

func TestRequestBatcherSubmitAfterCloseReturnsCanceled(t *testing.T) {
	s := &server{runSlots: make(chan struct{}, 1)}
	b := newRequestBatcher(s, 2, time.Millisecond)
	b.close()
	_, err := b.submit(context.Background(), generateRequest{Tokens: []int{1}, MaxNewTokens: 1})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err=%v want canceled", err)
	}
}

func TestRequestBatcherZeroWaitCollectsOnlyAvailableItems(t *testing.T) {
	b := newStoppedRequestBatcherForTest(nil, 3, 0)
	b.ch <- batchItem{ctx: context.Background(), done: make(chan batchItemResult, 1)}
	started := time.Now()
	batch := b.collect(batchItem{ctx: context.Background(), done: make(chan batchItemResult, 1)})
	if len(batch) != 2 {
		t.Fatalf("batch len=%d want 2", len(batch))
	}
	if elapsed := time.Since(started); elapsed > 50*time.Millisecond {
		t.Fatalf("zero-wait collect blocked for %s", elapsed)
	}
}

func TestRequestBatcherCloseCancelsPendingBatch(t *testing.T) {
	inferCalled := make(chan struct{}, 1)
	s := &server{
		runSlots: make(chan struct{}, 1),
		infer: func(ctx context.Context, req generateRequest, ids []int, imageData []byte, opts model.GenerateOptions) (model.GenerateResult, error) {
			inferCalled <- struct{}{}
			return model.GenerateResult{}, nil
		},
	}
	b := newRequestBatcher(s, 2, time.Hour)
	s.batcher = b
	resCh := make(chan error, 1)
	go func() {
		_, err := s.runGenerate(context.Background(), generateRequest{Tokens: []int{1}, MaxNewTokens: 1})
		resCh <- err
	}()
	time.Sleep(10 * time.Millisecond)
	b.close()
	select {
	case err := <-resCh:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("err=%v want canceled", err)
		}
	case <-time.After(time.Second):
		t.Fatal("pending auto-batched request did not return after close")
	}
	select {
	case <-inferCalled:
		t.Fatal("pending request should not run inference after close")
	default:
	}
}

func TestRequestBatcherCloseWaitsForInFlightProcess(t *testing.T) {
	entered := make(chan struct{}, 1)
	release := make(chan struct{})
	s := &server{
		runSlots: make(chan struct{}, 1),
		infer: func(ctx context.Context, req generateRequest, ids []int, imageData []byte, opts model.GenerateOptions) (model.GenerateResult, error) {
			entered <- struct{}{}
			<-release
			return model.GenerateResult{Tokens: []int{ids[0]}, PromptTokens: len(ids)}, nil
		},
	}
	b := newRequestBatcher(s, 1, 0)
	done := make(chan error, 1)
	go func() {
		_, err := b.submit(context.Background(), generateRequest{Tokens: []int{1}})
		done <- err
	}()
	select {
	case <-entered:
	case <-time.After(time.Second):
		t.Fatal("inference did not start")
	}
	closed := make(chan struct{})
	go func() {
		b.close()
		close(closed)
	}()
	select {
	case <-closed:
		t.Fatal("close returned before in-flight process finished")
	case <-time.After(20 * time.Millisecond):
	}
	close(release)
	select {
	case err := <-done:
		if err != nil && !errors.Is(err, context.Canceled) {
			t.Fatalf("submit err=%v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("submit did not finish")
	}
	select {
	case <-closed:
	case <-time.After(time.Second):
		t.Fatal("close did not finish")
	}
}

func TestRequestBatcherCloseCancelsBatchWaitingForProcessSlot(t *testing.T) {
	entered := make(chan struct{}, 1)
	release := make(chan struct{})
	s := &server{
		runSlots:    make(chan struct{}, 1),
		concurrency: 1,
		infer: func(ctx context.Context, req generateRequest, ids []int, imageData []byte, opts model.GenerateOptions) (model.GenerateResult, error) {
			entered <- struct{}{}
			<-release
			return model.GenerateResult{Tokens: []int{ids[0]}, PromptTokens: len(ids)}, nil
		},
	}
	b := newRequestBatcher(s, 1, 0)
	first := make(chan error, 1)
	second := make(chan error, 1)
	go func() {
		_, err := b.submit(context.Background(), generateRequest{Tokens: []int{1}})
		first <- err
	}()
	select {
	case <-entered:
	case <-time.After(time.Second):
		t.Fatal("first inference did not start")
	}
	go func() {
		_, err := b.submit(context.Background(), generateRequest{Tokens: []int{2}})
		second <- err
	}()
	time.Sleep(10 * time.Millisecond)
	closed := make(chan struct{})
	go func() {
		b.close()
		close(closed)
	}()
	select {
	case err := <-second:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("second err=%v want canceled", err)
		}
	case <-time.After(time.Second):
		close(release)
		t.Fatal("second submit was not canceled")
	}
	close(release)
	select {
	case <-first:
	case <-time.After(time.Second):
		t.Fatal("first submit did not finish")
	}
	select {
	case <-closed:
	case <-time.After(time.Second):
		t.Fatal("close did not finish")
	}
}

func TestRequestBatcherSubmitReturnsOnContextCancelAfterEnqueue(t *testing.T) {
	inferCalled := make(chan struct{}, 1)
	s := &server{
		runSlots: make(chan struct{}, 1),
		infer: func(ctx context.Context, req generateRequest, ids []int, imageData []byte, opts model.GenerateOptions) (model.GenerateResult, error) {
			inferCalled <- struct{}{}
			return model.GenerateResult{}, nil
		},
	}
	b := newRequestBatcher(s, 2, time.Hour)
	defer b.close()
	ctx, cancel := context.WithCancel(context.Background())
	errCh := make(chan error, 1)
	go func() {
		_, err := b.submit(ctx, generateRequest{Tokens: []int{1}, MaxNewTokens: 1})
		errCh <- err
	}()
	time.Sleep(10 * time.Millisecond)
	cancel()
	select {
	case err := <-errCh:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("err=%v want canceled", err)
		}
	case <-time.After(time.Second):
		t.Fatal("submit did not return after context cancellation")
	}
	select {
	case <-inferCalled:
		t.Fatal("canceled item should not run inference")
	default:
	}
}

func TestRequestBatcherSkipsCanceledItemBeforeInference(t *testing.T) {
	inferCalled := make(chan struct{}, 1)
	s := &server{
		runSlots: make(chan struct{}, 1),
		infer: func(ctx context.Context, req generateRequest, ids []int, imageData []byte, opts model.GenerateOptions) (model.GenerateResult, error) {
			inferCalled <- struct{}{}
			return model.GenerateResult{}, nil
		},
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	b := newStoppedRequestBatcherForTest(s, 1, time.Millisecond)
	done := make(chan batchItemResult, 1)
	b.process([]batchItem{{ctx: ctx, req: generateRequest{Tokens: []int{1}, MaxNewTokens: 1}, done: done}})
	got := <-done
	if !errors.Is(got.err, context.Canceled) {
		t.Fatalf("err=%v want canceled", got.err)
	}
	if got := s.metrics.batches.Load(); got != 0 {
		t.Fatalf("batches=%d want 0", got)
	}
	if got := s.metrics.batchItems.Load(); got != 0 {
		t.Fatalf("batchItems=%d want 0", got)
	}
	select {
	case <-inferCalled:
		t.Fatal("canceled item should not run inference")
	default:
	}
}

func newStoppedRequestBatcherForTest(s *server, maxSize int, wait time.Duration) *requestBatcher {
	done := make(chan struct{})
	close(done)
	return &requestBatcher{
		s:       s,
		maxSize: maxSize,
		wait:    wait,
		ch:      make(chan batchItem, maxSize*4),
		stop:    make(chan struct{}),
		done:    done,
		procSem: make(chan struct{}, 1),
	}
}

func TestBatchJSONSingleItemReleasesSlot(t *testing.T) {
	s := &server{requestLimit: 1 << 20, runSlots: make(chan struct{}, 1)}
	req := httptest.NewRequest(http.MethodPost, "/v1/batch", strings.NewReader(`{"requests":[{}]}`))
	rec := httptest.NewRecorder()
	s.batchJSON(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status=%d want %d", rec.Code, http.StatusBadRequest)
	}
	if got := len(s.runSlots); got != 0 {
		t.Fatalf("runSlots len=%d want 0", got)
	}
	if got := s.metrics.queued.Load(); got != 1 {
		t.Fatalf("queued=%d want 1", got)
	}
	if got := s.metrics.failed.Load(); got != 1 {
		t.Fatalf("failed=%d want 1", got)
	}
	if !strings.Contains(rec.Body.String(), "request 0:") {
		t.Fatalf("body=%q", rec.Body.String())
	}
}

func TestOCRMultipartRemovesTempFiles(t *testing.T) {
	tmp := t.TempDir()
	t.Setenv("TMP", tmp)
	t.Setenv("TEMP", tmp)
	t.Setenv("TMPDIR", tmp)

	var body bytes.Buffer
	mw := multipart.NewWriter(&body)
	fw, err := mw.CreateFormFile("image", "large.bin")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := fw.Write(bytes.Repeat([]byte("x"), 1024)); err != nil {
		t.Fatal(err)
	}
	if err := mw.WriteField("task", "bad-task"); err != nil {
		t.Fatal(err)
	}
	if err := mw.Close(); err != nil {
		t.Fatal(err)
	}

	s := &server{requestLimit: 1 << 20, multipartMem: 1, runSlots: make(chan struct{}, 1)}
	req := httptest.NewRequest(http.MethodPost, "/v1/ocr", &body)
	req.Header.Set("Content-Type", mw.FormDataContentType())
	rec := httptest.NewRecorder()
	s.ocrMultipart(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status=%d want %d body=%q", rec.Code, http.StatusBadRequest, rec.Body.String())
	}
	entries, err := os.ReadDir(tmp)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 0 {
		t.Fatalf("multipart temp files left behind: %v", entries)
	}
}

func TestReadyRejectsUninitializedServer(t *testing.T) {
	s := &server{runSlots: make(chan struct{}, 1), concurrency: 1}
	req := httptest.NewRequest(http.MethodGet, "/ready", nil)
	rec := httptest.NewRecorder()
	s.ready(rec, req)
	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("status=%d want %d", rec.Code, http.StatusServiceUnavailable)
	}
	if !strings.Contains(rec.Body.String(), "not_ready") {
		t.Fatalf("body=%q", rec.Body.String())
	}
}

func TestRunRejectsMaxNewLimit(t *testing.T) {
	s := &server{runSlots: make(chan struct{}, 1), maxNewLimit: 4}
	_, err := s.run(context.Background(), generateRequest{Tokens: []int{1}, MaxNewTokens: 5})
	if err == nil || !strings.Contains(err.Error(), "exceeds limit") {
		t.Fatalf("err=%v", err)
	}
}

func TestRunRejectsMaxInputLimit(t *testing.T) {
	s := &server{runSlots: make(chan struct{}, 1), maxInputLimit: 2}
	_, err := s.run(context.Background(), generateRequest{Tokens: []int{1, 2, 3}, MaxNewTokens: 1})
	if err == nil || !strings.Contains(err.Error(), "input tokens 3 exceeds limit 2") {
		t.Fatalf("err=%v", err)
	}
	if got := s.metrics.failed.Load(); got != 1 {
		t.Fatalf("failed=%d want 1", got)
	}
}

func TestRunRejectsNegativeInputToken(t *testing.T) {
	s := &server{runSlots: make(chan struct{}, 1)}
	_, err := s.run(context.Background(), generateRequest{Tokens: []int{1, -2}, MaxNewTokens: 1})
	if err == nil || !strings.Contains(err.Error(), "negative") {
		t.Fatalf("err=%v want negative token error", err)
	}
	if got := s.metrics.failed.Load(); got != 1 {
		t.Fatalf("failed=%d want 1", got)
	}
}

func TestRunRejectsBadSamplingOptions(t *testing.T) {
	s := &server{runSlots: make(chan struct{}, 1)}
	_, err := s.run(context.Background(), generateRequest{Tokens: []int{1}, MaxNewTokens: 1, Temperature: -1})
	if err == nil || !strings.Contains(err.Error(), "temperature") {
		t.Fatalf("err=%v", err)
	}
	_, err = s.run(context.Background(), generateRequest{Tokens: []int{1}, MaxNewTokens: 1, TopK: -1})
	if err == nil || !strings.Contains(err.Error(), "top_k") {
		t.Fatalf("err=%v", err)
	}
}

func TestDecodeTokenRangeClampsPromptTokens(t *testing.T) {
	tokens := []int{1, 2, 3}
	if got := decodeTokenRange(tokens, 1, false); !reflect.DeepEqual(got, tokens) {
		t.Fatalf("full range=%v", got)
	}
	if got := decodeTokenRange(tokens, 1, true); !reflect.DeepEqual(got, []int{2, 3}) {
		t.Fatalf("generated range=%v", got)
	}
	if got := decodeTokenRange(tokens, 99, true); len(got) != 0 {
		t.Fatalf("out of range=%v", got)
	}
	if got := decodeTokenRange(tokens, -1, true); !reflect.DeepEqual(got, tokens) {
		t.Fatalf("negative prompt range=%v", got)
	}
}

func TestValidateServerLimits(t *testing.T) {
	if err := validateServerLimits(1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0); err != nil {
		t.Fatalf("valid limits err=%v", err)
	}
	cases := []struct {
		name          string
		concurrency   int
		maxBatchSize  int
		autoBatchSize int
		maxInputLimit int
		requestLimit  int64
		multipartMem  int64
		timeout       time.Duration
		readTimeout   time.Duration
		idleTimeout   time.Duration
		shutdown      time.Duration
		autoBatchWait time.Duration
		want          string
	}{
		{"concurrency", 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, "-concurrency"},
		{"batch", 1, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, "-max-batch-size"},
		{"auto-batch", 1, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, "-auto-batch-size"},
		{"input", 1, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, "-max-input-tokens"},
		{"request", 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "-request-limit"},
		{"multipart", 1, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, "-multipart-memory"},
		{"timeout", 1, 0, 0, 0, 1, 0, -time.Second, 0, 0, 0, 0, "-timeout"},
		{"read-timeout", 1, 0, 0, 0, 1, 0, 0, -time.Second, 0, 0, 0, "-read-timeout"},
		{"idle-timeout", 1, 0, 0, 0, 1, 0, 0, 0, -time.Second, 0, 0, "-idle-timeout"},
		{"shutdown-timeout", 1, 0, 0, 0, 1, 0, 0, 0, 0, -time.Second, 0, "-shutdown-timeout"},
		{"auto-batch-wait", 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, -time.Second, "-auto-batch-wait"},
	}
	for _, tc := range cases {
		err := validateServerLimits(tc.concurrency, tc.maxBatchSize, tc.autoBatchSize, tc.maxInputLimit, tc.requestLimit, tc.multipartMem, tc.timeout, tc.readTimeout, tc.idleTimeout, tc.shutdown, tc.autoBatchWait)
		if err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Fatalf("%s err=%v want %s", tc.name, err, tc.want)
		}
	}
}

func TestNeedsSamplingSeed(t *testing.T) {
	if needsSamplingSeed(1, 0, 0) {
		t.Fatal("greedy temperature=0 should not need seed")
	}
	if needsSamplingSeed(1, 1, 1) {
		t.Fatal("top_k=1 should not need seed")
	}
	if needsSamplingSeed(0, 1, 0) {
		t.Fatal("max_new_tokens=0 should not need seed")
	}
	if !needsSamplingSeed(1, 1, 0) {
		t.Fatal("sampling should need seed")
	}
}

func TestAutoBatchSizeForConcurrency(t *testing.T) {
	if got := autoBatchSizeForConcurrency(2, 1); got != 0 {
		t.Fatalf("auto batch with single concurrency=%d want 0", got)
	}
	if got := autoBatchSizeForConcurrency(1, 4); got != 0 {
		t.Fatalf("disabled auto batch=%d want 0", got)
	}
	if got := autoBatchSizeForConcurrency(4, 2); got != 4 {
		t.Fatalf("auto batch=%d want 4", got)
	}
}

func TestBatchProcessLimit(t *testing.T) {
	if got := batchProcessLimit(&server{runSlots: make(chan struct{}, 4), concurrency: 4}, 2); got != 2 {
		t.Fatalf("process limit=%d want 2", got)
	}
	if got := batchProcessLimit(&server{runSlots: make(chan struct{}, 3), concurrency: 3}, 2); got != 2 {
		t.Fatalf("ceil process limit=%d want 2", got)
	}
	if got := batchProcessLimit(&server{runSlots: make(chan struct{}, 1), concurrency: 1}, 4); got != 1 {
		t.Fatalf("minimum process limit=%d want 1", got)
	}
}

func TestFormIntDefault(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/v1/ocr?max_new_tokens=%20123%20", nil)
	if got := formIntDefault(req, "max_new_tokens", 7); got != 123 {
		t.Fatalf("got %d", got)
	}
	bad := httptest.NewRequest(http.MethodPost, "/v1/ocr?max_new_tokens=0", nil)
	if got := formIntDefault(bad, "max_new_tokens", 7); got != 7 {
		t.Fatalf("fallback got %d", got)
	}
}

func BenchmarkFormIntDefault(b *testing.B) {
	req := httptest.NewRequest(http.MethodPost, "/v1/ocr?max_new_tokens=%201024%20", nil)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if got := formIntDefault(req, "max_new_tokens", 7); got != 1024 {
			b.Fatal(got)
		}
	}
}

func BenchmarkFormDefault(b *testing.B) {
	req := httptest.NewRequest(http.MethodPost, "/v1/ocr?task=%20ocr%20", nil)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if got := formDefault(req, "task", "ocr"); got != "ocr" {
			b.Fatal(got)
		}
	}
}

func TestStatusForRunError(t *testing.T) {
	if got := statusForRunError(context.DeadlineExceeded); got != http.StatusGatewayTimeout {
		t.Fatalf("deadline status=%d", got)
	}
	if got := statusForRunError(context.Canceled); got != http.StatusRequestTimeout {
		t.Fatalf("canceled status=%d", got)
	}
	if got := statusForRunError(errors.New("bad")); got != http.StatusBadRequest {
		t.Fatalf("bad status=%d", got)
	}
}

func TestFirstBatchErrorPrefersValidationError(t *testing.T) {
	i, err := firstBatchError([]error{context.Canceled, errors.New("bad request")})
	if i != 1 || err == nil || err.Error() != "bad request" {
		t.Fatalf("i=%d err=%v", i, err)
	}
	i, err = firstBatchError([]error{context.Canceled, nil})
	if i != 0 || !errors.Is(err, context.Canceled) {
		t.Fatalf("i=%d err=%v", i, err)
	}
}

func TestServeWithShutdownReturnsOnServerClose(t *testing.T) {
	srv := &http.Server{Addr: "127.0.0.1:0", Handler: http.NewServeMux()}
	done := make(chan error, 1)
	go func() {
		done <- serveWithShutdown(srv, time.Second)
	}()
	time.Sleep(50 * time.Millisecond)
	_ = srv.Close()
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("serveWithShutdown err=%v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("serveWithShutdown did not return")
	}
}

func TestServeWithShutdownStopChannel(t *testing.T) {
	srv := &http.Server{Addr: "127.0.0.1:0", Handler: http.NewServeMux()}
	stop := make(chan struct{})
	done := make(chan error, 1)
	go func() {
		done <- serveWithShutdownStop(srv, time.Second, stop)
	}()
	time.Sleep(50 * time.Millisecond)
	close(stop)
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("serveWithShutdownStop err=%v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("serveWithShutdownStop did not return")
	}
}

func TestAdminAPIKeyQuotaAndDisabled(t *testing.T) {
	st := loadAdminState(filepath.Join(t.TempDir(), "admin.json"))
	st.mu.Lock()
	key, managed, err := st.createAPIKeyLocked("agent", 1, 0)
	if err != nil {
		t.Fatal(err)
	}
	if err := st.saveLocked(); err != nil {
		t.Fatal(err)
	}
	st.mu.Unlock()
	if _, err := st.consumeAPIKey(key, "127.0.0.1"); err != nil {
		t.Fatalf("first consume err=%v", err)
	}
	keys := st.keyResponses()
	if len(keys) != 1 || keys[0].LastUsedIP != "127.0.0.1" || keys[0].LastUsedAt == "" {
		t.Fatalf("keys=%+v", keys)
	}
	if _, err := st.consumeAPIKey(key, "127.0.0.1"); !errors.Is(err, errAPIKeyQuota) {
		t.Fatalf("second consume err=%v want quota", err)
	}
	st.mu.Lock()
	for i := range st.cfg.APIKeys {
		if st.cfg.APIKeys[i].ID == managed.ID {
			st.cfg.APIKeys[i].Quota = 0
			st.cfg.APIKeys[i].Disabled = true
		}
	}
	st.mu.Unlock()
	if _, err := st.consumeAPIKey(key, "127.0.0.1"); !errors.Is(err, errAPIKeyDisabled) {
		t.Fatalf("disabled consume err=%v want disabled", err)
	}
}

func TestAdminLegacyAPIKeyMigratesToManagedKey(t *testing.T) {
	st := loadAdminState(filepath.Join(t.TempDir(), "admin.json"))
	salt, hash, err := saltedHash("legacy-secret")
	if err != nil {
		t.Fatal(err)
	}
	st.mu.Lock()
	st.cfg.AdminUser = "admin"
	st.cfg.PasswordSalt = "salt"
	st.cfg.PasswordHash = strings.Repeat("a", 64)
	st.cfg.APIKeySalt = salt
	st.cfg.APIKeyHash = hash
	st.cfg.APIKeyPreview = "lega...cret"
	if err := st.saveLocked(); err != nil {
		t.Fatal(err)
	}
	st.mu.Unlock()
	info, err := st.consumeAPIKey("legacy-secret", "127.0.0.1")
	if err != nil {
		t.Fatalf("first legacy consume err=%v", err)
	}
	if info.Name != "legacy" || info.ID == "" {
		t.Fatalf("legacy info=%+v", info)
	}
	keys := st.keyResponses()
	if len(keys) != 1 || keys[0].Name != "legacy" || keys[0].Used != 1 || keys[0].LastUsedIP != "127.0.0.1" {
		t.Fatalf("keys after migration=%+v", keys)
	}
	st.mu.RLock()
	legacyHash := st.cfg.APIKeyHash
	st.mu.RUnlock()
	if legacyHash != "" {
		t.Fatalf("legacy hash not cleared")
	}
	if _, err := st.consumeAPIKey("legacy-secret", "127.0.0.2"); err != nil {
		t.Fatalf("second legacy consume err=%v", err)
	}
	keys = st.keyResponses()
	if len(keys) != 1 || keys[0].Used != 2 || keys[0].LastUsedIP != "127.0.0.2" {
		t.Fatalf("keys after second consume=%+v", keys)
	}
}

func TestConsumeAPIKeyRollsBackWhenSaveFails(t *testing.T) {
	st := loadAdminState(filepath.Join(t.TempDir(), "admin.json"))
	st.mu.Lock()
	key, managed, err := st.createAPIKeyLocked("metered", 0, 1)
	if err != nil {
		t.Fatal(err)
	}
	if err := st.saveLocked(); err != nil {
		t.Fatal(err)
	}
	st.mu.Unlock()
	st.path = filepath.Join(t.TempDir(), "bad\x00name", "admin.json")
	if _, err := st.consumeAPIKey(key, "127.0.0.1"); err == nil {
		t.Fatal("expected save error")
	}
	keys := st.keyResponses()
	if len(keys) != 1 || keys[0].Used != 0 || keys[0].LastUsedIP != "" {
		t.Fatalf("key usage was not rolled back: %+v", keys)
	}
	st.mu.RLock()
	_, rateConsumed := st.rates[managed.ID]
	st.mu.RUnlock()
	if rateConsumed {
		t.Fatal("rate window was not rolled back")
	}
	st.path = filepath.Join(t.TempDir(), "admin.json")
	if _, err := st.consumeAPIKey(key, "127.0.0.2"); err != nil {
		t.Fatalf("consume after rollback err=%v", err)
	}
}

func TestLegacyAPIKeyMigrationRollsBackWhenSaveFails(t *testing.T) {
	st := loadAdminState(filepath.Join(t.TempDir(), "bad\x00name", "admin.json"))
	salt, hash, err := saltedHash("legacy-secret")
	if err != nil {
		t.Fatal(err)
	}
	st.mu.Lock()
	st.cfg.APIKeySalt = salt
	st.cfg.APIKeyHash = hash
	st.cfg.APIKeyPreview = "lega...cret"
	st.mu.Unlock()
	if _, err := st.consumeAPIKey("legacy-secret", "127.0.0.1"); err == nil {
		t.Fatal("expected save error")
	}
	st.mu.RLock()
	defer st.mu.RUnlock()
	if len(st.cfg.APIKeys) != 0 || st.cfg.APIKeyHash != hash || st.cfg.APIKeySalt != salt {
		t.Fatalf("legacy state not rolled back: %+v", st.cfg)
	}
}

func TestLoadAdminStateReportsBadJSON(t *testing.T) {
	path := filepath.Join(t.TempDir(), "admin.json")
	if err := os.WriteFile(path, []byte("{bad"), 0o600); err != nil {
		t.Fatal(err)
	}
	st := loadAdminState(path)
	if st.loadErr == nil {
		t.Fatal("expected loadErr")
	}
}

func TestLoadAdminStateRejectsDuplicateJSONKeys(t *testing.T) {
	path := filepath.Join(t.TempDir(), "admin.json")
	raw := []byte(`{"admin_user":"alice","admin_user":"bob"}`)
	if err := os.WriteFile(path, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	st := loadAdminState(path)
	if st.loadErr == nil || !strings.Contains(st.loadErr.Error(), "duplicate JSON key") {
		t.Fatalf("loadErr=%v want duplicate JSON key", st.loadErr)
	}
}

func TestLoadAdminStateRejectsHugeConfig(t *testing.T) {
	path := filepath.Join(t.TempDir(), "admin.json")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := f.Truncate(adminConfigFileLimit + 1); err != nil {
		_ = f.Close()
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	st := loadAdminState(path)
	if st.loadErr == nil || !strings.Contains(st.loadErr.Error(), "file too large") {
		t.Fatalf("loadErr=%v want file too large", st.loadErr)
	}
}

func TestWriteFileAtomicDoesNotRemoveDirectoryTarget(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "admin.json")
	if err := os.Mkdir(dir, 0700); err != nil {
		t.Fatal(err)
	}
	if err := writeFileAtomic(dir, []byte("{}"), 0600); err == nil {
		t.Fatal("expected error writing over directory")
	}
	st, err := os.Stat(dir)
	if err != nil {
		t.Fatal(err)
	}
	if !st.IsDir() {
		t.Fatal("directory target was removed")
	}
}

func TestAdminSecurityHeaders(t *testing.T) {
	st := loadAdminState(filepath.Join(t.TempDir(), "admin.json"))
	st.mu.Lock()
	if err := st.setPasswordLocked("admin", "password123"); err != nil {
		t.Fatal(err)
	}
	st.mu.Unlock()
	token, err := st.createSession()
	if err != nil {
		t.Fatal(err)
	}
	s := &server{admin: st}
	rec := httptest.NewRecorder()
	s.adminPage(rec, httptest.NewRequest(http.MethodGet, "/admin", nil))
	if got := rec.Header().Get("X-Frame-Options"); got != "DENY" {
		t.Fatalf("page X-Frame-Options=%q", got)
	}
	if got := rec.Header().Get("Cache-Control"); got != "no-store" {
		t.Fatalf("page Cache-Control=%q", got)
	}
	req := httptest.NewRequest(http.MethodGet, "/admin/api/config", nil)
	req.AddCookie(&http.Cookie{Name: "paddle_admin_session", Value: token})
	rec = httptest.NewRecorder()
	s.requireAdmin(s.adminConfigGet)(rec, req)
	if got := rec.Header().Get("Content-Security-Policy"); !strings.Contains(got, "frame-ancestors 'none'") {
		t.Fatalf("api CSP=%q", got)
	}
	if got := rec.Header().Get("X-Content-Type-Options"); got != "nosniff" {
		t.Fatalf("api nosniff=%q", got)
	}
}

func TestAdminSaveOverwritesExistingConfig(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "admin.json")
	st := loadAdminState(path)
	st.mu.Lock()
	if err := st.setPasswordLocked("admin", "password123"); err != nil {
		t.Fatal(err)
	}
	if err := st.saveLocked(); err != nil {
		t.Fatal(err)
	}
	st.cfg.AdminUser = "admin2"
	if err := st.saveLocked(); err != nil {
		t.Fatal(err)
	}
	st.mu.Unlock()
	reloaded := loadAdminState(path)
	if got := reloaded.response(""); got.AdminUser != "admin2" {
		t.Fatalf("admin_user=%q want admin2", got.AdminUser)
	}
	leftovers, err := filepath.Glob(filepath.Join(dir, "admin.json.tmp-*"))
	if err != nil {
		t.Fatal(err)
	}
	if len(leftovers) != 0 {
		t.Fatalf("leftover temp files=%v", leftovers)
	}
}

func TestAdminInitSetsSecureCookieBehindHTTPSProxy(t *testing.T) {
	modelDir := testModelDir(t)
	st := loadAdminState(filepath.Join(t.TempDir(), "admin.json"))
	s := &server{admin: st, modelDir: modelDir}
	req := httptest.NewRequest(http.MethodPost, "http://127.0.0.1:8080/admin/api/init", strings.NewReader(`{"admin_user":"admin","password":"password123"}`))
	req.Host = "127.0.0.1:8080"
	req.RemoteAddr = "127.0.0.1:1234"
	req.Header.Set("X-Forwarded-Proto", "https")
	rec := httptest.NewRecorder()
	s.adminInit(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%q", rec.Code, rec.Body.String())
	}
	c := responseCookie(t, rec, "paddle_admin_session")
	if !c.Secure {
		t.Fatalf("admin session cookie Secure=false")
	}
}

func TestAdminLoginAndLogoutCookieSecureFollowsRequestScheme(t *testing.T) {
	st := loadAdminState(filepath.Join(t.TempDir(), "admin.json"))
	st.mu.Lock()
	if err := st.setPasswordLocked("admin", "password123"); err != nil {
		t.Fatal(err)
	}
	st.mu.Unlock()
	s := &server{admin: st}
	req := httptest.NewRequest(http.MethodPost, "http://127.0.0.1:8080/admin/api/login", strings.NewReader(`{"admin_user":"admin","password":"password123"}`))
	req.RemoteAddr = "127.0.0.1:1234"
	req.Header.Set("X-Forwarded-Proto", "https")
	rec := httptest.NewRecorder()
	s.adminLogin(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("https login status=%d body=%q", rec.Code, rec.Body.String())
	}
	c := responseCookie(t, rec, "paddle_admin_session")
	if !c.Secure {
		t.Fatalf("https login cookie Secure=false")
	}
	req = httptest.NewRequest(http.MethodPost, "http://127.0.0.1:8080/admin/api/login", strings.NewReader(`{"admin_user":"admin","password":"password123"}`))
	rec = httptest.NewRecorder()
	s.adminLogin(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("http login status=%d body=%q", rec.Code, rec.Body.String())
	}
	c = responseCookie(t, rec, "paddle_admin_session")
	if c.Secure {
		t.Fatalf("http login cookie Secure=true")
	}
	req = httptest.NewRequest(http.MethodPost, "http://127.0.0.1:8080/admin/api/logout", nil)
	req.RemoteAddr = "127.0.0.1:1234"
	req.Header.Set("X-Forwarded-Proto", "https")
	req.AddCookie(&http.Cookie{Name: "paddle_admin_session", Value: c.Value})
	rec = httptest.NewRecorder()
	s.adminLogout(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("logout status=%d body=%q", rec.Code, rec.Body.String())
	}
	c = responseCookie(t, rec, "paddle_admin_session")
	if !c.Secure || c.MaxAge != -1 || c.Expires.IsZero() {
		t.Fatalf("logout cookie Secure=%v MaxAge=%d Expires=%v", c.Secure, c.MaxAge, c.Expires)
	}
}

func TestCreateSessionPrunesExpiredSessions(t *testing.T) {
	st := loadAdminState(filepath.Join(t.TempDir(), "admin.json"))
	st.mu.Lock()
	st.sessions["expired"] = time.Now().Add(-time.Hour)
	st.sessions["active"] = time.Now().Add(time.Hour)
	st.mu.Unlock()
	token, err := st.createSession()
	if err != nil {
		t.Fatal(err)
	}
	st.mu.RLock()
	defer st.mu.RUnlock()
	if _, ok := st.sessions["expired"]; ok {
		t.Fatal("expired session was not pruned")
	}
	if _, ok := st.sessions["active"]; !ok {
		t.Fatal("active session was pruned")
	}
	if _, ok := st.sessions[token]; !ok {
		t.Fatal("new session missing")
	}
}

func TestAdminSessionCookie(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "/admin", nil)
	if _, ok := adminSessionCookie(req); ok {
		t.Fatal("empty request should not have session")
	}
	req.Header.Add("Cookie", "x=1; paddle_admin_session=abc123; y=2")
	if got, ok := adminSessionCookie(req); !ok || got != "abc123" {
		t.Fatalf("session=%q ok=%v", got, ok)
	}
	req.Header.Set("Cookie", "paddle_admin_session=")
	if _, ok := adminSessionCookie(req); ok {
		t.Fatal("empty session should not be accepted")
	}
	req.Header = http.Header{"cookie": []string{"paddle_admin_session=lower"}}
	if got, ok := adminSessionCookie(req); !ok || got != "lower" {
		t.Fatalf("lowercase cookie session=%q ok=%v", got, ok)
	}
}

func BenchmarkAdminSessionCookie(b *testing.B) {
	req := httptest.NewRequest(http.MethodGet, "/admin", nil)
	req.Header.Set("Cookie", "x=1; paddle_admin_session=abc123; y=2")
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if got, ok := adminSessionCookie(req); !ok || got != "abc123" {
			b.Fatalf("session=%q ok=%v", got, ok)
		}
	}
}

func BenchmarkAdminRequestOK(b *testing.B) {
	st := loadAdminState(filepath.Join(b.TempDir(), "admin.json"))
	token, err := st.createSession()
	if err != nil {
		b.Fatal(err)
	}
	s := &server{admin: st}
	req := httptest.NewRequest(http.MethodGet, "/admin", nil)
	req.AddCookie(&http.Cookie{Name: "paddle_admin_session", Value: token})
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if !s.adminRequestOK(req) {
			b.Fatal("not authenticated")
		}
	}
}

func TestAdminTemplateEscapesKeyNameAttribute(t *testing.T) {
	if !strings.Contains(adminHTML, `value="'+escAttr(k.name)+'"`) {
		t.Fatal("key name input no longer uses escAttr")
	}
	if !strings.Contains(adminHTML, `function escAttr(s){return esc(String(s));}`) {
		t.Fatal("escAttr must HTML-escape double quotes for input value attributes")
	}
	if !strings.Contains(adminHTML, `function jsArg(v){return escAttr(JSON.stringify(String(v)));}`) {
		t.Fatal("onclick key id arguments must be JSON-stringified and HTML-escaped")
	}
	if !strings.Contains(adminHTML, `function keyField(prefix,id){return prefix+String(id).replace(/[^A-Za-z0-9_-]/g,'_');}`) {
		t.Fatal("key DOM ids must be sanitized")
	}
	if !strings.Contains(adminHTML, `let overview=null,overviewTimer=null;`) || !strings.Contains(adminHTML, `if(!overviewTimer)overviewTimer=setInterval(loadOverview,5000)`) {
		t.Fatal("overview polling timer must be single-instance")
	}
	if !strings.Contains(adminHTML, `function stopOverviewTimer(){if(overviewTimer){clearInterval(overviewTimer);overviewTimer=null;}}`) {
		t.Fatal("overview polling timer must be cleared when leaving the app")
	}
}

func testModelDir(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	configJSON := `{
		"vocab_size": 2,
		"hidden_size": 4,
		"intermediate_size": 8,
		"num_hidden_layers": 0,
		"num_attention_heads": 2,
		"num_key_value_heads": 1,
		"head_dim": 2,
		"vision_config": {"num_hidden_layers": 0}
	}`
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(configJSON), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "model.gguf"), []byte("weights"), 0600); err != nil {
		t.Fatal(err)
	}
	return dir
}

func responseCookie(t *testing.T, rec *httptest.ResponseRecorder, name string) *http.Cookie {
	t.Helper()
	for _, c := range rec.Result().Cookies() {
		if c.Name == name {
			return c
		}
	}
	t.Fatalf("missing response cookie %q in %v", name, rec.Header().Values("Set-Cookie"))
	return nil
}

func TestAdminRejectsCrossOriginWrites(t *testing.T) {
	st := loadAdminState(filepath.Join(t.TempDir(), "admin.json"))
	st.mu.Lock()
	if err := st.setPasswordLocked("admin", "password123"); err != nil {
		t.Fatal(err)
	}
	st.mu.Unlock()
	token, err := st.createSession()
	if err != nil {
		t.Fatal(err)
	}
	s := &server{admin: st}
	req := httptest.NewRequest(http.MethodPost, "http://127.0.0.1:8080/admin/api/audit/clear", nil)
	req.Host = "127.0.0.1:8080"
	req.Header.Set("Origin", "http://evil.example")
	req.AddCookie(&http.Cookie{Name: "paddle_admin_session", Value: token})
	rec := httptest.NewRecorder()
	s.requireAdmin(s.adminAuditClear)(rec, req)
	if rec.Code != http.StatusForbidden {
		t.Fatalf("cross-origin status=%d body=%q", rec.Code, rec.Body.String())
	}
	req = httptest.NewRequest(http.MethodPost, "http://127.0.0.1:8080/admin/api/audit/clear", nil)
	req.Host = "127.0.0.1:8080"
	req.Header.Set("Origin", "http://127.0.0.1:8080")
	req.AddCookie(&http.Cookie{Name: "paddle_admin_session", Value: token})
	rec = httptest.NewRecorder()
	s.requireAdmin(s.adminAuditClear)(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("same-origin status=%d body=%q", rec.Code, rec.Body.String())
	}
}

func BenchmarkAllowAdminWriteOrigin(b *testing.B) {
	req := httptest.NewRequest(http.MethodPost, "http://127.0.0.1:8080/admin/api/audit/clear", nil)
	req.Host = "127.0.0.1:8080"
	req.Header.Set("Origin", "http://127.0.0.1:8080")
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if !allowAdminWriteOrigin(req) {
			b.Fatal("blocked")
		}
	}
}

func BenchmarkSameOrigin(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if !sameOrigin("http://127.0.0.1:8080/admin", "http", "127.0.0.1:8080") {
			b.Fatal("not same")
		}
	}
}

func TestAdminLoginRejectsCrossOrigin(t *testing.T) {
	st := loadAdminState(filepath.Join(t.TempDir(), "admin.json"))
	st.mu.Lock()
	if err := st.setPasswordLocked("admin", "password123"); err != nil {
		t.Fatal(err)
	}
	st.mu.Unlock()
	s := &server{admin: st}
	req := httptest.NewRequest(http.MethodPost, "http://127.0.0.1:8080/admin/api/login", strings.NewReader(`{"admin_user":"admin","password":"password123"}`))
	req.Host = "127.0.0.1:8080"
	req.Header.Set("Origin", "http://evil.example")
	rec := httptest.NewRecorder()
	s.adminLogin(rec, req)
	if rec.Code != http.StatusForbidden {
		t.Fatalf("status=%d body=%q", rec.Code, rec.Body.String())
	}
}

func TestAdminWriteRejectsOversizedBody(t *testing.T) {
	st := loadAdminState(filepath.Join(t.TempDir(), "admin.json"))
	st.mu.Lock()
	if err := st.setPasswordLocked("admin", "password123"); err != nil {
		t.Fatal(err)
	}
	if err := st.saveLocked(); err != nil {
		t.Fatal(err)
	}
	st.mu.Unlock()
	s := &server{admin: st}
	body := `{"admin_user":"admin","password":"` + strings.Repeat("x", int(adminRequestLimit)) + `"}`
	req := httptest.NewRequest(http.MethodPost, "/admin/api/login", strings.NewReader(body))
	rec := httptest.NewRecorder()
	s.adminLogin(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("login status=%d body=%q", rec.Code, rec.Body.String())
	}

	token, err := st.createSession()
	if err != nil {
		t.Fatal(err)
	}
	req = httptest.NewRequest(http.MethodPost, "/admin/api/config/restore", strings.NewReader(strings.Repeat(" ", int(adminRequestLimit)+1)))
	req.AddCookie(&http.Cookie{Name: "paddle_admin_session", Value: token})
	rec = httptest.NewRecorder()
	s.requireAdmin(s.adminConfigRestore)(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("restore status=%d body=%q", rec.Code, rec.Body.String())
	}
}

func TestAdminLoginFailureLimit(t *testing.T) {
	st := loadAdminState(filepath.Join(t.TempDir(), "admin.json"))
	st.mu.Lock()
	if err := st.setPasswordLocked("admin", "password123"); err != nil {
		t.Fatal(err)
	}
	st.mu.Unlock()
	s := &server{admin: st}
	for i := 0; i < 5; i++ {
		req := httptest.NewRequest(http.MethodPost, "/admin/api/login", strings.NewReader(`{"admin_user":"admin","password":"bad"}`))
		req.RemoteAddr = "203.0.113.10:1234"
		rec := httptest.NewRecorder()
		s.adminLogin(rec, req)
		if rec.Code != http.StatusUnauthorized {
			t.Fatalf("attempt %d status=%d body=%q", i, rec.Code, rec.Body.String())
		}
	}
	req := httptest.NewRequest(http.MethodPost, "/admin/api/login", strings.NewReader(`{"admin_user":"admin","password":"password123"}`))
	req.RemoteAddr = "203.0.113.10:1234"
	rec := httptest.NewRecorder()
	s.adminLogin(rec, req)
	if rec.Code != http.StatusTooManyRequests {
		t.Fatalf("locked status=%d body=%q", rec.Code, rec.Body.String())
	}
}

func TestAdminUserNameLengthLimit(t *testing.T) {
	if _, err := normalizeAdminUser(strings.Repeat("a", 81)); err == nil {
		t.Fatal("expected long admin_user error")
	}
	modelDir := testModelDir(t)
	st := loadAdminState(filepath.Join(t.TempDir(), "admin.json"))
	s := &server{admin: st, modelDir: modelDir}
	longUser := strings.Repeat("a", 81)
	req := httptest.NewRequest(http.MethodPost, "/admin/api/init", strings.NewReader(`{"admin_user":"`+longUser+`","password":"password123","model_dir":"`+strings.ReplaceAll(modelDir, `\`, `\\`)+`"}`))
	rec := httptest.NewRecorder()
	s.adminInit(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("init status=%d body=%q", rec.Code, rec.Body.String())
	}

	st.mu.Lock()
	if err := st.setPasswordLocked("admin", "password123"); err != nil {
		t.Fatal(err)
	}
	if err := st.saveLocked(); err != nil {
		t.Fatal(err)
	}
	st.mu.Unlock()
	token, err := st.createSession()
	if err != nil {
		t.Fatal(err)
	}
	req = httptest.NewRequest(http.MethodPost, "/admin/api/config", strings.NewReader(`{"admin_user":"`+longUser+`","model_dir":"`+strings.ReplaceAll(modelDir, `\`, `\\`)+`"}`))
	req.AddCookie(&http.Cookie{Name: "paddle_admin_session", Value: token})
	rec = httptest.NewRecorder()
	s.requireAdmin(s.adminConfigSave)(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("config status=%d body=%q", rec.Code, rec.Body.String())
	}
	if got := st.response(modelDir); got.AdminUser != "admin" {
		t.Fatalf("admin_user changed after rejection: %+v", got)
	}
}

func TestAdminConfigSaveRollsBackUserOnPasswordValidationError(t *testing.T) {
	modelDir := testModelDir(t)
	st := loadAdminState(filepath.Join(t.TempDir(), "admin.json"))
	st.mu.Lock()
	if err := st.setPasswordLocked("admin", "password123"); err != nil {
		t.Fatal(err)
	}
	st.cfg.ModelDir = modelDir
	if err := st.saveLocked(); err != nil {
		t.Fatal(err)
	}
	st.mu.Unlock()
	token, err := st.createSession()
	if err != nil {
		t.Fatal(err)
	}
	s := &server{admin: st, modelDir: modelDir}
	req := httptest.NewRequest(http.MethodPost, "/admin/api/config", strings.NewReader(`{"admin_user":"changed","password":"short","model_dir":"`+strings.ReplaceAll(modelDir, `\`, `\\`)+`"}`))
	req.AddCookie(&http.Cookie{Name: "paddle_admin_session", Value: token})
	rec := httptest.NewRecorder()
	s.requireAdmin(s.adminConfigSave)(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status=%d body=%q", rec.Code, rec.Body.String())
	}
	if got := st.response(modelDir); got.AdminUser != "admin" {
		t.Fatalf("admin_user changed after password rejection: %+v", got)
	}
}

func TestAdminLoginSuccessClearsFailures(t *testing.T) {
	st := loadAdminState(filepath.Join(t.TempDir(), "admin.json"))
	st.mu.Lock()
	if err := st.setPasswordLocked("admin", "password123"); err != nil {
		t.Fatal(err)
	}
	st.mu.Unlock()
	s := &server{admin: st}
	req := httptest.NewRequest(http.MethodPost, "/admin/api/login", strings.NewReader(`{"admin_user":"admin","password":"bad"}`))
	req.RemoteAddr = "203.0.113.11:1234"
	rec := httptest.NewRecorder()
	s.adminLogin(rec, req)
	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("bad status=%d", rec.Code)
	}
	req = httptest.NewRequest(http.MethodPost, "/admin/api/login", strings.NewReader(`{"admin_user":"admin","password":"password123"}`))
	req.RemoteAddr = "203.0.113.11:1234"
	rec = httptest.NewRecorder()
	s.adminLogin(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("good status=%d body=%q", rec.Code, rec.Body.String())
	}
	if st.loginLocked("203.0.113.11", time.Now()) {
		t.Fatal("login failures were not cleared")
	}
}

func TestAdminAPIKeyRateLimit(t *testing.T) {
	st := loadAdminState(filepath.Join(t.TempDir(), "admin.json"))
	st.mu.Lock()
	key, managed, err := st.createAPIKeyLocked("agent", 0, 1)
	if err != nil {
		t.Fatal(err)
	}
	if err := st.saveLocked(); err != nil {
		t.Fatal(err)
	}
	st.mu.Unlock()
	if managed.RatePerMin != 1 {
		t.Fatalf("rate=%d", managed.RatePerMin)
	}
	if _, err := st.consumeAPIKey(key, "127.0.0.1"); err != nil {
		t.Fatalf("first consume err=%v", err)
	}
	if _, err := st.consumeAPIKey(key, "127.0.0.1"); !errors.Is(err, errAPIKeyRate) {
		t.Fatalf("second consume err=%v want rate", err)
	}
	keys := st.keyResponses()
	if len(keys) != 1 || keys[0].RatePerMin != 1 || keys[0].Used != 1 {
		t.Fatalf("keys=%+v", keys)
	}
}

func TestRequireAPIKeyStatusCodes(t *testing.T) {
	st := loadAdminState(filepath.Join(t.TempDir(), "admin.json"))
	st.mu.Lock()
	if err := st.setPasswordLocked("admin", "password123"); err != nil {
		t.Fatal(err)
	}
	key, managed, err := st.createAPIKeyLocked("limited", 1, 0)
	if err != nil {
		t.Fatal(err)
	}
	if err := st.saveLocked(); err != nil {
		t.Fatal(err)
	}
	st.mu.Unlock()
	s := &server{admin: st}
	handler := s.requireAPIKey(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	})
	rec := httptest.NewRecorder()
	handler(rec, httptest.NewRequest(http.MethodPost, "/v1/generate", nil))
	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("missing key status=%d", rec.Code)
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/generate", nil)
	req.Header.Set("Authorization", "Bearer "+key)
	rec = httptest.NewRecorder()
	handler(rec, req)
	if rec.Code != http.StatusNoContent {
		t.Fatalf("first key status=%d body=%q", rec.Code, rec.Body.String())
	}
	req = httptest.NewRequest(http.MethodPost, "/v1/generate", nil)
	req.Header.Set("Authorization", "Bearer "+key)
	rec = httptest.NewRecorder()
	handler(rec, req)
	if rec.Code != http.StatusTooManyRequests {
		t.Fatalf("quota status=%d body=%q", rec.Code, rec.Body.String())
	}
	st.mu.Lock()
	for i := range st.cfg.APIKeys {
		if st.cfg.APIKeys[i].ID == managed.ID {
			st.cfg.APIKeys[i].Quota = 0
			st.cfg.APIKeys[i].Disabled = true
		}
	}
	st.mu.Unlock()
	req = httptest.NewRequest(http.MethodPost, "/v1/generate", nil)
	req.Header.Set("Authorization", "Bearer "+key)
	rec = httptest.NewRecorder()
	handler(rec, req)
	if rec.Code != http.StatusForbidden {
		t.Fatalf("disabled status=%d body=%q", rec.Code, rec.Body.String())
	}
}

func TestRequireAPIKeyAcceptsXAPIKeyHeader(t *testing.T) {
	st := loadAdminState(filepath.Join(t.TempDir(), "admin.json"))
	st.mu.Lock()
	if err := st.setPasswordLocked("admin", "password123"); err != nil {
		t.Fatal(err)
	}
	key, _, err := st.createAPIKeyLocked("header", 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if err := st.saveLocked(); err != nil {
		t.Fatal(err)
	}
	st.mu.Unlock()
	s := &server{admin: st}
	handler := s.requireAPIKey(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/generate", nil)
	req.Header.Set("X-API-Key", key)
	rec := httptest.NewRecorder()
	handler(rec, req)
	if rec.Code != http.StatusNoContent {
		t.Fatalf("status=%d body=%q", rec.Code, rec.Body.String())
	}
}

func TestAPIKeyFromHeaders(t *testing.T) {
	h := http.Header{}
	if _, ok := apiKeyFromHeaders(h); ok {
		t.Fatal("empty headers should not contain key")
	}
	h.Set("Authorization", "Bearer   secret  ")
	if got, ok := apiKeyFromHeaders(h); !ok || got != "secret" {
		t.Fatalf("bearer got=%q ok=%v", got, ok)
	}
	h.Set("Authorization", "Basic secret")
	h.Set("X-API-Key", "  fallback  ")
	if got, ok := apiKeyFromHeaders(h); !ok || got != "fallback" {
		t.Fatalf("x-api-key got=%q ok=%v", got, ok)
	}
	h = http.Header{"authorization": []string{"Bearer lower-auth"}}
	if got, ok := apiKeyFromHeaders(h); !ok || got != "lower-auth" {
		t.Fatalf("lowercase authorization got=%q ok=%v", got, ok)
	}
	h = http.Header{"Authorization": []string{"bearer\tmixed-case"}}
	if got, ok := apiKeyFromHeaders(h); !ok || got != "mixed-case" {
		t.Fatalf("case-insensitive bearer got=%q ok=%v", got, ok)
	}
	h = http.Header{"x-api-key": []string{"lower-key"}}
	if got, ok := apiKeyFromHeaders(h); !ok || got != "lower-key" {
		t.Fatalf("lowercase x-api-key got=%q ok=%v", got, ok)
	}
}

func BenchmarkAPIKeyFromHeadersBearer(b *testing.B) {
	h := http.Header{}
	h.Set("Authorization", "Bearer paddle_key_123")
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if got, ok := apiKeyFromHeaders(h); !ok || got != "paddle_key_123" {
			b.Fatalf("got=%q ok=%v", got, ok)
		}
	}
}

func BenchmarkAPIKeyFromHeadersXAPIKey(b *testing.B) {
	h := http.Header{}
	h.Set("X-API-Key", "paddle_key_123")
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if got, ok := apiKeyFromHeaders(h); !ok || got != "paddle_key_123" {
			b.Fatalf("got=%q ok=%v", got, ok)
		}
	}
}

func TestRequireAPIKeyRejectsWhenInitializedWithoutKeys(t *testing.T) {
	st := loadAdminState(filepath.Join(t.TempDir(), "admin.json"))
	st.mu.Lock()
	if err := st.setPasswordLocked("admin", "password123"); err != nil {
		t.Fatal(err)
	}
	if err := st.saveLocked(); err != nil {
		t.Fatal(err)
	}
	st.mu.Unlock()
	s := &server{admin: st}
	handler := s.requireAPIKey(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/generate", nil)
	req.Header.Set("Authorization", "Bearer anything")
	rec := httptest.NewRecorder()
	handler(rec, req)
	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("status=%d body=%q", rec.Code, rec.Body.String())
	}
}

func TestRequireAPIKeyRecordsAudit(t *testing.T) {
	st := loadAdminState(filepath.Join(t.TempDir(), "admin.json"))
	st.mu.Lock()
	if err := st.setPasswordLocked("admin", "password123"); err != nil {
		t.Fatal(err)
	}
	key, _, err := st.createAPIKeyLocked("audit", 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if err := st.saveLocked(); err != nil {
		t.Fatal(err)
	}
	st.mu.Unlock()
	s := &server{admin: st}
	handler := s.requireAPIKey(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusCreated)
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/generate", nil)
	req.RemoteAddr = "127.0.0.1:1234"
	req.Header.Set("Authorization", "Bearer "+key)
	req.Header.Set("X-Real-IP", "198.51.100.9")
	rec := httptest.NewRecorder()
	handler(rec, req)
	if rec.Code != http.StatusCreated {
		t.Fatalf("status=%d", rec.Code)
	}
	entries := st.auditEntries()
	if len(entries) != 1 || entries[0].Status != http.StatusCreated || entries[0].KeyName != "audit" || entries[0].ClientIP != "198.51.100.9" {
		t.Fatalf("entries=%+v", entries)
	}
	errorHandler := s.requireAPIKey(func(w http.ResponseWriter, r *http.Request) {
		writeError(w, http.StatusBadRequest, errors.New("downstream validation failed"))
	})
	req = httptest.NewRequest(http.MethodPost, "/v1/generate", nil)
	req.Header.Set("Authorization", "Bearer "+key)
	rec = httptest.NewRecorder()
	errorHandler(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("error status=%d", rec.Code)
	}
	entries = st.auditEntries()
	if len(entries) != 2 || entries[0].Error != "downstream validation failed" {
		t.Fatalf("entries=%+v", entries)
	}
	rec = httptest.NewRecorder()
	handler(rec, httptest.NewRequest(http.MethodPost, "/v1/ocr", nil))
	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("missing status=%d", rec.Code)
	}
	entries = st.auditEntries()
	if len(entries) != 3 || entries[0].Status != http.StatusUnauthorized || entries[0].Error == "" {
		t.Fatalf("entries=%+v", entries)
	}
}

func TestRequireAPIKeyAuditKeepsFirstWrittenStatus(t *testing.T) {
	st := loadAdminState(filepath.Join(t.TempDir(), "admin.json"))
	st.mu.Lock()
	if err := st.setPasswordLocked("admin", "password123"); err != nil {
		t.Fatal(err)
	}
	key, _, err := st.createAPIKeyLocked("audit", 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if err := st.saveLocked(); err != nil {
		t.Fatal(err)
	}
	st.mu.Unlock()
	s := &server{admin: st}
	handler := s.requireAPIKey(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusAccepted)
		w.WriteHeader(http.StatusInternalServerError)
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/generate", nil)
	req.Header.Set("Authorization", "Bearer "+key)
	rec := httptest.NewRecorder()
	handler(rec, req)
	if rec.Code != http.StatusAccepted {
		t.Fatalf("status=%d", rec.Code)
	}
	entries := st.auditEntries()
	if len(entries) != 1 || entries[0].Status != http.StatusAccepted {
		t.Fatalf("entries=%+v", entries)
	}
}

func TestAdminAuditClear(t *testing.T) {
	st := loadAdminState(filepath.Join(t.TempDir(), "admin.json"))
	st.mu.Lock()
	if err := st.setPasswordLocked("admin", "password123"); err != nil {
		t.Fatal(err)
	}
	st.mu.Unlock()
	st.recordAudit(auditEntry{Time: time.Now().UTC().Format(time.RFC3339), Method: http.MethodPost, Path: "/v1/ocr", Status: http.StatusOK})
	token, err := st.createSession()
	if err != nil {
		t.Fatal(err)
	}
	s := &server{admin: st}
	req := httptest.NewRequest(http.MethodPost, "/admin/api/audit/clear", nil)
	req.AddCookie(&http.Cookie{Name: "paddle_admin_session", Value: token})
	rec := httptest.NewRecorder()
	s.requireAdmin(s.adminAuditClear)(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%q", rec.Code, rec.Body.String())
	}
	if got := st.auditEntries(); len(got) != 0 {
		t.Fatalf("audit entries=%+v", got)
	}
}

func TestAdminAuditCSV(t *testing.T) {
	st := loadAdminState(filepath.Join(t.TempDir(), "admin.json"))
	st.mu.Lock()
	if err := st.setPasswordLocked("admin", "password123"); err != nil {
		t.Fatal(err)
	}
	st.mu.Unlock()
	st.recordAudit(auditEntry{
		Time:       "2026-06-13T00:00:00Z",
		KeyID:      "key-1",
		KeyName:    "=agent",
		KeyPreview: "abcd...wxyz",
		ClientIP:   "203.0.113.9",
		Method:     http.MethodPost,
		Path:       "/v1/ocr",
		Status:     http.StatusTooManyRequests,
		DurationMS: 17,
		Error:      "API key quota exceeded",
	})
	token, err := st.createSession()
	if err != nil {
		t.Fatal(err)
	}
	s := &server{admin: st}
	req := httptest.NewRequest(http.MethodGet, "/admin/api/audit.csv", nil)
	req.AddCookie(&http.Cookie{Name: "paddle_admin_session", Value: token})
	rec := httptest.NewRecorder()
	s.requireAdmin(s.adminAuditCSV)(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%q", rec.Code, rec.Body.String())
	}
	body := rec.Body.String()
	if !strings.Contains(body, "time,key_id,key_name") || !strings.Contains(body, "'=agent") || !strings.Contains(body, "429") {
		t.Fatalf("csv=%q", body)
	}
	if ct := rec.Header().Get("Content-Type"); !strings.Contains(ct, "text/csv") {
		t.Fatalf("content-type=%q", ct)
	}
}

func TestAdminConfigBackupAndRestore(t *testing.T) {
	liveModelDir := testModelDir(t)
	restoredModelDir := testModelDir(t)
	st := loadAdminState(filepath.Join(t.TempDir(), "admin.json"))
	st.mu.Lock()
	if err := st.setPasswordLocked("admin", "password123"); err != nil {
		t.Fatal(err)
	}
	_, keyMeta, err := st.createAPIKeyLocked("backup", 9, 3)
	if err != nil {
		t.Fatal(err)
	}
	st.cfg.ModelDir = liveModelDir
	if err := st.saveLocked(); err != nil {
		t.Fatal(err)
	}
	st.mu.Unlock()
	token, err := st.createSession()
	if err != nil {
		t.Fatal(err)
	}
	s := &server{admin: st, modelDir: liveModelDir}
	req := httptest.NewRequest(http.MethodGet, "/admin/api/config/backup", nil)
	req.AddCookie(&http.Cookie{Name: "paddle_admin_session", Value: token})
	rec := httptest.NewRecorder()
	s.requireAdmin(s.adminConfigBackup)(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("backup status=%d body=%q", rec.Code, rec.Body.String())
	}
	var backup adminConfigBackup
	if err := json.Unmarshal(rec.Body.Bytes(), &backup); err != nil {
		t.Fatal(err)
	}
	if backup.Version != 1 || len(backup.Config.APIKeys) != 1 || backup.Config.APIKeys[0].ID != keyMeta.ID {
		t.Fatalf("backup=%+v", backup)
	}
	backup.Config.AdminUser = " restored "
	backup.Config.ModelDir = restoredModelDir
	backup.Config.PostProcessDir = " post "
	body, err := json.Marshal(backup)
	if err != nil {
		t.Fatal(err)
	}
	req = httptest.NewRequest(http.MethodPost, "/admin/api/config/restore", strings.NewReader(string(body)))
	req.AddCookie(&http.Cookie{Name: "paddle_admin_session", Value: token})
	rec = httptest.NewRecorder()
	s.requireAdmin(s.adminConfigRestore)(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("restore status=%d body=%q", rec.Code, rec.Body.String())
	}
	got := st.response(s.modelDir)
	if got.AdminUser != "restored" || got.ModelDir != restoredModelDir || got.PostProcessDir != "post" || !got.RestartRequired {
		t.Fatalf("config=%+v", got)
	}
}

func TestCloneAdminConfigFileCopiesAPIKeys(t *testing.T) {
	cfg := adminConfigFile{
		AdminUser: "admin",
		APIKeys:   []managedAPIKey{{ID: "key1", Name: "original"}},
	}
	cloned := cloneAdminConfigFile(cfg)
	cfg.APIKeys[0].Name = "mutated"
	if cloned.APIKeys[0].Name != "original" {
		t.Fatalf("clone shared APIKeys backing array: %+v", cloned.APIKeys)
	}
}

func TestAdminConfigRestoreRejectsBadBackup(t *testing.T) {
	st := loadAdminState(filepath.Join(t.TempDir(), "admin.json"))
	st.mu.Lock()
	if err := st.setPasswordLocked("admin", "password123"); err != nil {
		t.Fatal(err)
	}
	st.mu.Unlock()
	token, err := st.createSession()
	if err != nil {
		t.Fatal(err)
	}
	s := &server{admin: st}
	req := httptest.NewRequest(http.MethodPost, "/admin/api/config/restore", strings.NewReader(`{"version":99,"config":{}}`))
	req.AddCookie(&http.Cookie{Name: "paddle_admin_session", Value: token})
	rec := httptest.NewRecorder()
	s.requireAdmin(s.adminConfigRestore)(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status=%d body=%q", rec.Code, rec.Body.String())
	}
	backup := adminConfigBackup{
		Version: 1,
		Config: adminConfigFile{
			AdminUser:    "admin",
			PasswordSalt: "salt",
			PasswordHash: strings.Repeat("a", 64),
			APIKeys: []managedAPIKey{{
				ID:   `bad');alert(1)//`,
				Name: "bad",
				Salt: "salt",
				Hash: strings.Repeat("b", 64),
			}},
		},
	}
	body, err := json.Marshal(backup)
	if err != nil {
		t.Fatal(err)
	}
	req = httptest.NewRequest(http.MethodPost, "/admin/api/config/restore", strings.NewReader(string(body)))
	req.AddCookie(&http.Cookie{Name: "paddle_admin_session", Value: token})
	rec = httptest.NewRecorder()
	s.requireAdmin(s.adminConfigRestore)(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("malicious key id status=%d body=%q", rec.Code, rec.Body.String())
	}
	backup.Config.APIKeys = []managedAPIKey{
		{ID: "dup", Name: "a", Salt: "salt", Hash: strings.Repeat("b", 64)},
		{ID: "dup", Name: "b", Salt: "salt", Hash: strings.Repeat("c", 64)},
	}
	body, err = json.Marshal(backup)
	if err != nil {
		t.Fatal(err)
	}
	req = httptest.NewRequest(http.MethodPost, "/admin/api/config/restore", strings.NewReader(string(body)))
	req.AddCookie(&http.Cookie{Name: "paddle_admin_session", Value: token})
	rec = httptest.NewRecorder()
	s.requireAdmin(s.adminConfigRestore)(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("duplicate key id status=%d body=%q", rec.Code, rec.Body.String())
	}
	backup.Config.APIKeys = []managedAPIKey{{ID: "valid", Name: "bad-hash", Salt: "salt", Hash: "not-a-sha256"}}
	body, err = json.Marshal(backup)
	if err != nil {
		t.Fatal(err)
	}
	req = httptest.NewRequest(http.MethodPost, "/admin/api/config/restore", strings.NewReader(string(body)))
	req.AddCookie(&http.Cookie{Name: "paddle_admin_session", Value: token})
	rec = httptest.NewRecorder()
	s.requireAdmin(s.adminConfigRestore)(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("bad key hash status=%d body=%q", rec.Code, rec.Body.String())
	}
	backup.Config.APIKeys = []managedAPIKey{{ID: "valid", Name: strings.Repeat("x", 81), Salt: "salt", Hash: strings.Repeat("b", 64)}}
	body, err = json.Marshal(backup)
	if err != nil {
		t.Fatal(err)
	}
	req = httptest.NewRequest(http.MethodPost, "/admin/api/config/restore", strings.NewReader(string(body)))
	req.AddCookie(&http.Cookie{Name: "paddle_admin_session", Value: token})
	rec = httptest.NewRecorder()
	s.requireAdmin(s.adminConfigRestore)(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("long key name status=%d body=%q", rec.Code, rec.Body.String())
	}
	backup.Config.AdminUser = strings.Repeat("a", 81)
	backup.Config.APIKeys = []managedAPIKey{{ID: "valid", Name: "ok", Salt: "salt", Hash: strings.Repeat("b", 64)}}
	body, err = json.Marshal(backup)
	if err != nil {
		t.Fatal(err)
	}
	req = httptest.NewRequest(http.MethodPost, "/admin/api/config/restore", strings.NewReader(string(body)))
	req.AddCookie(&http.Cookie{Name: "paddle_admin_session", Value: token})
	rec = httptest.NewRecorder()
	s.requireAdmin(s.adminConfigRestore)(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("long admin_user status=%d body=%q", rec.Code, rec.Body.String())
	}
	backup.Config.AdminUser = "admin"
	backup.Config.APIKeys = []managedAPIKey{{ID: "valid", Name: "ok", Salt: "salt", Hash: strings.Repeat("b", 64)}}
	backup.Config.ModelDir = filepath.Join(t.TempDir(), "missing-model")
	body, err = json.Marshal(backup)
	if err != nil {
		t.Fatal(err)
	}
	req = httptest.NewRequest(http.MethodPost, "/admin/api/config/restore", strings.NewReader(string(body)))
	req.AddCookie(&http.Cookie{Name: "paddle_admin_session", Value: token})
	rec = httptest.NewRecorder()
	s.requireAdmin(s.adminConfigRestore)(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("missing model dir status=%d body=%q", rec.Code, rec.Body.String())
	}
}

func TestAdminConfigMutationsRollbackWhenSaveFails(t *testing.T) {
	modelDir := testModelDir(t)
	badPath := filepath.Join(t.TempDir(), "bad\x00name", "admin.json")
	st := loadAdminState(badPath)
	s := &server{admin: st, modelDir: modelDir}
	req := httptest.NewRequest(http.MethodPost, "/admin/api/init", strings.NewReader(`{"admin_user":"admin","password":"password123"}`))
	rec := httptest.NewRecorder()
	s.adminInit(rec, req)
	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("init status=%d body=%q", rec.Code, rec.Body.String())
	}
	if st.initialized() {
		t.Fatal("init save failure left admin initialized")
	}

	goodPath := filepath.Join(t.TempDir(), "admin.json")
	st = loadAdminState(goodPath)
	st.mu.Lock()
	if err := st.setPasswordLocked("admin", "password123"); err != nil {
		t.Fatal(err)
	}
	st.cfg.ModelDir = modelDir
	if err := st.saveLocked(); err != nil {
		t.Fatal(err)
	}
	st.mu.Unlock()
	token, err := st.createSession()
	if err != nil {
		t.Fatal(err)
	}
	st.path = badPath
	s = &server{admin: st, modelDir: modelDir}
	req = httptest.NewRequest(http.MethodPost, "/admin/api/config", strings.NewReader(`{"admin_user":"changed","password":"","model_dir":"`+strings.ReplaceAll(modelDir, `\`, `\\`)+`","post_process_dir":"post"}`))
	req.AddCookie(&http.Cookie{Name: "paddle_admin_session", Value: token})
	rec = httptest.NewRecorder()
	s.requireAdmin(s.adminConfigSave)(rec, req)
	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("config status=%d body=%q", rec.Code, rec.Body.String())
	}
	if got := st.response(modelDir); got.AdminUser != "admin" || got.PostProcessDir != "" {
		t.Fatalf("config rollback=%+v", got)
	}

	backup := adminConfigBackup{Version: 1, Config: st.cfg}
	backup.Config.AdminUser = "restored"
	body, err := json.Marshal(backup)
	if err != nil {
		t.Fatal(err)
	}
	req = httptest.NewRequest(http.MethodPost, "/admin/api/config/restore", strings.NewReader(string(body)))
	req.AddCookie(&http.Cookie{Name: "paddle_admin_session", Value: token})
	rec = httptest.NewRecorder()
	s.requireAdmin(s.adminConfigRestore)(rec, req)
	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("restore status=%d body=%q", rec.Code, rec.Body.String())
	}
	if got := st.response(modelDir); got.AdminUser != "admin" {
		t.Fatalf("restore rollback=%+v", got)
	}
}

func TestAdminKeysResetUsage(t *testing.T) {
	st := loadAdminState(filepath.Join(t.TempDir(), "admin.json"))
	st.mu.Lock()
	if err := st.setPasswordLocked("admin", "password123"); err != nil {
		t.Fatal(err)
	}
	key, managed, err := st.createAPIKeyLocked("periodic", 2, 0)
	if err != nil {
		t.Fatal(err)
	}
	if err := st.saveLocked(); err != nil {
		t.Fatal(err)
	}
	st.mu.Unlock()
	if _, err := st.consumeAPIKey(key, "10.0.0.8"); err != nil {
		t.Fatal(err)
	}
	token, err := st.createSession()
	if err != nil {
		t.Fatal(err)
	}
	s := &server{admin: st}
	req := httptest.NewRequest(http.MethodPost, "/admin/api/keys/reset", strings.NewReader(`{"id":"`+managed.ID+`"}`))
	req.AddCookie(&http.Cookie{Name: "paddle_admin_session", Value: token})
	rec := httptest.NewRecorder()
	s.requireAdmin(s.adminKeysReset)(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%q", rec.Code, rec.Body.String())
	}
	keys := st.keyResponses()
	if len(keys) != 1 || keys[0].Used != 0 || keys[0].Remaining != 2 {
		t.Fatalf("keys=%+v", keys)
	}
}

func TestAdminKeysRotate(t *testing.T) {
	st := loadAdminState(filepath.Join(t.TempDir(), "admin.json"))
	st.mu.Lock()
	if err := st.setPasswordLocked("admin", "password123"); err != nil {
		t.Fatal(err)
	}
	oldKey, managed, err := st.createAPIKeyLocked("rotating", 7, 2)
	if err != nil {
		t.Fatal(err)
	}
	if err := st.saveLocked(); err != nil {
		t.Fatal(err)
	}
	st.mu.Unlock()
	token, err := st.createSession()
	if err != nil {
		t.Fatal(err)
	}
	s := &server{admin: st}
	req := httptest.NewRequest(http.MethodPost, "/admin/api/keys/rotate", strings.NewReader(`{"id":"`+managed.ID+`"}`))
	req.AddCookie(&http.Cookie{Name: "paddle_admin_session", Value: token})
	rec := httptest.NewRecorder()
	s.requireAdmin(s.adminKeysRotate)(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%q", rec.Code, rec.Body.String())
	}
	var body struct {
		APIKey string `json:"api_key"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if body.APIKey == "" || body.APIKey == oldKey {
		t.Fatalf("api_key=%q old=%q", body.APIKey, oldKey)
	}
	if _, err := st.consumeAPIKey(oldKey, "127.0.0.1"); !errors.Is(err, errAPIKeyInvalid) {
		t.Fatalf("old key err=%v", err)
	}
	if _, err := st.consumeAPIKey(body.APIKey, "127.0.0.1"); err != nil {
		t.Fatalf("new key err=%v", err)
	}
	keys := st.keyResponses()
	if len(keys) != 1 || keys[0].Quota != 7 || keys[0].RatePerMin != 2 {
		t.Fatalf("keys=%+v", keys)
	}
}

func TestAdminKeysUpdateNameAndLimits(t *testing.T) {
	st := loadAdminState(filepath.Join(t.TempDir(), "admin.json"))
	st.mu.Lock()
	if err := st.setPasswordLocked("admin", "password123"); err != nil {
		t.Fatal(err)
	}
	_, managed, err := st.createAPIKeyLocked("old", 1, 1)
	if err != nil {
		t.Fatal(err)
	}
	if err := st.saveLocked(); err != nil {
		t.Fatal(err)
	}
	st.mu.Unlock()
	token, err := st.createSession()
	if err != nil {
		t.Fatal(err)
	}
	s := &server{admin: st}
	req := httptest.NewRequest(http.MethodPost, "/admin/api/keys/update", strings.NewReader(`{"id":"`+managed.ID+`","name":"new-name","quota":12,"rate_per_minute":4,"disabled":true}`))
	req.AddCookie(&http.Cookie{Name: "paddle_admin_session", Value: token})
	rec := httptest.NewRecorder()
	s.requireAdmin(s.adminKeysUpdate)(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%q", rec.Code, rec.Body.String())
	}
	keys := st.keyResponses()
	if len(keys) != 1 || keys[0].Name != "new-name" || keys[0].Quota != 12 || keys[0].RatePerMin != 4 || !keys[0].Disabled {
		t.Fatalf("keys=%+v", keys)
	}
}

func TestAdminKeysRejectLongName(t *testing.T) {
	st := loadAdminState(filepath.Join(t.TempDir(), "admin.json"))
	st.mu.Lock()
	if err := st.setPasswordLocked("admin", "password123"); err != nil {
		t.Fatal(err)
	}
	_, managed, err := st.createAPIKeyLocked("old", 1, 1)
	if err != nil {
		t.Fatal(err)
	}
	if err := st.saveLocked(); err != nil {
		t.Fatal(err)
	}
	st.mu.Unlock()
	token, err := st.createSession()
	if err != nil {
		t.Fatal(err)
	}
	s := &server{admin: st}
	longName := strings.Repeat("x", 81)
	req := httptest.NewRequest(http.MethodPost, "/admin/api/keys", strings.NewReader(`{"name":"`+longName+`"}`))
	req.AddCookie(&http.Cookie{Name: "paddle_admin_session", Value: token})
	rec := httptest.NewRecorder()
	s.requireAdmin(s.adminKeysCreate)(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("create status=%d body=%q", rec.Code, rec.Body.String())
	}
	req = httptest.NewRequest(http.MethodPost, "/admin/api/keys/update", strings.NewReader(`{"id":"`+managed.ID+`","name":"`+longName+`","quota":1,"rate_per_minute":1}`))
	req.AddCookie(&http.Cookie{Name: "paddle_admin_session", Value: token})
	rec = httptest.NewRecorder()
	s.requireAdmin(s.adminKeysUpdate)(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("update status=%d body=%q", rec.Code, rec.Body.String())
	}
}

func TestAdminKeysCreateRejectsTooManyKeys(t *testing.T) {
	st := loadAdminState(filepath.Join(t.TempDir(), "admin.json"))
	st.mu.Lock()
	if err := st.setPasswordLocked("admin", "password123"); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < maxAdminAPIKeys; i++ {
		st.cfg.APIKeys = append(st.cfg.APIKeys, managedAPIKey{ID: fmt.Sprintf("k%d", i), Name: "key", Salt: "salt", Hash: strings.Repeat("a", 64)})
	}
	if err := st.saveLocked(); err != nil {
		t.Fatal(err)
	}
	st.mu.Unlock()
	token, err := st.createSession()
	if err != nil {
		t.Fatal(err)
	}
	s := &server{admin: st}
	req := httptest.NewRequest(http.MethodPost, "/admin/api/keys", strings.NewReader(`{"name":"extra"}`))
	req.AddCookie(&http.Cookie{Name: "paddle_admin_session", Value: token})
	rec := httptest.NewRecorder()
	s.requireAdmin(s.adminKeysCreate)(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status=%d body=%q", rec.Code, rec.Body.String())
	}
	if len(st.cfg.APIKeys) != maxAdminAPIKeys {
		t.Fatalf("keys len=%d", len(st.cfg.APIKeys))
	}
}

func TestAdminConfigRestoreRejectsTooManyAPIKeys(t *testing.T) {
	cfg := adminConfigFile{
		AdminUser:    "admin",
		PasswordSalt: "salt",
		PasswordHash: strings.Repeat("a", 64),
		APIKeys:      make([]managedAPIKey, maxAdminAPIKeys+1),
	}
	for i := range cfg.APIKeys {
		cfg.APIKeys[i] = managedAPIKey{
			ID:   fmt.Sprintf("k%d", i),
			Name: "key",
			Salt: "salt",
			Hash: strings.Repeat("b", 64),
		}
	}
	err := validateAdminConfigBackup(cfg)
	if err == nil || !strings.Contains(err.Error(), "too many API keys") {
		t.Fatalf("err=%v want too many API keys", err)
	}
}

func TestAdminConfigRestoreRejectsInvalidLegacyAPIKeyMetadata(t *testing.T) {
	base := adminConfigFile{
		AdminUser:     "admin",
		PasswordSalt:  "salt",
		PasswordHash:  strings.Repeat("a", 64),
		APIKeySalt:    "legacy_salt",
		APIKeyHash:    strings.Repeat("b", 64),
		APIKeyPreview: "lega...cret",
	}
	cases := []struct {
		name string
		mut  func(*adminConfigFile)
	}{
		{name: "missing-salt", mut: func(cfg *adminConfigFile) { cfg.APIKeySalt = "" }},
		{name: "bad-salt", mut: func(cfg *adminConfigFile) { cfg.APIKeySalt = "bad salt" }},
		{name: "missing-hash", mut: func(cfg *adminConfigFile) { cfg.APIKeyHash = "" }},
		{name: "bad-hash", mut: func(cfg *adminConfigFile) { cfg.APIKeyHash = "not-a-sha256" }},
		{name: "preview-control", mut: func(cfg *adminConfigFile) { cfg.APIKeyPreview = "bad\npreview" }},
		{name: "preview-too-long", mut: func(cfg *adminConfigFile) { cfg.APIKeyPreview = strings.Repeat("x", 129) }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := base
			tc.mut(&cfg)
			err := validateAdminConfigBackup(cfg)
			if err == nil || !strings.Contains(err.Error(), "legacy API key metadata") {
				t.Fatalf("err=%v want legacy API key metadata", err)
			}
		})
	}
}

func TestAdminKeyMutationsRollbackWhenSaveFails(t *testing.T) {
	st := loadAdminState(filepath.Join(t.TempDir(), "admin.json"))
	st.mu.Lock()
	if err := st.setPasswordLocked("admin", "password123"); err != nil {
		t.Fatal(err)
	}
	key, managed, err := st.createAPIKeyLocked("stable", 5, 2)
	if err != nil {
		t.Fatal(err)
	}
	if err := st.saveLocked(); err != nil {
		t.Fatal(err)
	}
	st.mu.Unlock()
	token, err := st.createSession()
	if err != nil {
		t.Fatal(err)
	}
	badPath := filepath.Join(t.TempDir(), "bad\x00name", "admin.json")
	st.path = badPath
	s := &server{admin: st}

	req := httptest.NewRequest(http.MethodPost, "/admin/api/keys", strings.NewReader(`{"name":"new"}`))
	req.AddCookie(&http.Cookie{Name: "paddle_admin_session", Value: token})
	rec := httptest.NewRecorder()
	s.requireAdmin(s.adminKeysCreate)(rec, req)
	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("create status=%d body=%q", rec.Code, rec.Body.String())
	}
	if keys := st.keyResponses(); len(keys) != 1 || keys[0].ID != managed.ID {
		t.Fatalf("create rollback keys=%+v", keys)
	}

	req = httptest.NewRequest(http.MethodPost, "/admin/api/keys/update", strings.NewReader(`{"id":"`+managed.ID+`","name":"changed","quota":1,"rate_per_minute":1}`))
	req.AddCookie(&http.Cookie{Name: "paddle_admin_session", Value: token})
	rec = httptest.NewRecorder()
	s.requireAdmin(s.adminKeysUpdate)(rec, req)
	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("update status=%d body=%q", rec.Code, rec.Body.String())
	}
	keys := st.keyResponses()
	if len(keys) != 1 || keys[0].Name != "stable" || keys[0].Quota != 5 || keys[0].RatePerMin != 2 {
		t.Fatalf("update rollback keys=%+v", keys)
	}

	req = httptest.NewRequest(http.MethodPost, "/admin/api/keys/rotate", strings.NewReader(`{"id":"`+managed.ID+`"}`))
	req.AddCookie(&http.Cookie{Name: "paddle_admin_session", Value: token})
	rec = httptest.NewRecorder()
	s.requireAdmin(s.adminKeysRotate)(rec, req)
	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("rotate status=%d body=%q", rec.Code, rec.Body.String())
	}
	st.path = filepath.Join(t.TempDir(), "admin.json")
	if _, err := st.consumeAPIKey(key, "127.0.0.1"); err != nil {
		t.Fatalf("old key after failed rotate err=%v", err)
	}
	st.path = badPath

	req = httptest.NewRequest(http.MethodPost, "/admin/api/keys/delete", strings.NewReader(`{"id":"`+managed.ID+`"}`))
	req.AddCookie(&http.Cookie{Name: "paddle_admin_session", Value: token})
	rec = httptest.NewRecorder()
	s.requireAdmin(s.adminKeysDelete)(rec, req)
	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("delete status=%d body=%q", rec.Code, rec.Body.String())
	}
	if keys := st.keyResponses(); len(keys) != 1 || keys[0].ID != managed.ID {
		t.Fatalf("delete rollback keys=%+v", keys)
	}
}

func TestClientIPPrefersForwardedHeaders(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/v1/generate", nil)
	req.RemoteAddr = "127.0.0.1:1234"
	req.Header.Set("X-Forwarded-For", "203.0.113.5, 10.0.0.1")
	if got := clientIP(req); got != "203.0.113.5" {
		t.Fatalf("clientIP=%q", got)
	}
	req.Header.Del("X-Forwarded-For")
	req.Header.Set("X-Real-IP", "198.51.100.7")
	if got := clientIP(req); got != "198.51.100.7" {
		t.Fatalf("clientIP=%q", got)
	}
	req.Header.Del("X-Real-IP")
	if got := clientIP(req); got != "127.0.0.1" {
		t.Fatalf("clientIP=%q", got)
	}
	req.Header.Set("X-Forwarded-For", "not an ip")
	req.Header.Set("X-Real-IP", "also bad")
	if got := clientIP(req); got != "127.0.0.1" {
		t.Fatalf("invalid forwarded clientIP=%q", got)
	}
	req.Header = http.Header{"x-forwarded-for": []string{"198.51.100.8"}}
	if got := clientIP(req); got != "198.51.100.8" {
		t.Fatalf("lowercase forwarded clientIP=%q", got)
	}
	req.RemoteAddr = "192.0.2.10:1234"
	req.Header = http.Header{"X-Forwarded-For": []string{"198.51.100.9"}}
	if got := clientIP(req); got != "192.0.2.10" {
		t.Fatalf("untrusted forwarded clientIP=%q", got)
	}
	req.RemoteAddr = "2001:db8::1"
	req.Header = http.Header{}
	if got := clientIP(req); got != "2001:db8::1" {
		t.Fatalf("bare ipv6 clientIP=%q", got)
	}
}

func BenchmarkClientIPForwarded(b *testing.B) {
	req := httptest.NewRequest(http.MethodPost, "/v1/generate", nil)
	req.RemoteAddr = "127.0.0.1:1234"
	req.Header.Set("X-Forwarded-For", "203.0.113.5, 10.0.0.1")
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if got := clientIP(req); got != "203.0.113.5" {
			b.Fatalf("got=%q", got)
		}
	}
}

func BenchmarkClientIPRemoteAddr(b *testing.B) {
	req := httptest.NewRequest(http.MethodPost, "/v1/generate", nil)
	req.RemoteAddr = "192.0.2.10:1234"
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if got := clientIP(req); got != "192.0.2.10" {
			b.Fatalf("got=%q", got)
		}
	}
}

func TestOpenAPIJSONIncludesInferencePaths(t *testing.T) {
	s := &server{}
	req := httptest.NewRequest(http.MethodGet, "/doc/openapi.json", nil)
	req.Host = "127.0.0.1:8080"
	rec := httptest.NewRecorder()
	s.openapiJSON(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d", rec.Code)
	}
	var body map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if body["openapi"] != "3.1.0" {
		t.Fatalf("openapi=%v", body["openapi"])
	}
	security := body["security"].([]any)
	if len(security) != 2 {
		t.Fatalf("security=%v", security)
	}
	components := body["components"].(map[string]any)
	schemes := components["securitySchemes"].(map[string]any)
	if _, ok := schemes["bearerAuth"]; !ok {
		t.Fatalf("securitySchemes missing bearerAuth: %v", schemes)
	}
	if apiKey, ok := schemes["apiKeyHeader"].(map[string]any); !ok || apiKey["name"] != "X-API-Key" {
		t.Fatalf("apiKeyHeader=%v", schemes["apiKeyHeader"])
	}
	paths := body["paths"].(map[string]any)
	for _, path := range []string{"/v1/ocr", "/v1/generate", "/v1/batch", "/health", "/ready", "/stats"} {
		if _, ok := paths[path]; !ok {
			t.Fatalf("missing path %s", path)
		}
	}
	cases := []struct {
		path        string
		method      string
		operationID string
		tag         string
	}{
		{"/health", "get", "getHealth", "status"},
		{"/ready", "get", "getReady", "status"},
		{"/stats", "get", "getStats", "status"},
		{"/v1/ocr", "post", "ocrImage", "inference"},
		{"/v1/generate", "post", "generate", "inference"},
		{"/v1/batch", "post", "batchGenerate", "inference"},
	}
	for _, tc := range cases {
		op := paths[tc.path].(map[string]any)[tc.method].(map[string]any)
		if got := op["operationId"]; got != tc.operationID {
			t.Fatalf("%s %s operationId=%v want %s", tc.method, tc.path, got, tc.operationID)
		}
		tags := op["tags"].([]any)
		if len(tags) != 1 || tags[0] != tc.tag {
			t.Fatalf("%s %s tags=%v want %s", tc.method, tc.path, tags, tc.tag)
		}
	}
	generate := paths["/v1/generate"].(map[string]any)["post"].(map[string]any)
	genContent := generate["requestBody"].(map[string]any)["content"].(map[string]any)["application/json"].(map[string]any)
	if _, ok := genContent["examples"].(map[string]any)["ocrBase64"]; !ok {
		t.Fatalf("generate examples=%v", genContent["examples"])
	}
	batch := paths["/v1/batch"].(map[string]any)["post"].(map[string]any)
	batchContent := batch["requestBody"].(map[string]any)["content"].(map[string]any)["application/json"].(map[string]any)
	if _, ok := batchContent["examples"].(map[string]any)["mixed"]; !ok {
		t.Fatalf("batch examples=%v", batchContent["examples"])
	}
	ocr := paths["/v1/ocr"].(map[string]any)["post"].(map[string]any)
	ocrContent := ocr["requestBody"].(map[string]any)["content"].(map[string]any)["multipart/form-data"].(map[string]any)
	if _, ok := ocrContent["examples"].(map[string]any)["ocr"]; !ok {
		t.Fatalf("ocr examples=%v", ocrContent["examples"])
	}
	genResponses := generate["responses"].(map[string]any)
	genOK := genResponses["200"].(map[string]any)["content"].(map[string]any)["application/json"].(map[string]any)
	if _, ok := genOK["examples"].(map[string]any)["default"]; !ok {
		t.Fatalf("generate response examples=%v", genOK["examples"])
	}
	genErr := genResponses["401"].(map[string]any)["content"].(map[string]any)["application/json"].(map[string]any)
	if _, ok := genErr["examples"].(map[string]any)["default"]; !ok {
		t.Fatalf("error response examples=%v", genErr["examples"])
	}
}

func TestOpenAPIJSONNormalizesInvalidHost(t *testing.T) {
	s := &server{}
	req := httptest.NewRequest(http.MethodGet, "/doc/openapi.json", nil)
	req.Host = `host"with\chars`
	req.RemoteAddr = "127.0.0.1:1234"
	req.Header.Set("X-Forwarded-Proto", "https")
	rec := httptest.NewRecorder()
	s.openapiJSON(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d", rec.Code)
	}
	var body struct {
		Servers []struct {
			URL string `json:"url"`
		} `json:"servers"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if len(body.Servers) != 1 || body.Servers[0].URL != "https://localhost" {
		t.Fatalf("servers=%+v", body.Servers)
	}
}

func TestOpenAPIResponseBufferDropsOversized(t *testing.T) {
	baseCap := len(openAPIJSONParts[0]) + len(openAPIBasePlaceholder) + len(openAPIJSONParts[1]) + maxOpenAPIResponseBufferExtra
	buf := make([]byte, 0, baseCap+1)
	p := &buf
	putOpenAPIResponseBuffer(p, buf)
	if cap(*p) != baseCap+1 {
		t.Fatalf("oversized buffer was retained cap=%d", cap(*p))
	}
	buf = make([]byte, 0, baseCap)
	buf = append(buf, 'x')
	p = &buf
	putOpenAPIResponseBuffer(p, buf)
	if len(*p) != 0 || cap(*p) != baseCap {
		t.Fatalf("bounded buffer not reset len=%d cap=%d", len(*p), cap(*p))
	}
}

func TestAdminBaseURLAllowsValidHostPort(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "/doc/openapi.json", nil)
	req.Host = "[::1]:8080"
	req.RemoteAddr = "127.0.0.1:1234"
	req.Header.Set("X-Forwarded-Proto", "https")
	if got := adminBaseURL(req); got != "https://[::1]:8080" {
		t.Fatalf("adminBaseURL=%q", got)
	}
	req.Host = "127.0.0.1:70000"
	if got := adminBaseURL(req); got != "https://localhost" {
		t.Fatalf("invalid port adminBaseURL=%q", got)
	}
}

func TestRequestSchemeRejectsInvalidForwardedProto(t *testing.T) {
	s := &server{}
	req := httptest.NewRequest(http.MethodGet, "/doc/openapi.json", nil)
	req.Host = "127.0.0.1:8080"
	req.Header.Set("X-Forwarded-Proto", "javascript:alert(1)")
	rec := httptest.NewRecorder()
	s.openapiJSON(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d", rec.Code)
	}
	var body struct {
		Servers []struct {
			URL string `json:"url"`
		} `json:"servers"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if len(body.Servers) != 1 || body.Servers[0].URL != "http://127.0.0.1:8080" {
		t.Fatalf("servers=%+v", body.Servers)
	}
	req = httptest.NewRequest(http.MethodGet, "/doc/llms.txt", nil)
	req.Host = "127.0.0.1:8080"
	req.RemoteAddr = "127.0.0.1:1234"
	req.Header.Set("X-Forwarded-Proto", "https, http")
	rec = httptest.NewRecorder()
	s.llmsTXT(rec, req)
	if !strings.Contains(rec.Body.String(), "Base URL: https://127.0.0.1:8080") {
		t.Fatalf("llms.txt=%s", rec.Body.String())
	}
	req = httptest.NewRequest(http.MethodGet, "/doc/openapi.json", nil)
	req.RemoteAddr = "127.0.0.1:1234"
	req.Header = http.Header{"x-forwarded-proto": []string{"https"}}
	if got := requestScheme(req); got != "https" {
		t.Fatalf("lowercase forwarded proto scheme=%q", got)
	}
	req.RemoteAddr = "192.0.2.10:1234"
	req.Header = http.Header{"X-Forwarded-Proto": []string{"https"}}
	if got := requestScheme(req); got != "http" {
		t.Fatalf("untrusted forwarded proto scheme=%q", got)
	}
}

func BenchmarkAdminBaseURLHost(b *testing.B) {
	req := httptest.NewRequest(http.MethodGet, "/doc/openapi.json", nil)
	req.Host = "paddleocrvl.local:8080"
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if got := adminBaseURL(req); got != "http://paddleocrvl.local:8080" {
			b.Fatalf("got=%q", got)
		}
	}
}

func BenchmarkRequestSchemeForwarded(b *testing.B) {
	req := httptest.NewRequest(http.MethodGet, "/doc/openapi.json", nil)
	req.RemoteAddr = "127.0.0.1:1234"
	req.Header.Set("X-Forwarded-Proto", " HTTPS , http")
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if got := requestScheme(req); got != "https" {
			b.Fatalf("got=%q", got)
		}
	}
}

func BenchmarkValidRequestHostName(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if !validRequestHost("paddleocrvl.local:8080") {
			b.Fatal("invalid")
		}
	}
}

func BenchmarkValidRequestHostIPv4(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if !validRequestHost("127.0.0.1:8080") {
			b.Fatal("invalid")
		}
	}
}

func BenchmarkValidAdminASCII(b *testing.B) {
	hash := strings.Repeat("a", 64)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if !validAPIKeyID("key_123-ABC") || !validSalt("salt_123-ABC") || !validSHA256Hex(hash) || !validPort("8080") {
			b.Fatal("invalid")
		}
	}
}

func TestLLMSTXTIncludesAgentIntegrationSummary(t *testing.T) {
	s := &server{}
	req := httptest.NewRequest(http.MethodGet, "/doc/llms.txt", nil)
	req.Host = "127.0.0.1:8080"
	rec := httptest.NewRecorder()
	s.llmsTXT(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d", rec.Code)
	}
	if ct := rec.Header().Get("Content-Type"); !strings.Contains(ct, "text/plain") {
		t.Fatalf("content-type=%q", ct)
	}
	body := rec.Body.String()
	for _, want := range []string{
		"Base URL: http://127.0.0.1:8080",
		"OpenAPI: http://127.0.0.1:8080/doc/openapi.json",
		"Authorization: Bearer <API_KEY>",
		"X-API-Key: <API_KEY>",
		"ocrImage: POST /v1/ocr",
		"generate: POST /v1/generate",
		"batchGenerate: POST /v1/batch",
	} {
		if !strings.Contains(body, want) {
			t.Fatalf("llms.txt missing %q in:\n%s", want, body)
		}
	}
}

func TestLLMSTXTBufferDropsOversized(t *testing.T) {
	baseCap := llmsTXTBaseCap() + maxLLMSTXTBufferExtra
	buf := make([]byte, 0, baseCap+1)
	p := &buf
	putLLMSTXTBuffer(p, buf)
	if cap(*p) != baseCap+1 {
		t.Fatalf("oversized buffer was retained cap=%d", cap(*p))
	}
	buf = make([]byte, 0, baseCap)
	buf = append(buf, 'x')
	p = &buf
	putLLMSTXTBuffer(p, buf)
	if len(*p) != 0 || cap(*p) != baseCap {
		t.Fatalf("bounded buffer not reset len=%d cap=%d", len(*p), cap(*p))
	}
}

func TestDocPageIncludesBothAuthHeaderOptions(t *testing.T) {
	s := &server{}
	req := httptest.NewRequest(http.MethodGet, "/doc", nil)
	req.Host = "127.0.0.1:8080"
	rec := httptest.NewRecorder()
	s.docPage(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d", rec.Code)
	}
	body := rec.Body.String()
	for _, want := range []string{"Authorization: Bearer &lt;API_KEY&gt;", "X-API-Key: &lt;API_KEY&gt;"} {
		if !strings.Contains(body, want) {
			t.Fatalf("doc missing %q body=%s", want, body)
		}
	}
}

func BenchmarkOpenAPIJSON(b *testing.B) {
	s := &server{}
	req := httptest.NewRequest(http.MethodGet, "/doc/openapi.json", nil)
	req.Host = "127.0.0.1:8080"
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		rec := httptest.NewRecorder()
		s.openapiJSON(rec, req)
	}
}

func BenchmarkLLMSTXT(b *testing.B) {
	s := &server{}
	req := httptest.NewRequest(http.MethodGet, "/doc/llms.txt", nil)
	req.Host = "127.0.0.1:8080"
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		rec := httptest.NewRecorder()
		s.llmsTXT(rec, req)
	}
}

func BenchmarkLLMSTXTDirect(b *testing.B) {
	w := discardResponseWriter{h: http.Header{}}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		writeLLMsTXT(w, "http://127.0.0.1:8080")
	}
}

func BenchmarkLLMSTXTRequestDirect(b *testing.B) {
	w := discardResponseWriter{h: http.Header{}}
	req := httptest.NewRequest(http.MethodGet, "/doc/llms.txt", nil)
	req.Host = "127.0.0.1:8080"
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		writeLLMsTXTForRequest(w, req)
	}
}

func BenchmarkOpenAPIJSONDirect(b *testing.B) {
	s := &server{}
	req := httptest.NewRequest(http.MethodGet, "/doc/openapi.json", nil)
	req.Host = "127.0.0.1:8080"
	w := discardResponseWriter{h: http.Header{}}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		s.openapiJSON(w, req)
	}
}

func BenchmarkWriteError(b *testing.B) {
	err := errors.New("bad request")
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rec := httptest.NewRecorder()
		writeError(rec, http.StatusBadRequest, err)
		_, _ = io.Copy(io.Discard, rec.Result().Body)
		_ = rec.Result().Body.Close()
	}
}

func BenchmarkWriteErrorDirect(b *testing.B) {
	err := errors.New("bad request")
	w := discardResponseWriter{h: http.Header{}}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		writeError(w, http.StatusBadRequest, err)
	}
}

func BenchmarkWriteOKJSONDirect(b *testing.B) {
	w := discardResponseWriter{h: http.Header{}}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		writeOKJSON(w)
	}
}

func TestWriteOKErrorJSON(t *testing.T) {
	rec := httptest.NewRecorder()
	writeOKErrorJSON(rec, `bad "path" <x>`)
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d", rec.Code)
	}
	var got struct {
		OK    bool   `json:"ok"`
		Error string `json:"error"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.OK || got.Error != `bad "path" <x>` {
		t.Fatalf("got=%+v", got)
	}
	if !strings.Contains(rec.Body.String(), `<x>`) {
		t.Fatalf("body=%q", rec.Body.String())
	}
}

func BenchmarkWriteOKErrorJSONDirect(b *testing.B) {
	w := discardResponseWriter{h: http.Header{}}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		writeOKErrorJSON(w, "bad request")
	}
}

func TestWriteAdminSessionJSON(t *testing.T) {
	for _, tc := range []struct {
		initialized   bool
		authenticated bool
	}{
		{false, false},
		{true, false},
		{false, true},
		{true, true},
	} {
		rec := httptest.NewRecorder()
		writeAdminSessionJSON(rec, tc.initialized, tc.authenticated)
		if rec.Code != http.StatusOK {
			t.Fatalf("status=%d", rec.Code)
		}
		var got struct {
			Initialized   bool `json:"initialized"`
			Authenticated bool `json:"authenticated"`
		}
		if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
			t.Fatal(err)
		}
		if got.Initialized != tc.initialized || got.Authenticated != tc.authenticated {
			t.Fatalf("got=%+v want init=%v auth=%v", got, tc.initialized, tc.authenticated)
		}
	}
}

func BenchmarkWriteAdminSessionJSONDirect(b *testing.B) {
	w := discardResponseWriter{h: http.Header{}}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		writeAdminSessionJSON(w, true, true)
	}
}

func TestWriteHealthJSON(t *testing.T) {
	rec := httptest.NewRecorder()
	writeHealthJSON(rec, healthResponse{
		Status:       "ok",
		Quantization: "q4",
		WeightPath:   `D:\models\m"q4.gguf`,
		WeightSource: "existing_gguf",
		Backend:      "vulkan",
		VisionLoaded: true,
	})
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d", rec.Code)
	}
	var got healthResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Status != "ok" || got.Quantization != "q4" || got.WeightPath != `D:\models\m"q4.gguf` || got.WeightSource != "existing_gguf" || got.Backend != "vulkan" || !got.VisionLoaded {
		t.Fatalf("got=%+v", got)
	}
}

func BenchmarkWriteHealthJSONDirect(b *testing.B) {
	w := discardResponseWriter{h: http.Header{}}
	res := healthResponse{
		Status:       "ok",
		Quantization: "q4",
		WeightPath:   `D:\models\model-q4.gguf`,
		WeightSource: "existing_gguf",
		Backend:      "cpu",
		VisionLoaded: true,
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		writeHealthJSON(w, res)
	}
}

func BenchmarkWriteHealthJSONEncoderDirect(b *testing.B) {
	w := discardResponseWriter{h: http.Header{}}
	res := healthResponse{
		Status:       "ok",
		Quantization: "q4",
		WeightPath:   `D:\models\model-q4.gguf`,
		WeightSource: "existing_gguf",
		Backend:      "cpu",
		VisionLoaded: true,
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		writeJSON(w, http.StatusOK, res)
	}
}

func TestWriteReadyJSONMatchesEncoderShape(t *testing.T) {
	cases := []readyResponse{
		{Status: "not_ready", Reason: "model, tokenizer, or inference slots not initialized"},
		{Status: "ready", Quantization: "q4", WeightPath: `D:\models\m"q4.gguf`, WeightSource: "existing_gguf", Backend: "cpu", Concurrency: 4, AvailableSlots: 4},
		{Status: "ready", Quantization: "q8", WeightPath: "model.gguf", WeightSource: "converted_safetensors", Backend: "vulkan", VisionLoaded: true, Concurrency: 4, InFlight: 2, AvailableSlots: 2},
	}
	for _, tc := range cases {
		rec := httptest.NewRecorder()
		writeReadyJSON(rec, http.StatusOK, tc)
		var got map[string]any
		if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
			t.Fatal(err)
		}
		wantBytes, err := json.Marshal(tc)
		if err != nil {
			t.Fatal(err)
		}
		var want map[string]any
		if err := json.Unmarshal(wantBytes, &want); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("got=%v want=%v body=%q", got, want, rec.Body.String())
		}
	}
}

func BenchmarkWriteReadyJSONDirect(b *testing.B) {
	w := discardResponseWriter{h: http.Header{}}
	res := readyResponse{Status: "ready", Quantization: "q4", WeightPath: `D:\models\model-q4.gguf`, WeightSource: "existing_gguf", Backend: "cpu", Concurrency: 4, AvailableSlots: 4}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		writeReadyJSON(w, http.StatusOK, res)
	}
}

func BenchmarkWriteReadyJSONEncoderDirect(b *testing.B) {
	w := discardResponseWriter{h: http.Header{}}
	res := readyResponse{Status: "ready", Quantization: "q4", WeightPath: `D:\models\model-q4.gguf`, WeightSource: "existing_gguf", Backend: "cpu", Concurrency: 4, AvailableSlots: 4}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		writeJSON(w, http.StatusOK, res)
	}
}

type discardResponseWriter struct {
	h http.Header
}

func (w discardResponseWriter) Header() http.Header       { return w.h }
func (discardResponseWriter) Write(p []byte) (int, error) { return len(p), nil }
func (discardResponseWriter) WriteHeader(statusCode int)  {}

func BenchmarkChatPrompt(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = chatPrompt("OCR:", true)
	}
}
