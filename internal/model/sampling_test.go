package model

import (
	"context"
	"math"
	"math/rand"
	"reflect"
	"testing"
)

var benchEOS bool

func TestSampleTopKOneUsesArgmax(t *testing.T) {
	logits := []float32{-1, 3, 2}
	got := sampleTokenScratch(logits, GenerateOptions{Temperature: 1, TopK: 1}, rand.New(rand.NewSource(1)), nil)
	if got != 1 {
		t.Fatalf("sampleTokenScratch topK=1 got %d want 1", got)
	}
}

func TestTopKCandidatesSorted(t *testing.T) {
	logits := []float32{0.5, 9, -1, 7, 3, 8, 2}
	got := topKCandidates(logits, 3, nil)
	wantIDs := []int{1, 5, 3}
	for i, want := range wantIDs {
		if got[i].id != want {
			t.Fatalf("topK[%d] id=%d want %d candidates=%v", i, got[i].id, want, got)
		}
	}
}

func TestTopKCandidatesUnsortedContainsTopK(t *testing.T) {
	logits := []float32{0.5, 9, -1, 7, 3, 8, 2}
	got, maxScore := topKCandidatesUnsortedWithMax(logits, 3, nil)
	seen := map[int]bool{}
	for _, c := range got {
		seen[c.id] = true
	}
	for _, id := range []int{1, 5, 3} {
		if !seen[id] {
			t.Fatalf("missing id %d in %v", id, got)
		}
	}
	if maxScore != 9 {
		t.Fatalf("max score got %f want 9", maxScore)
	}
}

func TestTopKCandidatesUnsortedBounds(t *testing.T) {
	logits := []float32{0.2, 1.5, -0.1, 1.2}
	got := topKCandidatesUnsorted(logits, 8, &generationScratch{})
	if len(got) != len(logits) {
		t.Fatalf("len got %d want %d", len(got), len(logits))
	}
	seen := make(map[int]float32, len(got))
	for _, c := range got {
		seen[c.id] = c.score
	}
	for i, v := range logits {
		if seen[i] != v {
			t.Fatalf("candidate %d got %f want %f", i, seen[i], v)
		}
	}
}

func TestSampleTopKReusesWeightScratch(t *testing.T) {
	logits := []float32{0.5, 9, -1, 7, 3, 8, 2}
	scratch := &generationScratch{}
	_ = sampleTokenScratch(logits, GenerateOptions{Temperature: 0.8, TopK: 3}, rand.New(rand.NewSource(1)), scratch)
	if cap(scratch.weights) < 3 {
		t.Fatalf("weights cap=%d want >=3", cap(scratch.weights))
	}
	first := &scratch.weights[0]
	_ = sampleTokenScratch(logits, GenerateOptions{Temperature: 0.8, TopK: 3}, rand.New(rand.NewSource(2)), scratch)
	if &scratch.weights[0] != first {
		t.Fatal("expected weight scratch reuse")
	}
}

func TestSampleFullLogitsReusesWeightScratch(t *testing.T) {
	logits := []float32{0.5, 9, -1, 7, 3, 8, 2}
	scratch := &generationScratch{}
	_ = sampleTokenScratch(logits, GenerateOptions{Temperature: 0.8, TopK: 0}, rand.New(rand.NewSource(1)), scratch)
	if cap(scratch.weights) < len(logits) {
		t.Fatalf("weights cap=%d want >=%d", cap(scratch.weights), len(logits))
	}
	first := &scratch.weights[0]
	_ = sampleTokenScratch(logits, GenerateOptions{Temperature: 0.8, TopK: 0}, rand.New(rand.NewSource(2)), scratch)
	if &scratch.weights[0] != first {
		t.Fatal("expected full-logit weight scratch reuse")
	}
}

func TestSampleFullLogitsReturnsValidID(t *testing.T) {
	logits := []float32{-100, -100, 0}
	for seed := int64(0); seed < 8; seed++ {
		got := sampleFullLogits(logits, 1, rand.New(rand.NewSource(seed)))
		if got < 0 || got >= len(logits) {
			t.Fatalf("sampleFullLogits got invalid id %d", got)
		}
	}
}

func TestPickCumulativeFloat32(t *testing.T) {
	cdf := []float32{0.25, 0.75, 1}
	for pick, want := range map[float32]int{
		0.01: 0,
		0.25: 0,
		0.26: 1,
		0.75: 1,
		0.90: 2,
	} {
		if got := pickCumulativeFloat32(cdf, pick); got != want {
			t.Fatalf("pick %f got %d want %d", pick, got, want)
		}
	}
	if got := pickCumulativeFloat32(cdf, 1.1); got != -1 {
		t.Fatalf("out of range got %d want -1", got)
	}
}

func TestSampleFullLogitsTemperatureScaleReturnsValidID(t *testing.T) {
	logits := []float32{-4, 0.5, 3, -2}
	for _, temp := range []float32{0.5, 0.8, 1.5} {
		for seed := int64(0); seed < 8; seed++ {
			got := sampleFullLogits(logits, temp, rand.New(rand.NewSource(seed)))
			if got < 0 || got >= len(logits) {
				t.Fatalf("sampleFullLogits temp=%f got invalid id %d", temp, got)
			}
		}
	}
}

func TestIsEOSShortLists(t *testing.T) {
	if !isEOS(3, []int{2, 3}) || isEOS(4, []int{2, 3}) {
		t.Fatal("bad len2 eos check")
	}
	if !isEOS(4, []int{2, 3, 4}) || isEOS(5, []int{2, 3, 4}) {
		t.Fatal("bad len3 eos check")
	}
}

func TestSamplingRNGOnlyWhenNeeded(t *testing.T) {
	if rng := samplingRNG(GenerateOptions{MaxNewTokens: 1, Temperature: 0}); rng != nil {
		t.Fatal("greedy temperature=0 should not allocate rng")
	}
	if rng := samplingRNG(GenerateOptions{MaxNewTokens: 1, Temperature: 1, TopK: 1}); rng != nil {
		t.Fatal("top_k=1 should not allocate rng")
	}
	if rng := samplingRNG(GenerateOptions{MaxNewTokens: 0, Temperature: 1}); rng != nil {
		t.Fatal("max_new_tokens=0 should not allocate rng")
	}
	if rng := samplingRNG(GenerateOptions{MaxNewTokens: 1, Temperature: 1}); rng == nil {
		t.Fatal("sampling should allocate rng")
	}
}

func BenchmarkIsEOSShortList(b *testing.B) {
	ids := []int{2, 100257, 100258}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchEOS = isEOS(i&7, ids)
	}
}

func TestRankedCandidatesTopKZeroSortsAll(t *testing.T) {
	logits := []float32{2, 5, 1}
	got := rankedCandidates(logits, 0)
	if got[0].id != 1 || got[1].id != 0 || got[2].id != 2 {
		t.Fatalf("ranked=%v", got)
	}
}

func TestValidateGenerateOptions(t *testing.T) {
	if err := ValidateGenerateOptions(GenerateOptions{}); err != nil {
		t.Fatal(err)
	}
	if err := ValidateGenerateOptions(GenerateOptions{MaxNewTokens: -1}); err == nil {
		t.Fatal("expected max_new_tokens error")
	}
	if err := ValidateGenerateOptions(GenerateOptions{Temperature: -0.1}); err == nil {
		t.Fatal("expected temperature error")
	}
	if err := ValidateGenerateOptions(GenerateOptions{Temperature: math.NaN()}); err == nil {
		t.Fatal("expected nan temperature error")
	}
	if err := ValidateGenerateOptions(GenerateOptions{Temperature: math.Inf(1)}); err == nil {
		t.Fatal("expected inf temperature error")
	}
	if err := ValidateGenerateOptions(GenerateOptions{TopK: -1}); err == nil {
		t.Fatal("expected top_k error")
	}
}

func TestGenerateWithOptionsZeroNewReturnsInputCopy(t *testing.T) {
	rt := &Runtime{}
	input := []int{1, 2, 3}
	res, err := rt.GenerateWithOptions(context.Background(), input, GenerateOptions{MaxNewTokens: 0})
	if err != nil {
		t.Fatal(err)
	}
	if res.PromptTokens != len(input) || !reflect.DeepEqual(res.Tokens, input) {
		t.Fatalf("res=%+v input=%v", res, input)
	}
	input[0] = 9
	if res.Tokens[0] != 1 {
		t.Fatalf("result aliased input: %v", res.Tokens)
	}
}

func TestGenerateWithImageZeroNewReturnsInputCopyBeforeImageLoad(t *testing.T) {
	rt := &Runtime{}
	input := []int{1, 2, 3}
	res, err := rt.GenerateWithImageOptions(context.Background(), input, "missing.png", GenerateOptions{MaxNewTokens: 0})
	if err != nil {
		t.Fatal(err)
	}
	if res.PromptTokens != len(input) || !reflect.DeepEqual(res.Tokens, input) {
		t.Fatalf("res=%+v input=%v", res, input)
	}
	input[0] = 9
	if res.Tokens[0] != 1 {
		t.Fatalf("result aliased input: %v", res.Tokens)
	}
}

func TestGenerateWithImageBytesZeroNewReturnsInputCopyBeforeImageDecode(t *testing.T) {
	rt := &Runtime{}
	input := []int{4, 5}
	res, err := rt.GenerateWithImageBytesOptions(context.Background(), input, []byte("not an image"), GenerateOptions{MaxNewTokens: 0})
	if err != nil {
		t.Fatal(err)
	}
	if res.PromptTokens != len(input) || !reflect.DeepEqual(res.Tokens, input) {
		t.Fatalf("res=%+v input=%v", res, input)
	}
	input[0] = 9
	if res.Tokens[0] != 4 {
		t.Fatalf("result aliased input: %v", res.Tokens)
	}
}
