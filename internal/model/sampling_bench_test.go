package model

import (
	"math/rand"
	"testing"
)

var samplingBenchState uint64

func BenchmarkSampleFullLogits(b *testing.B) {
	logits := make([]float32, 128000)
	for i := range logits {
		logits[i] = float32(i%97) / 97
	}
	rng := rand.New(rand.NewSource(1))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = sampleTokenScratch(logits, GenerateOptions{Temperature: 0.8, TopK: 0}, rng, nil)
	}
}

func BenchmarkSampleFullLogitsScratch(b *testing.B) {
	logits := make([]float32, 128000)
	for i := range logits {
		logits[i] = float32(i%97) / 97
	}
	rng := rand.New(rand.NewSource(1))
	scratch := &generationScratch{}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = sampleTokenScratch(logits, GenerateOptions{Temperature: 0.8, TopK: 0}, rng, scratch)
	}
}

func BenchmarkSampleFullLogitsTemp1Scratch(b *testing.B) {
	logits := make([]float32, 128000)
	for i := range logits {
		logits[i] = float32(i%97) / 97
	}
	rng := rand.New(rand.NewSource(1))
	scratch := &generationScratch{}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = sampleTokenScratch(logits, GenerateOptions{Temperature: 1, TopK: 0}, rng, scratch)
	}
}

func BenchmarkTopKCandidates(b *testing.B) {
	logits := make([]float32, 128000)
	for i := range logits {
		logits[i] = float32(i%997) / 997
	}
	scratch := &generationScratch{}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = topKCandidates(logits, 50, scratch)
	}
}

func BenchmarkSampleTopK(b *testing.B) {
	logits := make([]float32, 128000)
	for i := range logits {
		logits[i] = float32(i%997) / 997
	}
	rng := rand.New(rand.NewSource(1))
	scratch := &generationScratch{}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = sampleTokenScratch(logits, GenerateOptions{Temperature: 0.8, TopK: 50}, rng, scratch)
	}
}

func BenchmarkSampleTopKTemp1(b *testing.B) {
	logits := make([]float32, 128000)
	for i := range logits {
		logits[i] = float32(i%997) / 997
	}
	rng := rand.New(rand.NewSource(1))
	scratch := &generationScratch{}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = sampleTokenScratch(logits, GenerateOptions{Temperature: 1, TopK: 50}, rng, scratch)
	}
}

func BenchmarkSamplingRNGGreedy(b *testing.B) {
	opts := GenerateOptions{MaxNewTokens: 64, Temperature: 0}
	var rng fastRNG
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if samplingRNGInto(opts, &rng) != nil {
			samplingBenchState = rng.state
		}
	}
}

func BenchmarkSamplingRNGSampling(b *testing.B) {
	opts := GenerateOptions{MaxNewTokens: 64, Temperature: 1, Seed: 1}
	var rng fastRNG
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		samplingBenchState = samplingRNGInto(opts, &rng).state
	}
}
