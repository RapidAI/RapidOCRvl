package tensor

import "testing"

func BenchmarkWeightedSum2_128(b *testing.B) {
	dst := make([]float32, 128)
	x0 := make([]float32, 128)
	x1 := make([]float32, 128)
	fillBench(x0)
	fillBench(x1)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		WeightedSum2(dst, x0, x1, 0.6, 0.4)
	}
}

func BenchmarkWeightedSum3_128(b *testing.B) {
	dst := make([]float32, 128)
	x0 := make([]float32, 128)
	x1 := make([]float32, 128)
	x2 := make([]float32, 128)
	fillBench(x0)
	fillBench(x1)
	fillBench(x2)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		WeightedSum3(dst, x0, x1, x2, 0.3, 0.4, 0.3)
	}
}

func BenchmarkWeightedSum4_128(b *testing.B) {
	dst := make([]float32, 128)
	x0 := make([]float32, 128)
	x1 := make([]float32, 128)
	x2 := make([]float32, 128)
	x3 := make([]float32, 128)
	fillBench(x0)
	fillBench(x1)
	fillBench(x2)
	fillBench(x3)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		WeightedSum4(dst, x0, x1, x2, x3, 0.25, 0.25, 0.25, 0.25)
	}
}

func BenchmarkWeightedSumAdd4_128(b *testing.B) {
	dst := make([]float32, 128)
	x0 := make([]float32, 128)
	x1 := make([]float32, 128)
	x2 := make([]float32, 128)
	x3 := make([]float32, 128)
	fillBench(dst)
	fillBench(x0)
	fillBench(x1)
	fillBench(x2)
	fillBench(x3)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		WeightedSumAdd4(dst, x0, x1, x2, x3, 0.25, 0.25, 0.25, 0.25)
	}
}

func BenchmarkScaleCopy_128(b *testing.B) {
	dst := make([]float32, 128)
	x := make([]float32, 128)
	fillBench(x)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ScaleCopy(dst, x, 0.5)
	}
}

func BenchmarkScaleAdd_128(b *testing.B) {
	dst := make([]float32, 128)
	x := make([]float32, 128)
	fillBench(dst)
	fillBench(x)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ScaleAdd(dst, x, 0.5)
	}
}