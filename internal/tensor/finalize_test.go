package tensor

import (
	"testing"
)

func TestFinalizeDotQ8VNNI(t *testing.T) {
	n := 16
	dots := make([]int32, n)
	rowSum := make([]int32, n)
	scale := make([]float32, n)
	out := make([]float32, n)
	scaleX := float32(0.5)

	for i := 0; i < n; i++ {
		dots[i] = int32(i*100 + 128)
		rowSum[i] = int32(i)
		scale[i] = float32(i+1) * 0.1
	}

	finalizeDotQ8VNNI(&dots[0], &rowSum[0], &scale[0], &out[0], n, scaleX)

	for i := 0; i < n; i++ {
		expected := float32(dots[i]-128*rowSum[i]) * scaleX * scale[i]
		diff := out[i] - expected
		if diff < -0.001 || diff > 0.001 {
			t.Errorf("i=%d: got %f expected %f diff %f", i, out[i], expected, diff)
		}
	}
}

func TestFinalizeDotQ8VNNIVarSizes(t *testing.T) {
	for _, n := range []int{1, 4, 7, 8, 9, 15, 16, 17, 32, 33, 100, 256} {
		dots := make([]int32, n)
		rowSum := make([]int32, n)
		scale := make([]float32, n)
		out := make([]float32, n)
		scaleX := float32(0.007)
		for i := 0; i < n; i++ {
			dots[i] = int32(i*37 - 128*(i%5))
			rowSum[i] = int32(i%5)
			scale[i] = float32(i+1) * 0.003
		}
		finalizeDotQ8VNNI(&dots[0], &rowSum[0], &scale[0], &out[0], n, scaleX)
		for i := 0; i < n; i++ {
			expected := float32(dots[i]-128*rowSum[i]) * scaleX * scale[i]
			diff := out[i] - expected
			if diff < -0.001 || diff > 0.001 {
				t.Errorf("n=%d i=%d: got %f expected %f diff %f", n, i, out[i], expected, diff)
			}
		}
	}
}
