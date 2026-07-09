package tensor

import (
	"math/rand"
	"testing"
)

func TestDotQ4FMALarge(t *testing.T) {
	cols := 256
	a := make([]float32, cols)
	b := make([]float32, cols)
	r := rand.New(rand.NewSource(42))
	for i := range a {
		a[i] = float32(r.Float64()*4 - 2)
		b[i] = float32(r.Float64()*4 - 2)
	}
	packedCols := (cols + 1) / 2
	q4a := make([]byte, packedCols)
	for i := 0; i < cols; i += 2 {
		lo := clampQ4(a[i])
		if i+1 < cols {
			hi := clampQ4(a[i+1])
			q4a[i/2] = byte((int(lo) & 15) | ((int(hi) & 15) << 4))
		} else {
			q4a[i/2] = byte(int(lo) & 15)
		}
	}
	if useDotFMA && useDotQ4AVX2 {
		got := dotQ4FMA(q4a, b, cols)
		want := float32(0)
		for i := 0; i < cols; i++ {
			q := unpackQ4(q4a, i)
			want += q * b[i]
		}
		if absf32(got-want) > 0.01*absf32(want)+1e-4 {
			t.Errorf("dotQ4FMA cols=%d: got %v want %v", cols, got, want)
		}
	}
}

func clampQ4(x float32) int {
	q := int(x + 8.5)
	if q < 0 { q = 0 }
	if q > 15 { q = 15 }
	return q
}

func unpackQ4(data []byte, idx int) float32 {
	b := data[idx/2]
	var nibble int
	if idx%2 == 0 {
		nibble = int(b & 15)
	} else {
		nibble = int((b >> 4) & 15)
	}
	return float32(nibble) - 8
}

func absf32(x float32) float32 {
	if x < 0 { return -x }
	return x
}