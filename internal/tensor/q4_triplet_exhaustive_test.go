package tensor

import (
	"testing"
)

func TestDotQ4TripletFMAExhaustive(t *testing.T) {
	if !useDotFMA || !useDotQ4AVX2 {
		t.Skip("FMA or Q4 AVX2 not available")
	}
	for cols := 8; cols <= 300; cols++ {
		a := make([]float32, cols)
		b := make([]float32, cols)
		c := make([]float32, cols)
		x := make([]float32, cols)
		for i := range a {
			a[i] = float32(i%15 - 7)
			b[i] = float32((i*3+5)%15 - 7)
			c[i] = float32((i*7+3)%15 - 7)
			x[i] = float32(i%13-6) / 6.0
		}
		packedCols := (cols + 1) / 2
		qa := make([]byte, packedCols)
		qb := make([]byte, packedCols)
		qc := make([]byte, packedCols)
		for i := 0; i < cols; i += 2 {
			for _, pair := range []struct {
				src []float32
				dst []byte
			}{{a, qa}, {b, qb}, {c, qc}} {
				lo := clampQ4(pair.src[i])
				pair.dst[i/2] = byte(int(lo) & 15)
				if i+1 < cols {
					hi := clampQ4(pair.src[i+1])
					pair.dst[i/2] |= byte((int(hi) & 15) << 4)
				}
			}
		}
		got0, got1, got2 := dotQ4TripletFMA(qa, qb, qc, x, cols)
		want0 := float32(0)
		want1 := float32(0)
		want2 := float32(0)
		for i := 0; i < cols; i++ {
			want0 += unpackQ4(qa, i) * x[i]
			want1 += unpackQ4(qb, i) * x[i]
			want2 += unpackQ4(qc, i) * x[i]
		}
		tol := float32(0.01 + 1e-4)
		if absf32(got0-want0) > tol*maxf32(absf32(want0), 1) ||
			absf32(got1-want1) > tol*maxf32(absf32(want1), 1) ||
			absf32(got2-want2) > tol*maxf32(absf32(want2), 1) {
			t.Errorf("cols=%d: g0=%v w0=%v | g1=%v w1=%v | g2=%v w2=%v", cols, got0, want0, got1, want1, got2, want2)
		}
	}
}
