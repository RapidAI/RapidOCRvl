package tensor

import (
	"testing"
)

func TestDotQ4PairFMAExhaustive(t *testing.T) {
	if !useDotFMA || !useDotQ4AVX2 {
		t.Skip("FMA or Q4 AVX2 not available")
	}
	for cols := 8; cols <= 300; cols++ {
		a := make([]float32, cols)
		b := make([]float32, cols)
		x := make([]float32, cols)
		for i := range a {
			a[i] = float32(i%15 - 7)
			b[i] = float32((i*3+5)%15 - 7)
			x[i] = float32(i%13-6) / 6.0
		}
		packedCols := (cols + 1) / 2
		qa := make([]byte, packedCols)
		qb := make([]byte, packedCols)
		for i := 0; i < cols; i += 2 {
			lo := clampQ4(a[i])
			qa[i/2] = byte(int(lo) & 15)
			if i+1 < cols {
				hi := clampQ4(a[i+1])
				qa[i/2] |= byte((int(hi) & 15) << 4)
			}
			lob := clampQ4(b[i])
			qb[i/2] = byte(int(lob) & 15)
			if i+1 < cols {
				hib := clampQ4(b[i+1])
				qb[i/2] |= byte((int(hib) & 15) << 4)
			}
		}
		got0, got1 := dotQ4PairFMA(qa, qb, x, cols)
		want0 := float32(0)
		want1 := float32(0)
		for i := 0; i < cols; i++ {
			want0 += unpackQ4(qa, i) * x[i]
			want1 += unpackQ4(qb, i) * x[i]
		}
		tol0 := 0.01*maxf32(absf32(want0), 1) + 1e-4
		tol1 := 0.01*maxf32(absf32(want1), 1) + 1e-4
		if absf32(got0-want0) > tol0 || absf32(got1-want1) > tol1 {
			t.Errorf("cols=%d: got0=%v want0=%v diff0=%v | got1=%v want1=%v diff1=%v", cols, got0, want0, got0-want0, got1, want1, got1-want1)
		}
	}
}
