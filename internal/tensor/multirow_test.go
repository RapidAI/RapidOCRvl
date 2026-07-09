package tensor

import (
	"testing"
)

func TestDotQ8VNNICoreMultiRowZMM(t *testing.T) {
	rows, cols := 4, 512
	a := make([]int8, rows*cols)
	xq := make([]uint8, cols)
	out := make([]int32, rows)

	// Fill with known pattern
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			a[r*cols+c] = int8((c % 17) - 8)
		}
	}
	for c := 0; c < cols; c++ {
		xq[c] = uint8((c % 13) + 120)
	}

	dotQ8VNNICoreMultiRowZMM(&a[0], &xq[0], &out[0], rows, cols)

	// Reference
	for r := 0; r < rows; r++ {
		ref := dotQ8VNNICoreZMM(&a[r*cols], &xq[0], cols)
		if out[r] != ref {
			t.Errorf("row %d: multirow=%d single=%d", r, out[r], ref)
		} else {
			t.Logf("row %d: OK %d", r, out[r])
		}
	}
}

func TestDotQ8VNNICoreMultiRowZMMVarSizes(t *testing.T) {
	for _, cols := range []int{64, 128, 256, 512, 1024, 2048} {
		for _, rows := range []int{1, 2, 3, 8, 16, 32} {
			a := make([]int8, rows*cols)
			xq := make([]uint8, cols)
			out := make([]int32, rows)
			for i := range a {
				a[i] = int8(i%17 - 8)
			}
			for i := range xq {
				xq[i] = uint8(i%13 + 120)
			}
			dotQ8VNNICoreMultiRowZMM(&a[0], &xq[0], &out[0], rows, cols)
			for r := 0; r < rows; r++ {
				ref := dotQ8VNNICoreZMM(&a[r*cols], &xq[0], cols)
				if out[r] != ref {
					t.Errorf("rows=%d cols=%d r=%d: multi=%d single=%d", rows, cols, r, out[r], ref)
				}
			}
		}
	}
}
