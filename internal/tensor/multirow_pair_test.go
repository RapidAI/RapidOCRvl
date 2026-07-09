package tensor

import (
	"testing"
)

func TestDotQ8PairVNNICoreMultiRowZMM(t *testing.T) {
	rows, cols := 4, 512
	a := make([]int8, rows*cols)
	b := make([]int8, rows*cols)
	xq := make([]uint8, cols)
	outA := make([]int32, rows)
	outB := make([]int32, rows)

	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			a[r*cols+c] = int8((c + r) % 17 - 8)
			b[r*cols+c] = int8((c*2 + r) % 17 - 8)
		}
	}
	for c := 0; c < cols; c++ {
		xq[c] = uint8((c % 13) + 120)
	}

	dotQ8PairVNNICoreMultiRowZMM(&a[0], &b[0], &xq[0], &outA[0], &outB[0], rows, cols)

	for r := 0; r < rows; r++ {
		refA, refB := dotQ8PairVNNICore(&a[r*cols], &b[r*cols], &xq[0], cols)
		if outA[r] != refA || outB[r] != refB {
			t.Errorf("row %d: A multi=%d single=%d, B multi=%d single=%d",
				r, outA[r], refA, outB[r], refB)
		} else {
			t.Logf("row %d: A=%d B=%d OK", r, outA[r], outB[r])
		}
	}
}

func TestDotQ8PairVNNICoreMultiRowZMMVarSizes(t *testing.T) {
	for _, cols := range []int{64, 128, 256, 512, 1024, 2048} {
		for _, rows := range []int{1, 2, 3, 8, 16, 32} {
			a := make([]int8, rows*cols)
			b := make([]int8, rows*cols)
			xq := make([]uint8, cols)
			outA := make([]int32, rows)
			outB := make([]int32, rows)
			for i := range a {
				a[i] = int8(i%17 - 8)
				b[i] = int8((i*3+1)%17 - 8)
			}
			for i := range xq {
				xq[i] = uint8(i%13 + 120)
			}
			dotQ8PairVNNICoreMultiRowZMM(&a[0], &b[0], &xq[0], &outA[0], &outB[0], rows, cols)
			for r := 0; r < rows; r++ {
				refA, refB := dotQ8PairVNNICore(&a[r*cols], &b[r*cols], &xq[0], cols)
				if outA[r] != refA || outB[r] != refB {
					t.Errorf("rows=%d cols=%d r=%d: A multi=%d single=%d B multi=%d single=%d",
						rows, cols, r, outA[r], refA, outB[r], refB)
				}
			}
		}
	}
}
