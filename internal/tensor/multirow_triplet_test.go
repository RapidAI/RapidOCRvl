package tensor

import (
	"testing"
)

func TestDotQ8TripletVNNICoreMultiRowZMM(t *testing.T) {
	rows, cols := 4, 512
	a := make([]int8, rows*cols)
	b := make([]int8, rows*cols)
	c := make([]int8, rows*cols)
	xq := make([]uint8, cols)
	outA := make([]int32, rows)
	outB := make([]int32, rows)
	outC := make([]int32, rows)

	for r := 0; r < rows; r++ {
		for col := 0; col < cols; col++ {
			a[r*cols+col] = int8((col + r) % 17 - 8)
			b[r*cols+col] = int8((col*2 + r) % 17 - 8)
			c[r*cols+col] = int8((col*3 + r) % 17 - 8)
		}
	}
	for col := 0; col < cols; col++ {
		xq[col] = uint8((col % 13) + 120)
	}

	dotQ8TripletVNNICoreMultiRowZMM(&a[0], &b[0], &c[0], &xq[0], &outA[0], &outB[0], &outC[0], rows, cols)

	for r := 0; r < rows; r++ {
		refA, refB, refC := dotQ8TripletVNNICore(&a[r*cols], &b[r*cols], &c[r*cols], &xq[0], cols)
		if outA[r] != refA || outB[r] != refB || outC[r] != refC {
			t.Errorf("row %d: A multi=%d single=%d, B multi=%d single=%d, C multi=%d single=%d",
				r, outA[r], refA, outB[r], refB, outC[r], refC)
		} else {
			t.Logf("row %d: A=%d B=%d C=%d OK", r, outA[r], outB[r], outC[r])
		}
	}
}

func TestDotQ8TripletVNNICoreMultiRowZMMVarSizes(t *testing.T) {
	for _, cols := range []int{64, 128, 256, 512, 1024, 2048} {
		for _, rows := range []int{1, 2, 3, 8, 16, 32} {
			a := make([]int8, rows*cols)
			b := make([]int8, rows*cols)
			c := make([]int8, rows*cols)
			xq := make([]uint8, cols)
			outA := make([]int32, rows)
			outB := make([]int32, rows)
			outC := make([]int32, rows)
			for i := range a {
				a[i] = int8(i%17 - 8)
				b[i] = int8((i*3+1)%17 - 8)
				c[i] = int8((i*7+2)%17 - 8)
			}
			for i := range xq {
				xq[i] = uint8(i%13 + 120)
			}
			dotQ8TripletVNNICoreMultiRowZMM(&a[0], &b[0], &c[0], &xq[0], &outA[0], &outB[0], &outC[0], rows, cols)
			for r := 0; r < rows; r++ {
				refA, refB, refC := dotQ8TripletVNNICore(&a[r*cols], &b[r*cols], &c[r*cols], &xq[0], cols)
				if outA[r] != refA || outB[r] != refB || outC[r] != refC {
					t.Errorf("rows=%d cols=%d r=%d: A multi=%d single=%d B multi=%d single=%d C multi=%d single=%d",
						rows, cols, r, outA[r], refA, outB[r], refB, outC[r], refC)
				}
			}
		}
	}
}
