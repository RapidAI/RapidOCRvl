//go:build windows

package backend

import "fmt"

func checkedMatVecF32WeightLenWin(rows, cols int, label string) (int, error) {
	wLen, ok := checkedMulInt(rows, cols)
	if !ok {
		return 0, fmt.Errorf("%s weight length overflows: rows=%d cols=%d", label, rows, cols)
	}
	return wLen, nil
}

type vulkanMatRowsBiasDimsWin struct {
	xLen   int
	wLen   int
	addLen int
	outLen int
}

func checkedMatRowsBiasDimsWin(batches, addRows, rows, cols int, label string) (vulkanMatRowsBiasDimsWin, error) {
	xLen, ok := checkedMulInt(batches, cols)
	if !ok {
		return vulkanMatRowsBiasDimsWin{}, fmt.Errorf("%s x length overflows: batches=%d cols=%d", label, batches, cols)
	}
	wLen, err := checkedMatVecF32WeightLenWin(rows, cols, label)
	if err != nil {
		return vulkanMatRowsBiasDimsWin{}, err
	}
	outLen, ok := checkedMulInt(batches, rows)
	if !ok {
		return vulkanMatRowsBiasDimsWin{}, fmt.Errorf("%s output length overflows: batches=%d rows=%d", label, batches, rows)
	}
	addLen := 0
	if addRows > 0 {
		addLen, ok = checkedMulInt(addRows, rows)
		if !ok {
			return vulkanMatRowsBiasDimsWin{}, fmt.Errorf("%s add length overflows: addRows=%d rows=%d", label, addRows, rows)
		}
	}
	return vulkanMatRowsBiasDimsWin{xLen: xLen, wLen: wLen, addLen: addLen, outLen: outLen}, nil
}

type vulkanMatRowsBias3DimsWin struct {
	xLen    int
	waLen   int
	wbLen   int
	wcLen   int
	outALen int
	outBLen int
	outCLen int
}

func checkedMatRowsBias3DimsWin(batches, rowsA, rowsB, rowsC, cols int, label string) (vulkanMatRowsBias3DimsWin, error) {
	xLen, ok := checkedMulInt(batches, cols)
	if !ok {
		return vulkanMatRowsBias3DimsWin{}, fmt.Errorf("%s x length overflows: batches=%d cols=%d", label, batches, cols)
	}
	waLen, err := checkedMatVecF32WeightLenWin(rowsA, cols, label+" wa")
	if err != nil {
		return vulkanMatRowsBias3DimsWin{}, err
	}
	wbLen, err := checkedMatVecF32WeightLenWin(rowsB, cols, label+" wb")
	if err != nil {
		return vulkanMatRowsBias3DimsWin{}, err
	}
	wcLen, err := checkedMatVecF32WeightLenWin(rowsC, cols, label+" wc")
	if err != nil {
		return vulkanMatRowsBias3DimsWin{}, err
	}
	outALen, ok := checkedMulInt(batches, rowsA)
	if !ok {
		return vulkanMatRowsBias3DimsWin{}, fmt.Errorf("%s outA length overflows: batches=%d rowsA=%d", label, batches, rowsA)
	}
	outBLen, ok := checkedMulInt(batches, rowsB)
	if !ok {
		return vulkanMatRowsBias3DimsWin{}, fmt.Errorf("%s outB length overflows: batches=%d rowsB=%d", label, batches, rowsB)
	}
	outCLen, ok := checkedMulInt(batches, rowsC)
	if !ok {
		return vulkanMatRowsBias3DimsWin{}, fmt.Errorf("%s outC length overflows: batches=%d rowsC=%d", label, batches, rowsC)
	}
	return vulkanMatRowsBias3DimsWin{xLen: xLen, waLen: waLen, wbLen: wbLen, wcLen: wcLen, outALen: outALen, outBLen: outBLen, outCLen: outCLen}, nil
}

type vulkanMatRowsGELU2DimsWin struct {
	xLen      int
	hiddenLen int
	outLen    int
	w1Len     int
	w2Len     int
}

func checkedMatRowsGELU2DimsWin(batches, hiddenRows, cols, outRows int, label string) (vulkanMatRowsGELU2DimsWin, error) {
	xLen, ok := checkedMulInt(batches, cols)
	if !ok {
		return vulkanMatRowsGELU2DimsWin{}, fmt.Errorf("%s x length overflows: batches=%d cols=%d", label, batches, cols)
	}
	hiddenLen, ok := checkedMulInt(batches, hiddenRows)
	if !ok {
		return vulkanMatRowsGELU2DimsWin{}, fmt.Errorf("%s hidden length overflows: batches=%d hiddenRows=%d", label, batches, hiddenRows)
	}
	outLen, ok := checkedMulInt(batches, outRows)
	if !ok {
		return vulkanMatRowsGELU2DimsWin{}, fmt.Errorf("%s output length overflows: batches=%d outRows=%d", label, batches, outRows)
	}
	w1Len, ok := checkedMulInt(hiddenRows, cols)
	if !ok {
		return vulkanMatRowsGELU2DimsWin{}, fmt.Errorf("%s w1 length overflows: hiddenRows=%d cols=%d", label, hiddenRows, cols)
	}
	w2Len, ok := checkedMulInt(outRows, hiddenRows)
	if !ok {
		return vulkanMatRowsGELU2DimsWin{}, fmt.Errorf("%s w2 length overflows: outRows=%d hiddenRows=%d", label, outRows, hiddenRows)
	}
	return vulkanMatRowsGELU2DimsWin{xLen: xLen, hiddenLen: hiddenLen, outLen: outLen, w1Len: w1Len, w2Len: w2Len}, nil
}
