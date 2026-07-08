package backend

import (
	"math"
	"os"
	"sort"
	"testing"

	"paddleocrvl-go/internal/tensor"
)

func TestVulkanDispatchSmokeTest(t *testing.T) {
	if os.Getenv("RAPIDOCRVL_VULKAN_SMOKE") != "1" {
		t.Skip("set RAPIDOCRVL_VULKAN_SMOKE=1 to run the Vulkan GPU dispatch smoke test")
	}
	if err := VulkanDispatchSmokeTest(); err != nil {
		t.Fatal(err)
	}
	results := VulkanDispatchSmokeSuite()
	if len(results) == 0 {
		t.Fatal("empty Vulkan dispatch smoke suite")
	}
	for _, result := range results {
		if !result.OK {
			t.Fatalf("probe %s failed: %s", result.Name, result.Error)
		}
	}
}

func TestSelectVulkanTopKCandidates(t *testing.T) {
	data := []float32{
		1, 3,
		5, 7,
		5, 2,
		4, 99,
		6, -1,
		3, 1,
		5, 4,
	}
	got := selectVulkanTopKCandidates(data, 10, 3)
	want := []VulkanTokenScore{
		{Token: 2, Score: 5},
		{Token: 4, Score: 5},
		{Token: 7, Score: 5},
	}
	if len(got) != len(want) {
		t.Fatalf("len=%d want %d got=%#v", len(got), len(want), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("top[%d]=%#v want %#v all=%#v", i, got[i], want[i], got)
		}
	}
	scratch := make([]VulkanTokenScore, 0, 8)
	got = selectVulkanTopKCandidatesInto(scratch, data, 10, 3)
	if len(got) != len(want) || cap(got) != cap(scratch) {
		t.Fatalf("scratch top-k len=%d cap=%d want len=%d cap=%d", len(got), cap(got), len(want), cap(scratch))
	}
	if len(got) > 0 && &got[:cap(got)][0] != &scratch[:cap(scratch)][0] {
		t.Fatal("top-k candidate scratch was not reused")
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("scratch top[%d]=%#v want %#v all=%#v", i, got[i], want[i], got)
		}
	}
}

func TestVulkanMatVecF32(t *testing.T) {
	if os.Getenv("RAPIDOCRVL_VULKAN_SMOKE") != "1" {
		t.Skip("set RAPIDOCRVL_VULKAN_SMOKE=1 to run the Vulkan GPU matvec test")
	}
	x := []float32{2, -1, 0.5, 4, -3}
	w := []float32{
		1, 0, 0, 0, 0,
		0, 1, 2, 0, 0,
		-1, 0, 0, 0.25, 1,
		0.5, 0.5, 0.5, 0.5, 0.5,
	}
	out := make([]float32, 4)
	if err := VulkanMatVecF32(out, x, w, 4, 5); err != nil {
		t.Fatal(err)
	}
	want := []float32{2, 0, -4, 1.25}
	for i := range want {
		if d := out[i] - want[i]; d < -1e-4 || d > 1e-4 {
			t.Fatalf("out[%d] = %.6f, want %.6f", i, out[i], want[i])
		}
	}
	xRepeat := []float32{1, 2, 3, 4, 5}
	wRepeat := []float32{
		1, 1, 1, 1, 1,
		1, 0, 0, 0, -1,
		0, 2, 0, -1, 0,
		-1, -1, -1, -1, -1,
	}
	if err := VulkanMatVecF32(out, xRepeat, wRepeat, 4, 5); err != nil {
		t.Fatal(err)
	}
	wantRepeat := []float32{15, -4, 0, -15}
	for i := range wantRepeat {
		if d := out[i] - wantRepeat[i]; d < -1e-4 || d > 1e-4 {
			t.Fatalf("repeat out[%d] = %.6f, want %.6f", i, out[i], wantRepeat[i])
		}
	}

	xSmall := []float32{3, -2, 5}
	wSmall := []float32{
		1, 2, 0,
		-1, 0, 1,
	}
	outSmall := make([]float32, 2)
	if err := VulkanMatVecF32(outSmall, xSmall, wSmall, 2, 3); err != nil {
		t.Fatal(err)
	}
	wantSmall := []float32{-1, 2}
	for i := range wantSmall {
		if d := outSmall[i] - wantSmall[i]; d < -1e-4 || d > 1e-4 {
			t.Fatalf("small out[%d] = %.6f, want %.6f", i, outSmall[i], wantSmall[i])
		}
	}
}

func TestVulkanMatVecArgmaxF32(t *testing.T) {
	if os.Getenv("RAPIDOCRVL_VULKAN_SMOKE") != "1" {
		t.Skip("set RAPIDOCRVL_VULKAN_SMOKE=1 to run the Vulkan GPU matvec argmax test")
	}
	x := []float32{2, -1, 0.5, 4, -3}
	w := []float32{
		1, 0, 0, 0, 0,
		0, 1, 2, 0, 0,
		-1, 0, 0, 0.25, 1,
		0.5, 0.5, 0.5, 0.5, 0.5,
	}
	token, score, err := VulkanMatVecArgmaxF32(x, w, 4, 5)
	if err != nil {
		t.Fatal(err)
	}
	if token != 0 || math.Abs(float64(score-2)) > 1e-4 {
		t.Fatalf("argmax token=%d score=%.6f want token=0 score=2", token, score)
	}
	wTie := []float32{
		1, 0, 0, 0, 0,
		1, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
	}
	token, score, err = VulkanMatVecArgmaxF32(x, wTie, 3, 5)
	if err != nil {
		t.Fatal(err)
	}
	if token != 0 || math.Abs(float64(score-2)) > 1e-4 {
		t.Fatalf("tie argmax token=%d score=%.6f want token=0 score=2", token, score)
	}
}

func TestVulkanMatVecTopKF32(t *testing.T) {
	if os.Getenv("RAPIDOCRVL_VULKAN_SMOKE") != "1" {
		t.Skip("set RAPIDOCRVL_VULKAN_SMOKE=1 to run the Vulkan GPU matvec top-k test")
	}
	rows, cols, k := 270, 5, 50
	x := []float32{2, -1, 0.5, 4, -3}
	w := make([]float32, rows*cols)
	for r := 0; r < rows; r++ {
		base := float32((r%17)-8) * 0.125
		for c := 0; c < cols; c++ {
			w[r*cols+c] = base + float32((r+c)%7-3)*0.25
		}
	}
	setRow := func(r int, weight []float32) {
		copy(w[r*cols:(r+1)*cols], weight)
	}
	setRow(3, []float32{1, 0, 0, 0, 0})      // 2
	setRow(42, []float32{0, 0, 0, 1, 0})     // 4
	setRow(199, []float32{0, -2, 1, 1, -1})  // 9.5
	setRow(256, []float32{2, 0, 0, 1, -0.5}) // 9.5, tie should prefer 199
	setRow(269, []float32{-1, 1, 0, 2, -2})  // 11
	setRow(128, []float32{0.5, -1, 0, 1, 1}) // 3
	logits := make([]float32, rows)
	tensor.MatVec(logits, x, w, rows, cols)
	gpuLogits := make([]float32, rows)
	if err := VulkanMatVecF32(gpuLogits, x, w, rows, cols); err != nil {
		t.Fatal(err)
	}
	for _, idx := range []int{0, 42, 128, 199, 256, 269} {
		if math.Abs(float64(gpuLogits[idx]-logits[idx])) > 1e-4 {
			t.Fatalf("matvec logits[%d] = %.6f, want %.6f", idx, gpuLogits[idx], logits[idx])
		}
	}
	want := make([]VulkanTokenScore, rows)
	for i, score := range logits {
		want[i] = VulkanTokenScore{Token: i, Score: score}
	}
	sort.Slice(want, func(i, j int) bool {
		if want[i].Score == want[j].Score {
			return want[i].Token < want[j].Token
		}
		return want[i].Score > want[j].Score
	})
	got, err := VulkanMatVecTopKF32(x, w, rows, cols, k)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != k {
		t.Fatalf("top-k len=%d want %d: %#v", len(got), k, got)
	}
	for i := 0; i < k; i++ {
		if got[i].Token != want[i].Token || math.Abs(float64(got[i].Score-want[i].Score)) > 1e-4 {
			t.Fatalf("top-k[%d] = token=%d score=%.6f, want token=%d score=%.6f, got=%#v", i, got[i].Token, got[i].Score, want[i].Token, want[i].Score, got)
		}
	}
}

func TestVulkanMatVecTopKQuantized(t *testing.T) {
	if os.Getenv("RAPIDOCRVL_VULKAN_SMOKE") != "1" {
		t.Skip("set RAPIDOCRVL_VULKAN_SMOKE=1 to run the Vulkan GPU quantized matvec top-k test")
	}
	rows, cols, k := 270, 5, 50
	x := []float32{2, -1, 0.5, 4, -3}
	w := make([]float32, rows*cols)
	for r := 0; r < rows; r++ {
		base := float32((r%17)-8) * 0.125
		for c := 0; c < cols; c++ {
			w[r*cols+c] = base + float32((r+c)%7-3)*0.25
		}
	}
	setRow := func(r int, weight []float32) {
		copy(w[r*cols:(r+1)*cols], weight)
	}
	setRow(3, []float32{1, 0, 0, 0, 0})
	setRow(42, []float32{0, 0, 0, 1, 0})
	setRow(199, []float32{0, -2, 1, 1, -1})
	setRow(256, []float32{2, 0, 0, 1, -0.5})
	setRow(269, []float32{-1, 1, 0, 2, -2})
	setRow(128, []float32{0.5, -1, 0, 1, 1})

	assertTopK := func(name string, got []VulkanTokenScore, logits []float32) {
		t.Helper()
		want := make([]VulkanTokenScore, rows)
		for i, score := range logits {
			want[i] = VulkanTokenScore{Token: i, Score: score}
		}
		sort.Slice(want, func(i, j int) bool {
			if want[i].Score == want[j].Score {
				return want[i].Token < want[j].Token
			}
			return want[i].Score > want[j].Score
		})
		if len(got) != k {
			t.Fatalf("%s top-k len=%d want %d: %#v", name, len(got), k, got)
		}
		for i := 0; i < k; i++ {
			if got[i].Token != want[i].Token || math.Abs(float64(got[i].Score-want[i].Score)) > 1e-4 {
				t.Fatalf("%s top-k[%d] = token=%d score=%.6f, want token=%d score=%.6f, got=%#v", name, i, got[i].Token, got[i].Score, want[i].Token, want[i].Score, got)
			}
		}
	}

	t.Run("q8", func(t *testing.T) {
		q := tensor.QuantizeQ8Row(w, rows, cols)
		logits := make([]float32, rows)
		tensor.MatVecQ8(logits, x, q)
		got, err := VulkanMatVecTopKQ8(x, q, k)
		if err != nil {
			t.Fatal(err)
		}
		assertTopK("q8", got, logits)
	})

	t.Run("q4", func(t *testing.T) {
		q := tensor.QuantizeQ4Row(w, rows, cols)
		logits := make([]float32, rows)
		tensor.MatVecQ4(logits, x, q)
		got, err := VulkanMatVecTopKQ4(x, q, k)
		if err != nil {
			t.Fatal(err)
		}
		assertTopK("q4", got, logits)
	})

	t.Run("q6", func(t *testing.T) {
		q := tensor.QuantizeQ6Row(w, rows, cols)
		logits := make([]float32, rows)
		tensor.MatVecQ6(logits, x, q)
		got, err := VulkanMatVecTopKQ6(x, q, k)
		if err != nil {
			t.Fatal(err)
		}
		assertTopK("q6", got, logits)
	})
}

func TestVulkanFusedMatVec2F32(t *testing.T) {
	if os.Getenv("RAPIDOCRVL_VULKAN_SMOKE") != "1" {
		t.Skip("set RAPIDOCRVL_VULKAN_SMOKE=1 to run the Vulkan GPU fused matvec2 test")
	}
	x := []float32{2, -1, 0.5, 4, -3}
	wb := []float32{
		1, 0, 0, 0, 0,
		0, 1, 2, 0, 0,
		-1, 0, 0, 0.25, 1,
	}
	wc := []float32{
		0.5, 0.5, 0.5, 0.5, 0.5,
		1, -1, 1, -1, 1,
	}
	outB := make([]float32, 3)
	outC := make([]float32, 2)
	if err := VulkanFusedMatVec2F32(outB, outC, x, wb, wc, 3, 2, 5); err != nil {
		t.Fatal(err)
	}
	wantB := make([]float32, 3)
	wantC := make([]float32, 2)
	tensor.MatVec(wantB, x, wb, 3, 5)
	tensor.MatVec(wantC, x, wc, 2, 5)
	assertCloseSliceTol(t, "fusedMatVec2F32B", outB, wantB, 1e-4)
	assertCloseSliceTol(t, "fusedMatVec2F32C", outC, wantC, 1e-4)
}

func TestVulkanFusedMatVec2MRoPEF32(t *testing.T) {
	if os.Getenv("RAPIDOCRVL_VULKAN_SMOKE") != "1" {
		t.Skip("set RAPIDOCRVL_VULKAN_SMOKE=1 to run the Vulkan GPU fused matvec2+mrope test")
	}
	x := []float32{2, -1, 0.5, 4}
	kvHeads, headDim := 1, 4
	kvRows := kvHeads * headDim
	wb := []float32{
		0, 0, 1, 1,
		-1, 2, 0, 0,
		0.5, 0.5, 0.5, 0.5,
		1, -1, 1, -1,
	}
	wc := []float32{
		2, 0, -1, 1,
		0, 1, 0, -1,
		1, 1, 1, 1,
		-1, 0, 2, 0,
	}
	cosTable := []float32{0.5, -0.25}
	sinTable := []float32{0.8660254, 0.9682458}
	outB := make([]float32, kvRows)
	outC := make([]float32, kvRows)
	wantB := make([]float32, kvRows)
	wantC := make([]float32, kvRows)
	tensor.MatVec(wantB, x, wb, kvRows, len(x))
	tensor.MatVec(wantC, x, wc, kvRows, len(x))
	applyMRoPEReference(wantB, kvHeads, headDim, cosTable, sinTable)
	if err := VulkanFusedMatVec2MRoPEF32(outB, outC, x, nil, wb, wc, cosTable, sinTable, kvRows, kvRows, len(x), kvHeads, headDim); err != nil {
		t.Fatal(err)
	}
	assertCloseSliceTol(t, "fusedMatVec2MRoPEF32B", outB, wantB, 1e-4)
	assertCloseSliceTol(t, "fusedMatVec2MRoPEF32C", outC, wantC, 1e-4)
}

func TestVulkanMatVecAddRMSNormF32(t *testing.T) {
	if os.Getenv("RAPIDOCRVL_VULKAN_SMOKE") != "1" {
		t.Skip("set RAPIDOCRVL_VULKAN_SMOKE=1 to run the Vulkan GPU matvec+add+rmsnorm test")
	}
	x := []float32{2, -1, 0.5, 4, -3}
	w := []float32{
		1, 0, 0, 0, 0,
		0, 1, 2, 0, 0,
		-1, 0, 0, 0.25, 1,
		0.5, 0.5, 0.5, 0.5, 0.5,
	}
	residual := []float32{0.25, -0.5, 1.5, -1}
	weight := []float32{1, 0.5, -1, 1.25}
	mat := make([]float32, 4)
	tensor.MatVec(mat, x, w, 4, 5)
	wantResidual := append([]float32(nil), residual...)
	want := make([]float32, 4)
	tensor.AddRMSNorm(want, wantResidual, mat, weight, 1e-6)
	got := make([]float32, 4)
	if err := VulkanMatVecAddRMSNormF32(got, residual, x, w, weight, 4, 5); err != nil {
		t.Fatal(err)
	}
	assertCloseSliceTol(t, "matVecAddRMSNormF32Out", got, want, 1e-4)
	assertCloseSliceTol(t, "matVecAddRMSNormF32Residual", residual, wantResidual, 1e-4)
}

func TestVulkanRMSNormF32(t *testing.T) {
	if os.Getenv("RAPIDOCRVL_VULKAN_SMOKE") != "1" {
		t.Skip("set RAPIDOCRVL_VULKAN_SMOKE=1 to run the Vulkan GPU rmsnorm test")
	}
	x := []float32{2, -1, 0.5, 4, -3, 0.25, -0.75, 1.5}
	weight := []float32{1, 0.5, -1, 1.25, 0.75, -0.5, 1.5, 0.25}
	out := make([]float32, len(x))
	want := make([]float32, len(x))
	tensor.RMSNorm(want, x, weight, 1e-6)
	if err := VulkanRMSNormF32(out, x, weight); err != nil {
		t.Fatal(err)
	}
	assertCloseSliceTol(t, "rmsnormF32", out, want, 1e-4)

	xRepeat := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	weightRepeat := []float32{0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2}
	tensor.RMSNorm(want, xRepeat, weightRepeat, 1e-6)
	if err := VulkanRMSNormF32(out, xRepeat, weightRepeat); err != nil {
		t.Fatal(err)
	}
	assertCloseSliceTol(t, "rmsnormF32Repeat", out, want, 1e-4)
}

func TestVulkanAddRMSNormF32(t *testing.T) {
	if os.Getenv("RAPIDOCRVL_VULKAN_SMOKE") != "1" {
		t.Skip("set RAPIDOCRVL_VULKAN_SMOKE=1 to run the Vulkan GPU add+rmsnorm test")
	}
	dst := []float32{2, -1, 0.5, 4, -3, 0.25, -0.75, 1.5}
	add := []float32{0.25, -0.5, 1.5, -1, 2, 0.5, -0.25, 1}
	weight := []float32{1, 0.5, -1, 1.25, 0.75, -0.5, 1.5, 0.25}
	out := make([]float32, len(dst))
	wantDst := append([]float32(nil), dst...)
	want := make([]float32, len(dst))
	tensor.AddRMSNorm(want, wantDst, add, weight, 1e-6)
	if err := VulkanAddRMSNormF32(out, dst, add, weight); err != nil {
		t.Fatal(err)
	}
	assertCloseSliceTol(t, "addRMSNormF32Out", out, want, 1e-4)
	assertCloseSliceTol(t, "addRMSNormF32Dst", dst, wantDst, 1e-4)

	dstRepeat := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	addRepeat := []float32{-1, 0.5, -0.25, 1.25, -2, 0.75, 1, -0.5}
	weightRepeat := []float32{0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2}
	wantDst = append(wantDst[:0], dstRepeat...)
	tensor.AddRMSNorm(want, wantDst, addRepeat, weightRepeat, 1e-6)
	if err := VulkanAddRMSNormF32(out, dstRepeat, addRepeat, weightRepeat); err != nil {
		t.Fatal(err)
	}
	assertCloseSliceTol(t, "addRMSNormF32RepeatOut", out, want, 1e-4)
	assertCloseSliceTol(t, "addRMSNormF32RepeatDst", dstRepeat, wantDst, 1e-4)
}

func TestVulkanAddRMSNormF32OutOnly(t *testing.T) {
	if os.Getenv("RAPIDOCRVL_VULKAN_SMOKE") != "1" {
		t.Skip("set RAPIDOCRVL_VULKAN_SMOKE=1 to run the Vulkan GPU add+rmsnorm out-only test")
	}
	dst := []float32{2, -1, 0.5, 4, -3, 0.25, -0.75, 1.5}
	originalDst := append([]float32(nil), dst...)
	add := []float32{0.25, -0.5, 1.5, -1, 2, 0.5, -0.25, 1}
	weight := []float32{1, 0.5, -1, 1.25, 0.75, -0.5, 1.5, 0.25}
	out := make([]float32, len(dst))
	wantDst := append([]float32(nil), dst...)
	want := make([]float32, len(dst))
	tensor.AddRMSNorm(want, wantDst, add, weight, 1e-6)
	if err := VulkanAddRMSNormF32OutOnly(out, dst, add, weight); err != nil {
		t.Fatal(err)
	}
	assertCloseSliceTol(t, "addRMSNormF32OutOnlyOut", out, want, 1e-4)
	assertCloseSliceTol(t, "addRMSNormF32OutOnlyDstUnchanged", dst, originalDst, 0)
}

func TestVulkanMRoPEF32(t *testing.T) {
	if os.Getenv("RAPIDOCRVL_VULKAN_SMOKE") != "1" {
		t.Skip("set RAPIDOCRVL_VULKAN_SMOKE=1 to run the Vulkan GPU mrope test")
	}
	heads, dim := 3, 8
	x := []float32{
		1, 2, 3, 4, -1, -2, -3, -4,
		0.5, -1.5, 2.5, -3.5, 1.25, -2.25, 3.25, -4.25,
		-0.75, 1.75, -2.75, 3.75, 0.25, -1.25, 2.25, -3.25,
	}
	cosTable := []float32{1, 0.5, -0.25, 0}
	sinTable := []float32{0, 0.8660254, 0.9682458, -1}
	want := append([]float32(nil), x...)
	applyMRoPEReference(want, heads, dim, cosTable, sinTable)
	if err := VulkanMRoPEF32(x, cosTable, sinTable, heads, dim); err != nil {
		t.Fatal(err)
	}
	assertCloseSliceTol(t, "mropeF32", x, want, 1e-4)

	xRepeat := []float32{
		2, 1, 0, -1, -2, -1, 0, 1,
		3, -3, 2, -2, 1, -1, 0.5, -0.5,
		4, 0.25, -4, -0.25, 2, -2, 1, -1,
	}
	want = append(want[:0], xRepeat...)
	applyMRoPEReference(want, heads, dim, cosTable, sinTable)
	if err := VulkanMRoPEF32(xRepeat, cosTable, sinTable, heads, dim); err != nil {
		t.Fatal(err)
	}
	assertCloseSliceTol(t, "mropeF32Repeat", xRepeat, want, 1e-4)
}

func TestVulkanMRoPEPairF32(t *testing.T) {
	if os.Getenv("RAPIDOCRVL_VULKAN_SMOKE") != "1" {
		t.Skip("set RAPIDOCRVL_VULKAN_SMOKE=1 to run the Vulkan GPU mrope pair test")
	}
	qHeads, kvHeads, dim := 4, 2, 8
	q := []float32{
		1, 2, 3, 4, -1, -2, -3, -4,
		0.5, -1.5, 2.5, -3.5, 1.25, -2.25, 3.25, -4.25,
		-0.75, 1.75, -2.75, 3.75, 0.25, -1.25, 2.25, -3.25,
		2, 1, 0, -1, -2, -1, 0, 1,
	}
	k := []float32{
		3, -3, 2, -2, 1, -1, 0.5, -0.5,
		4, 0.25, -4, -0.25, 2, -2, 1, -1,
	}
	cosTable := []float32{1, 0.5, -0.25, 0}
	sinTable := []float32{0, 0.8660254, 0.9682458, -1}
	wantQ := append([]float32(nil), q...)
	wantK := append([]float32(nil), k...)
	applyMRoPEReference(wantQ, qHeads, dim, cosTable, sinTable)
	applyMRoPEReference(wantK, kvHeads, dim, cosTable, sinTable)
	if err := VulkanMRoPEPairF32(q, k, cosTable, sinTable, qHeads, kvHeads, dim); err != nil {
		t.Fatal(err)
	}
	assertCloseSliceTol(t, "mropePairF32Q", q, wantQ, 1e-4)
	assertCloseSliceTol(t, "mropePairF32K", k, wantK, 1e-4)
}

func TestVulkanMatRowsBiasF32(t *testing.T) {
	if os.Getenv("RAPIDOCRVL_VULKAN_SMOKE") != "1" {
		t.Skip("set RAPIDOCRVL_VULKAN_SMOKE=1 to run the Vulkan GPU matrows+bias test")
	}
	xs := [][]float32{
		{2, -1, 0.5, 4, -3},
		{-1, 3, 2, 0.25, 1},
		{0, 1, -2, 3, 0.5},
	}
	w := []float32{
		1, 0, 0, 0, 0,
		0, 1, 2, 0, 0,
		-1, 0, 0, 0.25, 1,
		0.5, 0.5, 0.5, 0.5, 0.5,
	}
	bias := []float32{0.25, -0.5, 1, 2}
	out := makeRowsForBackendTest(3, 4)
	if err := VulkanMatRowsBiasF32(out, xs, w, bias, 4, 5); err != nil {
		t.Fatal(err)
	}
	want := makeRowsForBackendTest(3, 4)
	tensor.MatRowsBias(want, xs, w, bias, 4, 5)
	for i := range want {
		assertCloseSliceTol(t, "matrowsBiasF32", out[i], want[i], 1e-4)
	}

	xsRepeat := [][]float32{
		{1, 2, 3, 4, 5},
		{-1, -2, 0.5, 1, 0},
		{0.25, -0.5, 1.5, -1, 2},
	}
	wRepeat := []float32{
		1, 1, 1, 1, 1,
		1, 0, 0, 0, -1,
		0, 2, 0, -1, 0,
		-1, -1, -1, -1, -1,
	}
	biasRepeat := []float32{0, 0.5, -0.25, 1}
	if err := VulkanMatRowsBiasF32(out, xsRepeat, wRepeat, biasRepeat, 4, 5); err != nil {
		t.Fatal(err)
	}
	tensor.MatRowsBias(want, xsRepeat, wRepeat, biasRepeat, 4, 5)
	for i := range want {
		assertCloseSliceTol(t, "matrowsBiasF32Repeat", out[i], want[i], 1e-4)
	}
}

func TestVulkanMatRowsBiasAddRowsF32(t *testing.T) {
	if os.Getenv("RAPIDOCRVL_VULKAN_SMOKE") != "1" {
		t.Skip("set RAPIDOCRVL_VULKAN_SMOKE=1 to run the Vulkan GPU matrows+bias+addrows test")
	}
	xs := [][]float32{
		{2, -1, 0.5, 4, -3},
		{-1, 3, 2, 0.25, 1},
		{0, 1, -2, 3, 0.5},
	}
	w := []float32{
		1, 0, 0, 0, 0,
		0, 1, 2, 0, 0,
		-1, 0, 0, 0.25, 1,
		0.5, 0.5, 0.5, 0.5, 0.5,
	}
	bias := []float32{0.25, -0.5, 1, 2}
	add := [][]float32{
		{0.5, -1, 2, 0.25},
		{-0.75, 0.5, 0, 1.5},
	}
	out := makeRowsForBackendTest(3, 4)
	if err := VulkanMatRowsBiasAddRowsF32(out, xs, w, bias, add, 4, 5); err != nil {
		t.Fatal(err)
	}
	want := makeRowsForBackendTest(3, 4)
	tensor.MatRowsBiasAddRows(want, xs, w, bias, add, 4, 5)
	for i := range want {
		assertCloseSliceTol(t, "matrowsBiasAddRowsF32", out[i], want[i], 1e-4)
	}

	addRepeat := [][]float32{
		{1, 2, 3, 4},
		{-1, -2, -3, -4},
		{0.25, 0.5, 0.75, 1},
	}
	if err := VulkanMatRowsBiasAddRowsF32(out, xs, w, bias, addRepeat, 4, 5); err != nil {
		t.Fatal(err)
	}
	tensor.MatRowsBiasAddRows(want, xs, w, bias, addRepeat, 4, 5)
	for i := range want {
		assertCloseSliceTol(t, "matrowsBiasAddRowsF32Repeat", out[i], want[i], 1e-4)
	}
}

func TestVulkanMatRowsBias3F32(t *testing.T) {
	if os.Getenv("RAPIDOCRVL_VULKAN_SMOKE") != "1" {
		t.Skip("set RAPIDOCRVL_VULKAN_SMOKE=1 to run the Vulkan GPU matrows+bias3 test")
	}
	xs := [][]float32{
		{2, -1, 0.5, 4, -3},
		{-1, 3, 2, 0.25, 1},
		{0, 1, -2, 3, 0.5},
	}
	wa := []float32{
		1, 0, 0, 0, 0,
		0, 1, 1, 0, 0.5,
	}
	ba := []float32{0.25, -0.5}
	wb := []float32{
		0, 0, 1, 1, 0,
		-1, 2, 0, 0, 0,
		0.5, 0.5, 0.5, 0.5, 0.5,
	}
	bb := []float32{1, -1, 0.75}
	wc := []float32{
		2, 0, -1, 1, -0.5,
	}
	bc := []float32{-0.25}
	outA := makeRowsForBackendTest(3, 2)
	outB := makeRowsForBackendTest(3, 3)
	outC := makeRowsForBackendTest(3, 1)
	if err := VulkanMatRowsBias3F32(outA, outB, outC, xs, wa, ba, wb, bb, wc, bc, 2, 3, 1, 5); err != nil {
		t.Fatal(err)
	}
	wantA := makeRowsForBackendTest(3, 2)
	wantB := makeRowsForBackendTest(3, 3)
	wantC := makeRowsForBackendTest(3, 1)
	tensor.MatRowsBias3(wantA, wantB, wantC, xs, wa, ba, wb, bb, wc, bc, 2, 3, 1, 5)
	for i := range xs {
		assertCloseSliceTol(t, "matrowsBias3F32A", outA[i], wantA[i], 1e-4)
		assertCloseSliceTol(t, "matrowsBias3F32B", outB[i], wantB[i], 1e-4)
		assertCloseSliceTol(t, "matrowsBias3F32C", outC[i], wantC[i], 1e-4)
	}

	xsRepeat := [][]float32{
		{1, 2, 3, 4, 5},
		{-1, -2, 0.5, 1, 0},
		{0.25, -0.5, 1.5, -1, 2},
	}
	waRepeat := []float32{
		1, 1, 0, 0, 0.5,
		0, -1, 1, 0, -0.5,
	}
	baRepeat := []float32{0.1, -0.2}
	wbRepeat := []float32{
		1, 0, 1, 0, 0.25,
		0, 1, 0, 1, -0.25,
		-1, 0.5, 0, 1, 0,
	}
	bbRepeat := []float32{0.5, -0.5, 0.25}
	wcRepeat := []float32{
		0.5, 0.5, -1, 2, -0.5,
	}
	bcRepeat := []float32{0.75}
	if err := VulkanMatRowsBias3F32(outA, outB, outC, xsRepeat, waRepeat, baRepeat, wbRepeat, bbRepeat, wcRepeat, bcRepeat, 2, 3, 1, 5); err != nil {
		t.Fatal(err)
	}
	tensor.MatRowsBias3(wantA, wantB, wantC, xsRepeat, waRepeat, baRepeat, wbRepeat, bbRepeat, wcRepeat, bcRepeat, 2, 3, 1, 5)
	for i := range xsRepeat {
		assertCloseSliceTol(t, "matrowsBias3F32RepeatA", outA[i], wantA[i], 1e-4)
		assertCloseSliceTol(t, "matrowsBias3F32RepeatB", outB[i], wantB[i], 1e-4)
		assertCloseSliceTol(t, "matrowsBias3F32RepeatC", outC[i], wantC[i], 1e-4)
	}
}

func TestVulkanVisionAttentionF32(t *testing.T) {
	if os.Getenv("RAPIDOCRVL_VULKAN_SMOKE") != "1" {
		t.Skip("set RAPIDOCRVL_VULKAN_SMOKE=1 to run the Vulkan GPU vision attention test")
	}
	tokens, heads, headDim := 4, 2, 4
	hidden := heads * headDim
	q := makeRowsForBackendTest(tokens, hidden)
	k := makeRowsForBackendTest(tokens, hidden)
	v := makeRowsForBackendTest(tokens, hidden)
	for i := 0; i < tokens; i++ {
		for j := 0; j < hidden; j++ {
			q[i][j] = float32((i+1)*(j+2)%7-3) * 0.25
			k[i][j] = float32((i+3)*(j+1)%11-5) * 0.2
			v[i][j] = float32((i+2)*(j+4)%13-6) * 0.15
		}
	}
	out := makeRowsForBackendTest(tokens, hidden)
	if err := VulkanVisionAttentionF32(out, q, k, v, tokens, heads, headDim); err != nil {
		t.Fatal(err)
	}
	want := makeRowsForBackendTest(tokens, hidden)
	visionAttentionReference(want, q, k, v, tokens, heads, headDim)
	for i := range want {
		assertCloseSliceTol(t, "visionAttentionF32", out[i], want[i], 1e-4)
	}

	for i := 0; i < tokens; i++ {
		for j := 0; j < hidden; j++ {
			q[i][j] = float32((i+2)*(j+5)%9-4) * 0.18
			k[i][j] = float32((i+4)*(j+3)%10-5) * 0.16
			v[i][j] = float32((i+5)*(j+2)%12-6) * 0.13
		}
	}
	if err := VulkanVisionAttentionF32(out, q, k, v, tokens, heads, headDim); err != nil {
		t.Fatal(err)
	}
	visionAttentionReference(want, q, k, v, tokens, heads, headDim)
	for i := range want {
		assertCloseSliceTol(t, "visionAttentionF32Repeat", out[i], want[i], 1e-4)
	}
}

func TestVulkanVisionAttentionOutF32(t *testing.T) {
	if os.Getenv("RAPIDOCRVL_VULKAN_SMOKE") != "1" {
		t.Skip("set RAPIDOCRVL_VULKAN_SMOKE=1 to run the Vulkan GPU vision attention+out test")
	}
	tokens, heads, headDim := 4, 2, 4
	hidden := heads * headDim
	q := makeRowsForBackendTest(tokens, hidden)
	k := makeRowsForBackendTest(tokens, hidden)
	v := makeRowsForBackendTest(tokens, hidden)
	for i := 0; i < tokens; i++ {
		for j := 0; j < hidden; j++ {
			q[i][j] = float32((i+1)*(j+2)%7-3) * 0.25
			k[i][j] = float32((i+3)*(j+1)%11-5) * 0.2
			v[i][j] = float32((i+2)*(j+4)%13-6) * 0.15
		}
	}
	w := make([]float32, hidden*hidden)
	bias := make([]float32, hidden)
	for i := range w {
		w[i] = float32((i*5)%17-8) * 0.07
	}
	for i := range bias {
		bias[i] = float32(i%5-2) * 0.03
	}
	out := makeRowsForBackendTest(tokens, hidden)
	if err := VulkanVisionAttentionOutF32(out, q, k, v, w, bias, tokens, heads, headDim); err != nil {
		t.Fatal(err)
	}
	head := makeRowsForBackendTest(tokens, hidden)
	want := makeRowsForBackendTest(tokens, hidden)
	visionAttentionReference(head, q, k, v, tokens, heads, headDim)
	tensor.MatRowsBias(want, head, w, bias, hidden, hidden)
	for i := range want {
		assertCloseSliceTol(t, "visionAttentionOutF32", out[i], want[i], 1e-4)
	}

	for i := 0; i < tokens; i++ {
		for j := 0; j < hidden; j++ {
			q[i][j] = float32((i+2)*(j+5)%9-4) * 0.18
			k[i][j] = float32((i+4)*(j+3)%10-5) * 0.16
			v[i][j] = float32((i+5)*(j+2)%12-6) * 0.13
		}
	}
	wRepeat := make([]float32, hidden*hidden)
	biasRepeat := make([]float32, hidden)
	for i := range wRepeat {
		wRepeat[i] = float32((i*7)%19-9) * 0.05
	}
	for i := range biasRepeat {
		biasRepeat[i] = float32(i%7-3) * 0.025
	}
	if err := VulkanVisionAttentionOutF32(out, q, k, v, wRepeat, biasRepeat, tokens, heads, headDim); err != nil {
		t.Fatal(err)
	}
	visionAttentionReference(head, q, k, v, tokens, heads, headDim)
	tensor.MatRowsBias(want, head, wRepeat, biasRepeat, hidden, hidden)
	for i := range want {
		assertCloseSliceTol(t, "visionAttentionOutF32Repeat", out[i], want[i], 1e-4)
	}
}

func TestVulkanVisionRoPEPairF32(t *testing.T) {
	if os.Getenv("RAPIDOCRVL_VULKAN_SMOKE") != "1" {
		t.Skip("set RAPIDOCRVL_VULKAN_SMOKE=1 to run the Vulkan GPU vision rope pair test")
	}
	gridH, gridW := 2, 3
	tokens, heads, headDim := gridH*gridW, 2, 8
	hidden := heads * headDim
	q := makeRowsForBackendTest(tokens, hidden)
	k := makeRowsForBackendTest(tokens, hidden)
	for i := 0; i < tokens; i++ {
		for j := 0; j < hidden; j++ {
			q[i][j] = float32((i+1)*(j+3)%17-8) * 0.11
			k[i][j] = float32((i+2)*(j+5)%19-9) * 0.09
		}
	}
	wantQ := cloneRowsForBackendTest(q)
	wantK := cloneRowsForBackendTest(k)
	quarter := headDim / 4
	cosH, sinH := makeVisionRoPETablesForBackendTest(gridH, quarter, 0.17)
	cosW, sinW := makeVisionRoPETablesForBackendTest(gridW, quarter, 0.23)
	visionRoPEPairReference(wantQ, wantK, cosH, sinH, cosW, sinW, gridH, gridW, heads, headDim)
	if err := VulkanVisionRoPEPairF32(q, k, cosH, sinH, cosW, sinW, gridH, gridW, heads, headDim); err != nil {
		t.Fatal(err)
	}
	for i := range wantQ {
		assertCloseSliceTol(t, "visionRoPEPairQ", q[i], wantQ[i], 1e-4)
		assertCloseSliceTol(t, "visionRoPEPairK", k[i], wantK[i], 1e-4)
	}
}

func TestVulkanVisionRoPEAttentionOutF32(t *testing.T) {
	if os.Getenv("RAPIDOCRVL_VULKAN_SMOKE") != "1" {
		t.Skip("set RAPIDOCRVL_VULKAN_SMOKE=1 to run the Vulkan GPU vision rope+attention+out test")
	}
	gridH, gridW := 2, 3
	tokens, heads, headDim := gridH*gridW, 2, 8
	hidden := heads * headDim
	q := makeRowsForBackendTest(tokens, hidden)
	k := makeRowsForBackendTest(tokens, hidden)
	v := makeRowsForBackendTest(tokens, hidden)
	for i := 0; i < tokens; i++ {
		for j := 0; j < hidden; j++ {
			q[i][j] = float32((i+1)*(j+3)%17-8) * 0.11
			k[i][j] = float32((i+2)*(j+5)%19-9) * 0.09
			v[i][j] = float32((i+4)*(j+7)%23-11) * 0.07
		}
	}
	w := make([]float32, hidden*hidden)
	bias := make([]float32, hidden)
	for i := range w {
		w[i] = float32((i*7)%29-14) * 0.025
	}
	for i := range bias {
		bias[i] = float32(i%5-2) * 0.015
	}
	quarter := headDim / 4
	cosH, sinH := makeVisionRoPETablesForBackendTest(gridH, quarter, 0.17)
	cosW, sinW := makeVisionRoPETablesForBackendTest(gridW, quarter, 0.23)
	wantQ := cloneRowsForBackendTest(q)
	wantK := cloneRowsForBackendTest(k)
	visionRoPEPairReference(wantQ, wantK, cosH, sinH, cosW, sinW, gridH, gridW, heads, headDim)
	head := makeRowsForBackendTest(tokens, hidden)
	want := makeRowsForBackendTest(tokens, hidden)
	visionAttentionReference(head, wantQ, wantK, v, tokens, heads, headDim)
	tensor.MatRowsBias(want, head, w, bias, hidden, hidden)
	out := makeRowsForBackendTest(tokens, hidden)
	if err := VulkanVisionRoPEAttentionOutF32(out, q, k, v, w, bias, cosH, sinH, cosW, sinW, gridH, gridW, heads, headDim); err != nil {
		t.Fatal(err)
	}
	for i := range want {
		assertCloseSliceTol(t, "visionRoPEAttentionOut", out[i], want[i], 1e-4)
	}
}

func TestVulkanVisionQKVRoPEAttentionOutF32(t *testing.T) {
	if os.Getenv("RAPIDOCRVL_VULKAN_SMOKE") != "1" {
		t.Skip("set RAPIDOCRVL_VULKAN_SMOKE=1 to run the Vulkan GPU vision qkv+rope+attention+out test")
	}
	gridH, gridW := 2, 3
	tokens, heads, headDim := gridH*gridW, 2, 8
	hidden := heads * headDim
	x := makeRowsForBackendTest(tokens, hidden)
	for i := 0; i < tokens; i++ {
		for j := 0; j < hidden; j++ {
			x[i][j] = float32((i+2)*(j+3)%31-15) * 0.04
		}
	}
	qw, qb := makeVisionLinearForBackendTest(hidden, 0.017, 0.011)
	kw, kb := makeVisionLinearForBackendTest(hidden, 0.019, 0.013)
	vw, vb := makeVisionLinearForBackendTest(hidden, 0.023, 0.007)
	ow, ob := makeVisionLinearForBackendTest(hidden, 0.015, 0.009)
	quarter := headDim / 4
	cosH, sinH := makeVisionRoPETablesForBackendTest(gridH, quarter, 0.17)
	cosW, sinW := makeVisionRoPETablesForBackendTest(gridW, quarter, 0.23)
	q := makeRowsForBackendTest(tokens, hidden)
	k := makeRowsForBackendTest(tokens, hidden)
	v := makeRowsForBackendTest(tokens, hidden)
	tensor.MatRowsBias3(q, k, v, x, qw, qb, kw, kb, vw, vb, hidden, hidden, hidden, hidden)
	visionRoPEPairReference(q, k, cosH, sinH, cosW, sinW, gridH, gridW, heads, headDim)
	head := makeRowsForBackendTest(tokens, hidden)
	want := makeRowsForBackendTest(tokens, hidden)
	visionAttentionReference(head, q, k, v, tokens, heads, headDim)
	tensor.MatRowsBias(want, head, ow, ob, hidden, hidden)
	out := makeRowsForBackendTest(tokens, hidden)
	if err := VulkanVisionQKVRoPEAttentionOutF32(out, x, qw, qb, kw, kb, vw, vb, ow, ob, cosH, sinH, cosW, sinW, gridH, gridW, heads, headDim, hidden); err != nil {
		t.Fatal(err)
	}
	for i := range want {
		assertCloseSliceTol(t, "visionQKVRoPEAttentionOut", out[i], want[i], 1e-4)
	}
}

func TestVulkanMatRowsGELU2F32(t *testing.T) {
	if os.Getenv("RAPIDOCRVL_VULKAN_SMOKE") != "1" {
		t.Skip("set RAPIDOCRVL_VULKAN_SMOKE=1 to run the Vulkan GPU matrows gelu2 test")
	}
	xs := [][]float32{
		{2, -1, 0.5, 4, -3},
		{-1, 3, 2, 0.25, 1},
		{0, 1, -2, 3, 0.5},
	}
	w1 := []float32{
		1, 0, 0, 0, 0,
		0, 1, 1, 0, 0.5,
		0, 0, 1, 1, 0,
		-1, 2, 0, 0, 0,
	}
	b1 := []float32{0.25, -0.5, 1, -1}
	w2 := []float32{
		1, 0, 0.5, -0.25,
		0, 1, -1, 0.5,
		0.25, 0.25, 0.25, 0.25,
	}
	b2 := []float32{0.1, -0.2, 0.3}
	out := makeRowsForBackendTest(3, 3)
	if err := VulkanMatRowsGELU2F32(out, xs, w1, b1, w2, b2, 4, 5, 3); err != nil {
		t.Fatal(err)
	}
	hidden := makeRowsForBackendTest(3, 4)
	want := makeRowsForBackendTest(3, 3)
	tensor.MatRowsBias(hidden, xs, w1, b1, 4, 5)
	tensor.GELUTanhRowsInPlace(hidden)
	tensor.MatRowsBias(want, hidden, w2, b2, 3, 4)
	for i := range want {
		assertCloseSliceTol(t, "matrowsGELU2F32", out[i], want[i], 1e-4)
	}

	xsRepeat := [][]float32{
		{1, 2, 3, 4, 5},
		{-1, -2, 0.5, 1, 0},
		{0.25, -0.5, 1.5, -1, 2},
	}
	w1Repeat := []float32{
		1, 1, 0, 0, 0.5,
		0, -1, 1, 0, -0.5,
		1, 0, 1, 0, 0.25,
		0, 1, 0, 1, -0.25,
	}
	b1Repeat := []float32{0.1, -0.2, 0.3, -0.4}
	w2Repeat := []float32{
		1, 0, -1, 0.5,
		0, 1, 0.5, -0.5,
		0.25, -0.25, 0.75, 0,
	}
	b2Repeat := []float32{0.05, -0.1, 0.15}
	if err := VulkanMatRowsGELU2F32(out, xsRepeat, w1Repeat, b1Repeat, w2Repeat, b2Repeat, 4, 5, 3); err != nil {
		t.Fatal(err)
	}
	tensor.MatRowsBias(hidden, xsRepeat, w1Repeat, b1Repeat, 4, 5)
	tensor.GELUTanhRowsInPlace(hidden)
	tensor.MatRowsBias(want, hidden, w2Repeat, b2Repeat, 3, 4)
	for i := range want {
		assertCloseSliceTol(t, "matrowsGELU2F32Repeat", out[i], want[i], 1e-4)
	}
}

func TestVulkanProjectImageF32(t *testing.T) {
	if os.Getenv("RAPIDOCRVL_VULKAN_SMOKE") != "1" {
		t.Skip("set RAPIDOCRVL_VULKAN_SMOKE=1 to run the Vulkan GPU project image test")
	}
	const (
		gridT      = 2
		gridH      = 2
		gridW      = 4
		visionDim  = 5
		hiddenRows = 6
		outRows    = 3
	)
	tokens := gridT * gridH * gridW
	batches := gridT * (gridH / 2) * (gridW / 2)
	xs := makeRowsForBackendTest(tokens, visionDim)
	for i := range xs {
		for j := range xs[i] {
			xs[i][j] = float32((i*11+j*7)%23-11) / 9
		}
	}
	normW := []float32{1, 0.5, -0.25, 1.5, -1}
	normB := []float32{0.1, -0.2, 0.3, -0.4, 0.5}
	w1 := make([]float32, hiddenRows*visionDim*4)
	for i := range w1 {
		w1[i] = float32((i*5)%17-8) / 13
	}
	b1 := []float32{0.1, -0.2, 0.3, -0.4, 0.05, -0.15}
	w2 := make([]float32, outRows*hiddenRows)
	for i := range w2 {
		w2[i] = float32((i*3)%11-5) / 7
	}
	b2 := []float32{0.05, -0.1, 0.15}
	out := makeRowsForBackendTest(batches, outRows)
	if err := VulkanProjectImageF32(out, xs, normW, normB, w1, b1, w2, b2, gridT, gridH, gridW, visionDim, hiddenRows, outRows, 1e-5); err != nil {
		t.Fatal(err)
	}
	merged := makeRowsForBackendTest(batches, visionDim*4)
	for batch := 0; batch < batches; batch++ {
		blocksW := gridW / 2
		blocksPerT := (gridH / 2) * blocksW
		frame := batch / blocksPerT
		local := batch - frame*blocksPerT
		by := local / blocksW
		bx := local - by*blocksW
		base := frame*gridH*gridW + by*2*gridW + bx*2
		tensor.LayerNorm(merged[batch][:visionDim], xs[base], normW, normB, 1e-5)
		tensor.LayerNorm(merged[batch][visionDim:2*visionDim], xs[base+1], normW, normB, 1e-5)
		tensor.LayerNorm(merged[batch][2*visionDim:3*visionDim], xs[base+gridW], normW, normB, 1e-5)
		tensor.LayerNorm(merged[batch][3*visionDim:4*visionDim], xs[base+gridW+1], normW, normB, 1e-5)
	}
	hidden := makeRowsForBackendTest(batches, hiddenRows)
	want := makeRowsForBackendTest(batches, outRows)
	tensor.MatRowsBias(hidden, merged, w1, b1, hiddenRows, visionDim*4)
	tensor.GELUTanhRowsInPlace(hidden)
	tensor.MatRowsBias(want, hidden, w2, b2, outRows, hiddenRows)
	for i := range want {
		assertCloseSliceTol(t, "projectImageF32", out[i], want[i], 2e-4)
	}
}

func TestVulkanMatRowsGELU2AddLayerNormF32(t *testing.T) {
	if os.Getenv("RAPIDOCRVL_VULKAN_SMOKE") != "1" {
		t.Skip("set RAPIDOCRVL_VULKAN_SMOKE=1 to run the Vulkan GPU matrows gelu2+add layernorm test")
	}
	xs := [][]float32{
		{1, 2, -1, 0.5, 3},
		{-1, 0.25, 2, -0.5, 1},
		{0.5, -1.5, 1, 2, -2},
	}
	residual := [][]float32{
		{0.5, -0.25, 1, -1},
		{-0.5, 0.75, -1, 0.25},
		{1, 0.5, -0.75, -0.25},
	}
	w1 := []float32{
		1, 1, 0, 0, 0.5,
		0, -1, 1, 0, -0.5,
		1, 0, 1, 0, 0.25,
		0, 1, 0, 1, -0.25,
		-0.5, 0.25, 1, -1, 0.75,
		0.5, -0.75, 0.25, 1, -1,
	}
	b1 := []float32{0.1, -0.2, 0.3, -0.4, 0.05, -0.15}
	w2 := []float32{
		1, 0, -1, 0.5, 0.25, -0.25,
		0, 1, 0.5, -0.5, 0.75, 0,
		0.25, -0.25, 0.75, 0, -0.5, 1,
		-0.75, 0.5, 0, 1, 0.25, -0.25,
	}
	b2 := []float32{0.05, -0.1, 0.15, -0.2}
	normW := []float32{1, 0.5, -0.25, 2}
	normB := []float32{0.1, -0.2, 0.3, -0.4}
	out := makeRowsForBackendTest(3, 4)
	if err := VulkanMatRowsGELU2AddLayerNormF32(out, xs, residual, w1, b1, w2, b2, normW, normB, 6, 5, 4, 3e-5); err != nil {
		t.Fatal(err)
	}
	hidden := makeRowsForBackendTest(3, 6)
	mlp := makeRowsForBackendTest(3, 4)
	want := makeRowsForBackendTest(3, 4)
	tensor.MatRowsBias(hidden, xs, w1, b1, 6, 5)
	tensor.GELUTanhRowsInPlace(hidden)
	tensor.MatRowsBias(mlp, hidden, w2, b2, 4, 6)
	tensor.AddThenLayerNormRows(want, residual, mlp, normW, normB, 3e-5)
	for i := range want {
		assertCloseSliceTol(t, "matrowsGELU2AddLayerNormF32", out[i], want[i], 2e-4)
	}
}

func TestVulkanLayerNormRowsF32(t *testing.T) {
	if os.Getenv("RAPIDOCRVL_VULKAN_SMOKE") != "1" {
		t.Skip("set RAPIDOCRVL_VULKAN_SMOKE=1 to run the Vulkan GPU layernorm rows test")
	}
	xs := [][]float32{
		{2, -1, 0.5, 4, -3},
		{-1, 3, 2, 0.25, 1},
		{0, 1, -2, 3, 0.5},
	}
	weight := []float32{1, 0.5, -0.25, 2, -1}
	bias := []float32{0.1, -0.2, 0.3, -0.4, 0.5}
	out := makeRowsForBackendTest(3, 5)
	want := makeRowsForBackendTest(3, 5)
	eps := float32(3e-5)
	if err := VulkanLayerNormRowsF32(out, xs, weight, bias, 3, 5, eps); err != nil {
		t.Fatal(err)
	}
	tensor.LayerNormRows(want, xs, weight, bias, eps)
	for i := range want {
		assertCloseSliceTol(t, "layerNormRowsF32", out[i], want[i], 1e-4)
	}

	add := [][]float32{
		{0.5, -0.25, 1, -1, 0.75},
		{-0.5, 0.25, 0.5, 1, -0.75},
		{1, 0, -1, 0.5, -0.5},
	}
	if err := VulkanAddThenLayerNormRowsF32(out, xs, add, weight, bias, 3, 5, eps); err != nil {
		t.Fatal(err)
	}
	tensor.AddThenLayerNormRows(want, xs, add, weight, bias, eps)
	for i := range want {
		assertCloseSliceTol(t, "addLayerNormRowsF32", out[i], want[i], 1e-4)
	}
}

func TestVulkanTextAttentionF32(t *testing.T) {
	if os.Getenv("RAPIDOCRVL_VULKAN_SMOKE") != "1" {
		t.Skip("set RAPIDOCRVL_VULKAN_SMOKE=1 to run the Vulkan GPU text attention test")
	}
	cacheLen, numHeads, kvHeads, headDim := 7, 4, 2, 4
	qRows := numHeads * headDim
	kvDim := kvHeads * headDim
	q := make([]float32, qRows)
	k := make([]float32, cacheLen*kvDim)
	v := make([]float32, cacheLen*kvDim)
	for i := range q {
		q[i] = float32((i*3)%11-5) * 0.17
	}
	for i := range k {
		k[i] = float32((i*5)%13-6) * 0.11
		v[i] = float32((i*7)%17-8) * 0.09
	}
	out := make([]float32, qRows)
	warm := make([]float32, qRows)
	if err := VulkanTextAttentionF32(warm, q, k, v, 1, cacheLen-1, numHeads, kvHeads, headDim); err != nil {
		t.Fatal(err)
	}
	if err := VulkanTextAttentionF32(out, q, k, v, 1, cacheLen, numHeads, kvHeads, headDim); err != nil {
		t.Fatal(err)
	}
	want := make([]float32, qRows)
	textAttentionReference(want, q, k, v, cacheLen, numHeads, kvHeads, headDim)
	assertCloseSliceTol(t, "textAttentionF32", out, want, 1e-4)

	qRepeat := make([]float32, qRows)
	for i := range qRepeat {
		qRepeat[i] = float32((i*7)%19-9) * 0.13
	}
	outRepeat := make([]float32, qRows)
	if err := VulkanTextAttentionF32(outRepeat, qRepeat, k, v, 1, cacheLen, numHeads, kvHeads, headDim); err != nil {
		t.Fatal(err)
	}
	wantRepeat := make([]float32, qRows)
	textAttentionReference(wantRepeat, qRepeat, k, v, cacheLen, numHeads, kvHeads, headDim)
	assertCloseSliceTol(t, "textAttentionF32Repeat", outRepeat, wantRepeat, 1e-4)

	for i := range k {
		k[i] = float32((i*11)%29-14) * 0.075
		v[i] = float32((i*13)%31-15) * 0.065
	}
	outEpoch := make([]float32, qRows)
	if err := VulkanTextAttentionF32(outEpoch, q, k, v, 2, cacheLen, numHeads, kvHeads, headDim); err != nil {
		t.Fatal(err)
	}
	wantEpoch := make([]float32, qRows)
	textAttentionReference(wantEpoch, q, k, v, cacheLen, numHeads, kvHeads, headDim)
	assertCloseSliceTol(t, "textAttentionF32EpochRefresh", outEpoch, wantEpoch, 1e-4)
}

func TestVulkanTextAttentionOutF32(t *testing.T) {
	if os.Getenv("RAPIDOCRVL_VULKAN_SMOKE") != "1" {
		t.Skip("set RAPIDOCRVL_VULKAN_SMOKE=1 to run the Vulkan GPU text attention+out test")
	}
	cacheLen, numHeads, kvHeads, headDim := 7, 4, 2, 4
	qRows := numHeads * headDim
	kvDim := kvHeads * headDim
	q := make([]float32, qRows)
	k := make([]float32, cacheLen*kvDim)
	v := make([]float32, cacheLen*kvDim)
	for i := range q {
		q[i] = float32((i*3)%11-5) * 0.17
	}
	for i := range k {
		k[i] = float32((i*5)%13-6) * 0.11
		v[i] = float32((i*7)%17-8) * 0.09
	}
	w := make([]float32, qRows*qRows)
	bias := make([]float32, qRows)
	for i := range w {
		w[i] = float32((i*5)%17-8) * 0.07
	}
	for i := range bias {
		bias[i] = float32(i%5-2) * 0.03
	}
	warm := make([]float32, qRows)
	if err := VulkanTextAttentionOutF32(warm, q, k, v, w, bias, 1, cacheLen-1, numHeads, kvHeads, headDim); err != nil {
		t.Fatal(err)
	}
	out := make([]float32, qRows)
	if err := VulkanTextAttentionOutF32(out, q, k, v, w, bias, 1, cacheLen, numHeads, kvHeads, headDim); err != nil {
		t.Fatal(err)
	}
	head := make([]float32, qRows)
	want := make([]float32, qRows)
	textAttentionReference(head, q, k, v, cacheLen, numHeads, kvHeads, headDim)
	tensor.MatVecBias(want, head, w, bias, qRows, qRows)
	assertCloseSliceTol(t, "textAttentionOutF32", out, want, 1e-4)

	qRepeat := make([]float32, qRows)
	for i := range qRepeat {
		qRepeat[i] = float32((i*7)%19-9) * 0.13
	}
	wRepeat := make([]float32, qRows*qRows)
	biasRepeat := make([]float32, qRows)
	for i := range wRepeat {
		wRepeat[i] = float32((i*11)%23-11) * 0.05
	}
	for i := range biasRepeat {
		biasRepeat[i] = float32((i*3)%7-3) * 0.025
	}
	outRepeat := make([]float32, qRows)
	if err := VulkanTextAttentionOutF32(outRepeat, qRepeat, k, v, wRepeat, biasRepeat, 1, cacheLen, numHeads, kvHeads, headDim); err != nil {
		t.Fatal(err)
	}
	headRepeat := make([]float32, qRows)
	wantRepeat := make([]float32, qRows)
	textAttentionReference(headRepeat, qRepeat, k, v, cacheLen, numHeads, kvHeads, headDim)
	tensor.MatVecBias(wantRepeat, headRepeat, wRepeat, biasRepeat, qRows, qRows)
	assertCloseSliceTol(t, "textAttentionOutF32Repeat", outRepeat, wantRepeat, 1e-4)
}

func TestVulkanTextAttentionOutAddRMSNormF32(t *testing.T) {
	if os.Getenv("RAPIDOCRVL_VULKAN_SMOKE") != "1" {
		t.Skip("set RAPIDOCRVL_VULKAN_SMOKE=1 to run the Vulkan GPU text attention+out+add+rmsnorm test")
	}
	cacheLen, numHeads, kvHeads, headDim := 7, 4, 2, 4
	qRows := numHeads * headDim
	kvDim := kvHeads * headDim
	q := make([]float32, qRows)
	k := make([]float32, cacheLen*kvDim)
	v := make([]float32, cacheLen*kvDim)
	for i := range q {
		q[i] = float32((i*3)%11-5) * 0.17
	}
	for i := range k {
		k[i] = float32((i*5)%13-6) * 0.11
		v[i] = float32((i*7)%17-8) * 0.09
	}
	w := make([]float32, qRows*qRows)
	bias := make([]float32, qRows)
	residual := make([]float32, qRows)
	normWeight := make([]float32, qRows)
	for i := range w {
		w[i] = float32((i*5)%17-8) * 0.07
	}
	for i := range bias {
		bias[i] = float32(i%5-2) * 0.03
		residual[i] = float32((i*7)%13-6) * 0.08
		normWeight[i] = 0.5 + float32(i%7)*0.125
	}
	wantResidual := append([]float32(nil), residual...)
	head := make([]float32, qRows)
	attOut := make([]float32, qRows)
	wantNorm := make([]float32, qRows)
	textAttentionReference(head, q, k, v, cacheLen, numHeads, kvHeads, headDim)
	tensor.MatVecBias(attOut, head, w, bias, qRows, qRows)
	tensor.AddRMSNorm(wantNorm, wantResidual, attOut, normWeight, 1e-6)

	gotNorm := make([]float32, qRows)
	if err := VulkanTextAttentionOutAddRMSNormF32(gotNorm, residual, q, k, v, w, bias, normWeight, 3, cacheLen, numHeads, kvHeads, headDim); err != nil {
		t.Fatal(err)
	}
	assertCloseSliceTol(t, "textAttentionOutAddRMSNormF32Norm", gotNorm, wantNorm, 1e-4)
	assertCloseSliceTol(t, "textAttentionOutAddRMSNormF32Residual", residual, wantResidual, 1e-4)
}

func TestVulkanTextFirstTokenValueOutAddRMSNormF32(t *testing.T) {
	if os.Getenv("RAPIDOCRVL_VULKAN_SMOKE") != "1" {
		t.Skip("set RAPIDOCRVL_VULKAN_SMOKE=1 to run the Vulkan GPU first-token value+out+add+rmsnorm test")
	}
	numHeads, kvHeads, headDim := 4, 2, 4
	qRows := numHeads * headDim
	kvDim := kvHeads * headDim
	v := make([]float32, kvDim)
	k := make([]float32, kvDim)
	for i := range v {
		v[i] = float32((i*7)%17-8) * 0.09
		k[i] = float32((i*5)%13-6) * 0.11
	}
	w := make([]float32, qRows*qRows)
	residualBase := make([]float32, qRows)
	normWeight := make([]float32, qRows)
	for i := range w {
		w[i] = float32((i*5)%17-8) * 0.07
	}
	for i := range residualBase {
		residualBase[i] = float32((i*7)%13-6) * 0.08
		normWeight[i] = 0.5 + float32(i%7)*0.125
	}
	head := make([]float32, qRows)
	group := numHeads / kvHeads
	for h := 0; h < numHeads; h++ {
		kvh := h / group
		copy(head[h*headDim:(h+1)*headDim], v[kvh*headDim:(kvh+1)*headDim])
	}
	attOut := make([]float32, qRows)
	tensor.MatVec(attOut, head, w, qRows, qRows)
	wantResidual := append([]float32(nil), residualBase...)
	wantNorm := make([]float32, qRows)
	tensor.AddRMSNorm(wantNorm, wantResidual, attOut, normWeight, 1e-6)

	gotResidual := append([]float32(nil), residualBase...)
	gotNorm := make([]float32, qRows)
	const f32CacheEpoch = 11
	if err := VulkanTextFirstTokenValueOutAddRMSNormF32(gotNorm, gotResidual, k, v, w, make([]float32, qRows), normWeight, f32CacheEpoch, numHeads, kvHeads, headDim); err != nil {
		t.Fatal(err)
	}
	assertCloseSliceTol(t, "textFirstTokenValueOutAddRMSNormF32Norm", gotNorm, wantNorm, 1e-4)
	assertCloseSliceTol(t, "textFirstTokenValueOutAddRMSNormF32Residual", gotResidual, wantResidual, 1e-4)
}

func TestVulkanTextFirstTokenValueOutAddRMSNormQuantized(t *testing.T) {
	if os.Getenv("RAPIDOCRVL_VULKAN_SMOKE") != "1" {
		t.Skip("set RAPIDOCRVL_VULKAN_SMOKE=1 to run the Vulkan GPU quantized first-token value+out+add+rmsnorm test")
	}
	numHeads, kvHeads, headDim := 4, 2, 4
	qRows := numHeads * headDim
	kvDim := kvHeads * headDim
	v := make([]float32, kvDim)
	k := make([]float32, kvDim)
	for i := range v {
		v[i] = float32((i*7)%17-8) * 0.09
		k[i] = float32((i*5)%13-6) * 0.11
	}
	w := make([]float32, qRows*qRows)
	residualBase := make([]float32, qRows)
	normWeight := make([]float32, qRows)
	for i := range w {
		w[i] = float32((i*5)%17-8) * 0.07
	}
	for i := range residualBase {
		residualBase[i] = float32((i*7)%13-6) * 0.08
		normWeight[i] = 0.5 + float32(i%7)*0.125
	}
	head := make([]float32, qRows)
	group := numHeads / kvHeads
	for h := 0; h < numHeads; h++ {
		kvh := h / group
		copy(head[h*headDim:(h+1)*headDim], v[kvh*headDim:(kvh+1)*headDim])
	}
	check := func(name string, epoch uint64, project func([]float32), run func([]float32, []float32) error) {
		t.Helper()
		attOut := make([]float32, qRows)
		project(attOut)
		wantResidual := append([]float32(nil), residualBase...)
		wantNorm := make([]float32, qRows)
		tensor.AddRMSNorm(wantNorm, wantResidual, attOut, normWeight, 1e-6)
		gotResidual := append([]float32(nil), residualBase...)
		gotNorm := make([]float32, qRows)
		if err := run(gotNorm, gotResidual); err != nil {
			t.Fatal(err)
		}
		assertCloseSliceTol(t, name+"Norm", gotNorm, wantNorm, 1e-4)
		assertCloseSliceTol(t, name+"Residual", gotResidual, wantResidual, 1e-4)
		_ = epoch
	}
	q8 := tensor.QuantizeQ8Row(w, qRows, qRows)
	check("textFirstTokenValueOutAddRMSNormQ8", 12, func(out []float32) {
		tensor.MatVecQ8(out, head, q8)
	}, func(normOut, residual []float32) error {
		return VulkanTextFirstTokenValueOutAddRMSNormQ8(normOut, residual, k, v, q8, normWeight, 12, numHeads, kvHeads, headDim)
	})
	q6 := tensor.QuantizeQ6Row(w, qRows, qRows)
	check("textFirstTokenValueOutAddRMSNormQ6", 13, func(out []float32) {
		tensor.MatVecQ6(out, head, q6)
	}, func(normOut, residual []float32) error {
		return VulkanTextFirstTokenValueOutAddRMSNormQ6(normOut, residual, k, v, q6, normWeight, 13, numHeads, kvHeads, headDim)
	})
	q4 := tensor.QuantizeQ4Row(w, qRows, qRows)
	check("textFirstTokenValueOutAddRMSNormQ4", 14, func(out []float32) {
		tensor.MatVecQ4(out, head, q4)
	}, func(normOut, residual []float32) error {
		return VulkanTextFirstTokenValueOutAddRMSNormQ4(normOut, residual, k, v, q4, normWeight, 14, numHeads, kvHeads, headDim)
	})
}

func TestVulkanTextAttentionOutQ8(t *testing.T) {
	if os.Getenv("RAPIDOCRVL_VULKAN_SMOKE") != "1" {
		t.Skip("set RAPIDOCRVL_VULKAN_SMOKE=1 to run the Vulkan GPU q8 text attention+out test")
	}
	cacheLen, numHeads, kvHeads, headDim := 7, 4, 2, 4
	qRows := numHeads * headDim
	kvDim := kvHeads * headDim
	q := make([]float32, qRows)
	k := make([]float32, cacheLen*kvDim)
	v := make([]float32, cacheLen*kvDim)
	for i := range q {
		q[i] = float32((i*3)%11-5) * 0.17
	}
	for i := range k {
		k[i] = float32((i*5)%13-6) * 0.11
		v[i] = float32((i*7)%17-8) * 0.09
	}
	w := make([]float32, qRows*qRows)
	for i := range w {
		w[i] = float32((i*5)%17-8) * 0.07
	}
	qw := tensor.QuantizeQ8Row(w, qRows, qRows)
	warm := make([]float32, qRows)
	if err := VulkanTextAttentionOutQ8(warm, q, k, v, qw, 1, cacheLen-1, numHeads, kvHeads, headDim); err != nil {
		t.Fatal(err)
	}
	out := make([]float32, qRows)
	if err := VulkanTextAttentionOutQ8(out, q, k, v, qw, 1, cacheLen, numHeads, kvHeads, headDim); err != nil {
		t.Fatal(err)
	}
	head := make([]float32, qRows)
	want := make([]float32, qRows)
	textAttentionReference(head, q, k, v, cacheLen, numHeads, kvHeads, headDim)
	tensor.MatVecQ8(want, head, qw)
	assertCloseSliceTol(t, "textAttentionOutQ8", out, want, 1e-4)

	qRepeat := make([]float32, qRows)
	wRepeat := make([]float32, qRows*qRows)
	for i := range qRepeat {
		qRepeat[i] = float32((i*7)%19-9) * 0.13
	}
	for i := range wRepeat {
		wRepeat[i] = float32((i*11)%23-11) * 0.05
	}
	qwRepeat := tensor.QuantizeQ8Row(wRepeat, qRows, qRows)
	outRepeat := make([]float32, qRows)
	if err := VulkanTextAttentionOutQ8(outRepeat, qRepeat, k, v, qwRepeat, 1, cacheLen, numHeads, kvHeads, headDim); err != nil {
		t.Fatal(err)
	}
	headRepeat := make([]float32, qRows)
	wantRepeat := make([]float32, qRows)
	textAttentionReference(headRepeat, qRepeat, k, v, cacheLen, numHeads, kvHeads, headDim)
	tensor.MatVecQ8(wantRepeat, headRepeat, qwRepeat)
	assertCloseSliceTol(t, "textAttentionOutQ8Repeat", outRepeat, wantRepeat, 1e-4)
}

func TestVulkanTextAttentionOutQ6(t *testing.T) {
	if os.Getenv("RAPIDOCRVL_VULKAN_SMOKE") != "1" {
		t.Skip("set RAPIDOCRVL_VULKAN_SMOKE=1 to run the Vulkan GPU q6 text attention+out test")
	}
	cacheLen, numHeads, kvHeads, headDim := 7, 4, 2, 4
	qRows := numHeads * headDim
	kvDim := kvHeads * headDim
	q := make([]float32, qRows)
	k := make([]float32, cacheLen*kvDim)
	v := make([]float32, cacheLen*kvDim)
	for i := range q {
		q[i] = float32((i*3)%11-5) * 0.17
	}
	for i := range k {
		k[i] = float32((i*5)%13-6) * 0.11
		v[i] = float32((i*7)%17-8) * 0.09
	}
	w := make([]float32, qRows*qRows)
	for i := range w {
		w[i] = float32((i*5)%17-8) * 0.07
	}
	qw := tensor.QuantizeQ6Row(w, qRows, qRows)
	warm := make([]float32, qRows)
	if err := VulkanTextAttentionOutQ6(warm, q, k, v, qw, 1, cacheLen-1, numHeads, kvHeads, headDim); err != nil {
		t.Fatal(err)
	}
	out := make([]float32, qRows)
	if err := VulkanTextAttentionOutQ6(out, q, k, v, qw, 1, cacheLen, numHeads, kvHeads, headDim); err != nil {
		t.Fatal(err)
	}
	head := make([]float32, qRows)
	want := make([]float32, qRows)
	textAttentionReference(head, q, k, v, cacheLen, numHeads, kvHeads, headDim)
	tensor.MatVecQ6(want, head, qw)
	assertCloseSliceTol(t, "textAttentionOutQ6", out, want, 1e-4)

	qRepeat := make([]float32, qRows)
	wRepeat := make([]float32, qRows*qRows)
	for i := range qRepeat {
		qRepeat[i] = float32((i*7)%19-9) * 0.13
	}
	for i := range wRepeat {
		wRepeat[i] = float32((i*11)%23-11) * 0.05
	}
	qwRepeat := tensor.QuantizeQ6Row(wRepeat, qRows, qRows)
	outRepeat := make([]float32, qRows)
	if err := VulkanTextAttentionOutQ6(outRepeat, qRepeat, k, v, qwRepeat, 1, cacheLen, numHeads, kvHeads, headDim); err != nil {
		t.Fatal(err)
	}
	headRepeat := make([]float32, qRows)
	wantRepeat := make([]float32, qRows)
	textAttentionReference(headRepeat, qRepeat, k, v, cacheLen, numHeads, kvHeads, headDim)
	tensor.MatVecQ6(wantRepeat, headRepeat, qwRepeat)
	assertCloseSliceTol(t, "textAttentionOutQ6Repeat", outRepeat, wantRepeat, 1e-4)
}

func TestVulkanTextAttentionOutQ4(t *testing.T) {
	if os.Getenv("RAPIDOCRVL_VULKAN_SMOKE") != "1" {
		t.Skip("set RAPIDOCRVL_VULKAN_SMOKE=1 to run the Vulkan GPU q4 text attention+out test")
	}
	cacheLen, numHeads, kvHeads, headDim := 7, 4, 2, 4
	qRows := numHeads * headDim
	kvDim := kvHeads * headDim
	q := make([]float32, qRows)
	k := make([]float32, cacheLen*kvDim)
	v := make([]float32, cacheLen*kvDim)
	for i := range q {
		q[i] = float32((i*3)%11-5) * 0.17
	}
	for i := range k {
		k[i] = float32((i*5)%13-6) * 0.11
		v[i] = float32((i*7)%17-8) * 0.09
	}
	w := make([]float32, qRows*qRows)
	for i := range w {
		w[i] = float32((i*5)%17-8) * 0.07
	}
	qw := tensor.QuantizeQ4Row(w, qRows, qRows)
	warm := make([]float32, qRows)
	if err := VulkanTextAttentionOutQ4(warm, q, k, v, qw, 1, cacheLen-1, numHeads, kvHeads, headDim); err != nil {
		t.Fatal(err)
	}
	out := make([]float32, qRows)
	if err := VulkanTextAttentionOutQ4(out, q, k, v, qw, 1, cacheLen, numHeads, kvHeads, headDim); err != nil {
		t.Fatal(err)
	}
	head := make([]float32, qRows)
	want := make([]float32, qRows)
	textAttentionReference(head, q, k, v, cacheLen, numHeads, kvHeads, headDim)
	tensor.MatVecQ4(want, head, qw)
	assertCloseSliceTol(t, "textAttentionOutQ4", out, want, 1e-4)

	qRepeat := make([]float32, qRows)
	wRepeat := make([]float32, qRows*qRows)
	for i := range qRepeat {
		qRepeat[i] = float32((i*7)%19-9) * 0.13
	}
	for i := range wRepeat {
		wRepeat[i] = float32((i*11)%23-11) * 0.05
	}
	qwRepeat := tensor.QuantizeQ4Row(wRepeat, qRows, qRows)
	outRepeat := make([]float32, qRows)
	if err := VulkanTextAttentionOutQ4(outRepeat, qRepeat, k, v, qwRepeat, 1, cacheLen, numHeads, kvHeads, headDim); err != nil {
		t.Fatal(err)
	}
	headRepeat := make([]float32, qRows)
	wantRepeat := make([]float32, qRows)
	textAttentionReference(headRepeat, qRepeat, k, v, cacheLen, numHeads, kvHeads, headDim)
	tensor.MatVecQ4(wantRepeat, headRepeat, qwRepeat)
	assertCloseSliceTol(t, "textAttentionOutQ4Repeat", outRepeat, wantRepeat, 1e-4)
}

func TestVulkanTextAttentionOutAddRMSNormQuantized(t *testing.T) {
	if os.Getenv("RAPIDOCRVL_VULKAN_SMOKE") != "1" {
		t.Skip("set RAPIDOCRVL_VULKAN_SMOKE=1 to run the Vulkan GPU quantized text attention+out+add+rmsnorm test")
	}
	cacheLen, numHeads, kvHeads, headDim := 7, 4, 2, 4
	qRows := numHeads * headDim
	kvDim := kvHeads * headDim
	q := make([]float32, qRows)
	k := make([]float32, cacheLen*kvDim)
	v := make([]float32, cacheLen*kvDim)
	for i := range q {
		q[i] = float32((i*3)%11-5) * 0.17
	}
	for i := range k {
		k[i] = float32((i*5)%13-6) * 0.11
		v[i] = float32((i*7)%17-8) * 0.09
	}
	w := make([]float32, qRows*qRows)
	residualBase := make([]float32, qRows)
	normWeight := make([]float32, qRows)
	for i := range w {
		w[i] = float32((i*5)%17-8) * 0.07
	}
	for i := range residualBase {
		residualBase[i] = float32((i*7)%13-6) * 0.08
		normWeight[i] = 0.5 + float32(i%7)*0.125
	}
	head := make([]float32, qRows)
	textAttentionReference(head, q, k, v, cacheLen, numHeads, kvHeads, headDim)

	check := func(name string, project func([]float32), run func([]float32, []float32) error) {
		t.Helper()
		attOut := make([]float32, qRows)
		project(attOut)
		wantResidual := append([]float32(nil), residualBase...)
		wantNorm := make([]float32, qRows)
		tensor.AddRMSNorm(wantNorm, wantResidual, attOut, normWeight, 1e-6)
		gotResidual := append([]float32(nil), residualBase...)
		gotNorm := make([]float32, qRows)
		if err := run(gotNorm, gotResidual); err != nil {
			t.Fatal(err)
		}
		assertCloseSliceTol(t, name+"Norm", gotNorm, wantNorm, 1e-4)
		assertCloseSliceTol(t, name+"Residual", gotResidual, wantResidual, 1e-4)
	}

	q8 := tensor.QuantizeQ8Row(w, qRows, qRows)
	check("textAttentionOutAddRMSNormQ8", func(out []float32) {
		tensor.MatVecQ8(out, head, q8)
	}, func(normOut, residual []float32) error {
		return VulkanTextAttentionOutAddRMSNormQ8(normOut, residual, q, k, v, q8, normWeight, 5, cacheLen, numHeads, kvHeads, headDim)
	})
	q6 := tensor.QuantizeQ6Row(w, qRows, qRows)
	check("textAttentionOutAddRMSNormQ6", func(out []float32) {
		tensor.MatVecQ6(out, head, q6)
	}, func(normOut, residual []float32) error {
		return VulkanTextAttentionOutAddRMSNormQ6(normOut, residual, q, k, v, q6, normWeight, 6, cacheLen, numHeads, kvHeads, headDim)
	})
	q4 := tensor.QuantizeQ4Row(w, qRows, qRows)
	check("textAttentionOutAddRMSNormQ4", func(out []float32) {
		tensor.MatVecQ4(out, head, q4)
	}, func(normOut, residual []float32) error {
		return VulkanTextAttentionOutAddRMSNormQ4(normOut, residual, q, k, v, q4, normWeight, 7, cacheLen, numHeads, kvHeads, headDim)
	})
}

func TestVulkanTextAttentionSmallCache(t *testing.T) {
	if os.Getenv("RAPIDOCRVL_VULKAN_SMOKE") != "1" {
		t.Skip("set RAPIDOCRVL_VULKAN_SMOKE=1 to run the Vulkan GPU small-cache text attention test")
	}
	cacheLen, numHeads, kvHeads, headDim := 1, 4, 2, 4
	qRows := numHeads * headDim
	kvDim := kvHeads * headDim
	q := make([]float32, qRows)
	k := make([]float32, cacheLen*kvDim)
	v := make([]float32, cacheLen*kvDim)
	for i := range q {
		q[i] = float32((i*3)%11-5) * 0.17
	}
	for i := range k {
		k[i] = float32((i*5)%13-6) * 0.11
		v[i] = float32((i*7)%17-8) * 0.09
	}
	w := make([]float32, qRows*qRows)
	bias := make([]float32, qRows)
	for i := range w {
		w[i] = float32((i*5)%17-8) * 0.07
	}
	for i := range bias {
		bias[i] = float32(i%5-2) * 0.03
	}

	head := make([]float32, qRows)
	if err := VulkanTextAttentionF32(head, q, k, v, 1, cacheLen, numHeads, kvHeads, headDim); err != nil {
		t.Fatal(err)
	}
	wantHead := make([]float32, qRows)
	textAttentionReference(wantHead, q, k, v, cacheLen, numHeads, kvHeads, headDim)
	assertCloseSliceTol(t, "smallCacheTextAttentionF32", head, wantHead, 1e-4)

	out := make([]float32, qRows)
	want := make([]float32, qRows)
	if err := VulkanTextAttentionOutF32(out, q, k, v, w, bias, 1, cacheLen, numHeads, kvHeads, headDim); err != nil {
		t.Fatal(err)
	}
	tensor.MatVecBias(want, wantHead, w, bias, qRows, qRows)
	assertCloseSliceTol(t, "smallCacheTextAttentionOutF32", out, want, 1e-4)

	q8 := tensor.QuantizeQ8Row(w, qRows, qRows)
	clear(out)
	clear(want)
	if err := VulkanTextAttentionOutQ8(out, q, k, v, q8, 1, cacheLen, numHeads, kvHeads, headDim); err != nil {
		t.Fatal(err)
	}
	tensor.MatVecQ8(want, wantHead, q8)
	assertCloseSliceTol(t, "smallCacheTextAttentionOutQ8", out, want, 1e-4)

	q6 := tensor.QuantizeQ6Row(w, qRows, qRows)
	clear(out)
	clear(want)
	if err := VulkanTextAttentionOutQ6(out, q, k, v, q6, 1, cacheLen, numHeads, kvHeads, headDim); err != nil {
		t.Fatal(err)
	}
	tensor.MatVecQ6(want, wantHead, q6)
	assertCloseSliceTol(t, "smallCacheTextAttentionOutQ6", out, want, 1e-4)

	q4 := tensor.QuantizeQ4Row(w, qRows, qRows)
	clear(out)
	clear(want)
	if err := VulkanTextAttentionOutQ4(out, q, k, v, q4, 1, cacheLen, numHeads, kvHeads, headDim); err != nil {
		t.Fatal(err)
	}
	tensor.MatVecQ4(want, wantHead, q4)
	assertCloseSliceTol(t, "smallCacheTextAttentionOutQ4", out, want, 1e-4)
}

func TestVulkanFusedMatVec3F32(t *testing.T) {
	if os.Getenv("RAPIDOCRVL_VULKAN_SMOKE") != "1" {
		t.Skip("set RAPIDOCRVL_VULKAN_SMOKE=1 to run the Vulkan GPU fused matvec3 test")
	}
	x := []float32{2, -1, 0.5, 4}
	wa := []float32{
		1, 0, 0, 0,
		0, 1, 1, 0,
	}
	wb := []float32{
		0, 0, 1, 1,
		-1, 2, 0, 0,
		0.5, 0.5, 0.5, 0.5,
	}
	wc := []float32{
		2, 0, -1, 1,
	}
	outA := make([]float32, 2)
	outB := make([]float32, 3)
	outC := make([]float32, 1)
	if err := VulkanFusedMatVec3F32(outA, outB, outC, x, wa, wb, wc, 2, 3, 1, 4); err != nil {
		t.Fatal(err)
	}
	assertCloseSlice(t, "outA", outA, []float32{2, -0.5})
	assertCloseSlice(t, "outB", outB, []float32{4.5, -4, 2.75})
	assertCloseSlice(t, "outC", outC, []float32{7.5})

	xRepeat := []float32{1, 2, -1, 0.5}
	waRepeat := []float32{
		1, 1, 0, 0,
		0, -1, 1, 0,
	}
	wbRepeat := []float32{
		1, 0, 1, 0,
		0, 1, 0, 1,
		-1, 0.5, 0, 1,
	}
	wcRepeat := []float32{
		0.5, 0.5, -1, 2,
	}
	if err := VulkanFusedMatVec3F32(outA, outB, outC, xRepeat, waRepeat, wbRepeat, wcRepeat, 2, 3, 1, 4); err != nil {
		t.Fatal(err)
	}
	wantA := make([]float32, 2)
	wantB := make([]float32, 3)
	wantC := make([]float32, 1)
	tensor.FusedMatVec3(wantA, wantB, wantC, xRepeat, waRepeat, wbRepeat, wcRepeat, 2, 3, 1, 4)
	assertCloseSliceTol(t, "outRepeatA", outA, wantA, 1e-4)
	assertCloseSliceTol(t, "outRepeatB", outB, wantB, 1e-4)
	assertCloseSliceTol(t, "outRepeatC", outC, wantC, 1e-4)
}

func TestVulkanFusedMatVec3MRoPEF32(t *testing.T) {
	if os.Getenv("RAPIDOCRVL_VULKAN_SMOKE") != "1" {
		t.Skip("set RAPIDOCRVL_VULKAN_SMOKE=1 to run the Vulkan GPU fused matvec3+mrope test")
	}
	x := []float32{2, -1, 0.5, 4}
	qHeads, kvHeads, headDim := 2, 1, 4
	qRows, kvRows := qHeads*headDim, kvHeads*headDim
	wa := []float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1,
		1, 1, 0, 0,
		0, 1, 1, 0,
		0, 0, 1, 1,
		1, 0, 0, 1,
	}
	wb := []float32{
		0, 0, 1, 1,
		-1, 2, 0, 0,
		0.5, 0.5, 0.5, 0.5,
		1, -1, 1, -1,
	}
	wc := []float32{
		2, 0, -1, 1,
		0, 1, 0, -1,
		1, 1, 1, 1,
		-1, 0, 2, 0,
	}
	cosTable := []float32{0.5, -0.25}
	sinTable := []float32{0.8660254, 0.9682458}
	outA := make([]float32, qRows)
	outB := make([]float32, kvRows)
	outC := make([]float32, kvRows)
	wantA := make([]float32, qRows)
	wantB := make([]float32, kvRows)
	wantC := make([]float32, kvRows)
	tensor.FusedMatVec3(wantA, wantB, wantC, x, wa, wb, wc, qRows, kvRows, kvRows, len(x))
	applyMRoPEReference(wantA, qHeads, headDim, cosTable, sinTable)
	applyMRoPEReference(wantB, kvHeads, headDim, cosTable, sinTable)
	if err := VulkanFusedMatVec3MRoPEF32(outA, outB, outC, x, wa, wb, wc, cosTable, sinTable, qRows, kvRows, kvRows, len(x), qHeads, kvHeads, headDim); err != nil {
		t.Fatal(err)
	}
	assertCloseSliceTol(t, "fusedMRoPEA", outA, wantA, 1e-4)
	assertCloseSliceTol(t, "fusedMRoPEB", outB, wantB, 1e-4)
	assertCloseSliceTol(t, "fusedMRoPEC", outC, wantC, 1e-4)
}

func TestVulkanSwiGLUGateUpF32(t *testing.T) {
	if os.Getenv("RAPIDOCRVL_VULKAN_SMOKE") != "1" {
		t.Skip("set RAPIDOCRVL_VULKAN_SMOKE=1 to run the Vulkan GPU swiglu gate/up test")
	}
	x := []float32{2, -1, 0.5, 4}
	gate := []float32{
		1, 0, 0, 0,
		0, 1, 1, 0,
		-1, 0, 0.5, 0.25,
	}
	up := []float32{
		0, 0, 1, 1,
		-1, 2, 0, 0,
		0.5, 0.5, 0.5, 0.5,
	}
	out := make([]float32, 3)
	if err := VulkanSwiGLUGateUpF32(out, x, gate, up, 3, 4); err != nil {
		t.Fatal(err)
	}
	want := make([]float32, 3)
	for r := range want {
		var g, u float32
		for c := range x {
			g += gate[r*4+c] * x[c]
			u += up[r*4+c] * x[c]
		}
		want[r] = float32(float64(g)/(1+math.Exp(float64(-g)))) * u
	}
	assertCloseSliceTol(t, "swiglu", out, want, 1e-4)

	xRepeat := []float32{1, 2, -1, 0.5}
	gateRepeat := []float32{
		1, 1, 0, 0,
		0, -1, 1, 0,
		0.5, 0, 1, -1,
	}
	upRepeat := []float32{
		0, 1, 0, 1,
		1, 0, -1, 0,
		0.5, -0.5, 0.5, -0.5,
	}
	if err := VulkanSwiGLUGateUpF32(out, xRepeat, gateRepeat, upRepeat, 3, 4); err != nil {
		t.Fatal(err)
	}
	for r := range want {
		var g, u float32
		for c := range xRepeat {
			g += gateRepeat[r*4+c] * xRepeat[c]
			u += upRepeat[r*4+c] * xRepeat[c]
		}
		want[r] = float32(float64(g)/(1+math.Exp(float64(-g)))) * u
	}
	assertCloseSliceTol(t, "swigluRepeat", out, want, 1e-4)
}

func TestVulkanSwiGLUDownF32(t *testing.T) {
	if os.Getenv("RAPIDOCRVL_VULKAN_SMOKE") != "1" {
		t.Skip("set RAPIDOCRVL_VULKAN_SMOKE=1 to run the Vulkan GPU swiglu/down test")
	}
	x := []float32{2, -1, 0.5, 4}
	gate := []float32{
		1, 0, 0, 0,
		0, 1, 1, 0,
		-1, 0, 0.5, 0.25,
	}
	up := []float32{
		0, 0, 1, 1,
		-1, 2, 0, 0,
		0.5, 0.5, 0.5, 0.5,
	}
	down := []float32{
		1, 0, 0,
		0, 1, 0,
		0.5, -1, 2,
		-0.25, 0.5, 1,
	}
	out := make([]float32, 4)
	if err := VulkanSwiGLUDownF32(out, x, gate, up, down, 3, 4, 4); err != nil {
		t.Fatal(err)
	}
	hidden := make([]float32, 3)
	for r := range hidden {
		var g, u float32
		for c := range x {
			g += gate[r*4+c] * x[c]
			u += up[r*4+c] * x[c]
		}
		hidden[r] = float32(float64(g)/(1+math.Exp(float64(-g)))) * u
	}
	want := make([]float32, 4)
	for r := range want {
		for c := range hidden {
			want[r] += down[r*3+c] * hidden[c]
		}
	}
	assertCloseSliceTol(t, "swigluDown", out, want, 1e-4)

	xRepeat := []float32{1, 2, -1, 0.5}
	gateRepeat := []float32{
		1, 1, 0, 0,
		0, -1, 1, 0,
		0.5, 0, 1, -1,
	}
	upRepeat := []float32{
		0, 1, 0, 1,
		1, 0, -1, 0,
		0.5, -0.5, 0.5, -0.5,
	}
	downRepeat := []float32{
		1, 0, -1,
		0, 1, 0.5,
		0.5, -0.5, 1,
		-1, 0, 0.25,
	}
	if err := VulkanSwiGLUDownF32(out, xRepeat, gateRepeat, upRepeat, downRepeat, 3, 4, 4); err != nil {
		t.Fatal(err)
	}
	tensor.FusedSwiGLUF32Scratch(want, xRepeat, gateRepeat, upRepeat, downRepeat, 3, 4, 4, hidden)
	assertCloseSliceTol(t, "swigluDownRepeat", out, want, 1e-4)
}

func TestVulkanSwiGLUDownAddRMSNormF32(t *testing.T) {
	if os.Getenv("RAPIDOCRVL_VULKAN_SMOKE") != "1" {
		t.Skip("set RAPIDOCRVL_VULKAN_SMOKE=1 to run the Vulkan GPU swiglu/down+add+rmsnorm test")
	}
	x := []float32{2, -1, 0.5, 4}
	gate := []float32{
		1, 0, 0, 0,
		0, 1, 1, 0,
		-1, 0, 0.5, 0.25,
	}
	up := []float32{
		0, 0, 1, 1,
		-1, 2, 0, 0,
		0.5, 0.5, 0.5, 0.5,
	}
	down := []float32{
		1, 0, 0,
		0, 1, 0,
		0.5, -1, 2,
		-0.25, 0.5, 1,
	}
	residual := []float32{0.25, -0.5, 1.5, -1}
	normWeight := []float32{1, 0.5, -1, 1.25}
	mlp := make([]float32, 4)
	hidden := make([]float32, 3)
	tensor.FusedSwiGLUF32Scratch(mlp, x, gate, up, down, 3, 4, 4, hidden)
	wantResidual := append([]float32(nil), residual...)
	wantNorm := make([]float32, 4)
	tensor.AddRMSNorm(wantNorm, wantResidual, mlp, normWeight, 1e-6)

	gotNorm := make([]float32, 4)
	if err := VulkanSwiGLUDownAddRMSNormF32(gotNorm, residual, x, gate, up, down, normWeight, 3, 4, 4); err != nil {
		t.Fatal(err)
	}
	assertCloseSliceTol(t, "swigluDownAddRMSNormNorm", gotNorm, wantNorm, 1e-4)
	assertCloseSliceTol(t, "swigluDownAddRMSNormResidual", residual, wantResidual, 1e-4)

	outOnlyResidual := []float32{0.25, -0.5, 1.5, -1}
	outOnlyNorm := make([]float32, 4)
	if err := VulkanSwiGLUDownAddRMSNormF32OutOnly(outOnlyNorm, outOnlyResidual, x, gate, up, down, normWeight, 3, 4, 4); err != nil {
		t.Fatal(err)
	}
	assertCloseSliceTol(t, "swigluDownAddRMSNormOutOnlyNorm", outOnlyNorm, wantNorm, 1e-4)
	assertCloseSliceTol(t, "swigluDownAddRMSNormOutOnlyResidual", outOnlyResidual, []float32{0.25, -0.5, 1.5, -1}, 1e-4)
}

func TestVulkanMatVecQ8(t *testing.T) {
	if os.Getenv("RAPIDOCRVL_VULKAN_SMOKE") != "1" {
		t.Skip("set RAPIDOCRVL_VULKAN_SMOKE=1 to run the Vulkan GPU q8 matvec test")
	}
	x := []float32{2, -1, 0.5, 4, -3}
	w := []float32{
		1, 0, 0, 0, 0,
		0, 1, 2, 0, 0,
		-1, 0, 0, 0.25, 1,
		0.5, 0.5, 0.5, 0.5, 0.5,
	}
	q := tensor.QuantizeQ8Row(w, 4, 5)
	out := make([]float32, 4)
	if err := VulkanMatVecQ8(out, x, q); err != nil {
		t.Fatal(err)
	}
	want := make([]float32, 4)
	tensor.MatVecQ8(want, x, q)
	assertCloseSliceTol(t, "q8", out, want, 1e-4)

	xRepeat := []float32{1, 2, 3, 4, 5}
	qRepeat := tensor.QuantizeQ8Row([]float32{
		1, 1, 1, 1, 1,
		1, 0, 0, 0, -1,
		0, 2, 0, -1, 0,
		-1, -1, -1, -1, -1,
	}, 4, 5)
	if err := VulkanMatVecQ8(out, xRepeat, qRepeat); err != nil {
		t.Fatal(err)
	}
	tensor.MatVecQ8(want, xRepeat, qRepeat)
	assertCloseSliceTol(t, "q8Repeat", out, want, 1e-4)

	xSmall := []float32{3, -2, 5}
	qSmall := tensor.QuantizeQ8Row([]float32{
		1, 2, 0,
		-1, 0, 1,
	}, 2, 3)
	outSmall := make([]float32, 2)
	if err := VulkanMatVecQ8(outSmall, xSmall, qSmall); err != nil {
		t.Fatal(err)
	}
	wantSmall := make([]float32, 2)
	tensor.MatVecQ8(wantSmall, xSmall, qSmall)
	assertCloseSliceTol(t, "q8Small", outSmall, wantSmall, 1e-4)
}

func TestVulkanMatVecArgmaxQuantized(t *testing.T) {
	if os.Getenv("RAPIDOCRVL_VULKAN_SMOKE") != "1" {
		t.Skip("set RAPIDOCRVL_VULKAN_SMOKE=1 to run the Vulkan GPU quantized matvec+argmax test")
	}
	x := []float32{2, -1, 0.5, 4, -3}
	w := []float32{
		1, 0, 0, 0, 0,
		0, 1, 2, 0, 0,
		-1, 0, 0, 0.25, 1,
		0.5, 0.5, 0.5, 0.5, 0.5,
	}

	t.Run("q8", func(t *testing.T) {
		q := tensor.QuantizeQ8Row(w, 4, 5)
		want := make([]float32, 4)
		tensor.MatVecQ8(want, x, q)
		wantToken := tensor.Argmax(want)
		gotToken, gotScore, err := VulkanMatVecArgmaxQ8(x, q)
		if err != nil {
			t.Fatal(err)
		}
		if gotToken != wantToken {
			t.Fatalf("q8 token=%d want=%d logits=%v", gotToken, wantToken, want)
		}
		if math.Abs(float64(gotScore-want[wantToken])) > 1e-4 {
			t.Fatalf("q8 score=%g want=%g", gotScore, want[wantToken])
		}
	})

	t.Run("q4", func(t *testing.T) {
		q := tensor.QuantizeQ4Row(w, 4, 5)
		want := make([]float32, 4)
		tensor.MatVecQ4(want, x, q)
		wantToken := tensor.Argmax(want)
		gotToken, gotScore, err := VulkanMatVecArgmaxQ4(x, q)
		if err != nil {
			t.Fatal(err)
		}
		if gotToken != wantToken {
			t.Fatalf("q4 token=%d want=%d logits=%v", gotToken, wantToken, want)
		}
		if math.Abs(float64(gotScore-want[wantToken])) > 1e-4 {
			t.Fatalf("q4 score=%g want=%g", gotScore, want[wantToken])
		}
	})

	t.Run("q6", func(t *testing.T) {
		q := tensor.QuantizeQ6Row(w, 4, 5)
		want := make([]float32, 4)
		tensor.MatVecQ6(want, x, q)
		wantToken := tensor.Argmax(want)
		gotToken, gotScore, err := VulkanMatVecArgmaxQ6(x, q)
		if err != nil {
			t.Fatal(err)
		}
		if gotToken != wantToken {
			t.Fatalf("q6 token=%d want=%d logits=%v", gotToken, wantToken, want)
		}
		if math.Abs(float64(gotScore-want[wantToken])) > 1e-4 {
			t.Fatalf("q6 score=%g want=%g", gotScore, want[wantToken])
		}
	})
}

func TestVulkanMatVecAddRMSNormQ8(t *testing.T) {
	if os.Getenv("RAPIDOCRVL_VULKAN_SMOKE") != "1" {
		t.Skip("set RAPIDOCRVL_VULKAN_SMOKE=1 to run the Vulkan GPU q8 matvec+add+rmsnorm test")
	}
	x := []float32{2, -1, 0.5, 4, -3}
	q := tensor.QuantizeQ8Row([]float32{
		1, 0, 0, 0, 0,
		0, 1, 2, 0, 0,
		-1, 0, 0, 0.25, 1,
		0.5, 0.5, 0.5, 0.5, 0.5,
	}, 4, 5)
	residual := []float32{0.25, -0.5, 1.5, -1}
	weight := []float32{1, 0.5, -1, 1.25}
	mat := make([]float32, 4)
	tensor.MatVecQ8(mat, x, q)
	wantResidual := append([]float32(nil), residual...)
	want := make([]float32, 4)
	tensor.AddRMSNorm(want, wantResidual, mat, weight, 1e-6)
	got := make([]float32, 4)
	if err := VulkanMatVecAddRMSNormQ8(got, residual, x, q, weight); err != nil {
		t.Fatal(err)
	}
	assertCloseSliceTol(t, "q8MatVecAddRMSNormOut", got, want, 1e-4)
	assertCloseSliceTol(t, "q8MatVecAddRMSNormResidual", residual, wantResidual, 1e-4)
}

func TestVulkanMatVecQ4(t *testing.T) {
	if os.Getenv("RAPIDOCRVL_VULKAN_SMOKE") != "1" {
		t.Skip("set RAPIDOCRVL_VULKAN_SMOKE=1 to run the Vulkan GPU q4 matvec test")
	}
	x := []float32{2, -1, 0.5, 4, -3}
	w := []float32{
		1, 0, 0, 0, 0,
		0, 1, 2, 0, 0,
		-1, 0, 0, 0.25, 1,
		0.5, 0.5, 0.5, 0.5, 0.5,
	}
	q := tensor.QuantizeQ4Row(w, 4, 5)
	out := make([]float32, 4)
	if err := VulkanMatVecQ4(out, x, q); err != nil {
		t.Fatal(err)
	}
	want := make([]float32, 4)
	tensor.MatVecQ4(want, x, q)
	assertCloseSliceTol(t, "q4", out, want, 1e-4)

	xRepeat := []float32{1, 2, 3, 4, 5}
	qRepeat := tensor.QuantizeQ4Row([]float32{
		1, 1, 1, 1, 1,
		1, 0, 0, 0, -1,
		0, 2, 0, -1, 0,
		-1, -1, -1, -1, -1,
	}, 4, 5)
	if err := VulkanMatVecQ4(out, xRepeat, qRepeat); err != nil {
		t.Fatal(err)
	}
	tensor.MatVecQ4(want, xRepeat, qRepeat)
	assertCloseSliceTol(t, "q4Repeat", out, want, 1e-4)

	xSmall := []float32{3, -2, 5}
	qSmall := tensor.QuantizeQ4Row([]float32{
		1, 2, 0,
		-1, 0, 1,
	}, 2, 3)
	outSmall := make([]float32, 2)
	if err := VulkanMatVecQ4(outSmall, xSmall, qSmall); err != nil {
		t.Fatal(err)
	}
	wantSmall := make([]float32, 2)
	tensor.MatVecQ4(wantSmall, xSmall, qSmall)
	assertCloseSliceTol(t, "q4Small", outSmall, wantSmall, 1e-4)
}

func TestVulkanMatVecAddRMSNormQ4(t *testing.T) {
	if os.Getenv("RAPIDOCRVL_VULKAN_SMOKE") != "1" {
		t.Skip("set RAPIDOCRVL_VULKAN_SMOKE=1 to run the Vulkan GPU q4 matvec+add+rmsnorm test")
	}
	x := []float32{2, -1, 0.5, 4, -3}
	q := tensor.QuantizeQ4Row([]float32{
		1, 0, 0, 0, 0,
		0, 1, 2, 0, 0,
		-1, 0, 0, 0.25, 1,
		0.5, 0.5, 0.5, 0.5, 0.5,
	}, 4, 5)
	residual := []float32{0.25, -0.5, 1.5, -1}
	weight := []float32{1, 0.5, -1, 1.25}
	mat := make([]float32, 4)
	tensor.MatVecQ4(mat, x, q)
	wantResidual := append([]float32(nil), residual...)
	want := make([]float32, 4)
	tensor.AddRMSNorm(want, wantResidual, mat, weight, 1e-6)
	got := make([]float32, 4)
	if err := VulkanMatVecAddRMSNormQ4(got, residual, x, q, weight); err != nil {
		t.Fatal(err)
	}
	assertCloseSliceTol(t, "q4MatVecAddRMSNormOut", got, want, 1e-4)
	assertCloseSliceTol(t, "q4MatVecAddRMSNormResidual", residual, wantResidual, 1e-4)
}

func TestVulkanMatVecQ6(t *testing.T) {
	if os.Getenv("RAPIDOCRVL_VULKAN_SMOKE") != "1" {
		t.Skip("set RAPIDOCRVL_VULKAN_SMOKE=1 to run the Vulkan GPU q6 matvec test")
	}
	x := []float32{2, -1, 0.5, 4, -3}
	w := []float32{
		1, 0, 0, 0, 0,
		0, 1, 2, 0, 0,
		-1, 0, 0, 0.25, 1,
		0.5, 0.5, 0.5, 0.5, 0.5,
	}
	q := tensor.QuantizeQ6Row(w, 4, 5)
	out := make([]float32, 4)
	if err := VulkanMatVecQ6(out, x, q); err != nil {
		t.Fatal(err)
	}
	want := make([]float32, 4)
	tensor.MatVecQ6(want, x, q)
	assertCloseSliceTol(t, "q6", out, want, 1e-4)

	xRepeat := []float32{1, 2, 3, 4, 5}
	qRepeat := tensor.QuantizeQ6Row([]float32{
		1, 1, 1, 1, 1,
		1, 0, 0, 0, -1,
		0, 2, 0, -1, 0,
		-1, -1, -1, -1, -1,
	}, 4, 5)
	if err := VulkanMatVecQ6(out, xRepeat, qRepeat); err != nil {
		t.Fatal(err)
	}
	tensor.MatVecQ6(want, xRepeat, qRepeat)
	assertCloseSliceTol(t, "q6Repeat", out, want, 1e-4)

	xSmall := []float32{3, -2, 5}
	qSmall := tensor.QuantizeQ6Row([]float32{
		1, 2, 0,
		-1, 0, 1,
	}, 2, 3)
	outSmall := make([]float32, 2)
	if err := VulkanMatVecQ6(outSmall, xSmall, qSmall); err != nil {
		t.Fatal(err)
	}
	wantSmall := make([]float32, 2)
	tensor.MatVecQ6(wantSmall, xSmall, qSmall)
	assertCloseSliceTol(t, "q6Small", outSmall, wantSmall, 1e-4)
}

func TestVulkanMatVecAddRMSNormQ6(t *testing.T) {
	if os.Getenv("RAPIDOCRVL_VULKAN_SMOKE") != "1" {
		t.Skip("set RAPIDOCRVL_VULKAN_SMOKE=1 to run the Vulkan GPU q6 matvec+add+rmsnorm test")
	}
	x := []float32{2, -1, 0.5, 4, -3}
	q := tensor.QuantizeQ6Row([]float32{
		1, 0, 0, 0, 0,
		0, 1, 2, 0, 0,
		-1, 0, 0, 0.25, 1,
		0.5, 0.5, 0.5, 0.5, 0.5,
	}, 4, 5)
	residual := []float32{0.25, -0.5, 1.5, -1}
	weight := []float32{1, 0.5, -1, 1.25}
	mat := make([]float32, 4)
	tensor.MatVecQ6(mat, x, q)
	wantResidual := append([]float32(nil), residual...)
	want := make([]float32, 4)
	tensor.AddRMSNorm(want, wantResidual, mat, weight, 1e-6)
	got := make([]float32, 4)
	if err := VulkanMatVecAddRMSNormQ6(got, residual, x, q, weight); err != nil {
		t.Fatal(err)
	}
	assertCloseSliceTol(t, "q6MatVecAddRMSNormOut", got, want, 1e-4)
	assertCloseSliceTol(t, "q6MatVecAddRMSNormResidual", residual, wantResidual, 1e-4)
}

func TestVulkanFusedMatVec3Q8(t *testing.T) {
	if os.Getenv("RAPIDOCRVL_VULKAN_SMOKE") != "1" {
		t.Skip("set RAPIDOCRVL_VULKAN_SMOKE=1 to run the Vulkan GPU q8 fused matvec3 test")
	}
	x := []float32{2, -1, 0.5, 4}
	wa := tensor.QuantizeQ8Row([]float32{
		1, 0, 0, 0,
		0, 1, 1, 0,
	}, 2, 4)
	wb := tensor.QuantizeQ8Row([]float32{
		0, 0, 1, 1,
		-1, 2, 0, 0,
		0.5, 0.5, 0.5, 0.5,
	}, 3, 4)
	wc := tensor.QuantizeQ8Row([]float32{
		2, 0, -1, 1,
	}, 1, 4)
	outA := make([]float32, 2)
	outB := make([]float32, 3)
	outC := make([]float32, 1)
	if err := VulkanFusedMatVec3Q8(outA, outB, outC, x, wa, wb, wc); err != nil {
		t.Fatal(err)
	}
	wantA := make([]float32, 2)
	wantB := make([]float32, 3)
	wantC := make([]float32, 1)
	tensor.FusedMatVec3Q8(wantA, wantB, wantC, x, wa, wb, wc)
	assertCloseSliceTol(t, "q8A", outA, wantA, 1e-4)
	assertCloseSliceTol(t, "q8B", outB, wantB, 1e-4)
	assertCloseSliceTol(t, "q8C", outC, wantC, 1e-4)

	xRepeat := []float32{1, 2, -1, 0.5}
	waRepeat := tensor.QuantizeQ8Row([]float32{
		1, 1, 0, 0,
		0, -1, 1, 0,
	}, 2, 4)
	wbRepeat := tensor.QuantizeQ8Row([]float32{
		1, 0, 1, 0,
		0, 1, 0, 1,
		-1, 0.5, 0, 1,
	}, 3, 4)
	wcRepeat := tensor.QuantizeQ8Row([]float32{
		0.5, 0.5, -1, 2,
	}, 1, 4)
	if err := VulkanFusedMatVec3Q8(outA, outB, outC, xRepeat, waRepeat, wbRepeat, wcRepeat); err != nil {
		t.Fatal(err)
	}
	tensor.FusedMatVec3Q8(wantA, wantB, wantC, xRepeat, waRepeat, wbRepeat, wcRepeat)
	assertCloseSliceTol(t, "q8RepeatA", outA, wantA, 1e-4)
	assertCloseSliceTol(t, "q8RepeatB", outB, wantB, 1e-4)
	assertCloseSliceTol(t, "q8RepeatC", outC, wantC, 1e-4)
}

func TestVulkanFusedMatVec3MRoPEQ8(t *testing.T) {
	if os.Getenv("RAPIDOCRVL_VULKAN_SMOKE") != "1" {
		t.Skip("set RAPIDOCRVL_VULKAN_SMOKE=1 to run the Vulkan GPU q8 fused matvec3+mrope test")
	}
	x := []float32{2, -1, 0.5, 4}
	qHeads, kvHeads, headDim := 2, 1, 4
	qRows, kvRows := qHeads*headDim, kvHeads*headDim
	wa := tensor.QuantizeQ8Row([]float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1,
		1, 1, 0, 0,
		0, 1, 1, 0,
		0, 0, 1, 1,
		1, 0, 0, 1,
	}, qRows, len(x))
	wb := tensor.QuantizeQ8Row([]float32{
		0, 0, 1, 1,
		-1, 2, 0, 0,
		0.5, 0.5, 0.5, 0.5,
		1, -1, 1, -1,
	}, kvRows, len(x))
	wc := tensor.QuantizeQ8Row([]float32{
		2, 0, -1, 1,
		0, 1, 0, -1,
		1, 1, 1, 1,
		-1, 0, 2, 0,
	}, kvRows, len(x))
	cosTable := []float32{0.5, -0.25}
	sinTable := []float32{0.8660254, 0.9682458}
	outA := make([]float32, qRows)
	outB := make([]float32, kvRows)
	outC := make([]float32, kvRows)
	wantA := make([]float32, qRows)
	wantB := make([]float32, kvRows)
	wantC := make([]float32, kvRows)
	tensor.FusedMatVec3Q8(wantA, wantB, wantC, x, wa, wb, wc)
	applyMRoPEReference(wantA, qHeads, headDim, cosTable, sinTable)
	applyMRoPEReference(wantB, kvHeads, headDim, cosTable, sinTable)
	if err := VulkanFusedMatVec3MRoPEQ8(outA, outB, outC, x, wa, wb, wc, cosTable, sinTable, qHeads, kvHeads, headDim); err != nil {
		t.Fatal(err)
	}
	assertCloseSliceTol(t, "q8FusedMRoPEA", outA, wantA, 1e-4)
	assertCloseSliceTol(t, "q8FusedMRoPEB", outB, wantB, 1e-4)
	assertCloseSliceTol(t, "q8FusedMRoPEC", outC, wantC, 1e-4)
}

func TestVulkanFusedMatVec3MRoPEQ6(t *testing.T) {
	if os.Getenv("RAPIDOCRVL_VULKAN_SMOKE") != "1" {
		t.Skip("set RAPIDOCRVL_VULKAN_SMOKE=1 to run the Vulkan GPU q6 fused matvec3+mrope test")
	}
	x := []float32{2, -1, 0.5, 4, -3}
	qHeads, kvHeads, headDim := 2, 1, 4
	qRows, kvRows := qHeads*headDim, kvHeads*headDim
	wa := tensor.QuantizeQ6Row([]float32{
		1, 0, 0, 0, 0,
		0, 1, 0, 0, 0.5,
		0, 0, 1, 0, -0.5,
		0, 0, 0, 1, 0,
		1, 1, 0, 0, 0,
		0, 1, 1, 0, 1,
		0, 0, 1, 1, -1,
		1, 0, 0, 1, 0.5,
	}, qRows, len(x))
	wb := tensor.QuantizeQ6Row([]float32{
		0, 0, 1, 1, 0,
		-1, 2, 0, 0, 0.5,
		0.5, 0.5, 0.5, 0.5, 0.5,
		1, -1, 1, -1, 0,
	}, kvRows, len(x))
	wc := tensor.QuantizeQ6Row([]float32{
		2, 0, -1, 1, -0.5,
		0, 1, 0, -1, 0.25,
		1, 1, 1, 1, 0,
		-1, 0, 2, 0, 0.75,
	}, kvRows, len(x))
	cosTable := []float32{0.5, -0.25}
	sinTable := []float32{0.8660254, 0.9682458}
	outA := make([]float32, qRows)
	outB := make([]float32, kvRows)
	outC := make([]float32, kvRows)
	wantA := make([]float32, qRows)
	wantB := make([]float32, kvRows)
	wantC := make([]float32, kvRows)
	tensor.FusedMatVec3Q6(wantA, wantB, wantC, x, wa, wb, wc)
	applyMRoPEReference(wantA, qHeads, headDim, cosTable, sinTable)
	applyMRoPEReference(wantB, kvHeads, headDim, cosTable, sinTable)
	if err := VulkanFusedMatVec3MRoPEQ6(outA, outB, outC, x, wa, wb, wc, cosTable, sinTable, qHeads, kvHeads, headDim); err != nil {
		t.Fatal(err)
	}
	assertCloseSliceTol(t, "q6FusedMRoPEA", outA, wantA, 1e-4)
	assertCloseSliceTol(t, "q6FusedMRoPEB", outB, wantB, 1e-4)
	assertCloseSliceTol(t, "q6FusedMRoPEC", outC, wantC, 1e-4)
}

func TestVulkanFusedMatVec3MRoPEQ4(t *testing.T) {
	if os.Getenv("RAPIDOCRVL_VULKAN_SMOKE") != "1" {
		t.Skip("set RAPIDOCRVL_VULKAN_SMOKE=1 to run the Vulkan GPU q4 fused matvec3+mrope test")
	}
	x := []float32{2, -1, 0.5, 4, -3}
	qHeads, kvHeads, headDim := 2, 1, 4
	qRows, kvRows := qHeads*headDim, kvHeads*headDim
	wa := tensor.QuantizeQ4Row([]float32{
		1, 0, 0, 0, 0,
		0, 1, 0, 0, 0.5,
		0, 0, 1, 0, -0.5,
		0, 0, 0, 1, 0,
		1, 1, 0, 0, 0,
		0, 1, 1, 0, 1,
		0, 0, 1, 1, -1,
		1, 0, 0, 1, 0.5,
	}, qRows, len(x))
	wb := tensor.QuantizeQ4Row([]float32{
		0, 0, 1, 1, 0,
		-1, 2, 0, 0, 0.5,
		0.5, 0.5, 0.5, 0.5, 0.5,
		1, -1, 1, -1, 0,
	}, kvRows, len(x))
	wc := tensor.QuantizeQ4Row([]float32{
		2, 0, -1, 1, -0.5,
		0, 1, 0, -1, 0.25,
		1, 1, 1, 1, 0,
		-1, 0, 2, 0, 0.75,
	}, kvRows, len(x))
	cosTable := []float32{0.5, -0.25}
	sinTable := []float32{0.8660254, 0.9682458}
	outA := make([]float32, qRows)
	outB := make([]float32, kvRows)
	outC := make([]float32, kvRows)
	wantA := make([]float32, qRows)
	wantB := make([]float32, kvRows)
	wantC := make([]float32, kvRows)
	tensor.FusedMatVec3Q4(wantA, wantB, wantC, x, wa, wb, wc)
	applyMRoPEReference(wantA, qHeads, headDim, cosTable, sinTable)
	applyMRoPEReference(wantB, kvHeads, headDim, cosTable, sinTable)
	if err := VulkanFusedMatVec3MRoPEQ4(outA, outB, outC, x, wa, wb, wc, cosTable, sinTable, qHeads, kvHeads, headDim); err != nil {
		t.Fatal(err)
	}
	assertCloseSliceTol(t, "q4FusedMRoPEA", outA, wantA, 1e-4)
	assertCloseSliceTol(t, "q4FusedMRoPEB", outB, wantB, 1e-4)
	assertCloseSliceTol(t, "q4FusedMRoPEC", outC, wantC, 1e-4)
}

func TestVulkanFusedMatVec2MRoPEQuantized(t *testing.T) {
	if os.Getenv("RAPIDOCRVL_VULKAN_SMOKE") != "1" {
		t.Skip("set RAPIDOCRVL_VULKAN_SMOKE=1 to run the Vulkan GPU quantized fused matvec2+mrope tests")
	}
	x := []float32{2, -1, 0.5, 4, -3}
	kvHeads, headDim := 1, 4
	kvRows := kvHeads * headDim
	wbData := []float32{
		0, 0, 1, 1, 0,
		-1, 2, 0, 0, 0.5,
		0.5, 0.5, 0.5, 0.5, 0.5,
		1, -1, 1, -1, 0,
	}
	wcData := []float32{
		2, 0, -1, 1, -0.5,
		0, 1, 0, -1, 0.25,
		1, 1, 1, 1, 0,
		-1, 0, 2, 0, 0.75,
	}
	cosTable := []float32{0.5, -0.25}
	sinTable := []float32{0.8660254, 0.9682458}

	t.Run("q8", func(t *testing.T) {
		wa := &tensor.Q8Matrix{Cols: len(x)}
		wb := tensor.QuantizeQ8Row(wbData, kvRows, len(x))
		wc := tensor.QuantizeQ8Row(wcData, kvRows, len(x))
		outB := make([]float32, kvRows)
		outC := make([]float32, kvRows)
		wantB := make([]float32, kvRows)
		wantC := make([]float32, kvRows)
		tensor.FusedMatVec2Q8(wantB, wantC, x, wb, wc)
		applyMRoPEReference(wantB, kvHeads, headDim, cosTable, sinTable)
		if err := VulkanFusedMatVec2MRoPEQ8(outB, outC, x, wa, wb, wc, cosTable, sinTable, kvHeads, headDim); err != nil {
			t.Fatal(err)
		}
		assertCloseSliceTol(t, "q8FusedMatVec2MRoPEB", outB, wantB, 1e-4)
		assertCloseSliceTol(t, "q8FusedMatVec2MRoPEC", outC, wantC, 1e-4)
	})
	t.Run("q6", func(t *testing.T) {
		wa := &tensor.Q6Matrix{Cols: len(x)}
		wb := tensor.QuantizeQ6Row(wbData, kvRows, len(x))
		wc := tensor.QuantizeQ6Row(wcData, kvRows, len(x))
		outB := make([]float32, kvRows)
		outC := make([]float32, kvRows)
		wantB := make([]float32, kvRows)
		wantC := make([]float32, kvRows)
		tensor.FusedMatVec2Q6(wantB, wantC, x, wb, wc)
		applyMRoPEReference(wantB, kvHeads, headDim, cosTable, sinTable)
		if err := VulkanFusedMatVec2MRoPEQ6(outB, outC, x, wa, wb, wc, cosTable, sinTable, kvHeads, headDim); err != nil {
			t.Fatal(err)
		}
		assertCloseSliceTol(t, "q6FusedMatVec2MRoPEB", outB, wantB, 1e-4)
		assertCloseSliceTol(t, "q6FusedMatVec2MRoPEC", outC, wantC, 1e-4)
	})
	t.Run("q4", func(t *testing.T) {
		wa := &tensor.Q4Matrix{Cols: len(x)}
		wb := tensor.QuantizeQ4Row(wbData, kvRows, len(x))
		wc := tensor.QuantizeQ4Row(wcData, kvRows, len(x))
		outB := make([]float32, kvRows)
		outC := make([]float32, kvRows)
		wantB := make([]float32, kvRows)
		wantC := make([]float32, kvRows)
		tensor.FusedMatVec2Q4(wantB, wantC, x, wb, wc)
		applyMRoPEReference(wantB, kvHeads, headDim, cosTable, sinTable)
		if err := VulkanFusedMatVec2MRoPEQ4(outB, outC, x, wa, wb, wc, cosTable, sinTable, kvHeads, headDim); err != nil {
			t.Fatal(err)
		}
		assertCloseSliceTol(t, "q4FusedMatVec2MRoPEB", outB, wantB, 1e-4)
		assertCloseSliceTol(t, "q4FusedMatVec2MRoPEC", outC, wantC, 1e-4)
	})
}

func TestVulkanFusedMatVec2Quantized(t *testing.T) {
	if os.Getenv("RAPIDOCRVL_VULKAN_SMOKE") != "1" {
		t.Skip("set RAPIDOCRVL_VULKAN_SMOKE=1 to run the Vulkan GPU quantized fused matvec2 tests")
	}
	x := []float32{2, -1, 0.5, 4, -3}
	wbData := []float32{
		0, 0, 1, 1, 0,
		-1, 2, 0, 0, 0.5,
		0.5, 0.5, 0.5, 0.5, 0.5,
	}
	wcData := []float32{
		2, 0, -1, 1, -0.5,
		0, 1, 0, -1, 0.25,
	}
	rowsB, rowsC := 3, 2

	t.Run("q8", func(t *testing.T) {
		wa := &tensor.Q8Matrix{Cols: len(x)}
		wb := tensor.QuantizeQ8Row(wbData, rowsB, len(x))
		wc := tensor.QuantizeQ8Row(wcData, rowsC, len(x))
		outB := make([]float32, rowsB)
		outC := make([]float32, rowsC)
		wantB := make([]float32, rowsB)
		wantC := make([]float32, rowsC)
		tensor.FusedMatVec2Q8(wantB, wantC, x, wb, wc)
		if err := VulkanFusedMatVec2Q8(outB, outC, x, wa, wb, wc); err != nil {
			t.Fatal(err)
		}
		assertCloseSliceTol(t, "q8FusedMatVec2B", outB, wantB, 1e-4)
		assertCloseSliceTol(t, "q8FusedMatVec2C", outC, wantC, 1e-4)
	})
	t.Run("q6", func(t *testing.T) {
		wa := &tensor.Q6Matrix{Cols: len(x)}
		wb := tensor.QuantizeQ6Row(wbData, rowsB, len(x))
		wc := tensor.QuantizeQ6Row(wcData, rowsC, len(x))
		outB := make([]float32, rowsB)
		outC := make([]float32, rowsC)
		wantB := make([]float32, rowsB)
		wantC := make([]float32, rowsC)
		tensor.FusedMatVec2Q6(wantB, wantC, x, wb, wc)
		if err := VulkanFusedMatVec2Q6(outB, outC, x, wa, wb, wc); err != nil {
			t.Fatal(err)
		}
		assertCloseSliceTol(t, "q6FusedMatVec2B", outB, wantB, 1e-4)
		assertCloseSliceTol(t, "q6FusedMatVec2C", outC, wantC, 1e-4)
	})
	t.Run("q4", func(t *testing.T) {
		wa := &tensor.Q4Matrix{Cols: len(x)}
		wb := tensor.QuantizeQ4Row(wbData, rowsB, len(x))
		wc := tensor.QuantizeQ4Row(wcData, rowsC, len(x))
		outB := make([]float32, rowsB)
		outC := make([]float32, rowsC)
		wantB := make([]float32, rowsB)
		wantC := make([]float32, rowsC)
		tensor.FusedMatVec2Q4(wantB, wantC, x, wb, wc)
		if err := VulkanFusedMatVec2Q4(outB, outC, x, wa, wb, wc); err != nil {
			t.Fatal(err)
		}
		assertCloseSliceTol(t, "q4FusedMatVec2B", outB, wantB, 1e-4)
		assertCloseSliceTol(t, "q4FusedMatVec2C", outC, wantC, 1e-4)
	})
}

func TestVulkanFusedMatVec3Q4(t *testing.T) {
	if os.Getenv("RAPIDOCRVL_VULKAN_SMOKE") != "1" {
		t.Skip("set RAPIDOCRVL_VULKAN_SMOKE=1 to run the Vulkan GPU q4 fused matvec3 test")
	}
	x := []float32{2, -1, 0.5, 4, -3}
	wa := tensor.QuantizeQ4Row([]float32{
		1, 0, 0, 0, 0,
		0, 1, 1, 0, 0.5,
	}, 2, 5)
	wb := tensor.QuantizeQ4Row([]float32{
		0, 0, 1, 1, 0,
		-1, 2, 0, 0, 0,
		0.5, 0.5, 0.5, 0.5, 0.5,
	}, 3, 5)
	wc := tensor.QuantizeQ4Row([]float32{
		2, 0, -1, 1, -0.5,
	}, 1, 5)
	outA := make([]float32, 2)
	outB := make([]float32, 3)
	outC := make([]float32, 1)
	if err := VulkanFusedMatVec3Q4(outA, outB, outC, x, wa, wb, wc); err != nil {
		t.Fatal(err)
	}
	wantA := make([]float32, 2)
	wantB := make([]float32, 3)
	wantC := make([]float32, 1)
	tensor.FusedMatVec3Q4(wantA, wantB, wantC, x, wa, wb, wc)
	assertCloseSliceTol(t, "q4A", outA, wantA, 1e-4)
	assertCloseSliceTol(t, "q4B", outB, wantB, 1e-4)
	assertCloseSliceTol(t, "q4C", outC, wantC, 1e-4)

	xRepeat := []float32{1, 2, -1, 0.5, -0.25}
	waRepeat := tensor.QuantizeQ4Row([]float32{
		1, 1, 0, 0, 0.5,
		0, -1, 1, 0, -0.5,
	}, 2, 5)
	wbRepeat := tensor.QuantizeQ4Row([]float32{
		1, 0, 1, 0, 0.25,
		0, 1, 0, 1, -0.25,
		-1, 0.5, 0, 1, 0,
	}, 3, 5)
	wcRepeat := tensor.QuantizeQ4Row([]float32{
		0.5, 0.5, -1, 2, -0.5,
	}, 1, 5)
	if err := VulkanFusedMatVec3Q4(outA, outB, outC, xRepeat, waRepeat, wbRepeat, wcRepeat); err != nil {
		t.Fatal(err)
	}
	tensor.FusedMatVec3Q4(wantA, wantB, wantC, xRepeat, waRepeat, wbRepeat, wcRepeat)
	assertCloseSliceTol(t, "q4RepeatA", outA, wantA, 1e-4)
	assertCloseSliceTol(t, "q4RepeatB", outB, wantB, 1e-4)
	assertCloseSliceTol(t, "q4RepeatC", outC, wantC, 1e-4)
}

func TestVulkanFusedMatVec3Q6(t *testing.T) {
	if os.Getenv("RAPIDOCRVL_VULKAN_SMOKE") != "1" {
		t.Skip("set RAPIDOCRVL_VULKAN_SMOKE=1 to run the Vulkan GPU q6 fused matvec3 test")
	}
	x := []float32{2, -1, 0.5, 4, -3}
	wa := tensor.QuantizeQ6Row([]float32{
		1, 0, 0, 0, 0,
		0, 1, 1, 0, 0.5,
	}, 2, 5)
	wb := tensor.QuantizeQ6Row([]float32{
		0, 0, 1, 1, 0,
		-1, 2, 0, 0, 0,
		0.5, 0.5, 0.5, 0.5, 0.5,
	}, 3, 5)
	wc := tensor.QuantizeQ6Row([]float32{
		2, 0, -1, 1, -0.5,
	}, 1, 5)
	outA := make([]float32, 2)
	outB := make([]float32, 3)
	outC := make([]float32, 1)
	if err := VulkanFusedMatVec3Q6(outA, outB, outC, x, wa, wb, wc); err != nil {
		t.Fatal(err)
	}
	wantA := make([]float32, 2)
	wantB := make([]float32, 3)
	wantC := make([]float32, 1)
	tensor.FusedMatVec3Q6(wantA, wantB, wantC, x, wa, wb, wc)
	assertCloseSliceTol(t, "q6A", outA, wantA, 1e-4)
	assertCloseSliceTol(t, "q6B", outB, wantB, 1e-4)
	assertCloseSliceTol(t, "q6C", outC, wantC, 1e-4)

	xRepeat := []float32{1, 2, -1, 0.5, -0.25}
	waRepeat := tensor.QuantizeQ6Row([]float32{
		1, 1, 0, 0, 0.5,
		0, -1, 1, 0, -0.5,
	}, 2, 5)
	wbRepeat := tensor.QuantizeQ6Row([]float32{
		1, 0, 1, 0, 0.25,
		0, 1, 0, 1, -0.25,
		-1, 0.5, 0, 1, 0,
	}, 3, 5)
	wcRepeat := tensor.QuantizeQ6Row([]float32{
		0.5, 0.5, -1, 2, -0.5,
	}, 1, 5)
	if err := VulkanFusedMatVec3Q6(outA, outB, outC, xRepeat, waRepeat, wbRepeat, wcRepeat); err != nil {
		t.Fatal(err)
	}
	tensor.FusedMatVec3Q6(wantA, wantB, wantC, xRepeat, waRepeat, wbRepeat, wcRepeat)
	assertCloseSliceTol(t, "q6RepeatA", outA, wantA, 1e-4)
	assertCloseSliceTol(t, "q6RepeatB", outB, wantB, 1e-4)
	assertCloseSliceTol(t, "q6RepeatC", outC, wantC, 1e-4)
}

func TestVulkanSwiGLUDownQ8(t *testing.T) {
	if os.Getenv("RAPIDOCRVL_VULKAN_SMOKE") != "1" {
		t.Skip("set RAPIDOCRVL_VULKAN_SMOKE=1 to run the Vulkan GPU q8 swiglu/down test")
	}
	x := []float32{2, -1, 0.5, 4}
	gate := tensor.QuantizeQ8Row([]float32{
		1, 0, 0, 0,
		0, 1, 1, 0,
		-1, 0, 0.5, 0.25,
	}, 3, 4)
	up := tensor.QuantizeQ8Row([]float32{
		0, 0, 1, 1,
		-1, 2, 0, 0,
		0.5, 0.5, 0.5, 0.5,
	}, 3, 4)
	down := tensor.QuantizeQ8Row([]float32{
		1, 0, 0,
		0, 1, 0,
		0.5, -1, 2,
		-0.25, 0.5, 1,
	}, 4, 3)
	out := make([]float32, 4)
	if err := VulkanSwiGLUDownQ8(out, x, gate, up, down); err != nil {
		t.Fatal(err)
	}
	want := make([]float32, 4)
	scratch := make([]float32, gate.Rows)
	tensor.FusedSwiGLUQ8Scratch(want, x, gate, up, down, scratch)
	assertCloseSliceTol(t, "q8SwiGLUDown", out, want, 1e-4)

	xRepeat := []float32{1, 2, -1, 0.5}
	gateRepeat := tensor.QuantizeQ8Row([]float32{
		1, 1, 0, 0,
		0, -1, 1, 0,
		0.5, 0, 1, -1,
	}, 3, 4)
	upRepeat := tensor.QuantizeQ8Row([]float32{
		0, 1, 0, 1,
		1, 0, -1, 0,
		0.5, -0.5, 0.5, -0.5,
	}, 3, 4)
	downRepeat := tensor.QuantizeQ8Row([]float32{
		1, 0, -1,
		0, 1, 0.5,
		0.5, -0.5, 1,
		-1, 0, 0.25,
	}, 4, 3)
	if err := VulkanSwiGLUDownQ8(out, xRepeat, gateRepeat, upRepeat, downRepeat); err != nil {
		t.Fatal(err)
	}
	tensor.FusedSwiGLUQ8Scratch(want, xRepeat, gateRepeat, upRepeat, downRepeat, scratch)
	assertCloseSliceTol(t, "q8SwiGLUDownRepeat", out, want, 1e-4)
}

func TestVulkanSwiGLUDownAddRMSNormQ8(t *testing.T) {
	if os.Getenv("RAPIDOCRVL_VULKAN_SMOKE") != "1" {
		t.Skip("set RAPIDOCRVL_VULKAN_SMOKE=1 to run the Vulkan GPU q8 swiglu/down+add+rmsnorm test")
	}
	x := []float32{2, -1, 0.5, 4}
	gate := tensor.QuantizeQ8Row([]float32{
		1, 0, 0, 0,
		0, 1, 1, 0,
		-1, 0, 0.5, 0.25,
	}, 3, 4)
	up := tensor.QuantizeQ8Row([]float32{
		0, 0, 1, 1,
		-1, 2, 0, 0,
		0.5, 0.5, 0.5, 0.5,
	}, 3, 4)
	down := tensor.QuantizeQ8Row([]float32{
		1, 0, 0,
		0, 1, 0,
		0.5, -1, 2,
		-0.25, 0.5, 1,
	}, 4, 3)
	residual := []float32{0.25, -0.5, 0.75, 1.25}
	normWeight := []float32{1, 1.25, 0.75, 1.5}
	gotResidual := append([]float32(nil), residual...)
	gotNorm := make([]float32, 4)
	if err := VulkanSwiGLUDownAddRMSNormQ8(gotNorm, gotResidual, x, gate, up, down, normWeight); err != nil {
		t.Fatal(err)
	}

	mlpOut := make([]float32, 4)
	wantResidual := append([]float32(nil), residual...)
	wantNorm := make([]float32, 4)
	scratch := make([]float32, gate.Rows)
	tensor.FusedSwiGLUQ8Scratch(mlpOut, x, gate, up, down, scratch)
	tensor.AddRMSNorm(wantNorm, wantResidual, mlpOut, normWeight, 1e-6)
	assertCloseSliceTol(t, "q8SwiGLUDownAddRMSNormResidual", gotResidual, wantResidual, 1e-4)
	assertCloseSliceTol(t, "q8SwiGLUDownAddRMSNorm", gotNorm, wantNorm, 1e-4)

	outOnlyResidual := append([]float32(nil), residual...)
	outOnlyNorm := make([]float32, 4)
	if err := VulkanSwiGLUDownAddRMSNormQ8OutOnly(outOnlyNorm, outOnlyResidual, x, gate, up, down, normWeight); err != nil {
		t.Fatal(err)
	}
	assertCloseSliceTol(t, "q8SwiGLUDownAddRMSNormOutOnlyResidual", outOnlyResidual, residual, 1e-4)
	assertCloseSliceTol(t, "q8SwiGLUDownAddRMSNormOutOnly", outOnlyNorm, wantNorm, 1e-4)
}

func TestVulkanSwiGLUDownQ4(t *testing.T) {
	if os.Getenv("RAPIDOCRVL_VULKAN_SMOKE") != "1" {
		t.Skip("set RAPIDOCRVL_VULKAN_SMOKE=1 to run the Vulkan GPU q4 swiglu/down test")
	}
	x := []float32{2, -1, 0.5, 4, -3}
	gate := tensor.QuantizeQ4Row([]float32{
		1, 0, 0, 0, 0,
		0, 1, 1, 0, 0.5,
		-1, 0, 0.5, 0.25, -0.5,
	}, 3, 5)
	up := tensor.QuantizeQ4Row([]float32{
		0, 0, 1, 1, 0,
		-1, 2, 0, 0, 0,
		0.5, 0.5, 0.5, 0.5, 0.5,
	}, 3, 5)
	down := tensor.QuantizeQ4Row([]float32{
		1, 0, 0,
		0, 1, 0,
		0.5, -1, 2,
		-0.25, 0.5, 1,
	}, 4, 3)
	out := make([]float32, 4)
	if err := VulkanSwiGLUDownQ4(out, x, gate, up, down); err != nil {
		t.Fatal(err)
	}
	want := make([]float32, 4)
	scratch := make([]float32, gate.Rows)
	tensor.FusedSwiGLUQ4Scratch(want, x, gate, up, down, scratch)
	assertCloseSliceTol(t, "q4SwiGLUDown", out, want, 1e-4)

	xRepeat := []float32{1, 2, -1, 0.5, -0.25}
	gateRepeat := tensor.QuantizeQ4Row([]float32{
		1, 1, 0, 0, 0.5,
		0, -1, 1, 0, -0.5,
		0.5, 0, 1, -1, 0.25,
	}, 3, 5)
	upRepeat := tensor.QuantizeQ4Row([]float32{
		0, 1, 0, 1, 0,
		1, 0, -1, 0, 0.5,
		0.5, -0.5, 0.5, -0.5, 0.25,
	}, 3, 5)
	downRepeat := tensor.QuantizeQ4Row([]float32{
		1, 0, -1,
		0, 1, 0.5,
		0.5, -0.5, 1,
		-1, 0, 0.25,
	}, 4, 3)
	if err := VulkanSwiGLUDownQ4(out, xRepeat, gateRepeat, upRepeat, downRepeat); err != nil {
		t.Fatal(err)
	}
	tensor.FusedSwiGLUQ4Scratch(want, xRepeat, gateRepeat, upRepeat, downRepeat, scratch)
	assertCloseSliceTol(t, "q4SwiGLUDownRepeat", out, want, 1e-4)
}

func TestVulkanSwiGLUDownAddRMSNormQ4(t *testing.T) {
	if os.Getenv("RAPIDOCRVL_VULKAN_SMOKE") != "1" {
		t.Skip("set RAPIDOCRVL_VULKAN_SMOKE=1 to run the Vulkan GPU q4 swiglu/down+add+rmsnorm test")
	}
	x := []float32{2, -1, 0.5, 4, -3}
	gate := tensor.QuantizeQ4Row([]float32{
		1, 0, 0, 0, 0,
		0, 1, 1, 0, 0.5,
		-1, 0, 0.5, 0.25, -0.5,
	}, 3, 5)
	up := tensor.QuantizeQ4Row([]float32{
		0, 0, 1, 1, 0,
		-1, 2, 0, 0, 0,
		0.5, 0.5, 0.5, 0.5, 0.5,
	}, 3, 5)
	down := tensor.QuantizeQ4Row([]float32{
		1, 0, 0,
		0, 1, 0,
		0.5, -1, 2,
		-0.25, 0.5, 1,
	}, 4, 3)
	residual := []float32{0.25, -0.5, 0.75, 1.25}
	normWeight := []float32{1, 1.25, 0.75, 1.5}
	gotResidual := append([]float32(nil), residual...)
	gotNorm := make([]float32, 4)
	if err := VulkanSwiGLUDownAddRMSNormQ4(gotNorm, gotResidual, x, gate, up, down, normWeight); err != nil {
		t.Fatal(err)
	}

	mlpOut := make([]float32, 4)
	wantResidual := append([]float32(nil), residual...)
	wantNorm := make([]float32, 4)
	scratch := make([]float32, gate.Rows)
	tensor.FusedSwiGLUQ4Scratch(mlpOut, x, gate, up, down, scratch)
	tensor.AddRMSNorm(wantNorm, wantResidual, mlpOut, normWeight, 1e-6)
	assertCloseSliceTol(t, "q4SwiGLUDownAddRMSNormResidual", gotResidual, wantResidual, 1e-4)
	assertCloseSliceTol(t, "q4SwiGLUDownAddRMSNorm", gotNorm, wantNorm, 1e-4)

	outOnlyResidual := append([]float32(nil), residual...)
	outOnlyNorm := make([]float32, 4)
	if err := VulkanSwiGLUDownAddRMSNormQ4OutOnly(outOnlyNorm, outOnlyResidual, x, gate, up, down, normWeight); err != nil {
		t.Fatal(err)
	}
	assertCloseSliceTol(t, "q4SwiGLUDownAddRMSNormOutOnlyResidual", outOnlyResidual, residual, 1e-4)
	assertCloseSliceTol(t, "q4SwiGLUDownAddRMSNormOutOnly", outOnlyNorm, wantNorm, 1e-4)
}

func TestVulkanSwiGLUDownQ6(t *testing.T) {
	if os.Getenv("RAPIDOCRVL_VULKAN_SMOKE") != "1" {
		t.Skip("set RAPIDOCRVL_VULKAN_SMOKE=1 to run the Vulkan GPU q6 swiglu/down test")
	}
	x := []float32{2, -1, 0.5, 4, -3}
	gate := tensor.QuantizeQ6Row([]float32{
		1, 0, 0, 0, 0,
		0, 1, 1, 0, 0.5,
		-1, 0, 0.5, 0.25, -0.5,
	}, 3, 5)
	up := tensor.QuantizeQ6Row([]float32{
		0, 0, 1, 1, 0,
		-1, 2, 0, 0, 0,
		0.5, 0.5, 0.5, 0.5, 0.5,
	}, 3, 5)
	down := tensor.QuantizeQ6Row([]float32{
		1, 0, 0,
		0, 1, 0,
		0.5, -1, 2,
		-0.25, 0.5, 1,
	}, 4, 3)
	out := make([]float32, 4)
	if err := VulkanSwiGLUDownQ6(out, x, gate, up, down); err != nil {
		t.Fatal(err)
	}
	want := make([]float32, 4)
	scratch := make([]float32, gate.Rows)
	tensor.FusedSwiGLUQ6Scratch(want, x, gate, up, down, scratch)
	assertCloseSliceTol(t, "q6SwiGLUDown", out, want, 1e-4)

	xRepeat := []float32{1, 2, -1, 0.5, -0.25}
	gateRepeat := tensor.QuantizeQ6Row([]float32{
		1, 1, 0, 0, 0.5,
		0, -1, 1, 0, -0.5,
		0.5, 0, 1, -1, 0.25,
	}, 3, 5)
	upRepeat := tensor.QuantizeQ6Row([]float32{
		0, 1, 0, 1, 0,
		1, 0, -1, 0, 0.5,
		0.5, -0.5, 0.5, -0.5, 0.25,
	}, 3, 5)
	downRepeat := tensor.QuantizeQ6Row([]float32{
		1, 0, -1,
		0, 1, 0.5,
		0.5, -0.5, 1,
		-1, 0, 0.25,
	}, 4, 3)
	if err := VulkanSwiGLUDownQ6(out, xRepeat, gateRepeat, upRepeat, downRepeat); err != nil {
		t.Fatal(err)
	}
	tensor.FusedSwiGLUQ6Scratch(want, xRepeat, gateRepeat, upRepeat, downRepeat, scratch)
	assertCloseSliceTol(t, "q6SwiGLUDownRepeat", out, want, 1e-4)
}

func TestVulkanSwiGLUDownAddRMSNormQ6(t *testing.T) {
	if os.Getenv("RAPIDOCRVL_VULKAN_SMOKE") != "1" {
		t.Skip("set RAPIDOCRVL_VULKAN_SMOKE=1 to run the Vulkan GPU q6 swiglu/down+add+rmsnorm test")
	}
	x := []float32{2, -1, 0.5, 4, -3}
	gate := tensor.QuantizeQ6Row([]float32{
		1, 0, 0, 0, 0,
		0, 1, 1, 0, 0.5,
		-1, 0, 0.5, 0.25, -0.5,
	}, 3, 5)
	up := tensor.QuantizeQ6Row([]float32{
		0, 0, 1, 1, 0,
		-1, 2, 0, 0, 0,
		0.5, 0.5, 0.5, 0.5, 0.5,
	}, 3, 5)
	down := tensor.QuantizeQ6Row([]float32{
		1, 0, 0,
		0, 1, 0,
		0.5, -1, 2,
		-0.25, 0.5, 1,
	}, 4, 3)
	residual := []float32{0.25, -0.5, 0.75, 1.25}
	normWeight := []float32{1, 1.25, 0.75, 1.5}
	gotResidual := append([]float32(nil), residual...)
	gotNorm := make([]float32, 4)
	if err := VulkanSwiGLUDownAddRMSNormQ6(gotNorm, gotResidual, x, gate, up, down, normWeight); err != nil {
		t.Fatal(err)
	}

	mlpOut := make([]float32, 4)
	wantResidual := append([]float32(nil), residual...)
	wantNorm := make([]float32, 4)
	scratch := make([]float32, gate.Rows)
	tensor.FusedSwiGLUQ6Scratch(mlpOut, x, gate, up, down, scratch)
	tensor.AddRMSNorm(wantNorm, wantResidual, mlpOut, normWeight, 1e-6)
	assertCloseSliceTol(t, "q6SwiGLUDownAddRMSNormResidual", gotResidual, wantResidual, 1e-4)
	assertCloseSliceTol(t, "q6SwiGLUDownAddRMSNorm", gotNorm, wantNorm, 1e-4)

	outOnlyResidual := append([]float32(nil), residual...)
	outOnlyNorm := make([]float32, 4)
	if err := VulkanSwiGLUDownAddRMSNormQ6OutOnly(outOnlyNorm, outOnlyResidual, x, gate, up, down, normWeight); err != nil {
		t.Fatal(err)
	}
	assertCloseSliceTol(t, "q6SwiGLUDownAddRMSNormOutOnlyResidual", outOnlyResidual, residual, 1e-4)
	assertCloseSliceTol(t, "q6SwiGLUDownAddRMSNormOutOnly", outOnlyNorm, wantNorm, 1e-4)
}

func assertCloseSlice(t *testing.T, name string, got, want []float32) {
	t.Helper()
	assertCloseSliceTol(t, name, got, want, 1e-4)
}

func assertCloseSliceTol(t *testing.T, name string, got, want []float32, tol float32) {
	t.Helper()
	for i := range want {
		if d := got[i] - want[i]; d < -tol || d > tol {
			t.Fatalf("%s[%d] = %.6f, want %.6f", name, i, got[i], want[i])
		}
	}
}

func makeRowsForBackendTest(rows, cols int) [][]float32 {
	out := make([][]float32, rows)
	data := make([]float32, rows*cols)
	for i := range out {
		out[i] = data[i*cols : (i+1)*cols]
	}
	return out
}

func cloneRowsForBackendTest(in [][]float32) [][]float32 {
	if len(in) == 0 {
		return nil
	}
	cols := len(in[0])
	out := makeRowsForBackendTest(len(in), cols)
	for i := range in {
		copy(out[i], in[i])
	}
	return out
}

func makeVisionRoPETablesForBackendTest(rows, quarter int, step float64) ([]float32, []float32) {
	cosv := make([]float32, rows*quarter)
	sinv := make([]float32, rows*quarter)
	for r := 0; r < rows; r++ {
		for i := 0; i < quarter; i++ {
			angle := float64(r+1) * float64(i+1) * step
			cosv[r*quarter+i] = float32(math.Cos(angle))
			sinv[r*quarter+i] = float32(math.Sin(angle))
		}
	}
	return cosv, sinv
}

func makeVisionLinearForBackendTest(hidden int, wScale, bScale float32) ([]float32, []float32) {
	w := make([]float32, hidden*hidden)
	b := make([]float32, hidden)
	for i := range w {
		w[i] = float32((i*7)%37-18) * wScale
	}
	for i := range b {
		b[i] = float32(i%11-5) * bScale
	}
	return w, b
}

func visionRoPEPairReference(q, k [][]float32, cosH, sinH, cosW, sinW []float32, gridH, gridW, heads, headDim int) {
	half := headDim / 2
	quarter := half / 2
	period := gridH * gridW
	for token := range q {
		pos := token % period
		hy := pos / gridW
		wx := pos % gridW
		for h := 0; h < heads; h++ {
			base := h * headDim
			for i := 0; i < quarter; i++ {
				rotateVisionRoPEPairReference(q[token], k[token], base+i, base+quarter+i, cosH[hy*quarter+i], sinH[hy*quarter+i])
				rotateVisionRoPEPairReference(q[token], k[token], base+half+i, base+half+quarter+i, cosW[wx*quarter+i], sinW[wx*quarter+i])
			}
		}
	}
}

func rotateVisionRoPEPairReference(q, k []float32, a, b int, cs, sn float32) {
	qa, qb := q[a], q[b]
	ka, kb := k[a], k[b]
	q[a] = qa*cs - qb*sn
	q[b] = qb*cs + qa*sn
	k[a] = ka*cs - kb*sn
	k[b] = kb*cs + ka*sn
}

func visionAttentionReference(out, q, k, v [][]float32, tokens, heads, headDim int) {
	scale := float32(1 / math.Sqrt(float64(headDim)))
	scores := make([]float32, tokens)
	for i := 0; i < tokens; i++ {
		for h := 0; h < heads; h++ {
			offset := h * headDim
			maxScore := float32(math.Inf(-1))
			for j := 0; j < tokens; j++ {
				var score float32
				for d := 0; d < headDim; d++ {
					score += q[i][offset+d] * k[j][offset+d]
				}
				score *= scale
				scores[j] = score
				if score > maxScore {
					maxScore = score
				}
			}
			var denom float32
			for j := 0; j < tokens; j++ {
				w := float32(math.Exp(float64(scores[j] - maxScore)))
				scores[j] = w
				denom += w
			}
			for d := 0; d < headDim; d++ {
				var sum float32
				for j := 0; j < tokens; j++ {
					sum += scores[j] * v[j][offset+d]
				}
				out[i][offset+d] = sum / denom
			}
		}
	}
}

func textAttentionReference(out, q, k, v []float32, cacheLen, numHeads, kvHeads, headDim int) {
	scale := float32(1 / math.Sqrt(float64(headDim)))
	group := numHeads / kvHeads
	kvDim := kvHeads * headDim
	scores := make([]float32, cacheLen)
	for h := 0; h < numHeads; h++ {
		kvh := h / group
		qBase := h * headDim
		kvHeadBase := kvh * headDim
		maxScore := float32(math.Inf(-1))
		for t := 0; t < cacheLen; t++ {
			kBase := t*kvDim + kvHeadBase
			var score float32
			for d := 0; d < headDim; d++ {
				score += q[qBase+d] * k[kBase+d]
			}
			score *= scale
			scores[t] = score
			if score > maxScore {
				maxScore = score
			}
		}
		var denom float32
		for t := 0; t < cacheLen; t++ {
			w := float32(math.Exp(float64(scores[t] - maxScore)))
			scores[t] = w
			denom += w
		}
		for d := 0; d < headDim; d++ {
			var sum float32
			for t := 0; t < cacheLen; t++ {
				sum += scores[t] * v[t*kvDim+kvHeadBase+d]
			}
			out[qBase+d] = sum / denom
		}
	}
}

func applyMRoPEReference(x []float32, heads, dim int, cosTable, sinTable []float32) {
	half := dim / 2
	for h := 0; h < heads; h++ {
		base := h * dim
		for i := 0; i < half; i++ {
			cs, sn := cosTable[i], sinTable[i]
			a, b := x[base+i], x[base+half+i]
			x[base+i] = a*cs - b*sn
			x[base+half+i] = b*cs + a*sn
		}
	}
}
