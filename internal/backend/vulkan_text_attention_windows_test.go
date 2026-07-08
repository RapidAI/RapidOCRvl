//go:build windows

package backend

import (
	"os"
	"testing"

	"paddleocrvl-go/internal/tensor"
)

func TestVulkanTextAttentionWinByteSizeChecks(t *testing.T) {
	if got, ok := checkedFloat32ByteLenWin(7); !ok || got != 28 {
		t.Fatalf("float32 bytes got=%d ok=%v want=28 true", got, ok)
	}
	if got, ok := textAttentionCacheBufferBytesWin(make([]float32, 8), 4); !ok || got != 32 {
		t.Fatalf("cache bytes len preference got=%d ok=%v want=32 true", got, ok)
	}
	if got, ok := textAttentionCacheBufferBytesWin(make([]float32, 2), 4); !ok || got != 16 {
		t.Fatalf("cache bytes min preference got=%d ok=%v want=16 true", got, ok)
	}
	if _, ok := checkedFloat32ByteLenWin(maxInt()/4 + 1); ok {
		t.Fatal("overflowing float32 byte length should be rejected")
	}
	if _, ok := checkedFloat32ByteLenWinSquare(maxInt()/2 + 1); ok {
		t.Fatal("overflowing square float32 byte length should be rejected")
	}
	if _, _, _, err := checkedTextAttentionWinDims(maxInt()/2+1, 1, 3); err == nil {
		t.Fatal("overflowing q rows should be rejected")
	}
	if _, _, _, err := checkedTextAttentionWinDims(1, maxInt()/2+1, 3); err == nil {
		t.Fatal("overflowing kv dim should be rejected")
	}
	if _, err := checkedTextAttentionWinCacheElems(maxInt()/2+1, 3); err == nil {
		t.Fatal("overflowing cache element count should be rejected")
	}
	if _, ok := textAttentionCacheBufferBytesWin(nil, maxInt()); ok {
		t.Fatal("overflowing cache byte length should be rejected")
	}
	if dataLen, dataBytes, err := checkedTextAttentionQ8DataBytesWin(3, 5); err != nil || dataLen != 15 || dataBytes != 16 {
		t.Fatalf("q8 data bytes got len=%d bytes=%d err=%v want len=15 bytes=16 nil", dataLen, dataBytes, err)
	}
	q6Packed := tensor.PackedQ6Cols(8)
	if dataLen, dataBytes, err := checkedTextAttentionQ6DataBytesWin(2, 8); err != nil || dataLen != 2*q6Packed || dataBytes != uint64(alignUpInt(2*q6Packed, 4)) {
		t.Fatalf("q6 data bytes got len=%d bytes=%d err=%v want len=%d bytes=%d nil", dataLen, dataBytes, err, 2*q6Packed, alignUpInt(2*q6Packed, 4))
	}
	if dataLen, dataBytes, err := checkedTextAttentionQ4DataBytesWin(3, 5); err != nil || dataLen != 9 || dataBytes != 12 {
		t.Fatalf("q4 data bytes got len=%d bytes=%d err=%v want len=9 bytes=12 nil", dataLen, dataBytes, err)
	}
	if _, _, err := checkedTextAttentionQ8DataBytesWin(maxInt()/2+1, 3); err == nil {
		t.Fatal("overflowing q8 data length should be rejected")
	}
	if _, _, err := checkedTextAttentionQ6DataBytesWin(1, (maxInt()-7)/6+1); err == nil {
		t.Fatal("overflowing q6 packed cols should be rejected")
	}
	if _, _, err := checkedTextAttentionQ6DataBytesWin(maxInt()/2+1, 8); err == nil {
		t.Fatal("overflowing q6 data length should be rejected")
	}
	if _, _, err := checkedTextAttentionQ4DataBytesWin(1, maxInt()); err == nil {
		t.Fatal("overflowing q4 packed cols should be rejected")
	}
	if _, _, err := checkedTextAttentionQ4DataBytesWin(maxInt()/2+1, 4); err == nil {
		t.Fatal("overflowing q4 data length should be rejected")
	}
}

func TestVulkanPackedDataLenWinChecks(t *testing.T) {
	if got, err := checkedPackedQ4DataLenWin(3, 5, "test q4"); err != nil || got != 9 {
		t.Fatalf("q4 packed data len got=%d err=%v want=9 nil", got, err)
	}
	q6Packed := tensor.PackedQ6Cols(8)
	if got, err := checkedPackedQ6DataLenWin(2, 8, "test q6"); err != nil || got != 2*q6Packed {
		t.Fatalf("q6 packed data len got=%d err=%v want=%d nil", got, err, 2*q6Packed)
	}
	if _, err := checkedPackedQ4DataLenWin(1, maxInt(), "test q4"); err == nil {
		t.Fatal("overflowing q4 packed cols should be rejected")
	}
	if _, err := checkedPackedQ4DataLenWin(maxInt()/2+1, 4, "test q4"); err == nil {
		t.Fatal("overflowing q4 packed data length should be rejected")
	}
	if _, err := checkedPackedQ6DataLenWin(1, (maxInt()-7)/6+1, "test q6"); err == nil {
		t.Fatal("overflowing q6 packed cols should be rejected")
	}
	if _, err := checkedPackedQ6DataLenWin(maxInt()/2+1, 8, "test q6"); err == nil {
		t.Fatal("overflowing q6 packed data length should be rejected")
	}
	if _, err := checkedPackedRowsWin(maxInt()/2+1, 3, "test rows"); err == nil {
		t.Fatal("overflowing packed row product should be rejected")
	}
}

func TestVulkanTextFirstTokenWarmsAttentionCache(t *testing.T) {
	if os.Getenv("RAPIDOCRVL_VULKAN_SMOKE") != "1" {
		t.Skip("set RAPIDOCRVL_VULKAN_SMOKE=1 to run the Vulkan GPU first-token cache warm test")
	}
	runFirstTokenWarmSmoke := func(t *testing.T, epoch uint64, fn func(normOut, residual, k, v, normWeight []float32) error) {
		t.Helper()
		numHeads, kvHeads, headDim := 4, 2, 4
		qRows := numHeads * headDim
		kvDim := kvHeads * headDim
		k := make([]float32, kvDim, 4*kvDim)
		v := make([]float32, kvDim, 4*kvDim)
		for i := range k {
			k[i] = float32((i*5)%13-6) * 0.11
			v[i] = float32((i*7)%17-8) * 0.09
		}
		kCache := k[:cap(k)]
		vCache := v[:cap(v)]
		residual := make([]float32, qRows)
		normWeight := make([]float32, qRows)
		for i := range normWeight {
			normWeight[i] = 1
		}
		if err := fn(make([]float32, qRows), residual, kCache, vCache, normWeight); err != nil {
			t.Fatal(err)
		}
		runner, err := getVulkanTextAttentionF32RunnerWindows()
		if err != nil {
			t.Fatal(err)
		}
		runner.mu.Lock()
		defer runner.mu.Unlock()
		if runner.kCacheKey != float32SliceKey(kCache) || runner.vCacheKey != float32SliceKey(vCache) || runner.cacheEpoch != epoch || runner.cacheUploaded != 1 || runner.cacheKVDim != kvDim {
			t.Fatalf("runner cache warm state key=(%x,%x) epoch=%d uploaded=%d kvDim=%d, want key=(%x,%x) epoch=%d uploaded=1 kvDim=%d",
				runner.kCacheKey, runner.vCacheKey, runner.cacheEpoch, runner.cacheUploaded, runner.cacheKVDim,
				float32SliceKey(kCache), float32SliceKey(vCache), epoch, kvDim)
		}
		if runner.kBuf.size < uint64(cap(k)*4) || runner.vBuf.size < uint64(cap(v)*4) {
			t.Fatalf("runner cache buffer sizes k=%d v=%d, want at least k=%d v=%d", runner.kBuf.size, runner.vBuf.size, cap(k)*4, cap(v)*4)
		}
	}
	numHeads, kvHeads, headDim := 4, 2, 4
	qRows := numHeads * headDim
	w := make([]float32, qRows*qRows)
	for i := range w {
		w[i] = float32((i*5)%17-8) * 0.07
	}
	t.Run("f32", func(t *testing.T) {
		const epoch = 101
		runFirstTokenWarmSmoke(t, epoch, func(normOut, residual, k, v, normWeight []float32) error {
			return VulkanTextFirstTokenValueOutAddRMSNormF32(normOut, residual, k, v, w, make([]float32, qRows), normWeight, epoch, numHeads, kvHeads, headDim)
		})
	})
	t.Run("q8", func(t *testing.T) {
		const epoch = 102
		qw := tensor.QuantizeQ8Row(w, qRows, qRows)
		runFirstTokenWarmSmoke(t, epoch, func(normOut, residual, k, v, normWeight []float32) error {
			return VulkanTextFirstTokenValueOutAddRMSNormQ8(normOut, residual, k, v, qw, normWeight, epoch, numHeads, kvHeads, headDim)
		})
	})
	t.Run("q6", func(t *testing.T) {
		const epoch = 103
		qw := tensor.QuantizeQ6Row(w, qRows, qRows)
		runFirstTokenWarmSmoke(t, epoch, func(normOut, residual, k, v, normWeight []float32) error {
			return VulkanTextFirstTokenValueOutAddRMSNormQ6(normOut, residual, k, v, qw, normWeight, epoch, numHeads, kvHeads, headDim)
		})
	})
	t.Run("q4", func(t *testing.T) {
		const epoch = 104
		qw := tensor.QuantizeQ4Row(w, qRows, qRows)
		runFirstTokenWarmSmoke(t, epoch, func(normOut, residual, k, v, normWeight []float32) error {
			return VulkanTextFirstTokenValueOutAddRMSNormQ4(normOut, residual, k, v, qw, normWeight, epoch, numHeads, kvHeads, headDim)
		})
	})
}

func TestVulkanTextFirstTokenCacheIncrementalSecondToken(t *testing.T) {
	if os.Getenv("RAPIDOCRVL_VULKAN_SMOKE") != "1" {
		t.Skip("set RAPIDOCRVL_VULKAN_SMOKE=1 to run the Vulkan GPU incremental cache smoke test")
	}
	const epoch = 205
	numHeads, kvHeads, headDim := 4, 2, 4
	qRows := numHeads * headDim
	kvDim := kvHeads * headDim
	k := make([]float32, kvDim, 4*kvDim)
	v := make([]float32, kvDim, 4*kvDim)
	for i := range k {
		k[i] = float32((i*5)%13-6) * 0.11
		v[i] = float32((i*7)%17-8) * 0.09
	}
	kCache := k[:cap(k)]
	vCache := v[:cap(v)]
	w := make([]float32, qRows*qRows)
	for i := 0; i < qRows; i++ {
		w[i*qRows+i] = 1
	}
	normWeight := make([]float32, qRows)
	for i := range normWeight {
		normWeight[i] = 1
	}
	if err := VulkanTextFirstTokenValueOutAddRMSNormF32(make([]float32, qRows), make([]float32, qRows), kCache, vCache, w, make([]float32, qRows), normWeight, epoch, numHeads, kvHeads, headDim); err != nil {
		t.Fatal(err)
	}
	runner, err := getVulkanTextAttentionF32RunnerWindows()
	if err != nil {
		t.Fatal(err)
	}
	runner.mu.Lock()
	kBuffer, vBuffer := runner.kBuf.buffer, runner.vBuf.buffer
	kSize, vSize := runner.kBuf.size, runner.vBuf.size
	runner.mu.Unlock()

	k = kCache[:2*kvDim]
	v = vCache[:2*kvDim]
	for i := kvDim; i < 2*kvDim; i++ {
		k[i] = float32((i*3)%11-5) * 0.13
		v[i] = float32((i*4)%9-4) * 0.07
	}
	q := make([]float32, qRows)
	for i := range q {
		q[i] = float32((i*2)%7-3) * 0.05
	}
	if err := VulkanTextAttentionOutAddRMSNormF32(make([]float32, qRows), make([]float32, qRows), q, kCache, vCache, w, make([]float32, qRows), normWeight, epoch, 2, numHeads, kvHeads, headDim); err != nil {
		t.Fatal(err)
	}
	runner.mu.Lock()
	defer runner.mu.Unlock()
	if runner.kBuf.buffer != kBuffer || runner.vBuf.buffer != vBuffer || runner.kBuf.size != kSize || runner.vBuf.size != vSize {
		t.Fatalf("runner cache buffers changed k=(%x,%d)->(%x,%d) v=(%x,%d)->(%x,%d)",
			kBuffer, kSize, runner.kBuf.buffer, runner.kBuf.size,
			vBuffer, vSize, runner.vBuf.buffer, runner.vBuf.size)
	}
	if runner.cacheUploaded != 2 || runner.cacheEpoch != epoch || runner.cacheKVDim != kvDim || runner.kCacheKey != float32SliceKey(kCache) || runner.vCacheKey != float32SliceKey(vCache) {
		t.Fatalf("runner cache state uploaded=%d epoch=%d kvDim=%d key=(%x,%x), want uploaded=2 epoch=%d kvDim=%d key=(%x,%x)",
			runner.cacheUploaded, runner.cacheEpoch, runner.cacheKVDim, runner.kCacheKey, runner.vCacheKey,
			epoch, kvDim, float32SliceKey(kCache), float32SliceKey(vCache))
	}
}
