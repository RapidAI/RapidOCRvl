package tensor

import "sync"

// CumulativeSum computes out[i] = sum(x[0..=i]) writing to out.
// Uses a parallel block approach for large arrays: splits into 4 blocks,
// computes prefix sums within each block in parallel, then fixes up
// the inter-block offsets sequentially.
func CumulativeSum(out, x []float32) float32 {
	n := min(len(out), len(x))
	if n == 0 {
		return 0
	}
	if n < 4096 {
		var cs float32
		for i := 0; i < n; i++ {
			cs += x[i]
			out[i] = cs
		}
		return cs
	}
	blockSize := (n + 3) / 4
	var blockSums [4]float32
	var wg sync.WaitGroup
	for b := 0; b < 4; b++ {
		start := b * blockSize
		end := start + blockSize
		if end > n {
			end = n
		}
		if start >= end {
			continue
		}
		wg.Add(1)
		go func(b, start, end int) {
			defer wg.Done()
			var cs float32
			for i := start; i < end; i++ {
				cs += x[i]
				out[i] = cs
			}
			blockSums[b] = cs
		}(b, start, end)
	}
	wg.Wait()
	offset := blockSums[0]
	for b := 1; b < 4; b++ {
		start := b * blockSize
		end := start + blockSize
		if end > n {
			end = n
		}
		for i := start; i < end; i++ {
			out[i] += offset
		}
		offset += blockSums[b]
	}
	return offset
}
