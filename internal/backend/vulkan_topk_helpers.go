package backend

// vulkanMatVecTopKMaxK is the maximum number of top-k candidates each
// 256-row block can produce. Each candidate is stored as two float32
// values (score, token_id) in the readback buffer.
const vulkanMatVecTopKMaxK = 64

// ensureVulkanFloat32Scratch ensures scratch has at least n elements,
// reusing the existing backing array when possible.
func ensureVulkanFloat32Scratch(scratch []float32, n int) []float32 {
	if cap(scratch) >= n {
		return scratch[:n]
	}
	return make([]float32, n)
}

// selectVulkanTopKCandidates reads interleaved (score, token) pairs from
// data and returns the top-k by score (ties broken by token id ascending).
// It allocates a fresh result slice.
func selectVulkanTopKCandidates(data []float32, rows, k int) []VulkanTokenScore {
	return selectVulkanTopKCandidatesInto(nil, data, rows, k)
}

// selectVulkanTopKCandidatesInto reads interleaved (score, token) pairs
// from data and returns the top-k by score (ties broken by token id).
// The result reuses dst's backing array when possible.
func selectVulkanTopKCandidatesInto(dst []VulkanTokenScore, data []float32, rows, k int) []VulkanTokenScore {
	pairs := len(data) / 2
	if k <= 0 {
		return dst[:0]
	}
	dst = ensureVulkanTokenScoreScratch(dst, k)
	used := make([]bool, pairs)
	n := 0
	// Partial selection sort over valid (token >= 0) candidates, picking the
	// top-k by score with ties broken by ascending token id.  The input data
	// is left unmodified so callers can reuse it across invocations.
	for picked := 0; picked < k; picked++ {
		best := -1
		var bestScore float32
		var bestToken int
		for j := 0; j < pairs; j++ {
			if used[j] {
				continue
			}
			score := data[j*2]
			token := int(data[j*2+1])
			if token < 0 {
				continue
			}
			if best == -1 || score > bestScore || (score == bestScore && token < bestToken) {
				best = j
				bestScore = score
				bestToken = token
			}
		}
		if best == -1 {
			break
		}
		used[best] = true
		dst[n] = VulkanTokenScore{Token: bestToken, Score: bestScore}
		n++
	}
	return dst[:n]
}

// ensureVulkanTokenScoreScratch ensures scratch has capacity for n elements.
func ensureVulkanTokenScoreScratch(scratch []VulkanTokenScore, n int) []VulkanTokenScore {
	if cap(scratch) >= n {
		return scratch[:n]
	}
	return make([]VulkanTokenScore, n)
}
