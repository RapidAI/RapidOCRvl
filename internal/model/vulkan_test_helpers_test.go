package model

import "sync"

// resetVulkanMinWorkCacheForTest resets the sync.Once guards for the
// vulkan min-work caches so that tests can override env vars.
func resetVulkanMinWorkCacheForTest() {
	vulkanMatVecMinWorkOnce = sync.Once{}
	vulkanMatVecMinWorkValue = 0
	vulkanTextAttentionMinWorkOnce = sync.Once{}
	vulkanTextAttentionMinWorkValue = 0
	vulkanVectorMinWorkOnce = sync.Once{}
	vulkanVectorMinWorkValue = 0
	vulkanVisionMinWorkOnce = sync.Once{}
	vulkanVisionMinWorkValue = 0
}
