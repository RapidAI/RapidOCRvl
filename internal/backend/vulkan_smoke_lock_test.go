package backend

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestMain(m *testing.M) {
	unlock, err := acquireVulkanSmokeTestLock()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	code := m.Run()
	if unlock != nil {
		unlock()
	}
	os.Exit(code)
}

func acquireVulkanSmokeTestLock() (func(), error) {
	if os.Getenv("RAPIDOCRVL_VULKAN_SMOKE") != "1" {
		return nil, nil
	}
	lockDir := filepath.Join(os.TempDir(), "rapidocrvl-vulkan-smoke.lock")
	deadline := time.Now().Add(2 * time.Minute)
	for {
		if err := os.Mkdir(lockDir, 0o700); err == nil {
			return func() { _ = os.Remove(lockDir) }, nil
		}
		if st, err := os.Stat(lockDir); err == nil && time.Since(st.ModTime()) > 10*time.Minute {
			_ = os.Remove(lockDir)
			continue
		}
		if time.Now().After(deadline) {
			return nil, fmt.Errorf("timed out waiting for Vulkan smoke test lock %s", lockDir)
		}
		time.Sleep(50 * time.Millisecond)
	}
}
