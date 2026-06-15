package backend

import (
	"fmt"
	"runtime"
	"runtime/debug"
)

type CPUInfo struct {
	GOOS       string   `json:"goos"`
	GOARCH     string   `json:"goarch"`
	NumCPU     int      `json:"num_cpu"`
	GOMAXPROCS int      `json:"gomaxprocs"`
	Features   []string `json:"features"`
}

type MemoryInfo struct {
	AllocBytes      uint64 `json:"alloc_bytes"`
	TotalAllocBytes uint64 `json:"total_alloc_bytes"`
	SysBytes        uint64 `json:"sys_bytes"`
	HeapAllocBytes  uint64 `json:"heap_alloc_bytes"`
	HeapSysBytes    uint64 `json:"heap_sys_bytes"`
	HeapIdleBytes   uint64 `json:"heap_idle_bytes"`
	HeapInuseBytes  uint64 `json:"heap_inuse_bytes"`
	HeapObjects     uint64 `json:"heap_objects"`
	NumGC           uint32 `json:"num_gc"`
}

func CPU() CPUInfo {
	return CPUInfo{
		GOOS:       runtime.GOOS,
		GOARCH:     runtime.GOARCH,
		NumCPU:     runtime.NumCPU(),
		GOMAXPROCS: runtime.GOMAXPROCS(0),
		Features:   cpuFeatures(),
	}
}

func Memory() MemoryInfo {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return MemoryInfo{
		AllocBytes:      m.Alloc,
		TotalAllocBytes: m.TotalAlloc,
		SysBytes:        m.Sys,
		HeapAllocBytes:  m.HeapAlloc,
		HeapSysBytes:    m.HeapSys,
		HeapIdleBytes:   m.HeapIdle,
		HeapInuseBytes:  m.HeapInuse,
		HeapObjects:     m.HeapObjects,
		NumGC:           m.NumGC,
	}
}

func SetGOMAXPROCS(n int) (int, error) {
	if n < 0 {
		return runtime.GOMAXPROCS(0), fmt.Errorf("gomaxprocs must be >= 0")
	}
	if n == 0 {
		return runtime.GOMAXPROCS(0), nil
	}
	return runtime.GOMAXPROCS(n), nil
}

func SetGCPercent(n int) int {
	if n == 0 {
		current := debug.SetGCPercent(-1)
		debug.SetGCPercent(current)
		return current
	}
	return debug.SetGCPercent(n)
}
