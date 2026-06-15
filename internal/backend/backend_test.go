package backend

import (
	"runtime"
	"testing"
)

func TestSelectCPU(t *testing.T) {
	sel, err := Select("cpu")
	if err != nil {
		t.Fatal(err)
	}
	if sel.Active != "cpu" || sel.Requested != "cpu" {
		t.Fatalf("selection=%+v", sel)
	}
	if sel.Vulkan.Name != "vulkan" || sel.Vulkan.ComputeReason == "" {
		t.Fatalf("vulkan info=%+v", sel.Vulkan)
	}
}

func TestSelectNormalizesRequestedBackend(t *testing.T) {
	sel, err := Select(" CPU ")
	if err != nil {
		t.Fatal(err)
	}
	if sel.Active != "cpu" || sel.Requested != "cpu" {
		t.Fatalf("selection=%+v", sel)
	}
}

func TestCPUInfoIncludesSchedulerShape(t *testing.T) {
	cpu := CPU()
	if cpu.GOOS == "" || cpu.GOARCH == "" {
		t.Fatalf("cpu=%+v", cpu)
	}
	if cpu.NumCPU < 1 || cpu.GOMAXPROCS < 1 {
		t.Fatalf("cpu=%+v", cpu)
	}
	if len(cpu.Features) == 0 {
		t.Fatalf("cpu=%+v", cpu)
	}
}

func TestMemoryInfo(t *testing.T) {
	mem := Memory()
	if mem.SysBytes == 0 {
		t.Fatalf("memory=%+v", mem)
	}
	if mem.HeapObjects == 0 && mem.HeapAllocBytes == 0 {
		t.Fatalf("memory=%+v", mem)
	}
}

func TestSetGOMAXPROCS(t *testing.T) {
	orig := runtime.GOMAXPROCS(0)
	defer runtime.GOMAXPROCS(orig)
	prev, err := SetGOMAXPROCS(1)
	if err != nil {
		t.Fatal(err)
	}
	if prev < 1 || runtime.GOMAXPROCS(0) != 1 {
		t.Fatalf("prev=%d current=%d", prev, runtime.GOMAXPROCS(0))
	}
	if _, err := SetGOMAXPROCS(-1); err == nil {
		t.Fatal("expected negative gomaxprocs error")
	}
}

func TestSetGCPercent(t *testing.T) {
	orig := SetGCPercent(0)
	defer SetGCPercent(orig)
	prev := SetGCPercent(50)
	if prev != orig {
		t.Fatalf("prev=%d want %d", prev, orig)
	}
	if got := SetGCPercent(0); got != 50 {
		t.Fatalf("current=%d want 50", got)
	}
}

func TestSelectUnknown(t *testing.T) {
	if _, err := Select("metal"); err == nil {
		t.Fatal("expected error")
	}
}
