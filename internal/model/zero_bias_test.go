package model

import "testing"

func TestRuntimeZeroBiasReusesZeroBuffer(t *testing.T) {
	rt := &Runtime{}

	first := rt.zeroBias(4)
	if len(first) != 4 {
		t.Fatalf("zeroBias(4) len = %d, want 4", len(first))
	}
	for i, v := range first {
		if v != 0 {
			t.Fatalf("zeroBias(4)[%d] = %v, want 0", i, v)
		}
	}

	smaller := rt.zeroBias(2)
	if len(smaller) != 2 {
		t.Fatalf("zeroBias(2) len = %d, want 2", len(smaller))
	}
	if &smaller[0] != &first[0] {
		t.Fatal("zeroBias should reuse the existing buffer when capacity is sufficient")
	}

	grown := rt.zeroBias(8)
	if len(grown) != 8 {
		t.Fatalf("zeroBias(8) len = %d, want 8", len(grown))
	}
	for i, v := range grown {
		if v != 0 {
			t.Fatalf("zeroBias(8)[%d] = %v, want 0", i, v)
		}
	}
}

func TestRuntimeZeroBiasRejectsNonPositiveLength(t *testing.T) {
	rt := &Runtime{}
	if got := rt.zeroBias(0); got != nil {
		t.Fatalf("zeroBias(0) = %v, want nil", got)
	}
	if got := rt.zeroBias(-1); got != nil {
		t.Fatalf("zeroBias(-1) = %v, want nil", got)
	}
}

func TestRuntimeZeroBiasLargeUsesGrowableBuffer(t *testing.T) {
	rt := &Runtime{}
	first := rt.zeroBias(maxZeroBiasSmall + 1)
	if len(first) != maxZeroBiasSmall+1 {
		t.Fatalf("large zeroBias len=%d want %d", len(first), maxZeroBiasSmall+1)
	}
	for i, v := range first {
		if v != 0 {
			t.Fatalf("large zeroBias[%d] = %v, want 0", i, v)
		}
	}
	second := rt.zeroBias(maxZeroBiasSmall + 1)
	if len(second) != len(first) || &second[0] != &first[0] {
		t.Fatal("large zeroBias should reuse the growable buffer")
	}
}
