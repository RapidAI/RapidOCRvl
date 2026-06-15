package model

import (
	"reflect"
	"testing"

	"paddleocrvl-go/internal/config"
)

func TestMultimodalPositionsSingleImage(t *testing.T) {
	rt := &Runtime{cfg: &config.Config{
		ImageTokenID: 100295,
		VisionConfig: config.Vision{
			SpatialMergeSize: 2,
		},
	}}
	input := []int{10, 101305, 100295, 100295, 100295, 100295, 11}
	got, delta := rt.multimodalPositions(input, [3]int{1, 4, 4})
	want := []ropePos{
		{0, 0, 0},
		{1, 1, 1},
		{2, 2, 2},
		{2, 2, 3},
		{2, 3, 2},
		{2, 3, 3},
		{4, 4, 4},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("positions=%v want %v", got, want)
	}
	if delta != -2 {
		t.Fatalf("delta=%d want -2", delta)
	}
}

func TestMultimodalPositionsMultipleImages(t *testing.T) {
	rt := &Runtime{cfg: &config.Config{
		ImageTokenID: 7,
		VisionConfig: config.Vision{
			SpatialMergeSize: 2,
		},
	}}
	input := []int{1, 7, 7, 7, 7, 2, 3, 7, 7, 7, 7, 4}
	got, delta := rt.multimodalPositions(input, [3]int{1, 4, 4})
	want := []ropePos{
		{0, 0, 0},
		{1, 1, 1},
		{1, 1, 2},
		{1, 2, 1},
		{1, 2, 2},
		{3, 3, 3},
		{4, 4, 4},
		{5, 5, 5},
		{5, 5, 6},
		{5, 6, 5},
		{5, 6, 6},
		{7, 7, 7},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("positions=%v want %v", got, want)
	}
	if delta != -4 {
		t.Fatalf("delta=%d want -4", delta)
	}
}

func BenchmarkMultimodalPositionsManyImages(b *testing.B) {
	rt := &Runtime{cfg: &config.Config{
		ImageTokenID: 7,
		VisionConfig: config.Vision{
			SpatialMergeSize: 2,
		},
	}}
	input := make([]int, 0, 1024)
	for i := 0; i < 128; i++ {
		input = append(input, 1, 2, 7, 7, 7, 7)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = rt.multimodalPositions(input, [3]int{1, 4, 4})
	}
}

func BenchmarkMultimodalPositionsManyImagesReuse(b *testing.B) {
	rt := &Runtime{cfg: &config.Config{
		ImageTokenID: 7,
		VisionConfig: config.Vision{
			SpatialMergeSize: 2,
		},
	}}
	input := make([]int, 0, 1024)
	for i := 0; i < 128; i++ {
		input = append(input, 1, 2, 7, 7, 7, 7)
	}
	buf := make([]ropePos, 0, len(input))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var positions []ropePos
		positions, _ = rt.multimodalPositionsInto(input, [3]int{1, 4, 4}, buf)
		buf = positions
	}
}

func TestExpandSingleImagePlaceholderIntoReusesBuffer(t *testing.T) {
	rt := &Runtime{cfg: &config.Config{ImageTokenID: 7}}
	buf := make([]int, 0, 16)
	got, expanded := rt.expandSingleImagePlaceholderInto([]int{1, 7, 2}, 4, buf)
	if !expanded {
		t.Fatal("expected expansion")
	}
	want := []int{1, 7, 7, 7, 7, 2}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("expanded=%v want %v", got, want)
	}
	if len(got) == 0 || &got[0] != &buf[:1][0] {
		t.Fatal("expected buffer reuse")
	}
}

func BenchmarkExpandSingleImagePlaceholderReuse(b *testing.B) {
	rt := &Runtime{cfg: &config.Config{ImageTokenID: 7}}
	input := []int{1, 2, 7, 3, 4}
	buf := make([]int, 0, 512)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var expanded bool
		buf, expanded = rt.expandSingleImagePlaceholderInto(input, 256, buf)
		if !expanded || len(buf) != 260 {
			b.Fatalf("expanded=%v len=%d", expanded, len(buf))
		}
	}
}
