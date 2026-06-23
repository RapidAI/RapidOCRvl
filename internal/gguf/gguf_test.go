package gguf

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestOpenRejectsHugeCounts(t *testing.T) {
	dir := t.TempDir()
	for _, tc := range []struct {
		name        string
		tensorCount uint64
		kvCount     uint64
	}{
		{name: "tensors", tensorCount: maxTensorCount + 1},
		{name: "metadata", kvCount: maxMetadataKVCount + 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(dir, tc.name+".gguf")
			writeGGUFCountsForTest(t, path, tc.tensorCount, tc.kvCount)
			if _, err := Open(path); err == nil {
				t.Fatal("expected count error")
			}
		})
	}
}

func TestOpenRejectsHugeString(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "huge-string.gguf")
	var data []byte
	data = append(data, Magic...)
	data = binary.LittleEndian.AppendUint32(data, Version)
	data = binary.LittleEndian.AppendUint64(data, 0)
	data = binary.LittleEndian.AppendUint64(data, 1)
	data = binary.LittleEndian.AppendUint64(data, maxStringBytes+1)
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := Open(path); err == nil || !strings.Contains(err.Error(), "string too large") {
		t.Fatalf("err=%v", err)
	}
}

func TestOpenRejectsInvalidMetadataKeys(t *testing.T) {
	dir := t.TempDir()
	tests := []struct {
		name string
		meta []ggufMetadataForTest
	}{
		{
			name: "empty-key",
			meta: []ggufMetadataForTest{{key: "", typ: metaUint32, value: uint32(1)}},
		},
		{
			name: "duplicate-key",
			meta: []ggufMetadataForTest{
				{key: "general.alignment", typ: metaUint32, value: uint32(Alignment)},
				{key: "general.alignment", typ: metaUint32, value: uint32(16)},
			},
		},
		{
			name: "control-character-key",
			meta: []ggufMetadataForTest{{key: "general\nname", typ: metaString, value: "bad"}},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(dir, tc.name+".gguf")
			writeGGUFMetadataForTest(t, path, tc.meta)
			if _, err := Open(path); err == nil {
				t.Fatal("expected invalid metadata key error")
			}
		})
	}
}

func TestOpenRejectsInvalidTensorMetadata(t *testing.T) {
	dir := t.TempDir()
	tests := []struct {
		name   string
		dims   []uint64
		typ    uint32
		offset uint64
		data   []byte
	}{
		{
			name: "too-many-dims",
			dims: []uint64{1, 1, 1, 1, 1, 1, 1, 1, 1},
			typ:  typeF32,
		},
		{
			name: "negative-dim-after-cast",
			dims: []uint64{math.MaxUint64},
			typ:  typeF32,
		},
		{
			name: "unknown-type",
			dims: []uint64{1},
			typ:  math.MaxUint32,
			data: make([]byte, 4),
		},
		{
			name:   "offset-exceeds-file",
			dims:   []uint64{1},
			typ:    typeF32,
			offset: 8,
			data:   make([]byte, 4),
		},
		{
			name: "tensor-exceeds-file",
			dims: []uint64{2},
			typ:  typeF32,
			data: make([]byte, 4),
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(dir, tc.name+".gguf")
			writeGGUFTensorForTest(t, path, "x.weight", tc.dims, tc.typ, tc.offset, tc.data)
			if _, err := Open(path); err == nil {
				t.Fatal("expected invalid tensor metadata error")
			}
		})
	}
}

func TestOpenRejectsInvalidTensorNames(t *testing.T) {
	dir := t.TempDir()
	tests := []struct {
		name    string
		tensors []ggufTensorForTest
	}{
		{
			name: "empty-name",
			tensors: []ggufTensorForTest{
				{name: "", dims: []uint64{1}, typ: typeF32, payload: make([]byte, 4)},
			},
		},
		{
			name: "duplicate-name",
			tensors: []ggufTensorForTest{
				{name: "x.weight", dims: []uint64{1}, typ: typeF32, payload: make([]byte, 4)},
				{name: "x.weight", dims: []uint64{1}, typ: typeF32, offset: 4, payload: make([]byte, 4)},
			},
		},
		{
			name: "control-character-name",
			tensors: []ggufTensorForTest{
				{name: "x\nweight", dims: []uint64{1}, typ: typeF32, payload: make([]byte, 4)},
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(dir, tc.name+".gguf")
			writeGGUFTensorsForTest(t, path, tc.tensors)
			if _, err := Open(path); err == nil {
				t.Fatal("expected invalid tensor name error")
			}
		})
	}
}

func TestTensorBytesRejectsOverflow(t *testing.T) {
	if got := TensorBytes(TensorMeta{Shape: []int64{math.MaxInt64, 2}, Type: typeF32}); got != 0 {
		t.Fatalf("bytes=%d", got)
	}
}

func writeGGUFCountsForTest(t testing.TB, path string, tensorCount, kvCount uint64) {
	t.Helper()
	var data []byte
	data = append(data, Magic...)
	data = binary.LittleEndian.AppendUint32(data, Version)
	data = binary.LittleEndian.AppendUint64(data, tensorCount)
	data = binary.LittleEndian.AppendUint64(data, kvCount)
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
}

type ggufMetadataForTest struct {
	key   string
	typ   uint32
	value any
}

func writeGGUFMetadataForTest(t testing.TB, path string, meta []ggufMetadataForTest) {
	t.Helper()
	var data []byte
	data = append(data, Magic...)
	data = binary.LittleEndian.AppendUint32(data, Version)
	data = binary.LittleEndian.AppendUint64(data, 0)
	data = binary.LittleEndian.AppendUint64(data, uint64(len(meta)))
	for _, kv := range meta {
		data = appendGGUFStringForTest(data, kv.key)
		data = binary.LittleEndian.AppendUint32(data, kv.typ)
		switch kv.typ {
		case metaUint32:
			data = binary.LittleEndian.AppendUint32(data, kv.value.(uint32))
		case metaUint64:
			data = binary.LittleEndian.AppendUint64(data, kv.value.(uint64))
		case metaString:
			data = appendGGUFStringForTest(data, kv.value.(string))
		default:
			t.Fatalf("unsupported test metadata type %d", kv.typ)
		}
	}
	if pad := int(align(int64(len(data)), uint64(Alignment))) - len(data); pad > 0 {
		data = append(data, make([]byte, pad)...)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
}

type ggufTensorForTest struct {
	name    string
	dims    []uint64
	typ     uint32
	offset  uint64
	payload []byte
}

func writeGGUFTensorForTest(t testing.TB, path, name string, dims []uint64, typ uint32, offset uint64, payload []byte) {
	t.Helper()
	writeGGUFTensorsForTest(t, path, []ggufTensorForTest{
		{name: name, dims: dims, typ: typ, offset: offset, payload: payload},
	})
}

func writeGGUFTensorsForTest(t testing.TB, path string, tensors []ggufTensorForTest) {
	t.Helper()
	var data []byte
	data = append(data, Magic...)
	data = binary.LittleEndian.AppendUint32(data, Version)
	data = binary.LittleEndian.AppendUint64(data, uint64(len(tensors)))
	data = binary.LittleEndian.AppendUint64(data, 0)
	for _, tensor := range tensors {
		data = appendGGUFStringForTest(data, tensor.name)
		data = binary.LittleEndian.AppendUint32(data, uint32(len(tensor.dims)))
		for _, dim := range tensor.dims {
			data = binary.LittleEndian.AppendUint64(data, dim)
		}
		data = binary.LittleEndian.AppendUint32(data, tensor.typ)
		data = binary.LittleEndian.AppendUint64(data, tensor.offset)
	}
	if pad := int(align(int64(len(data)), uint64(Alignment))) - len(data); pad > 0 {
		data = append(data, make([]byte, pad)...)
	}
	for _, tensor := range tensors {
		data = append(data, tensor.payload...)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
}

func appendGGUFStringForTest(data []byte, s string) []byte {
	data = binary.LittleEndian.AppendUint64(data, uint64(len(s)))
	return append(data, s...)
}
