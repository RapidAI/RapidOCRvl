package jsonutil

import (
	"bytes"
	"strings"
	"testing"
)

func TestRejectDuplicateKeys(t *testing.T) {
	tests := []struct {
		name    string
		data    string
		wantErr string
	}{
		{name: "valid", data: `{"a":1,"b":{"c":2},"d":[{"e":3}]}`},
		{name: "same-key-in-sibling-objects", data: `{"a":{"x":1},"b":{"x":2}}`},
		{name: "same-key-in-array-objects", data: `[{"x":1},{"x":2}]`},
		{name: "top-level-duplicate", data: `{"a":1,"a":2}`, wantErr: "duplicate JSON key"},
		{name: "nested-duplicate", data: `{"a":{"b":1,"b":2}}`, wantErr: "duplicate JSON key"},
		{name: "array-object-duplicate", data: `{"a":[{"b":1,"b":2}]}`, wantErr: "duplicate JSON key"},
		{name: "trailing-data", data: `{"a":1}{"b":2}`, wantErr: "trailing JSON data"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := RejectDuplicateKeys([]byte(tc.data), "test.json")
			if tc.wantErr == "" {
				if err != nil {
					t.Fatalf("err=%v", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("err=%v want %q", err, tc.wantErr)
			}
		})
	}
}

func TestRejectDuplicateKeysRejectsDeepNesting(t *testing.T) {
	var b bytes.Buffer
	for i := 0; i < maxJSONDepth+2; i++ {
		b.WriteByte('[')
	}
	b.WriteByte('0')
	for i := 0; i < maxJSONDepth+2; i++ {
		b.WriteByte(']')
	}
	err := RejectDuplicateKeys(b.Bytes(), "deep.json")
	if err == nil || !strings.Contains(err.Error(), "nesting too deep") {
		t.Fatalf("err=%v want nesting too deep", err)
	}
}

func TestRejectDuplicateKeysAllowsDepthLimit(t *testing.T) {
	var b bytes.Buffer
	for i := 0; i < maxJSONDepth; i++ {
		b.WriteByte('[')
	}
	b.WriteByte('0')
	for i := 0; i < maxJSONDepth; i++ {
		b.WriteByte(']')
	}
	if err := RejectDuplicateKeys(b.Bytes(), "limit.json"); err != nil {
		t.Fatalf("err=%v", err)
	}
}

func BenchmarkRejectDuplicateKeys(b *testing.B) {
	data := []byte(`{"model":{"vocab":{"<unk>":0,"hello":1,"world":2},"merges":[["h","e"],["he","llo"]]},"added_tokens":[{"id":3,"content":"<s>","special":true}]}`)
	b.SetBytes(int64(len(data)))
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if err := RejectDuplicateKeys(data, "bench.json"); err != nil {
			b.Fatal(err)
		}
	}
}
