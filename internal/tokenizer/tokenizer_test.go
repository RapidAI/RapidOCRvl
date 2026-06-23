package tokenizer

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestEncodeDecodeBPEAndSpecials(t *testing.T) {
	dir := t.TempDir()
	writeTokenizer(t, dir, `{
	  "added_tokens": [
	    {"id": 1, "content": "<s>", "special": true},
	    {"id": 2, "content": "</s>", "special": true}
	  ],
	  "model": {
	    "vocab": {
	      "<unk>": 0,
	      "<s>": 1,
	      "</s>": 2,
	      "h": 3,
	      "e": 4,
	      "l": 5,
	      "o": 6,
	      "he": 7,
	      "ll": 8,
	      "hello": 9,
	      "\u809d": 10,
	      "\u809dhello": 11,
	      "<0xE4>": 12,
	      "<0xBD>": 13,
	      "<0xA0>": 14
	    },
	    "merges": [["h", "e"], ["l", "l"], ["he", "ll"], ["hell", "o"], ["\u809d", "hello"]]
	  }
	}`)
	tok, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	got := tok.Encode("<s> hello\u4f60")
	want := []int{1, 11, 12, 13, 14}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Encode()=%v want %v", got, want)
	}
	if text := tok.Decode(got, false); text != "<s> hello\u4f60" {
		t.Fatalf("Decode()=%q", text)
	}
	if text := tok.Decode(got, true); text != " hello\u4f60" {
		t.Fatalf("Decode(skipSpecial)=%q", text)
	}
	if hint := tok.decodedLenHint(got, true); hint < len(" hello\u4f60") {
		t.Fatalf("decodedLenHint=%d too small", hint)
	}
	if text := tok.Decode([]int{9}, true); text != "hello" {
		t.Fatalf("Decode(single)=%q", text)
	}
	if text := tok.Decode([]int{12}, true); text != "\xe4" {
		t.Fatalf("Decode(single byte)=%q", text)
	}
	if text := tok.Decode([]int{1}, true); text != "" {
		t.Fatalf("Decode(single special)=%q", text)
	}
	if len(tok.specialEntries) != 2 || tok.specialEntries[0].id == 0 || tok.specialEntries[0].token == "" {
		t.Fatalf("specialEntries=%+v", tok.specialEntries)
	}
	if ids := tok.Encode(""); len(ids) != 0 {
		t.Fatalf("Encode(empty)=%v", ids)
	}
	if text := tok.Decode(nil, true); text != "" {
		t.Fatalf("Decode(empty)=%q", text)
	}
}

func TestLoadRejectsInvalidTokenIDs(t *testing.T) {
	for _, body := range []string{
		`{"model":{"vocab":{"x":-1},"merges":[]}}`,
		`{"model":{"vocab":{"x":16777217},"merges":[]}}`,
		`{"added_tokens":[{"id":16777217,"content":"<s>","special":true}],"model":{"vocab":{"x":0},"merges":[]}}`,
		`{"model":{"vocab":{"x":0,"y":0},"merges":[]}}`,
		`{"added_tokens":[{"id":0,"content":"<s>","special":true}],"model":{"vocab":{"x":0},"merges":[]}}`,
		`{"added_tokens":[{"id":1,"content":"<s>","special":true},{"id":2,"content":"<s>","special":true}],"model":{"vocab":{"x":0},"merges":[]}}`,
	} {
		dir := t.TempDir()
		writeTokenizer(t, dir, body)
		if _, err := Load(dir); err == nil {
			t.Fatalf("expected invalid tokenizer error for %s", body)
		}
	}
}

func TestLoadRejectsDuplicateJSONKeys(t *testing.T) {
	for _, tc := range []struct {
		name string
		body string
	}{
		{
			name: "top-level",
			body: `{"model":{"vocab":{"x":0},"merges":[]},"model":{"vocab":{"y":1},"merges":[]}}`,
		},
		{
			name: "vocab",
			body: `{"model":{"vocab":{"x":0,"x":1},"merges":[]}}`,
		},
		{
			name: "added-token",
			body: `{"added_tokens":[{"id":1,"content":"<s>","content":"</s>","special":true}],"model":{"vocab":{"x":0},"merges":[]}}`,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			writeTokenizer(t, dir, tc.body)
			if _, err := Load(dir); err == nil || !strings.Contains(err.Error(), "duplicate JSON key") {
				t.Fatalf("err=%v want duplicate JSON key", err)
			}
		})
	}
}

func TestMakeByteToken(t *testing.T) {
	cases := map[byte]string{
		0x00: "<0x00>",
		0x0f: "<0x0F>",
		0xa0: "<0xA0>",
		0xff: "<0xFF>",
	}
	for in, want := range cases {
		got := makeByteToken(in)
		if got != want {
			t.Fatalf("makeByteToken(%#x)=%q want %q", in, got, want)
		}
		by, ok := byteToken(got)
		if !ok || by != in {
			t.Fatalf("byteToken(%q)=(%#x,%v) want (%#x,true)", got, by, ok, in)
		}
	}
}

func TestEncodeCacheReturnsCopies(t *testing.T) {
	dir := t.TempDir()
	writeTokenizer(t, dir, `{
	  "added_tokens": [],
	  "model": {
	    "vocab": {"<unk>": 0, "h": 1, "i": 2, "hi": 3},
	    "merges": [["h", "i"]]
	  }
	}`)
	tok, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	first := tok.Encode("hi")
	if len(first) != 1 || first[0] != 3 {
		t.Fatalf("first=%v", first)
	}
	first[0] = 99
	second := tok.Encode("hi")
	if len(second) != 1 || second[0] != 3 {
		t.Fatalf("cache aliased result: %v", second)
	}
	stats := tok.CacheStats()
	if stats.EncodeEntries != 1 || stats.NormalEntries != 1 {
		t.Fatalf("stats=%+v", stats)
	}
}

func TestEncodeReadOnlyReturnsCachedSlice(t *testing.T) {
	dir := t.TempDir()
	writeTokenizer(t, dir, `{
	  "added_tokens": [],
	  "model": {
	    "vocab": {"<unk>": 0, "h": 1, "i": 2, "hi": 3},
	    "merges": [["h", "i"]]
	  }
	}`)
	tok, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	_ = tok.EncodeReadOnly("hi")
	first := tok.EncodeReadOnly("hi")
	second := tok.EncodeReadOnly("hi")
	if len(first) != 1 || first[0] != 3 || len(second) != 1 || second[0] != 3 {
		t.Fatalf("bad readonly encodes first=%v second=%v", first, second)
	}
	if &first[0] != &second[0] {
		t.Fatalf("readonly cache did not reuse slice")
	}
	copied := tok.Encode("hi")
	if &copied[0] == &second[0] {
		t.Fatalf("Encode returned cached readonly slice")
	}
}

func TestEncodeCacheSkipsOversizedTokenResults(t *testing.T) {
	tok := &Tokenizer{
		Vocab:       map[string]int{"a": 1},
		cache:       map[string][]int{},
		encodeCache: map[string][]int{},
	}
	ids := make([]int, maxCacheableTokenLen+1)
	tok.storeEncodeCache("big", ids)
	if stats := tok.CacheStats(); stats.EncodeEntries != 0 {
		t.Fatalf("oversized encode cache stored stats=%+v", stats)
	}
	got := tok.encodeNormal(strings.Repeat("a", maxCacheableTokenLen+1))
	if len(got) != maxCacheableTokenLen+1 {
		t.Fatalf("len=%d", len(got))
	}
	if stats := tok.CacheStats(); stats.NormalEntries != 0 {
		t.Fatalf("oversized normal cache stored stats=%+v", stats)
	}
	tok.storeEncodeCache("small", ids[:maxCacheableTokenLen])
	if stats := tok.CacheStats(); stats.EncodeEntries != 1 {
		t.Fatalf("small encode cache skipped stats=%+v", stats)
	}
}

func TestEncodeNormalCacheSkipsOversizedInput(t *testing.T) {
	tok := &Tokenizer{
		Vocab:       map[string]int{"a": 1},
		cache:       map[string][]int{},
		encodeCache: map[string][]int{},
	}
	long := strings.Repeat("a", maxCacheableInputLen+1)
	if got := tok.encodeNormal(long); len(got) != len(long) {
		t.Fatalf("len=%d want %d", len(got), len(long))
	}
	if stats := tok.CacheStats(); stats.NormalEntries != 0 {
		t.Fatalf("oversized input cached stats=%+v", stats)
	}
	short := strings.Repeat("a", maxCacheableInputLen)
	if got := tok.encodeNormal(short); len(got) != len(short) {
		t.Fatalf("len=%d want %d", len(got), len(short))
	}
	if stats := tok.CacheStats(); stats.NormalEntries != 1 {
		t.Fatalf("cacheable input skipped stats=%+v", stats)
	}
}

func BenchmarkEncodeCached(b *testing.B) {
	dir := b.TempDir()
	writeTokenizer(b, dir, `{
	  "added_tokens": [{"id": 4, "content": "<s>", "special": true}],
	  "model": {
	    "vocab": {"<unk>": 0, "h": 1, "i": 2, "hi": 3, "<s>": 4, "\u809d": 5},
	    "merges": [["h", "i"]]
	  }
	}`)
	tok, err := Load(dir)
	if err != nil {
		b.Fatal(err)
	}
	prompt := "<s> hi hi hi hi hi"
	_ = tok.Encode(prompt)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = tok.Encode(prompt)
	}
}

func BenchmarkEncodeCachedReadOnly(b *testing.B) {
	dir := b.TempDir()
	writeTokenizer(b, dir, `{
	  "added_tokens": [{"id": 4, "content": "<s>", "special": true}],
	  "model": {
	    "vocab": {"<unk>": 0, "h": 1, "i": 2, "hi": 3, "<s>": 4, "\u809d": 5},
	    "merges": [["h", "i"]]
	  }
	}`)
	tok, err := Load(dir)
	if err != nil {
		b.Fatal(err)
	}
	prompt := "<s> hi hi hi hi hi"
	_ = tok.EncodeReadOnly(prompt)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = tok.EncodeReadOnly(prompt)
	}
}

func BenchmarkEncodeEmpty(b *testing.B) {
	dir := b.TempDir()
	writeTokenizer(b, dir, `{
	  "added_tokens": [{"id": 4, "content": "<s>", "special": true}],
	  "model": {
	    "vocab": {"<unk>": 0, "h": 1, "i": 2, "hi": 3, "<s>": 4},
	    "merges": [["h", "i"]]
	  }
	}`)
	tok, err := Load(dir)
	if err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = tok.Encode("")
	}
}

func BenchmarkDecodeEmpty(b *testing.B) {
	dir := b.TempDir()
	writeTokenizer(b, dir, `{
	  "added_tokens": [{"id": 4, "content": "<s>", "special": true}],
	  "model": {
	    "vocab": {"<unk>": 0, "h": 1, "i": 2, "hi": 3, "<s>": 4},
	    "merges": [["h", "i"]]
	  }
	}`)
	tok, err := Load(dir)
	if err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = tok.Decode(nil, true)
	}
}

func BenchmarkEncodeNoSpecialUncached(b *testing.B) {
	dir := b.TempDir()
	writeTokenizer(b, dir, `{
	  "added_tokens": [],
	  "model": {
	    "vocab": {"<unk>": 0, "h": 1, "i": 2, "hi": 3, " ": 4, "\u809d": 5},
	    "merges": [["h", "i"]]
	  }
	}`)
	tok, err := Load(dir)
	if err != nil {
		b.Fatal(err)
	}
	prompts := []string{"hi", "hi hi", "hi hi hi", "hihihi"}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = tok.Encode(prompts[i&3])
	}
}

func BenchmarkEncodeUnicodeFallbackUncached(b *testing.B) {
	dir := b.TempDir()
	writeTokenizer(b, dir, `{
	  "added_tokens": [],
	  "model": {
	    "vocab": {"<unk>": 0, "<0xE4>": 1, "<0xBD>": 2, "<0xA0>": 3, "<0xE5>": 4, "<0xA5>": 5, "<0xBD>": 6},
	    "merges": []
	  }
	}`)
	tok, err := Load(dir)
	if err != nil {
		b.Fatal(err)
	}
	prompts := []string{"你好", "你好你好", "你好abc", "abc你好"}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = tok.Encode(prompts[i&3])
	}
}

func BenchmarkDecodeNoSpaces(b *testing.B) {
	dir := b.TempDir()
	writeTokenizer(b, dir, `{
	  "added_tokens": [],
	  "model": {
	    "vocab": {"<unk>": 0, "h": 1, "i": 2, "hi": 3},
	    "merges": [["h", "i"]]
	  }
	}`)
	tok, err := Load(dir)
	if err != nil {
		b.Fatal(err)
	}
	ids := []int{3, 3, 3, 3, 3, 3, 3, 3}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = tok.Decode(ids, true)
	}
}

func BenchmarkDecodeSingleToken(b *testing.B) {
	dir := b.TempDir()
	writeTokenizer(b, dir, `{
	  "added_tokens": [],
	  "model": {
	    "vocab": {"<unk>": 0, "hello": 1},
	    "merges": []
	  }
	}`)
	tok, err := Load(dir)
	if err != nil {
		b.Fatal(err)
	}
	ids := []int{1}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = tok.Decode(ids, true)
	}
}

func BenchmarkDecodeSingleByteToken(b *testing.B) {
	dir := b.TempDir()
	writeTokenizer(b, dir, `{
	  "added_tokens": [],
	  "model": {
	    "vocab": {"<unk>": 0, "<0xE4>": 1},
	    "merges": []
	  }
	}`)
	tok, err := Load(dir)
	if err != nil {
		b.Fatal(err)
	}
	ids := []int{1}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = tok.Decode(ids, true)
	}
}

func BenchmarkDecodeByteTokens(b *testing.B) {
	dir := b.TempDir()
	writeTokenizer(b, dir, `{
	  "added_tokens": [],
	  "model": {
	    "vocab": {"<unk>": 0, "<0xE4>": 1, "<0xBD>": 2, "<0xA0>": 3, "<0xE5>": 4, "<0xA5>": 5},
	    "merges": []
	  }
	}`)
	tok, err := Load(dir)
	if err != nil {
		b.Fatal(err)
	}
	ids := []int{1, 2, 3, 4, 5, 2, 1, 2, 3, 4, 5, 2}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = tok.Decode(ids, true)
	}
}

func BenchmarkDecodeMixedByteTokens(b *testing.B) {
	dir := b.TempDir()
	writeTokenizer(b, dir, `{
	  "added_tokens": [{"id": 9, "content": "<s>", "special": true}],
	  "model": {
	    "vocab": {"<unk>": 0, "hello": 1, "\u809d": 2, "<0xE4>": 3, "<0xBD>": 4, "<0xA0>": 5, "<s>": 9},
	    "merges": []
	  }
	}`)
	tok, err := Load(dir)
	if err != nil {
		b.Fatal(err)
	}
	ids := []int{9, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = tok.Decode(ids, true)
	}
}

type testHelper interface {
	Helper()
	Fatal(args ...any)
}

func writeTokenizer(t testHelper, dir, body string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, "tokenizer.json"), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}
