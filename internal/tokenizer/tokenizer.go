package tokenizer

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"unicode/utf8"

	"paddleocrvl-go/internal/jsonutil"
)

type Tokenizer struct {
	Vocab          map[string]int
	IDToToken      map[int]string
	Special        map[string]int
	SpecialIDs     map[int]bool
	idTokens       []string
	specialFlags   []bool
	byteValues     []byte
	byteFlags      []bool
	specialOrder   []string
	specialEntries []specialEntry
	byteTokens     [256]string
	idToByte       map[int]byte
	hasByteTokens  bool
	hasSpaceToken  bool
	ranks          map[pair]int
	cache          map[string][]int
	encodeCache    map[string][]int
	cacheMu        sync.RWMutex
}

type pair struct {
	A string
	B string
}

type specialEntry struct {
	token string
	id    int
}

type rawTokenizer struct {
	AddedTokens []struct {
		ID      int    `json:"id"`
		Content string `json:"content"`
		Special bool   `json:"special"`
	} `json:"added_tokens"`
	Model struct {
		Vocab  map[string]int `json:"vocab"`
		Merges [][]string     `json:"merges"`
	} `json:"model"`
}

const (
	maxCacheEntries      = 4096
	maxCacheableInputLen = 8192
	maxCacheableTokenLen = 8192
	maxTokenizerBytes    = 256 << 20
	maxTokenID           = 1 << 24
)

type CacheStats struct {
	NormalEntries int `json:"normal_entries"`
	EncodeEntries int `json:"encode_entries"`
}

func Load(dir string) (*Tokenizer, error) {
	path := filepath.Join(dir, "tokenizer.json")
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		return nil, err
	}
	if st.Size() > maxTokenizerBytes {
		return nil, fmt.Errorf("tokenizer.json too large: %d bytes exceeds %d", st.Size(), maxTokenizerBytes)
	}
	b, err := io.ReadAll(io.LimitReader(f, maxTokenizerBytes+1))
	if err != nil {
		return nil, err
	}
	if len(b) > maxTokenizerBytes {
		return nil, fmt.Errorf("tokenizer.json too large: exceeds %d bytes", maxTokenizerBytes)
	}
	if err := jsonutil.RejectDuplicateKeys(b, path); err != nil {
		return nil, err
	}
	var raw rawTokenizer
	if err := json.Unmarshal(b, &raw); err != nil {
		return nil, err
	}
	if err := validateRawTokenizer(raw); err != nil {
		return nil, err
	}
	t := &Tokenizer{
		Vocab:       raw.Model.Vocab,
		IDToToken:   make(map[int]string, len(raw.Model.Vocab)+len(raw.AddedTokens)),
		Special:     make(map[string]int, len(raw.AddedTokens)),
		SpecialIDs:  make(map[int]bool, len(raw.AddedTokens)),
		idToByte:    make(map[int]byte, 256),
		ranks:       make(map[pair]int, len(raw.Model.Merges)),
		cache:       map[string][]int{},
		encodeCache: map[string][]int{},
	}
	for tok, id := range t.Vocab {
		t.IDToToken[id] = tok
		if strings.Contains(tok, "\u809d") {
			t.hasSpaceToken = true
		}
		if by, ok := byteToken(tok); ok {
			t.idToByte[id] = by
			t.hasByteTokens = true
			t.byteTokens[by] = tok
		}
	}
	for _, at := range raw.AddedTokens {
		t.IDToToken[at.ID] = at.Content
		if strings.Contains(at.Content, "\u809d") {
			t.hasSpaceToken = true
		}
		if by, ok := byteToken(at.Content); ok {
			t.idToByte[at.ID] = by
			t.hasByteTokens = true
			t.byteTokens[by] = at.Content
		}
		if _, ok := t.Vocab[at.Content]; !ok {
			t.Vocab[at.Content] = at.ID
		}
		if at.Special {
			t.Special[at.Content] = at.ID
			t.SpecialIDs[at.ID] = true
			t.specialOrder = append(t.specialOrder, at.Content)
		}
	}
	sort.Slice(t.specialOrder, func(i, j int) bool {
		return len(t.specialOrder[i]) > len(t.specialOrder[j])
	})
	t.specialEntries = make([]specialEntry, len(t.specialOrder))
	for i, tok := range t.specialOrder {
		t.specialEntries[i] = specialEntry{token: tok, id: t.Special[tok]}
	}
	for i, m := range raw.Model.Merges {
		if len(m) == 2 {
			t.ranks[pair{m[0], m[1]}] = i
		}
	}
	for i := range t.byteTokens {
		if t.byteTokens[i] == "" {
			t.byteTokens[i] = makeByteToken(byte(i))
		}
	}
	t.buildIDTables()
	return t, nil
}

func validateRawTokenizer(raw rawTokenizer) error {
	if len(raw.Model.Vocab) == 0 {
		return fmt.Errorf("tokenizer vocab must not be empty")
	}
	idOwners := make(map[int]string, len(raw.Model.Vocab)+len(raw.AddedTokens))
	for tok, id := range raw.Model.Vocab {
		if tok == "" {
			return fmt.Errorf("tokenizer vocab contains empty token")
		}
		if !validTokenID(id) {
			return fmt.Errorf("tokenizer vocab token %q has invalid id %d", tok, id)
		}
		if owner, ok := idOwners[id]; ok && owner != tok {
			return fmt.Errorf("tokenizer id %d is assigned to both %q and %q", id, owner, tok)
		}
		idOwners[id] = tok
	}
	addedContent := make(map[string]int, len(raw.AddedTokens))
	for _, at := range raw.AddedTokens {
		if at.Content == "" {
			return fmt.Errorf("tokenizer added token content must not be empty")
		}
		if !validTokenID(at.ID) {
			return fmt.Errorf("tokenizer added token %q has invalid id %d", at.Content, at.ID)
		}
		if owner, ok := idOwners[at.ID]; ok && owner != at.Content {
			return fmt.Errorf("tokenizer id %d is assigned to both %q and %q", at.ID, owner, at.Content)
		}
		idOwners[at.ID] = at.Content
		if prev, ok := addedContent[at.Content]; ok && prev != at.ID {
			return fmt.Errorf("tokenizer added token %q has conflicting ids %d and %d", at.Content, prev, at.ID)
		}
		addedContent[at.Content] = at.ID
	}
	return nil
}

func validTokenID(id int) bool {
	return id >= 0 && id <= maxTokenID
}

func (t *Tokenizer) buildIDTables() {
	maxID := -1
	for id := range t.IDToToken {
		if id > maxID {
			maxID = id
		}
	}
	if maxID < 0 {
		return
	}
	t.idTokens = make([]string, maxID+1)
	t.specialFlags = make([]bool, maxID+1)
	t.byteValues = make([]byte, maxID+1)
	t.byteFlags = make([]bool, maxID+1)
	for id, tok := range t.IDToToken {
		if id >= 0 {
			t.idTokens[id] = tok
		}
	}
	for id := range t.SpecialIDs {
		if id >= 0 && id < len(t.specialFlags) {
			t.specialFlags[id] = true
		}
	}
	for id, by := range t.idToByte {
		if id >= 0 && id < len(t.byteFlags) {
			t.byteValues[id] = by
			t.byteFlags[id] = true
		}
	}
}

func (t *Tokenizer) Encode(s string) []int {
	return t.encode(s, true)
}

func (t *Tokenizer) EncodeReadOnly(s string) []int {
	return t.encode(s, false)
}

func (t *Tokenizer) encode(s string, copyCached bool) []int {
	if s == "" {
		return nil
	}
	original := s
	if len(original) <= maxCacheableInputLen {
		t.cacheMu.RLock()
		if cached, ok := t.encodeCache[original]; ok {
			t.cacheMu.RUnlock()
			if !copyCached {
				return cached
			}
			return append([]int(nil), cached...)
		}
		t.cacheMu.RUnlock()
	}
	if len(t.specialOrder) == 0 {
		ids := t.encodeNormal(s)
		if len(original) <= maxCacheableInputLen {
			t.storeEncodeCache(original, ids)
		}
		return ids
	}
	var ids []int
	for len(s) > 0 {
		if tok, id, ok := t.matchSpecial(s); ok {
			ids = append(ids, id)
			s = s[len(tok):]
			continue
		}
		next := len(s)
		for _, sp := range t.specialOrder {
			if idx := strings.Index(s, sp); idx >= 0 && idx < next {
				next = idx
			}
		}
		if next == 0 {
			_, size := utf8.DecodeRuneInString(s)
			next = size
		}
		ids = append(ids, t.encodeNormal(s[:next])...)
		s = s[next:]
	}
	if len(original) <= maxCacheableInputLen {
		t.storeEncodeCache(original, ids)
	}
	return ids
}

func (t *Tokenizer) Decode(ids []int, skipSpecial bool) string {
	if len(ids) == 0 {
		return ""
	}
	includeAll := !skipSpecial || len(t.SpecialIDs) == 0
	if len(ids) == 1 {
		id := ids[0]
		if !includeAll && t.isSpecialID(id) {
			return ""
		}
		if t.hasByteTokens {
			if by, ok := t.byteByID(id); ok {
				return decodedByteStrings[by]
			}
		}
		if tok, ok := t.tokenByID(id); ok {
			return t.normalizeDecodedSpaces(tok)
		}
		return ""
	}
	if !t.hasByteTokens {
		return t.decodeTokenStrings(ids, includeAll)
	}
	return t.decodeWithByteTokens(ids, includeAll)
}

func (t *Tokenizer) decodeWithByteTokens(ids []int, includeAll bool) string {
	var smallBytes [256]byte
	b := smallBytes[:0]
	if len(ids) > len(smallBytes) {
		b = make([]byte, 0, len(ids))
	}
	var sb strings.Builder
	hasBuilder := false
	for _, id := range ids {
		if !includeAll && t.isSpecialID(id) {
			continue
		}
		if by, ok := t.byteByID(id); ok {
			b = append(b, by)
			continue
		}
		tok, ok := t.tokenByID(id)
		if !ok {
			continue
		}
		if !hasBuilder {
			sb.Grow(len(ids) * 2)
			hasBuilder = true
		}
		if len(b) > 0 {
			_, _ = sb.Write(b)
			b = b[:0]
		}
		if t.hasSpaceToken {
			t.writeDecodedToken(&sb, tok)
		} else {
			sb.WriteString(tok)
		}
	}
	if !hasBuilder {
		return string(b)
	}
	if len(b) > 0 {
		_, _ = sb.Write(b)
	}
	return sb.String()
}

func (t *Tokenizer) decodeTokenStrings(ids []int, includeAll bool) string {
	var sb strings.Builder
	if !t.hasSpaceToken {
		if includeAll && len(ids) <= 16 {
			var toks [16]string
			n := 0
			total := 0
			for _, id := range ids {
				if tok, ok := t.tokenByID(id); ok {
					toks[n] = tok
					total += len(tok)
					n++
				}
			}
			sb.Grow(total)
			for i := 0; i < n; i++ {
				sb.WriteString(toks[i])
			}
			return sb.String()
		}
		sb.Grow(t.decodedLenHint(ids, includeAll))
		for _, id := range ids {
			if !includeAll && t.isSpecialID(id) {
				continue
			}
			if tok, ok := t.tokenByID(id); ok {
				sb.WriteString(tok)
			}
		}
	} else {
		sb.Grow(t.decodedLenHint(ids, includeAll))
		for _, id := range ids {
			if !includeAll && t.isSpecialID(id) {
				continue
			}
			if tok, ok := t.tokenByID(id); ok {
				t.writeDecodedToken(&sb, tok)
			}
		}
	}
	return sb.String()
}

func normalizeDecodedSpaces(s string) string {
	if strings.Contains(s, "\u809d") {
		return strings.ReplaceAll(s, "\u809d", " ")
	}
	return s
}

func (t *Tokenizer) normalizeDecodedSpaces(s string) string {
	if !t.hasSpaceToken {
		return s
	}
	return normalizeDecodedSpaces(s)
}

func (t *Tokenizer) writeDecodedToken(sb *strings.Builder, tok string) {
	if !t.hasSpaceToken || !strings.Contains(tok, "\u809d") {
		sb.WriteString(tok)
		return
	}
	for len(tok) > 0 {
		i := strings.Index(tok, "\u809d")
		if i < 0 {
			sb.WriteString(tok)
			return
		}
		sb.WriteString(tok[:i])
		sb.WriteByte(' ')
		tok = tok[i+len("\u809d"):]
	}
}

func (t *Tokenizer) encodeNormal(s string) []int {
	if strings.Contains(s, " ") {
		s = strings.ReplaceAll(s, " ", "肝")
	}
	cacheableInput := len(s) <= maxCacheableInputLen
	if cacheableInput {
		t.cacheMu.RLock()
		if cached, ok := t.cache[s]; ok {
			t.cacheMu.RUnlock()
			return append([]int(nil), cached...)
		}
		t.cacheMu.RUnlock()
	}
	var smallPieces [128]string
	pieces := smallPieces[:0]
	if len(s) > len(smallPieces) {
		pieces = make([]string, 0, len(s))
	}
	for _, r := range s {
		part := string(r)
		if _, ok := t.Vocab[part]; ok {
			pieces = append(pieces, part)
			continue
		}
		var buf [utf8.UTFMax]byte
		n := utf8.EncodeRune(buf[:], r)
		for _, by := range buf[:n] {
			pieces = append(pieces, t.byteTokens[by])
		}
	}
	if len(t.ranks) > 0 {
		for {
			bestRank := int(^uint(0) >> 1)
			best := -1
			for i := 0; i+1 < len(pieces); i++ {
				if rank, ok := t.ranks[pair{pieces[i], pieces[i+1]}]; ok && rank < bestRank {
					bestRank = rank
					best = i
				}
			}
			if best < 0 {
				break
			}
			merged := pieces[best] + pieces[best+1]
			pieces[best] = merged
			copy(pieces[best+1:], pieces[best+2:])
			pieces = pieces[:len(pieces)-1]
		}
	}
	ids := make([]int, 0, len(pieces))
	for _, p := range pieces {
		if id, ok := t.Vocab[p]; ok {
			ids = append(ids, id)
		} else {
			ids = append(ids, 0)
		}
	}
	if cacheableInput && len(ids) <= maxCacheableTokenLen {
		t.cacheMu.Lock()
		if len(t.cache) >= maxCacheEntries {
			t.cache = make(map[string][]int, maxCacheEntries)
		}
		t.cache[s] = append([]int(nil), ids...)
		t.cacheMu.Unlock()
	}
	return ids
}

func (t *Tokenizer) storeEncodeCache(s string, ids []int) {
	if len(ids) > maxCacheableTokenLen {
		return
	}
	t.cacheMu.Lock()
	if len(t.encodeCache) >= maxCacheEntries {
		t.encodeCache = make(map[string][]int, maxCacheEntries)
	}
	t.encodeCache[s] = append([]int(nil), ids...)
	t.cacheMu.Unlock()
}

func (t *Tokenizer) CacheStats() CacheStats {
	t.cacheMu.RLock()
	defer t.cacheMu.RUnlock()
	return CacheStats{NormalEntries: len(t.cache), EncodeEntries: len(t.encodeCache)}
}

func (t *Tokenizer) matchSpecial(s string) (string, int, bool) {
	for _, sp := range t.specialEntries {
		if strings.HasPrefix(s, sp.token) {
			return sp.token, sp.id, true
		}
	}
	return "", 0, false
}

func (t *Tokenizer) decodedLenHint(ids []int, includeAll bool) int {
	n := 0
	if includeAll && !t.hasByteTokens {
		for _, id := range ids {
			if tok, ok := t.tokenByID(id); ok {
				n += len(tok)
			}
		}
		return n
	}
	for _, id := range ids {
		if !includeAll && t.isSpecialID(id) {
			continue
		}
		if _, ok := t.byteByID(id); ok {
			n++
			continue
		}
		if tok, ok := t.tokenByID(id); ok {
			n += len(tok)
		}
	}
	return n
}

func (t *Tokenizer) tokenByID(id int) (string, bool) {
	if id >= 0 && id < len(t.idTokens) {
		tok := t.idTokens[id]
		return tok, tok != ""
	}
	tok, ok := t.IDToToken[id]
	return tok, ok
}

func (t *Tokenizer) isSpecialID(id int) bool {
	if id >= 0 && id < len(t.specialFlags) {
		return t.specialFlags[id]
	}
	return t.SpecialIDs[id]
}

func (t *Tokenizer) byteByID(id int) (byte, bool) {
	if id >= 0 && id < len(t.byteFlags) {
		return t.byteValues[id], t.byteFlags[id]
	}
	by, ok := t.idToByte[id]
	return by, ok
}

var decodedByteStrings = func() [256]string {
	var out [256]string
	for i := range out {
		out[i] = string([]byte{byte(i)})
	}
	return out
}()

func makeByteToken(v byte) string {
	const hex = "0123456789ABCDEF"
	return string([]byte{'<', '0', 'x', hex[v>>4], hex[v&0x0F], '>'})
}

func byteToken(tok string) (byte, bool) {
	if len(tok) != 6 || !strings.HasPrefix(tok, "<0x") || tok[5] != '>' {
		return 0, false
	}
	var v byte
	for i := 3; i < 5; i++ {
		c := tok[i]
		v <<= 4
		switch {
		case c >= '0' && c <= '9':
			v |= c - '0'
		case c >= 'A' && c <= 'F':
			v |= c - 'A' + 10
		case c >= 'a' && c <= 'f':
			v |= c - 'a' + 10
		default:
			return 0, false
		}
	}
	return v, true
}
