package main

import (
	"bytes"
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/base64"
	"encoding/csv"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"html/template"
	"io"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"paddleocrvl-go/internal/backend"
)

type adminConfigFile struct {
	AdminUser       string          `json:"admin_user"`
	PasswordSalt    string          `json:"password_salt"`
	PasswordHash    string          `json:"password_hash"`
	APIKeySalt      string          `json:"api_key_salt,omitempty"`
	APIKeyHash      string          `json:"api_key_hash,omitempty"`
	APIKeyPreview   string          `json:"api_key_preview,omitempty"`
	APIKeys         []managedAPIKey `json:"api_keys"`
	ModelDir        string          `json:"model_dir"`
	PostProcessDir  string          `json:"post_process_dir"`
	UpdatedAt       string          `json:"updated_at"`
	RestartRequired bool            `json:"restart_required"`
}

type managedAPIKey struct {
	ID         string `json:"id"`
	Name       string `json:"name"`
	Salt       string `json:"salt"`
	Hash       string `json:"hash"`
	Preview    string `json:"preview"`
	Quota      int64  `json:"quota"`
	Used       int64  `json:"used"`
	RatePerMin int64  `json:"rate_per_minute,omitempty"`
	Disabled   bool   `json:"disabled"`
	LastUsedAt string `json:"last_used_at,omitempty"`
	LastUsedIP string `json:"last_used_ip,omitempty"`
	CreatedAt  string `json:"created_at"`
	UpdatedAt  string `json:"updated_at"`
}

type adminState struct {
	mu       sync.RWMutex
	path     string
	cfg      adminConfigFile
	sessions map[string]time.Time
	rates    map[string]apiKeyRateWindow
	logins   map[string]loginFailureWindow
	audit    []auditEntry
}

type apiKeyRateWindow struct {
	start time.Time
	count int64
}

type loginFailureWindow struct {
	start       time.Time
	count       int
	lockedUntil time.Time
}

type apiKeyAuditInfo struct {
	ID      string `json:"id"`
	Name    string `json:"name"`
	Preview string `json:"preview"`
}

type auditEntry struct {
	Time       string `json:"time"`
	KeyID      string `json:"key_id,omitempty"`
	KeyName    string `json:"key_name,omitempty"`
	KeyPreview string `json:"key_preview,omitempty"`
	ClientIP   string `json:"client_ip"`
	Method     string `json:"method"`
	Path       string `json:"path"`
	Status     int    `json:"status"`
	DurationMS int64  `json:"duration_ms"`
	Error      string `json:"error,omitempty"`
}

type adminConfigResponse struct {
	Initialized     bool   `json:"initialized"`
	AdminUser       string `json:"admin_user"`
	ModelDir        string `json:"model_dir"`
	PostProcessDir  string `json:"post_process_dir"`
	RestartRequired bool   `json:"restart_required"`
	UpdatedAt       string `json:"updated_at"`
}

type apiKeyResponse struct {
	ID         string `json:"id"`
	Name       string `json:"name"`
	Preview    string `json:"preview"`
	Quota      int64  `json:"quota"`
	Used       int64  `json:"used"`
	Remaining  int64  `json:"remaining"`
	RatePerMin int64  `json:"rate_per_minute"`
	Disabled   bool   `json:"disabled"`
	LastUsedAt string `json:"last_used_at"`
	LastUsedIP string `json:"last_used_ip"`
	CreatedAt  string `json:"created_at"`
	UpdatedAt  string `json:"updated_at"`
}

type adminOverviewResponse struct {
	Admin         adminConfigResponse `json:"admin"`
	APIBaseURL    string              `json:"api_base_url"`
	CurrentHost   string              `json:"current_host"`
	ModelDir      string              `json:"model_dir"`
	Stats         statsResponse       `json:"stats"`
	Ready         readyResponse       `json:"ready"`
	ReadyHTTPCode int                 `json:"ready_http_code"`
}

type adminConfigBackup struct {
	Version    int             `json:"version"`
	ExportedAt string          `json:"exported_at"`
	Config     adminConfigFile `json:"config"`
}

var (
	errAPIKeyDisabled = errors.New("API key disabled")
	errAPIKeyQuota    = errors.New("API key quota exceeded")
	errAPIKeyRate     = errors.New("API key rate limit exceeded")
	errAPIKeyInvalid  = errors.New("invalid API key")
)

func loadAdminState(path string) *adminState {
	st := &adminState{path: path, sessions: make(map[string]time.Time), rates: make(map[string]apiKeyRateWindow), logins: make(map[string]loginFailureWindow)}
	b, err := os.ReadFile(path)
	if err == nil {
		_ = json.Unmarshal(b, &st.cfg)
	}
	return st
}

func (a *adminState) initialized() bool {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.cfg.PasswordHash != "" && a.cfg.AdminUser != ""
}

func (a *adminState) response(fallbackModelDir string) adminConfigResponse {
	a.mu.RLock()
	defer a.mu.RUnlock()
	modelDir := a.cfg.ModelDir
	if modelDir == "" {
		modelDir = fallbackModelDir
	}
	return adminConfigResponse{
		Initialized:     a.cfg.PasswordHash != "" && a.cfg.AdminUser != "",
		AdminUser:       a.cfg.AdminUser,
		ModelDir:        modelDir,
		PostProcessDir:  a.cfg.PostProcessDir,
		RestartRequired: a.cfg.RestartRequired,
		UpdatedAt:       a.cfg.UpdatedAt,
	}
}

func (a *adminState) keyResponses() []apiKeyResponse {
	a.mu.RLock()
	defer a.mu.RUnlock()
	out := make([]apiKeyResponse, 0, len(a.cfg.APIKeys))
	for _, key := range a.cfg.APIKeys {
		out = append(out, key.response())
	}
	return out
}

func (k managedAPIKey) response() apiKeyResponse {
	remaining := int64(-1)
	if k.Quota > 0 {
		remaining = max(0, k.Quota-k.Used)
	}
	return apiKeyResponse{
		ID:         k.ID,
		Name:       k.Name,
		Preview:    k.Preview,
		Quota:      k.Quota,
		Used:       k.Used,
		Remaining:  remaining,
		RatePerMin: k.RatePerMin,
		Disabled:   k.Disabled,
		LastUsedAt: k.LastUsedAt,
		LastUsedIP: k.LastUsedIP,
		CreatedAt:  k.CreatedAt,
		UpdatedAt:  k.UpdatedAt,
	}
}

func (a *adminState) saveLocked() error {
	a.cfg.UpdatedAt = time.Now().UTC().Format(time.RFC3339)
	b, err := json.MarshalIndent(a.cfg, "", "  ")
	if err != nil {
		return err
	}
	dir := filepath.Dir(a.path)
	if dir != "." && dir != "" {
		if err := os.MkdirAll(dir, 0700); err != nil {
			return err
		}
	}
	return writeFileAtomic(a.path, b, 0600)
}

func writeFileAtomic(path string, data []byte, perm os.FileMode) error {
	dir := filepath.Dir(path)
	f, err := os.CreateTemp(dir, filepath.Base(path)+".tmp-*")
	if err != nil {
		return err
	}
	tmp := f.Name()
	defer func() {
		_ = os.Remove(tmp)
	}()
	if _, err := f.Write(data); err != nil {
		_ = f.Close()
		return err
	}
	if err := f.Chmod(perm); err != nil {
		_ = f.Close()
		return err
	}
	if err := f.Sync(); err != nil {
		_ = f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	if err := os.Rename(tmp, path); err != nil {
		st, statErr := os.Stat(path)
		if statErr != nil {
			return err
		}
		if st.IsDir() {
			return err
		}
		if removeErr := os.Remove(path); removeErr != nil {
			return err
		}
		if renameErr := os.Rename(tmp, path); renameErr != nil {
			return renameErr
		}
	}
	return nil
}

func (a *adminState) setPasswordLocked(user, password string) error {
	user, err := normalizeAdminUser(user)
	if err != nil {
		return err
	}
	if len(password) < 8 {
		return fmt.Errorf("password must be at least 8 characters")
	}
	salt, hash, err := saltedHash(password)
	if err != nil {
		return err
	}
	a.cfg.AdminUser = user
	a.cfg.PasswordSalt = salt
	a.cfg.PasswordHash = hash
	return nil
}

func normalizeAdminUser(user string) (string, error) {
	user = strings.TrimSpace(user)
	if user == "" {
		return "", fmt.Errorf("admin_user is required")
	}
	if len(user) > 80 {
		return "", fmt.Errorf("admin_user must be at most 80 bytes")
	}
	return user, nil
}

func (a *adminState) createAPIKeyLocked(name string, quota, ratePerMin int64) (string, managedAPIKey, error) {
	name, err := normalizeAPIKeyName(name)
	if err != nil {
		return "", managedAPIKey{}, err
	}
	if name == "" {
		name = "default"
	}
	apiKey, salt, hash, preview, err := newAPIKeySecret()
	if err != nil {
		return "", managedAPIKey{}, err
	}
	now := time.Now().UTC().Format(time.RFC3339)
	id, err := randomToken(12)
	if err != nil {
		return "", managedAPIKey{}, err
	}
	key := managedAPIKey{
		ID:         id,
		Name:       name,
		Salt:       salt,
		Hash:       hash,
		Preview:    preview,
		Quota:      quota,
		RatePerMin: ratePerMin,
		CreatedAt:  now,
		UpdatedAt:  now,
	}
	a.cfg.APIKeys = append(a.cfg.APIKeys, key)
	return apiKey, key, nil
}

func newAPIKeySecret() (apiKey, salt, hash, preview string, err error) {
	apiKey, err = randomToken(32)
	if err != nil {
		return "", "", "", "", err
	}
	salt, hash, err = saltedHash(apiKey)
	if err != nil {
		return "", "", "", "", err
	}
	return apiKey, salt, hash, previewSecret(apiKey), nil
}

func normalizeAPIKeyName(name string) (string, error) {
	name = strings.TrimSpace(name)
	if len(name) > 80 {
		return "", fmt.Errorf("api key name must be at most 80 bytes")
	}
	return name, nil
}

func (a *adminState) checkPassword(user, password string) bool {
	a.mu.RLock()
	defer a.mu.RUnlock()
	if user != a.cfg.AdminUser || a.cfg.PasswordHash == "" {
		return false
	}
	return checkSaltedHash(password, a.cfg.PasswordSalt, a.cfg.PasswordHash)
}

func (a *adminState) loginLocked(ip string, now time.Time) bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	win := a.logins[ip]
	if !win.lockedUntil.IsZero() && now.Before(win.lockedUntil) {
		return true
	}
	if !win.lockedUntil.IsZero() && !now.Before(win.lockedUntil) {
		delete(a.logins, ip)
	}
	return false
}

func (a *adminState) recordLoginFailure(ip string, now time.Time) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.logins == nil {
		a.logins = make(map[string]loginFailureWindow)
	}
	win := a.logins[ip]
	if win.start.IsZero() || now.Sub(win.start) > 5*time.Minute {
		win = loginFailureWindow{start: now}
	}
	win.count++
	if win.count >= 5 {
		win.lockedUntil = now.Add(10 * time.Minute)
	}
	a.logins[ip] = win
}

func (a *adminState) clearLoginFailures(ip string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	delete(a.logins, ip)
}

func (a *adminState) consumeAPIKey(apiKey, clientIP string) (apiKeyAuditInfo, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if len(a.cfg.APIKeys) == 0 && a.cfg.APIKeyHash == "" {
		return apiKeyAuditInfo{}, errAPIKeyInvalid
	}
	if a.cfg.APIKeyHash != "" && checkSaltedHash(apiKey, a.cfg.APIKeySalt, a.cfg.APIKeyHash) {
		oldCfg := a.cfg
		now := time.Now().UTC().Format(time.RFC3339)
		id, err := randomToken(12)
		if err != nil {
			return apiKeyAuditInfo{Name: "legacy", Preview: "legacy"}, err
		}
		preview := a.cfg.APIKeyPreview
		if preview == "" {
			preview = previewSecret(apiKey)
		}
		key := managedAPIKey{
			ID:         id,
			Name:       "legacy",
			Salt:       a.cfg.APIKeySalt,
			Hash:       a.cfg.APIKeyHash,
			Preview:    preview,
			Used:       1,
			LastUsedAt: now,
			LastUsedIP: clientIP,
			CreatedAt:  now,
			UpdatedAt:  now,
		}
		a.cfg.APIKeys = append(a.cfg.APIKeys, key)
		a.cfg.APIKeySalt = ""
		a.cfg.APIKeyHash = ""
		a.cfg.APIKeyPreview = ""
		if err := a.saveLocked(); err != nil {
			a.cfg = oldCfg
			return key.auditInfo(), err
		}
		return key.auditInfo(), nil
	}
	for i := range a.cfg.APIKeys {
		key := &a.cfg.APIKeys[i]
		if !checkSaltedHash(apiKey, key.Salt, key.Hash) {
			continue
		}
		if key.Disabled {
			return key.auditInfo(), errAPIKeyDisabled
		}
		if key.Quota > 0 && key.Used >= key.Quota {
			return key.auditInfo(), errAPIKeyQuota
		}
		t := time.Now()
		oldKey := *key
		oldRate, hadRate := a.rates[key.ID]
		if key.RatePerMin > 0 && !a.allowRateLocked(key.ID, key.RatePerMin, t) {
			return key.auditInfo(), errAPIKeyRate
		}
		now := t.UTC().Format(time.RFC3339)
		key.Used++
		key.LastUsedAt = now
		key.LastUsedIP = clientIP
		key.UpdatedAt = now
		if err := a.saveLocked(); err != nil {
			*key = oldKey
			if hadRate {
				a.rates[key.ID] = oldRate
			} else {
				delete(a.rates, key.ID)
			}
			return key.auditInfo(), err
		}
		return key.auditInfo(), nil
	}
	return apiKeyAuditInfo{}, errAPIKeyInvalid
}

func (k managedAPIKey) auditInfo() apiKeyAuditInfo {
	return apiKeyAuditInfo{ID: k.ID, Name: k.Name, Preview: k.Preview}
}

func (a *adminState) recordAudit(e auditEntry) {
	a.mu.Lock()
	defer a.mu.Unlock()
	const maxAuditEntries = 100
	if len(a.audit) >= maxAuditEntries {
		copy(a.audit, a.audit[1:])
		a.audit[len(a.audit)-1] = e
		return
	}
	a.audit = append(a.audit, e)
}

func (a *adminState) auditEntries() []auditEntry {
	a.mu.RLock()
	defer a.mu.RUnlock()
	out := make([]auditEntry, len(a.audit))
	for i := range a.audit {
		out[len(a.audit)-1-i] = a.audit[i]
	}
	return out
}

func (a *adminState) clearAudit() {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.audit = nil
}

func (a *adminState) allowRateLocked(id string, limit int64, now time.Time) bool {
	if a.rates == nil {
		a.rates = make(map[string]apiKeyRateWindow)
	}
	win := a.rates[id]
	if win.start.IsZero() || now.Sub(win.start) >= time.Minute {
		a.rates[id] = apiKeyRateWindow{start: now, count: 1}
		return true
	}
	if win.count >= limit {
		return false
	}
	win.count++
	a.rates[id] = win
	return true
}

func (a *adminState) createSession() (string, error) {
	token, err := randomToken(32)
	if err != nil {
		return "", err
	}
	now := time.Now()
	a.mu.Lock()
	defer a.mu.Unlock()
	for existing, expires := range a.sessions {
		if now.After(expires) {
			delete(a.sessions, existing)
		}
	}
	a.sessions[token] = now.Add(12 * time.Hour)
	return token, nil
}

func (a *adminState) deleteSession(token string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	delete(a.sessions, token)
}

func (a *adminState) validSession(token string) bool {
	if token == "" {
		return false
	}
	now := time.Now()
	a.mu.Lock()
	defer a.mu.Unlock()
	expires, ok := a.sessions[token]
	if !ok {
		return false
	}
	if now.After(expires) {
		delete(a.sessions, token)
		return false
	}
	a.sessions[token] = now.Add(12 * time.Hour)
	return true
}

func saltedHash(secret string) (string, string, error) {
	saltBytes := make([]byte, 16)
	if _, err := rand.Read(saltBytes); err != nil {
		return "", "", err
	}
	salt := base64.RawURLEncoding.EncodeToString(saltBytes)
	sum := sha256.Sum256([]byte(salt + ":" + secret))
	return salt, hex.EncodeToString(sum[:]), nil
}

func checkSaltedHash(secret, salt, want string) bool {
	sum := sha256.Sum256([]byte(salt + ":" + secret))
	got := hex.EncodeToString(sum[:])
	return subtle.ConstantTimeCompare([]byte(got), []byte(want)) == 1
}

func randomToken(n int) (string, error) {
	b := make([]byte, n)
	if _, err := rand.Read(b); err != nil {
		return "", err
	}
	return base64.RawURLEncoding.EncodeToString(b), nil
}

func previewSecret(s string) string {
	s = strings.TrimSpace(s)
	if len(s) <= 8 {
		return "********"
	}
	return s[:4] + "..." + s[len(s)-4:]
}

func csvCell(s string) string {
	if s == "" {
		return s
	}
	switch s[0] {
	case '=', '+', '-', '@':
		return "'" + s
	default:
		return s
	}
}

func (s *server) adminPage(w http.ResponseWriter, r *http.Request) {
	setAdminSecurityHeaders(w)
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	_ = adminTemplate.Execute(w, nil)
}

func (s *server) docPage(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	_ = docTemplate.Execute(w, map[string]string{"BaseURL": adminBaseURL(r)})
}

func (s *server) openapiJSON(w http.ResponseWriter, r *http.Request) {
	writeOpenAPIJSONForRequest(w, r)
}

func (s *server) llmsTXT(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	w.WriteHeader(http.StatusOK)
	writeLLMsTXTForRequest(w, r)
}

const llmsTXTBasePlaceholder = "__PADDLEOCRVL_BASE_URL__"

var llmsTXTParts = strings.Split(`# PaddleOCR-VL Local Inference API

Base URL: __PADDLEOCRVL_BASE_URL__
OpenAPI: __PADDLEOCRVL_BASE_URL__/doc/openapi.json
Human docs: __PADDLEOCRVL_BASE_URL__/doc
Admin console: __PADDLEOCRVL_BASE_URL__/admin

Authentication:
Use an API key generated in the admin console.
Send it as:
Authorization: Bearer <API_KEY>
or:
X-API-Key: <API_KEY>

Recommended operations:
- getHealth: GET /health
- getReady: GET /ready
- getStats: GET /stats
- ocrImage: POST /v1/ocr multipart/form-data
- generate: POST /v1/generate application/json
- batchGenerate: POST /v1/batch application/json

Tasks:
ocr, table, formula, chart

OCR multipart example:
curl -X POST "__PADDLEOCRVL_BASE_URL__/v1/ocr" \
  -H "Authorization: Bearer <API_KEY>" \
  -F "image=@page.png" \
  -F "task=ocr" \
  -F "max_new_tokens=1024"

JSON OCR example:
curl -X POST "__PADDLEOCRVL_BASE_URL__/v1/generate" \
  -H "Authorization: Bearer <API_KEY>" \
  -H "Content-Type: application/json" \
  -d "{\"task\":\"ocr\",\"image_base64\":\"<base64-image>\",\"max_new_tokens\":1024,\"decode\":true,\"decode_generated_only\":true,\"skip_special\":true}"

Batch example:
curl -X POST "__PADDLEOCRVL_BASE_URL__/v1/batch" \
  -H "Authorization: Bearer <API_KEY>" \
  -H "Content-Type: application/json" \
  -d "{\"requests\":[{\"prompt\":\"<|begin_of_sentence|>hello\",\"max_new_tokens\":1},{\"task\":\"ocr\",\"image_path\":\"D:\\docs\\page.png\",\"decode\":true,\"decode_generated_only\":true,\"skip_special\":true}]}"

Responses:
Generate returns tokens, prompt_tokens, generated_tokens, and optional text.
Errors are JSON: {"error":"message"}.
`, llmsTXTBasePlaceholder)
var llmsTXTBufferPool = sync.Pool{New: func() any {
	b := make([]byte, 0, len(llmsTXTParts[0])+len(llmsTXTBasePlaceholder)*7)
	return &b
}}

func writeLLMsTXT(w io.Writer, base string) {
	p := llmsTXTBufferPool.Get().(*[]byte)
	buf := (*p)[:0]
	need := 0
	for _, part := range llmsTXTParts {
		need += len(part)
	}
	need += len(base) * (len(llmsTXTParts) - 1)
	if cap(buf) < need {
		buf = make([]byte, 0, need)
	}
	for i, part := range llmsTXTParts {
		if i > 0 {
			buf = append(buf, base...)
		}
		buf = append(buf, part...)
	}
	_, _ = w.Write(buf)
	*p = buf
	llmsTXTBufferPool.Put(p)
}

func writeLLMsTXTForRequest(w io.Writer, r *http.Request) {
	scheme := requestScheme(r)
	host := requestHost(r)
	p := llmsTXTBufferPool.Get().(*[]byte)
	buf := (*p)[:0]
	need := 0
	for _, part := range llmsTXTParts {
		need += len(part)
	}
	need += (len(scheme) + 3 + len(host)) * (len(llmsTXTParts) - 1)
	if cap(buf) < need {
		buf = make([]byte, 0, need)
	}
	for i, part := range llmsTXTParts {
		if i > 0 {
			buf = append(buf, scheme...)
			buf = append(buf, "://"...)
			buf = append(buf, host...)
		}
		buf = append(buf, part...)
	}
	_, _ = w.Write(buf)
	*p = buf
	llmsTXTBufferPool.Put(p)
}

const openAPIBasePlaceholder = "__PADDLEOCRVL_BASE_URL__"
const adminRequestLimit int64 = 4 << 20

var openAPIJSONParts = buildOpenAPIJSONParts()
var openAPIResponseBufferPool = sync.Pool{New: func() any {
	b := make([]byte, 0, len(openAPIJSONParts[0])+len(openAPIBasePlaceholder)+len(openAPIJSONParts[1])+1)
	return &b
}}

func buildOpenAPIJSONParts() [2][]byte {
	raw, err := json.Marshal(buildOpenAPISpec())
	if err != nil {
		panic(err)
	}
	needle, err := json.Marshal(openAPIBasePlaceholder)
	if err != nil {
		panic(err)
	}
	idx := bytes.Index(raw, needle)
	if idx < 0 {
		panic("openapi base URL placeholder missing")
	}
	return [2][]byte{raw[:idx], raw[idx+len(needle):]}
}

func writeOpenAPIJSON(w http.ResponseWriter, base string) {
	w.Header()["Content-Type"] = contentTypeJSONHeader
	w.WriteHeader(http.StatusOK)
	p := openAPIResponseBufferPool.Get().(*[]byte)
	buf := appendOpenAPIJSON((*p)[:0], base)
	_, _ = w.Write(buf)
	*p = buf
	openAPIResponseBufferPool.Put(p)
}

func writeOpenAPIJSONForRequest(w http.ResponseWriter, r *http.Request) {
	w.Header()["Content-Type"] = contentTypeJSONHeader
	w.WriteHeader(http.StatusOK)
	p := openAPIResponseBufferPool.Get().(*[]byte)
	buf := (*p)[:0]
	scheme := requestScheme(r)
	host := requestHost(r)
	need := len(openAPIJSONParts[0]) + len(scheme) + len(host) + 5 + len(openAPIJSONParts[1]) + 1
	if cap(buf) < need {
		buf = make([]byte, 0, need)
	}
	buf = append(buf, openAPIJSONParts[0]...)
	buf = append(buf, '"')
	buf = appendJSONStringPart(buf, scheme)
	buf = append(buf, "://"...)
	buf = appendJSONStringPart(buf, host)
	buf = append(buf, '"')
	buf = append(buf, openAPIJSONParts[1]...)
	buf = append(buf, '\n')
	_, _ = w.Write(buf)
	*p = buf
	openAPIResponseBufferPool.Put(p)
}

func appendOpenAPIJSON(buf []byte, base string) []byte {
	need := len(openAPIJSONParts[0]) + len(base) + 2 + len(openAPIJSONParts[1]) + 1
	if cap(buf) < need {
		buf = make([]byte, 0, need)
	}
	buf = append(buf, openAPIJSONParts[0]...)
	buf = strconv.AppendQuote(buf, base)
	buf = append(buf, openAPIJSONParts[1]...)
	buf = append(buf, '\n')
	return buf
}

func appendJSONStringPart(buf []byte, s string) []byte {
	const hex = "0123456789abcdef"
	for i := 0; i < len(s); i++ {
		c := s[i]
		switch c {
		case '\\', '"':
			buf = append(buf, '\\', c)
		case '\n':
			buf = append(buf, '\\', 'n')
		case '\r':
			buf = append(buf, '\\', 'r')
		case '\t':
			buf = append(buf, '\\', 't')
		default:
			if c < 0x20 {
				buf = append(buf, '\\', 'u', '0', '0', hex[c>>4], hex[c&0xf])
			} else {
				buf = append(buf, c)
			}
		}
	}
	return buf
}

func buildOpenAPISpec() map[string]any {
	return map[string]any{
		"openapi": "3.1.0",
		"info": map[string]any{
			"title":       "PaddleOCR-VL Local Inference API",
			"version":     "1.0.0",
			"description": "Local OCR/document parsing inference service. Generate API keys in /admin.",
		},
		"servers": []map[string]string{{"url": openAPIBasePlaceholder}},
		"tags": []map[string]string{
			{"name": "status", "description": "Health, readiness, and runtime statistics."},
			{"name": "inference", "description": "OCR and generation inference endpoints."},
		},
		"security": []map[string]any{
			{"bearerAuth": []string{}},
			{"apiKeyHeader": []string{}},
		},
		"paths": map[string]any{
			"/health": map[string]any{
				"get": map[string]any{
					"operationId": "getHealth",
					"tags":        []string{"status"},
					"summary":     "Health check",
					"security":    []map[string]any{},
					"responses": map[string]any{
						"200": jsonResponseRef("HealthResponse"),
					},
				},
			},
			"/ready": map[string]any{
				"get": map[string]any{
					"operationId": "getReady",
					"tags":        []string{"status"},
					"summary":     "Readiness check",
					"security":    []map[string]any{},
					"responses": map[string]any{
						"200": jsonResponseRef("ReadyResponse"),
						"503": jsonResponseRef("ReadyResponse"),
					},
				},
			},
			"/stats": map[string]any{
				"get": map[string]any{
					"operationId": "getStats",
					"tags":        []string{"status"},
					"summary":     "Runtime stats",
					"security":    []map[string]any{},
					"responses": map[string]any{
						"200": jsonResponseRef("StatsResponse"),
					},
				},
			},
			"/v1/ocr": map[string]any{
				"post": map[string]any{
					"operationId": "ocrImage",
					"tags":        []string{"inference"},
					"summary":     "OCR from multipart image upload",
					"requestBody": map[string]any{
						"required": true,
						"content": map[string]any{
							"multipart/form-data": map[string]any{
								"schema": map[string]any{
									"type":     "object",
									"required": []string{"image"},
									"properties": map[string]any{
										"image":          map[string]any{"type": "string", "format": "binary"},
										"task":           map[string]any{"type": "string", "enum": []string{"ocr", "table", "formula", "chart"}, "default": "ocr"},
										"max_new_tokens": map[string]any{"type": "integer", "default": 1024, "minimum": 1},
									},
								},
								"encoding": map[string]any{
									"image": map[string]any{"contentType": "image/png, image/jpeg, image/webp"},
								},
								"examples": map[string]any{
									"ocr": map[string]any{
										"summary": "OCR image upload",
										"value": map[string]any{
											"task":           "ocr",
											"max_new_tokens": 1024,
										},
									},
								},
							},
						},
					},
					"responses": standardGenerateResponses(),
				},
			},
			"/v1/generate": map[string]any{
				"post": map[string]any{
					"operationId": "generate",
					"tags":        []string{"inference"},
					"summary":     "Generate from JSON prompt, tokens, task, image path, or base64 image",
					"requestBody": map[string]any{
						"required": true,
						"content": map[string]any{
							"application/json": map[string]any{
								"schema": schemaRef("GenerateRequest"),
								"examples": map[string]any{
									"ocrBase64": map[string]any{
										"summary": "OCR with base64 image",
										"value": map[string]any{
											"task":                  "ocr",
											"image_base64":          "<base64-image>",
											"max_new_tokens":        1024,
											"decode":                true,
											"decode_generated_only": true,
											"skip_special":          true,
										},
									},
									"textPrompt": map[string]any{
										"summary": "Text prompt generation",
										"value": map[string]any{
											"prompt":         "<|begin_of_sentence|>hello",
											"max_new_tokens": 16,
											"decode":         true,
										},
									},
								},
							},
						},
					},
					"responses": standardGenerateResponses(),
				},
			},
			"/v1/batch": map[string]any{
				"post": map[string]any{
					"operationId": "batchGenerate",
					"tags":        []string{"inference"},
					"summary":     "Batch generate; response order matches request order",
					"requestBody": map[string]any{
						"required": true,
						"content": map[string]any{
							"application/json": map[string]any{
								"schema": schemaRef("BatchRequest"),
								"examples": map[string]any{
									"mixed": map[string]any{
										"summary": "Mixed batch",
										"value": map[string]any{
											"requests": []map[string]any{
												{
													"prompt":         "<|begin_of_sentence|>hello",
													"max_new_tokens": 1,
												},
												{
													"task":                  "ocr",
													"image_path":            `D:\docs\page.png`,
													"max_new_tokens":        1024,
													"decode":                true,
													"decode_generated_only": true,
													"skip_special":          true,
												},
											},
										},
									},
								},
							},
						},
					},
					"responses": map[string]any{
						"200": jsonResponseRef("BatchResponse"),
						"400": jsonResponseRef("ErrorResponse"),
						"401": jsonResponseRef("ErrorResponse"),
						"403": jsonResponseRef("ErrorResponse"),
						"429": jsonResponseRef("ErrorResponse"),
						"408": jsonResponseRef("ErrorResponse"),
						"504": jsonResponseRef("ErrorResponse"),
					},
				},
			},
		},
		"components": map[string]any{
			"securitySchemes": map[string]any{
				"bearerAuth":   map[string]any{"type": "http", "scheme": "bearer"},
				"apiKeyHeader": map[string]any{"type": "apiKey", "in": "header", "name": "X-API-Key"},
			},
			"schemas": openapiSchemas(),
		},
	}
}

func schemaRef(name string) map[string]string {
	return map[string]string{"$ref": "#/components/schemas/" + name}
}

func jsonResponseRef(name string) map[string]any {
	out := map[string]any{
		"description": name,
		"content": map[string]any{
			"application/json": map[string]any{
				"schema": schemaRef(name),
			},
		},
	}
	if ex := responseExample(name); ex != nil {
		out["content"].(map[string]any)["application/json"].(map[string]any)["examples"] = map[string]any{
			"default": map[string]any{"value": ex},
		}
	}
	return out
}

func responseExample(name string) any {
	switch name {
	case "GenerateResponse":
		return map[string]any{
			"tokens":           []int{100273, 1234, 5678},
			"prompt_tokens":    12,
			"generated_tokens": 32,
			"text":             "识别结果",
		}
	case "BatchResponse":
		return map[string]any{
			"responses": []map[string]any{
				{
					"tokens":           []int{100273, 1234},
					"prompt_tokens":    8,
					"generated_tokens": 1,
					"text":             "hello",
				},
			},
			"items":            1,
			"generated_tokens": 1,
		}
	case "HealthResponse":
		return map[string]any{
			"status":        "ok",
			"quantization":  "q8",
			"weight_path":   `D:\models\PaddleOCR-VL\model-q8.gguf`,
			"weight_source": "existing_gguf",
			"backend":       "cpu",
			"vision_loaded": true,
		}
	case "ReadyResponse":
		return map[string]any{
			"status":          "ready",
			"quantization":    "q8",
			"weight_path":     `D:\models\PaddleOCR-VL\model-q8.gguf`,
			"weight_source":   "existing_gguf",
			"backend":         "cpu",
			"vision_loaded":   true,
			"concurrency":     1,
			"in_flight":       0,
			"available_slots": 1,
		}
	case "ErrorResponse":
		return map[string]any{"error": "missing API key"}
	default:
		return nil
	}
}

func standardGenerateResponses() map[string]any {
	return map[string]any{
		"200": jsonResponseRef("GenerateResponse"),
		"400": jsonResponseRef("ErrorResponse"),
		"401": jsonResponseRef("ErrorResponse"),
		"403": jsonResponseRef("ErrorResponse"),
		"429": jsonResponseRef("ErrorResponse"),
		"408": jsonResponseRef("ErrorResponse"),
		"504": jsonResponseRef("ErrorResponse"),
	}
}

func openapiSchemas() map[string]any {
	return map[string]any{
		"GenerateRequest": map[string]any{
			"type": "object",
			"properties": map[string]any{
				"prompt":                map[string]any{"type": "string", "examples": []string{"<|begin_of_sentence|>hello"}},
				"task":                  map[string]any{"type": "string", "enum": []string{"ocr", "table", "formula", "chart"}, "examples": []string{"ocr"}},
				"image_path":            map[string]any{"type": "string", "description": "Server-local image path.", "examples": []string{`D:\docs\page.png`}},
				"image_base64":          map[string]any{"type": "string", "description": "Base64 image data; data URL prefix is accepted.", "examples": []string{"<base64-image>"}},
				"tokens":                map[string]any{"type": "array", "items": map[string]any{"type": "integer"}},
				"max_new_tokens":        map[string]any{"type": "integer", "minimum": 0, "default": 16},
				"temperature":           map[string]any{"type": "number", "minimum": 0, "default": 0},
				"top_k":                 map[string]any{"type": "integer", "minimum": 0, "default": 0},
				"seed":                  map[string]any{"type": "integer", "format": "int64"},
				"eos_token_ids":         map[string]any{"type": "array", "items": map[string]any{"type": "integer"}},
				"decode":                map[string]any{"type": "boolean", "default": false},
				"decode_generated_only": map[string]any{"type": "boolean", "default": false},
				"skip_special":          map[string]any{"type": "boolean", "default": false},
			},
		},
		"GenerateResponse": map[string]any{
			"type":     "object",
			"required": []string{"tokens", "prompt_tokens", "generated_tokens"},
			"properties": map[string]any{
				"tokens":           map[string]any{"type": "array", "items": map[string]any{"type": "integer"}},
				"prompt_tokens":    map[string]any{"type": "integer"},
				"generated_tokens": map[string]any{"type": "integer"},
				"text":             map[string]any{"type": "string"},
			},
		},
		"BatchRequest": map[string]any{
			"type":     "object",
			"required": []string{"requests"},
			"properties": map[string]any{
				"requests": map[string]any{"type": "array", "minItems": 1, "items": schemaRef("GenerateRequest")},
			},
		},
		"BatchResponse": map[string]any{
			"type":     "object",
			"required": []string{"responses", "items", "generated_tokens"},
			"properties": map[string]any{
				"responses":        map[string]any{"type": "array", "items": schemaRef("GenerateResponse")},
				"items":            map[string]any{"type": "integer"},
				"generated_tokens": map[string]any{"type": "integer"},
			},
		},
		"HealthResponse": map[string]any{
			"type": "object",
			"properties": map[string]any{
				"status":        map[string]any{"type": "string"},
				"quantization":  map[string]any{"type": "string"},
				"weight_path":   map[string]any{"type": "string"},
				"weight_source": map[string]any{"type": "string"},
				"backend":       map[string]any{"type": "string"},
				"vision_loaded": map[string]any{"type": "boolean"},
			},
		},
		"ReadyResponse": map[string]any{
			"type": "object",
			"properties": map[string]any{
				"status":          map[string]any{"type": "string"},
				"reason":          map[string]any{"type": "string"},
				"quantization":    map[string]any{"type": "string"},
				"weight_path":     map[string]any{"type": "string"},
				"weight_source":   map[string]any{"type": "string"},
				"backend":         map[string]any{"type": "string"},
				"vision_loaded":   map[string]any{"type": "boolean"},
				"concurrency":     map[string]any{"type": "integer"},
				"in_flight":       map[string]any{"type": "integer"},
				"available_slots": map[string]any{"type": "integer"},
			},
		},
		"StatsResponse": map[string]any{
			"type":                 "object",
			"description":          "Runtime stats. Includes model, memory, requests, cache, backend, and weight metadata.",
			"additionalProperties": true,
		},
		"ErrorResponse": map[string]any{
			"type":     "object",
			"required": []string{"error"},
			"properties": map[string]any{
				"error": map[string]any{"type": "string"},
			},
		},
	}
}

func (s *server) adminInit(w http.ResponseWriter, r *http.Request) {
	setAdminSecurityHeaders(w)
	if !allowAdminWriteOrigin(r) {
		writeError(w, http.StatusForbidden, fmt.Errorf("invalid admin request origin"))
		return
	}
	limitAdminRequestBody(w, r)
	if s.admin.initialized() {
		writeError(w, http.StatusConflict, fmt.Errorf("admin already initialized"))
		return
	}
	var req struct {
		AdminUser      string `json:"admin_user"`
		Password       string `json:"password"`
		ModelDir       string `json:"model_dir"`
		PostProcessDir string `json:"post_process_dir"`
	}
	if err := decodeJSON(r.Body, &req); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	modelDir := strings.TrimSpace(req.ModelDir)
	if modelDir == "" {
		modelDir = s.modelDir
	}
	if err := validateModelDir(modelDir); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	s.admin.mu.Lock()
	if s.admin.cfg.PasswordHash != "" {
		s.admin.mu.Unlock()
		writeError(w, http.StatusConflict, fmt.Errorf("admin already initialized"))
		return
	}
	oldCfg := s.admin.cfg
	if err := s.admin.setPasswordLocked(req.AdminUser, req.Password); err != nil {
		s.admin.mu.Unlock()
		writeError(w, http.StatusBadRequest, err)
		return
	}
	apiKey, _, err := s.admin.createAPIKeyLocked("default", 0, 0)
	if err != nil {
		s.admin.cfg = oldCfg
		s.admin.mu.Unlock()
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	s.admin.cfg.ModelDir = modelDir
	s.admin.cfg.PostProcessDir = strings.TrimSpace(req.PostProcessDir)
	s.admin.cfg.RestartRequired = modelDir != s.modelDir
	if err := s.admin.saveLocked(); err != nil {
		s.admin.cfg = oldCfg
		s.admin.mu.Unlock()
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	s.admin.mu.Unlock()
	token, err := s.admin.createSession()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	setAdminCookie(w, r, token)
	writeJSON(w, http.StatusOK, map[string]any{"config": s.admin.response(s.modelDir), "api_key": apiKey})
}

func (s *server) adminLogin(w http.ResponseWriter, r *http.Request) {
	setAdminSecurityHeaders(w)
	if !allowAdminWriteOrigin(r) {
		writeError(w, http.StatusForbidden, fmt.Errorf("invalid admin request origin"))
		return
	}
	limitAdminRequestBody(w, r)
	ip := clientIP(r)
	now := time.Now()
	if s.admin.loginLocked(ip, now) {
		writeError(w, http.StatusTooManyRequests, fmt.Errorf("too many failed login attempts"))
		return
	}
	var req struct {
		AdminUser string `json:"admin_user"`
		Password  string `json:"password"`
	}
	if err := decodeJSON(r.Body, &req); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	if !s.admin.initialized() {
		writeError(w, http.StatusPreconditionRequired, fmt.Errorf("admin is not initialized"))
		return
	}
	if !s.admin.checkPassword(strings.TrimSpace(req.AdminUser), req.Password) {
		s.admin.recordLoginFailure(ip, now)
		writeError(w, http.StatusUnauthorized, fmt.Errorf("invalid admin credentials"))
		return
	}
	s.admin.clearLoginFailures(ip)
	token, err := s.admin.createSession()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	setAdminCookie(w, r, token)
	writeOKJSON(w)
}

func (s *server) adminLogout(w http.ResponseWriter, r *http.Request) {
	setAdminSecurityHeaders(w)
	if !allowAdminWriteOrigin(r) {
		writeError(w, http.StatusForbidden, fmt.Errorf("invalid admin request origin"))
		return
	}
	if token, ok := adminSessionCookie(r); ok {
		s.admin.deleteSession(token)
	}
	http.SetCookie(w, &http.Cookie{Name: "paddle_admin_session", Value: "", Path: "/admin", MaxAge: -1, Expires: time.Unix(0, 0), HttpOnly: true, Secure: requestScheme(r) == "https", SameSite: http.SameSiteLaxMode})
	writeOKJSON(w)
}

func (s *server) adminSession(w http.ResponseWriter, r *http.Request) {
	setAdminSecurityHeaders(w)
	authed := s.adminRequestOK(r)
	writeAdminSessionJSON(w, s.admin.initialized(), authed)
}

func (s *server) adminOverview(w http.ResponseWriter, r *http.Request) {
	code, ready := s.readyState()
	writeJSON(w, http.StatusOK, adminOverviewResponse{
		Admin:         s.admin.response(s.modelDir),
		APIBaseURL:    adminBaseURL(r),
		CurrentHost:   requestHost(r),
		ModelDir:      s.modelDir,
		Stats:         s.statsSnapshot(),
		Ready:         ready,
		ReadyHTTPCode: code,
	})
}

func (s *server) adminConfigGet(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, s.admin.response(s.modelDir))
}

func (s *server) adminConfigSave(w http.ResponseWriter, r *http.Request) {
	var req struct {
		AdminUser      string `json:"admin_user"`
		Password       string `json:"password"`
		ModelDir       string `json:"model_dir"`
		PostProcessDir string `json:"post_process_dir"`
	}
	if err := decodeJSON(r.Body, &req); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	modelDir := strings.TrimSpace(req.ModelDir)
	if modelDir == "" {
		modelDir = s.modelDir
	}
	if err := validateModelDir(modelDir); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	s.admin.mu.Lock()
	oldCfg := s.admin.cfg
	if strings.TrimSpace(req.AdminUser) != "" {
		adminUser, err := normalizeAdminUser(req.AdminUser)
		if err != nil {
			s.admin.mu.Unlock()
			writeError(w, http.StatusBadRequest, err)
			return
		}
		s.admin.cfg.AdminUser = adminUser
	}
	if req.Password != "" {
		if err := s.admin.setPasswordLocked(s.admin.cfg.AdminUser, req.Password); err != nil {
			s.admin.cfg = oldCfg
			s.admin.mu.Unlock()
			writeError(w, http.StatusBadRequest, err)
			return
		}
	}
	s.admin.cfg.ModelDir = modelDir
	s.admin.cfg.PostProcessDir = strings.TrimSpace(req.PostProcessDir)
	s.admin.cfg.RestartRequired = modelDir != s.modelDir
	if err := s.admin.saveLocked(); err != nil {
		s.admin.cfg = oldCfg
		s.admin.mu.Unlock()
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	s.admin.mu.Unlock()
	writeJSON(w, http.StatusOK, s.admin.response(s.modelDir))
}

func (s *server) adminConfigBackup(w http.ResponseWriter, r *http.Request) {
	s.admin.mu.RLock()
	cfg := cloneAdminConfigFile(s.admin.cfg)
	s.admin.mu.RUnlock()
	w.Header().Set("Content-Disposition", `attachment; filename="paddleocrvl-admin-backup.json"`)
	writeJSON(w, http.StatusOK, adminConfigBackup{
		Version:    1,
		ExportedAt: time.Now().UTC().Format(time.RFC3339),
		Config:     cfg,
	})
}

func cloneAdminConfigFile(cfg adminConfigFile) adminConfigFile {
	cfg.APIKeys = append([]managedAPIKey(nil), cfg.APIKeys...)
	return cfg
}

func (s *server) adminConfigRestore(w http.ResponseWriter, r *http.Request) {
	var backup adminConfigBackup
	if err := decodeJSON(r.Body, &backup); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	if backup.Version != 1 {
		writeError(w, http.StatusBadRequest, fmt.Errorf("unsupported backup version %d", backup.Version))
		return
	}
	if err := validateAdminConfigBackup(backup.Config); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	cfg := backup.Config
	adminUser, err := normalizeAdminUser(cfg.AdminUser)
	if err != nil {
		writeError(w, http.StatusBadRequest, fmt.Errorf("backup %w", err))
		return
	}
	cfg.AdminUser = adminUser
	cfg.ModelDir = strings.TrimSpace(cfg.ModelDir)
	if cfg.ModelDir != "" {
		if err := validateModelDir(cfg.ModelDir); err != nil {
			writeError(w, http.StatusBadRequest, err)
			return
		}
	}
	cfg.PostProcessDir = strings.TrimSpace(cfg.PostProcessDir)
	cfg.RestartRequired = cfg.ModelDir != "" && cfg.ModelDir != s.modelDir
	s.admin.mu.Lock()
	oldCfg := s.admin.cfg
	oldRates := s.admin.rates
	s.admin.cfg = cfg
	s.admin.rates = make(map[string]apiKeyRateWindow)
	err = s.admin.saveLocked()
	if err != nil {
		s.admin.cfg = oldCfg
		s.admin.rates = oldRates
	}
	s.admin.mu.Unlock()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	writeJSON(w, http.StatusOK, s.admin.response(s.modelDir))
}

func (s *server) adminAuditList(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{"items": s.admin.auditEntries()})
}

func (s *server) adminAuditCSV(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/csv; charset=utf-8")
	w.Header().Set("Content-Disposition", `attachment; filename="paddleocrvl-audit.csv"`)
	cw := csv.NewWriter(w)
	_ = cw.Write([]string{"time", "key_id", "key_name", "key_preview", "client_ip", "method", "path", "status", "duration_ms", "error"})
	for _, e := range s.admin.auditEntries() {
		_ = cw.Write([]string{
			csvCell(e.Time),
			csvCell(e.KeyID),
			csvCell(e.KeyName),
			csvCell(e.KeyPreview),
			csvCell(e.ClientIP),
			csvCell(e.Method),
			csvCell(e.Path),
			strconv.Itoa(e.Status),
			strconv.FormatInt(e.DurationMS, 10),
			csvCell(e.Error),
		})
	}
	cw.Flush()
}

func (s *server) adminAuditClear(w http.ResponseWriter, r *http.Request) {
	s.admin.clearAudit()
	writeOKJSON(w)
}

func (s *server) adminKeysList(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{"keys": s.admin.keyResponses()})
}

func (s *server) adminKeysCreate(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Name       string `json:"name"`
		Quota      int64  `json:"quota"`
		RatePerMin int64  `json:"rate_per_minute"`
	}
	if err := decodeJSON(r.Body, &req); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	if req.Quota < 0 || req.RatePerMin < 0 {
		writeError(w, http.StatusBadRequest, fmt.Errorf("quota and rate_per_minute must be >= 0"))
		return
	}
	s.admin.mu.Lock()
	oldLen := len(s.admin.cfg.APIKeys)
	apiKey, key, err := s.admin.createAPIKeyLocked(req.Name, req.Quota, req.RatePerMin)
	if err == nil {
		err = s.admin.saveLocked()
		if err != nil {
			s.admin.cfg.APIKeys = s.admin.cfg.APIKeys[:oldLen]
		}
	}
	s.admin.mu.Unlock()
	if err != nil {
		if strings.Contains(err.Error(), "api key name") {
			writeError(w, http.StatusBadRequest, err)
			return
		}
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"api_key": apiKey, "key": key.response()})
}

func (s *server) adminKeysUpdate(w http.ResponseWriter, r *http.Request) {
	var req struct {
		ID         string `json:"id"`
		Name       string `json:"name"`
		Quota      int64  `json:"quota"`
		RatePerMin int64  `json:"rate_per_minute"`
		Disabled   bool   `json:"disabled"`
	}
	if err := decodeJSON(r.Body, &req); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	if req.ID == "" || req.Quota < 0 || req.RatePerMin < 0 {
		writeError(w, http.StatusBadRequest, fmt.Errorf("id is required; quota and rate_per_minute must be >= 0"))
		return
	}
	s.admin.mu.Lock()
	var found bool
	foundIndex := -1
	var oldKey managedAPIKey
	now := time.Now().UTC().Format(time.RFC3339)
	name, err := normalizeAPIKeyName(req.Name)
	if err != nil {
		s.admin.mu.Unlock()
		writeError(w, http.StatusBadRequest, err)
		return
	}
	for i := range s.admin.cfg.APIKeys {
		key := &s.admin.cfg.APIKeys[i]
		if key.ID != req.ID {
			continue
		}
		foundIndex = i
		oldKey = *key
		if name != "" {
			key.Name = name
		}
		key.Quota = req.Quota
		key.RatePerMin = req.RatePerMin
		key.Disabled = req.Disabled
		key.UpdatedAt = now
		found = true
		break
	}
	err = nil
	if found {
		err = s.admin.saveLocked()
		if err != nil && foundIndex >= 0 {
			s.admin.cfg.APIKeys[foundIndex] = oldKey
		}
	}
	s.admin.mu.Unlock()
	if !found {
		writeError(w, http.StatusNotFound, fmt.Errorf("api key not found"))
		return
	}
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"keys": s.admin.keyResponses()})
}

func (s *server) adminKeysRotate(w http.ResponseWriter, r *http.Request) {
	var req struct {
		ID string `json:"id"`
	}
	if err := decodeJSON(r.Body, &req); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	if req.ID == "" {
		writeError(w, http.StatusBadRequest, fmt.Errorf("id is required"))
		return
	}
	apiKey, salt, hash, preview, err := newAPIKeySecret()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	s.admin.mu.Lock()
	var found bool
	foundIndex := -1
	var oldKey managedAPIKey
	var oldRate apiKeyRateWindow
	var hadRate bool
	now := time.Now().UTC().Format(time.RFC3339)
	for i := range s.admin.cfg.APIKeys {
		key := &s.admin.cfg.APIKeys[i]
		if key.ID != req.ID {
			continue
		}
		foundIndex = i
		oldKey = *key
		oldRate, hadRate = s.admin.rates[key.ID]
		key.Salt = salt
		key.Hash = hash
		key.Preview = preview
		key.LastUsedAt = ""
		key.LastUsedIP = ""
		key.UpdatedAt = now
		delete(s.admin.rates, key.ID)
		found = true
		break
	}
	if found {
		err = s.admin.saveLocked()
		if err != nil && foundIndex >= 0 {
			s.admin.cfg.APIKeys[foundIndex] = oldKey
			if hadRate {
				s.admin.rates[oldKey.ID] = oldRate
			} else {
				delete(s.admin.rates, oldKey.ID)
			}
		}
	}
	s.admin.mu.Unlock()
	if !found {
		writeError(w, http.StatusNotFound, fmt.Errorf("api key not found"))
		return
	}
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"api_key": apiKey, "keys": s.admin.keyResponses()})
}

func (s *server) adminKeysReset(w http.ResponseWriter, r *http.Request) {
	var req struct {
		ID string `json:"id"`
	}
	if err := decodeJSON(r.Body, &req); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	s.admin.mu.Lock()
	var found bool
	foundIndex := -1
	var oldKey managedAPIKey
	now := time.Now().UTC().Format(time.RFC3339)
	for i := range s.admin.cfg.APIKeys {
		key := &s.admin.cfg.APIKeys[i]
		if key.ID != req.ID {
			continue
		}
		foundIndex = i
		oldKey = *key
		key.Used = 0
		key.UpdatedAt = now
		found = true
		break
	}
	var err error
	if found {
		err = s.admin.saveLocked()
		if err != nil && foundIndex >= 0 {
			s.admin.cfg.APIKeys[foundIndex] = oldKey
		}
	}
	s.admin.mu.Unlock()
	if !found {
		writeError(w, http.StatusNotFound, fmt.Errorf("api key not found"))
		return
	}
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"keys": s.admin.keyResponses()})
}

func (s *server) adminKeysDelete(w http.ResponseWriter, r *http.Request) {
	var req struct {
		ID string `json:"id"`
	}
	if err := decodeJSON(r.Body, &req); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	s.admin.mu.Lock()
	var found bool
	oldKeys := append([]managedAPIKey(nil), s.admin.cfg.APIKeys...)
	next := s.admin.cfg.APIKeys[:0]
	for _, key := range s.admin.cfg.APIKeys {
		if key.ID == req.ID {
			found = true
			continue
		}
		next = append(next, key)
	}
	s.admin.cfg.APIKeys = next
	var err error
	if found {
		err = s.admin.saveLocked()
		if err != nil {
			s.admin.cfg.APIKeys = oldKeys
		}
	}
	s.admin.mu.Unlock()
	if !found {
		writeError(w, http.StatusNotFound, fmt.Errorf("api key not found"))
		return
	}
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"keys": s.admin.keyResponses()})
}

func (s *server) adminValidateModelDir(w http.ResponseWriter, r *http.Request) {
	var req struct {
		ModelDir string `json:"model_dir"`
	}
	if err := decodeJSON(r.Body, &req); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	err := validateModelDir(strings.TrimSpace(req.ModelDir))
	if err != nil {
		writeOKErrorJSON(w, err.Error())
		return
	}
	writeOKJSON(w)
}

func (s *server) requireAdmin(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		setAdminSecurityHeaders(w)
		if isUnsafeMethod(r.Method) && !allowAdminWriteOrigin(r) {
			writeError(w, http.StatusForbidden, fmt.Errorf("invalid admin request origin"))
			return
		}
		if isUnsafeMethod(r.Method) {
			limitAdminRequestBody(w, r)
		}
		if !s.admin.initialized() {
			writeError(w, http.StatusPreconditionRequired, fmt.Errorf("admin is not initialized"))
			return
		}
		if !s.adminRequestOK(r) {
			writeError(w, http.StatusUnauthorized, fmt.Errorf("admin login required"))
			return
		}
		next(w, r)
	}
}

func limitAdminRequestBody(w http.ResponseWriter, r *http.Request) {
	if r.Body != nil {
		r.Body = http.MaxBytesReader(w, r.Body, adminRequestLimit)
	}
}

func setAdminSecurityHeaders(w http.ResponseWriter) {
	h := w.Header()
	h.Set("Cache-Control", "no-store")
	h.Set("X-Content-Type-Options", "nosniff")
	h.Set("X-Frame-Options", "DENY")
	h.Set("Referrer-Policy", "same-origin")
	h.Set("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'")
}

func isUnsafeMethod(method string) bool {
	return method != http.MethodGet && method != http.MethodHead && method != http.MethodOptions
}

func allowAdminWriteOrigin(r *http.Request) bool {
	if !isUnsafeMethod(r.Method) {
		return true
	}
	expectedScheme := requestScheme(r)
	expectedHost := requestHost(r)
	if origin := trimASCIIForm(firstHeaderValue(r.Header, "Origin")); origin != "" {
		return sameOrigin(origin, expectedScheme, expectedHost)
	}
	if referer := trimASCIIForm(firstHeaderValue(r.Header, "Referer")); referer != "" {
		return sameOrigin(referer, expectedScheme, expectedHost)
	}
	return true
}

func sameOrigin(got, expectedScheme, expectedHost string) bool {
	scheme, host, ok := originSchemeHost(got)
	if !ok {
		return false
	}
	return asciiEqualFold(scheme, expectedScheme) && asciiEqualFold(host, expectedHost)
}

func originSchemeHost(s string) (scheme, host string, ok bool) {
	i := strings.Index(s, "://")
	if i <= 0 {
		return "", "", false
	}
	scheme = s[:i]
	rest := s[i+3:]
	if rest == "" {
		return "", "", false
	}
	end := len(rest)
	for j := 0; j < len(rest); j++ {
		switch rest[j] {
		case '/', '?', '#':
			end = j
			goto done
		}
	}
done:
	host = rest[:end]
	if !validRequestHost(host) {
		return "", "", false
	}
	return scheme, host, true
}

func (s *server) adminRequestOK(r *http.Request) bool {
	token, ok := adminSessionCookie(r)
	if !ok {
		return false
	}
	return s.admin.validSession(token)
}

func adminSessionCookie(r *http.Request) (string, bool) {
	values := r.Header["Cookie"]
	if len(values) == 0 {
		for k, v := range r.Header {
			if strings.EqualFold(k, "Cookie") {
				values = v
				break
			}
		}
	}
	for _, line := range values {
		for len(line) > 0 {
			for len(line) > 0 && (line[0] == ' ' || line[0] == ';') {
				line = line[1:]
			}
			end := strings.IndexByte(line, ';')
			part := line
			if end >= 0 {
				part = line[:end]
				line = line[end+1:]
			} else {
				line = ""
			}
			if len(part) <= len("paddle_admin_session=") || !strings.HasPrefix(part, "paddle_admin_session=") {
				continue
			}
			value := part[len("paddle_admin_session="):]
			if value != "" {
				return value, true
			}
		}
	}
	return "", false
}

func (s *server) requireAPIKey(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		started := time.Now()
		ip := clientIP(r)
		entry := auditEntry{
			Time:     started.UTC().Format(time.RFC3339),
			ClientIP: ip,
			Method:   r.Method,
			Path:     r.URL.Path,
		}
		if s.admin == nil || !s.admin.initialized() {
			rec := &statusRecorder{ResponseWriter: w, status: http.StatusOK}
			next(rec, r)
			entry.Status = rec.status
			entry.DurationMS = time.Since(started).Milliseconds()
			entry.Error = rec.auditError()
			if s.admin != nil {
				s.admin.recordAudit(entry)
			}
			return
		}
		key, ok := apiKeyFromHeaders(r.Header)
		if !ok {
			err := fmt.Errorf("missing API key")
			entry.Status = http.StatusUnauthorized
			entry.DurationMS = time.Since(started).Milliseconds()
			entry.Error = err.Error()
			s.admin.recordAudit(entry)
			writeError(w, http.StatusUnauthorized, err)
			return
		}
		info, err := s.admin.consumeAPIKey(key, ip)
		entry.KeyID = info.ID
		entry.KeyName = info.Name
		entry.KeyPreview = info.Preview
		if err != nil {
			status := statusForAPIKeyError(err)
			entry.Status = status
			entry.DurationMS = time.Since(started).Milliseconds()
			entry.Error = err.Error()
			s.admin.recordAudit(entry)
			writeError(w, status, err)
			return
		}
		rec := &statusRecorder{ResponseWriter: w, status: http.StatusOK}
		next(rec, r)
		entry.Status = rec.status
		entry.DurationMS = time.Since(started).Milliseconds()
		entry.Error = rec.auditError()
		s.admin.recordAudit(entry)
	}
}

func apiKeyFromHeaders(h http.Header) (string, bool) {
	if auth := firstHeaderValue(h, "Authorization"); auth != "" {
		auth = trimASCIIForm(auth)
		if len(auth) > len("Bearer ") && asciiEqualFold(auth[:len("Bearer")], "bearer") && isASCIISpace(auth[len("Bearer")]) {
			key := trimASCIIForm(auth[len("Bearer"):])
			return key, key != ""
		}
	}
	key := trimASCIIForm(firstHeaderValue2(h, "X-Api-Key", "X-API-Key"))
	return key, key != ""
}

type statusRecorder struct {
	http.ResponseWriter
	status      int
	wroteHeader bool
	body        bytes.Buffer
}

func (w *statusRecorder) WriteHeader(status int) {
	if w.wroteHeader {
		return
	}
	w.wroteHeader = true
	w.status = status
	w.ResponseWriter.WriteHeader(status)
}

func (w *statusRecorder) Write(p []byte) (int, error) {
	if !w.wroteHeader {
		w.wroteHeader = true
		w.status = http.StatusOK
	}
	if w.status >= 400 && w.body.Len() < 1024 {
		n := min(len(p), 1024-w.body.Len())
		_, _ = w.body.Write(p[:n])
	}
	return w.ResponseWriter.Write(p)
}

func (w *statusRecorder) auditError() string {
	if w.status < 400 {
		return ""
	}
	if w.body.Len() > 0 {
		var msg struct {
			Error string `json:"error"`
		}
		if err := json.Unmarshal(w.body.Bytes(), &msg); err == nil && msg.Error != "" {
			return msg.Error
		}
	}
	return http.StatusText(w.status)
}

func clientIP(r *http.Request) string {
	if forwarded := trimASCIIForm(firstHeaderValue(r.Header, "X-Forwarded-For")); forwarded != "" {
		if i := strings.IndexByte(forwarded, ','); i >= 0 {
			forwarded = forwarded[:i]
		}
		if ip := trimASCIIForm(forwarded); ip != "" {
			if validIPv4Literal(ip) || strings.Contains(ip, ":") && net.ParseIP(ip) != nil {
				return ip
			}
		}
	}
	if realIP := trimASCIIForm(firstHeaderValue2(r.Header, "X-Real-Ip", "X-Real-IP")); realIP != "" {
		if validIPv4Literal(realIP) || strings.Contains(realIP, ":") && net.ParseIP(realIP) != nil {
			return realIP
		}
	}
	if host, ok := splitHostPortASCII(r.RemoteAddr); ok {
		if validIPv4Literal(host) || strings.Contains(host, ":") && net.ParseIP(host) != nil {
			return host
		}
	}
	if validIPv4Literal(r.RemoteAddr) || strings.Contains(r.RemoteAddr, ":") && net.ParseIP(r.RemoteAddr) != nil {
		return r.RemoteAddr
	}
	return r.RemoteAddr
}

func firstHeaderValue(h http.Header, key string) string {
	if values := h[key]; len(values) > 0 {
		return values[0]
	}
	for k, values := range h {
		if len(values) > 0 && strings.EqualFold(k, key) {
			return values[0]
		}
	}
	return ""
}

func firstHeaderValue2(h http.Header, keyA, keyB string) string {
	if values := h[keyA]; len(values) > 0 {
		return values[0]
	}
	if values := h[keyB]; len(values) > 0 {
		return values[0]
	}
	for k, values := range h {
		if len(values) > 0 && (strings.EqualFold(k, keyA) || strings.EqualFold(k, keyB)) {
			return values[0]
		}
	}
	return ""
}

func splitHostPortASCII(addr string) (string, bool) {
	i := strings.LastIndexByte(addr, ':')
	if i <= 0 || i+1 >= len(addr) {
		return "", false
	}
	for j := i + 1; j < len(addr); j++ {
		c := addr[j]
		if c < '0' || c > '9' {
			return "", false
		}
	}
	host := addr[:i]
	if strings.HasPrefix(host, "[") && strings.HasSuffix(host, "]") {
		host = host[1 : len(host)-1]
	}
	return host, host != ""
}

func statusForAPIKeyError(err error) int {
	switch {
	case errors.Is(err, errAPIKeyDisabled):
		return http.StatusForbidden
	case errors.Is(err, errAPIKeyQuota):
		return http.StatusTooManyRequests
	case errors.Is(err, errAPIKeyRate):
		return http.StatusTooManyRequests
	default:
		return http.StatusUnauthorized
	}
}

func setAdminCookie(w http.ResponseWriter, r *http.Request, token string) {
	http.SetCookie(w, &http.Cookie{
		Name:     "paddle_admin_session",
		Value:    token,
		Path:     "/admin",
		HttpOnly: true,
		Secure:   requestScheme(r) == "https",
		SameSite: http.SameSiteLaxMode,
		Expires:  time.Now().Add(12 * time.Hour),
	})
}

func validateModelDir(dir string) error {
	if dir == "" {
		return fmt.Errorf("model_dir is required")
	}
	st, err := os.Stat(dir)
	if err != nil {
		return err
	}
	if !st.IsDir() {
		return fmt.Errorf("model_dir is not a directory")
	}
	required := []string{"config.json"}
	for _, name := range required {
		if _, err := os.Stat(filepath.Join(dir, name)); err != nil {
			return fmt.Errorf("missing %s", name)
		}
	}
	weights := []string{"model.gguf", "model-q8.gguf", "model-q6.gguf", "model-q4.gguf", "model.safetensors", "model.safetensors.index.json"}
	for _, name := range weights {
		if _, err := os.Stat(filepath.Join(dir, name)); err == nil {
			return nil
		}
	}
	return errors.New("missing model weights: expected GGUF or safetensors")
}

func validateAdminConfigBackup(cfg adminConfigFile) error {
	if _, err := normalizeAdminUser(cfg.AdminUser); err != nil {
		return fmt.Errorf("backup %w", err)
	}
	if cfg.PasswordSalt == "" || cfg.PasswordHash == "" {
		return fmt.Errorf("backup password hash is required")
	}
	if !validSalt(cfg.PasswordSalt) || !validSHA256Hex(cfg.PasswordHash) {
		return fmt.Errorf("backup password hash is invalid")
	}
	seenIDs := make(map[string]struct{}, len(cfg.APIKeys))
	for _, key := range cfg.APIKeys {
		if _, err := normalizeAPIKeyName(key.Name); err != nil {
			return fmt.Errorf("backup contains invalid API key name")
		}
		if key.ID == "" || key.Hash == "" || key.Salt == "" {
			return fmt.Errorf("backup contains invalid API key metadata")
		}
		if !validSalt(key.Salt) || !validSHA256Hex(key.Hash) {
			return fmt.Errorf("backup contains invalid API key hash")
		}
		if !validAPIKeyID(key.ID) {
			return fmt.Errorf("backup contains invalid API key id")
		}
		if _, ok := seenIDs[key.ID]; ok {
			return fmt.Errorf("backup contains duplicate API key id")
		}
		seenIDs[key.ID] = struct{}{}
		if key.Quota < 0 || key.RatePerMin < 0 || key.Used < 0 {
			return fmt.Errorf("backup contains invalid API key limits")
		}
	}
	return nil
}

func validAPIKeyID(id string) bool {
	if id == "" || len(id) > 128 {
		return false
	}
	for i := 0; i < len(id); i++ {
		c := id[i]
		if c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || c == '-' || c == '_' {
			continue
		}
		return false
	}
	return true
}

func validSalt(s string) bool {
	if s == "" || len(s) > 128 {
		return false
	}
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || c == '-' || c == '_' {
			continue
		}
		return false
	}
	return true
}

func validSHA256Hex(s string) bool {
	if len(s) != sha256.Size*2 {
		return false
	}
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c >= '0' && c <= '9' || c >= 'a' && c <= 'f' || c >= 'A' && c <= 'F' {
			continue
		}
		return false
	}
	return true
}

func adminBaseURL(r *http.Request) string {
	return requestScheme(r) + "://" + requestHost(r)
}

func requestHost(r *http.Request) string {
	if host := trimASCIIForm(r.Host); validRequestHost(host) {
		return host
	}
	if r.URL != nil {
		if host := trimASCIIForm(r.URL.Host); validRequestHost(host) {
			return host
		}
	}
	return "localhost"
}

func validRequestHost(host string) bool {
	if host == "" || len(host) > 255 || containsInvalidHostByte(host) {
		return false
	}
	name := host
	if strings.HasPrefix(name, "[") {
		end := strings.IndexByte(name, ']')
		if end < 0 || net.ParseIP(name[1:end]) == nil {
			return false
		}
		if rest := name[end+1:]; rest != "" {
			if !strings.HasPrefix(rest, ":") || !validPort(rest[1:]) {
				return false
			}
		}
		return true
	}
	if i := strings.LastIndexByte(name, ':'); i >= 0 {
		if strings.Contains(name[:i], ":") || !validPort(name[i+1:]) {
			return false
		}
		name = name[:i]
	}
	if name == "" {
		return false
	}
	if maybeIPv4Literal(name) && net.ParseIP(name) != nil {
		return true
	}
	start := 0
	for i := 0; i <= len(name); i++ {
		if i < len(name) && name[i] != '.' {
			continue
		}
		if i == start || name[start] == '-' || name[i-1] == '-' {
			return false
		}
		for j := start; j < i; j++ {
			c := name[j]
			if c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || c == '-' {
				continue
			}
			return false
		}
		start = i + 1
	}
	return true
}

func containsInvalidHostByte(host string) bool {
	for i := 0; i < len(host); i++ {
		switch host[i] {
		case ' ', '\t', '\r', '\n', '/', '\\', '@', '?', '#':
			return true
		}
	}
	return false
}

func maybeIPv4Literal(host string) bool {
	if host == "" {
		return false
	}
	for i := 0; i < len(host); i++ {
		c := host[i]
		if c >= '0' && c <= '9' || c == '.' {
			continue
		}
		return false
	}
	return true
}

func validIPv4Literal(host string) bool {
	parts := 0
	value := 0
	digits := 0
	for i := 0; i <= len(host); i++ {
		if i < len(host) && host[i] != '.' {
			c := host[i]
			if c < '0' || c > '9' {
				return false
			}
			value = value*10 + int(c-'0')
			digits++
			if value > 255 || digits > 3 {
				return false
			}
			continue
		}
		if digits == 0 {
			return false
		}
		parts++
		value = 0
		digits = 0
	}
	return parts == 4
}

func validPort(port string) bool {
	if port == "" || len(port) > 5 {
		return false
	}
	n := 0
	for i := 0; i < len(port); i++ {
		c := port[i]
		if c < '0' || c > '9' {
			return false
		}
		n = n*10 + int(c-'0')
	}
	return n > 0 && n <= 65535
}

func requestScheme(r *http.Request) string {
	if proto := firstHeaderValue(r.Header, "X-Forwarded-Proto"); proto != "" {
		start, end := 0, len(proto)
		for start < end && isASCIISpace(proto[start]) {
			start++
		}
		for i := start; i < end; i++ {
			if proto[i] == ',' {
				end = i
				break
			}
		}
		for end > start && isASCIISpace(proto[end-1]) {
			end--
		}
		switch {
		case asciiEqualFold(proto[start:end], "http"):
			return "http"
		case asciiEqualFold(proto[start:end], "https"):
			return "https"
		}
	}
	if r.TLS != nil {
		return "https"
	}
	return "http"
}

func (s *server) statsSnapshot() statsResponse {
	cfg := s.rt.Config()
	inFlight := len(s.runSlots)
	cmdPlan := s.rt.VulkanCommandPlan()
	cmdPlanErr := backend.ValidateVulkanCommandPlan(cmdPlan)
	return statsResponse{
		Status:               "ok",
		UptimeSeconds:        int64(time.Since(s.started).Seconds()),
		Concurrency:          s.concurrency,
		InFlight:             inFlight,
		AvailableSlots:       s.concurrency - inFlight,
		RequestLimit:         s.requestLimit,
		MultipartMemory:      s.multipartMem,
		MaxNewLimit:          s.maxNewLimit,
		MaxInputTokens:       s.maxInputLimit,
		MaxBatchSize:         s.maxBatchSize,
		TimeoutSeconds:       int64(s.timeout.Seconds()),
		Quantization:         s.rt.Quantization(),
		RequestedQuant:       s.rt.RequestedQuantization(),
		WeightPath:           s.rt.WeightPath(),
		WeightSource:         s.rt.WeightSource(),
		WeightSHA256:         s.weightSHA256,
		Weights:              s.rt.WeightStats(),
		LoadStats:            s.rt.LoadStats(),
		Backend:              s.rt.Backend(),
		VisionLoaded:         s.rt.VisionLoaded(),
		Backends:             s.backendSel,
		VulkanPlans:          s.rt.VulkanPlans(),
		VulkanPlanSummary:    s.rt.VulkanPlanSummary(),
		VulkanExecutionGraph: s.rt.VulkanExecutionGraph(),
		VulkanPipelinePlan:   s.rt.VulkanPipelinePlan(),
		VulkanCommandPlan:    cmdPlan,
		VulkanCommandPlanOK:  cmdPlanErr == "",
		VulkanCommandPlanErr: cmdPlanErr,
		CPU:                  s.cpuInfo,
		Memory:               backendMemory(),
		Requests: statsRequests{
			Queued:          s.metrics.queued.Load(),
			Started:         s.metrics.started.Load(),
			Succeeded:       s.metrics.succeeded.Load(),
			Failed:          s.metrics.failed.Load(),
			Canceled:        s.metrics.canceled.Load(),
			Batches:         s.metrics.batches.Load(),
			BatchItems:      s.metrics.batchItems.Load(),
			GeneratedTokens: s.metrics.generatedTokens.Load(),
			AvgLatencyMS:    s.avgLatencyMillis(),
			AvgQueueWaitMS:  s.avgQueueWaitMillis(),
			LastError:       s.lastError(),
		},
		Model: statsModel{
			VocabSize: cfg.VocabSize,
			Text: statsTextModel{
				Layers:  cfg.NumHiddenLayers,
				Hidden:  cfg.HiddenSize,
				Heads:   cfg.NumAttentionHeads,
				KVHeads: cfg.NumKeyValueHeads,
				HeadDim: cfg.HeadDim,
			},
			Vision: statsVisionModel{
				Layers: cfg.VisionConfig.NumHiddenLayers,
				Hidden: cfg.VisionConfig.HiddenSize,
				Heads:  cfg.VisionConfig.NumAttentionHeads,
				Patch:  cfg.VisionConfig.PatchSize,
			},
		},
		Cache: statsCache{
			TaskPrompts: len(s.taskIDs),
			Runtime:     s.rt.CacheStats(),
			Tokenizer:   s.tok.CacheStats(),
		},
	}
}

func backendMemory() backend.MemoryInfo {
	return backend.Memory()
}

var adminTemplate = template.Must(template.New("admin").Parse(adminHTML))
var docTemplate = template.Must(template.New("doc").Parse(docHTML))

const docHTML = `<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PaddleOCR-VL API 文档</title>
<style>
body{margin:0;background:#f6f8fb;color:#17202a;font-family:Inter,Segoe UI,Arial,sans-serif;letter-spacing:0}.wrap{max-width:1060px;margin:0 auto;padding:34px 22px 70px}.hero{border-bottom:1px solid #d8dee6;padding-bottom:20px;margin-bottom:22px}h1{font-size:30px;margin:0 0 8px}h2{font-size:20px;margin:28px 0 12px}h3{font-size:16px;margin:18px 0 10px}p,li{color:#475467;line-height:1.65}.card{background:#fff;border:1px solid #d8dee6;border-radius:8px;padding:18px;margin:14px 0;box-shadow:0 10px 24px rgba(16,24,40,.06)}code,pre{font-family:Consolas,Menlo,monospace}code{background:#eef2f6;border-radius:4px;padding:2px 5px;color:#0f3f3a}pre{overflow:auto;background:#101828;color:#e5e7eb;border-radius:8px;padding:15px;line-height:1.55}.pill{display:inline-block;border:1px solid #99d6cf;color:#0f766e;background:#ecfdf9;border-radius:999px;padding:3px 9px;font-size:12px;font-weight:700}.grid{display:grid;grid-template-columns:180px 1fr;gap:0;border:1px solid #d8dee6;border-radius:8px;overflow:hidden}.grid div{padding:10px 12px;border-bottom:1px solid #e8edf3}.grid div:nth-child(odd){background:#f9fafb;font-weight:700;color:#344054}.grid div:nth-last-child(-n+2){border-bottom:0}a{color:#0f766e;font-weight:700}@media(max-width:760px){.grid{grid-template-columns:1fr}.grid div:nth-child(odd){border-bottom:0}}
</style>
</head>
<body>
<main class="wrap">
  <section class="hero">
    <span class="pill">Local Inference API</span>
    <h1>PaddleOCR-VL API 文档</h1>
    <p>Base URL: <code>{{.BaseURL}}</code>。推理接口需要后台生成的 API Key：<code>Authorization: Bearer &lt;API_KEY&gt;</code> 或 <code>X-API-Key: &lt;API_KEY&gt;</code>。</p>
    <p>机器可读 OpenAPI：<a href="/doc/openapi.json">/doc/openapi.json</a>；AI 接入说明：<a href="/doc/llms.txt">/doc/llms.txt</a></p>
  </section>

  <section class="card">
    <h2>认证</h2>
    <p>在管理后台 <code>/admin</code> 的“凭证”页生成 API Key。Key 明文只显示一次，可设置请求额度，额度为 <code>0</code> 表示不限量。</p>
    <pre>Authorization: Bearer &lt;API_KEY&gt;
X-API-Key: &lt;API_KEY&gt;</pre>
  </section>

  <section class="card">
    <h2>端点总览</h2>
    <div class="grid">
      <div>GET /health</div><div>基础健康检查，不需要 API Key。</div>
      <div>GET /ready</div><div>模型、tokenizer、并发槽位就绪状态，不需要 API Key。</div>
      <div>GET /stats</div><div>运行指标、模型信息、权重路径、内存、请求统计，不需要 API Key。</div>
      <div>POST /v1/ocr</div><div>multipart 图片 OCR，适合最简单接入。</div>
      <div>POST /v1/generate</div><div>JSON 推理，支持 prompt、tokens、task、image_path、image_base64。</div>
      <div>POST /v1/batch</div><div>批量 JSON 推理，顺序返回。</div>
    </div>
  </section>

  <section class="card">
    <h2>POST /v1/ocr</h2>
    <p>Content-Type: <code>multipart/form-data</code></p>
    <div class="grid">
      <div>image</div><div>必填，上传图片文件。</div>
      <div>task</div><div>可选，<code>ocr</code>、<code>table</code>、<code>formula</code>、<code>chart</code>，默认 <code>ocr</code>。</div>
      <div>max_new_tokens</div><div>可选，默认 <code>1024</code>。</div>
    </div>
    <pre>curl -X POST "{{.BaseURL}}/v1/ocr" \
  -H "Authorization: Bearer &lt;API_KEY&gt;" \
  -F "image=@page.png" \
  -F "task=ocr" \
  -F "max_new_tokens=1024"</pre>
  </section>

  <section class="card">
    <h2>POST /v1/generate</h2>
    <p>Content-Type: <code>application/json</code></p>
    <div class="grid">
      <div>prompt</div><div>文本 prompt，与 <code>tokens</code>、<code>task</code> 三选一。</div>
      <div>task</div><div><code>ocr</code>、<code>table</code>、<code>formula</code>、<code>chart</code>。</div>
      <div>tokens</div><div>整数 token id 数组。</div>
      <div>image_path</div><div>服务器本机图片路径。</div>
      <div>image_base64</div><div>base64 图片，可含 data URL 前缀。</div>
      <div>max_new_tokens</div><div>生成 token 上限，默认 <code>16</code>，受服务启动参数限制。</div>
      <div>temperature</div><div>采样温度，<code>0</code> 为贪心。</div>
      <div>top_k</div><div>Top-K 采样。</div>
      <div>decode</div><div>是否返回文本。</div>
      <div>decode_generated_only</div><div>只解码新增 token。</div>
      <div>skip_special</div><div>解码时跳过特殊 token。</div>
    </div>
    <pre>curl -X POST "{{.BaseURL}}/v1/generate" \
  -H "Authorization: Bearer &lt;API_KEY&gt;" \
  -H "Content-Type: application/json" \
  -d "{\"task\":\"ocr\",\"image_base64\":\"&lt;base64&gt;\",\"max_new_tokens\":1024,\"decode\":true,\"decode_generated_only\":true,\"skip_special\":true}"</pre>
  </section>

  <section class="card">
    <h2>POST /v1/batch</h2>
    <p>批量请求字段为 <code>requests</code>，每项结构同 <code>/v1/generate</code>。响应顺序与请求顺序一致。</p>
    <pre>curl -X POST "{{.BaseURL}}/v1/batch" \
  -H "Authorization: Bearer &lt;API_KEY&gt;" \
  -H "Content-Type: application/json" \
  -d "{\"requests\":[{\"prompt\":\"&lt;|begin_of_sentence|&gt;hello\",\"max_new_tokens\":1},{\"task\":\"ocr\",\"image_path\":\"D:\\docs\\page.png\",\"decode\":true,\"decode_generated_only\":true,\"skip_special\":true}]}"</pre>
  </section>

  <section class="card">
    <h2>响应格式</h2>
    <h3>生成响应</h3>
    <pre>{
  "tokens": [1, 2, 3],
  "prompt_tokens": 12,
  "generated_tokens": 32,
  "text": "识别结果"
}</pre>
    <h3>批量响应</h3>
    <pre>{
  "responses": [],
  "items": 2,
  "generated_tokens": 64
}</pre>
    <h3>错误响应</h3>
    <pre>{"error":"missing API key"}</pre>
  </section>

  <section class="card">
    <h2>状态与运维</h2>
    <pre>curl "{{.BaseURL}}/health"
curl "{{.BaseURL}}/ready"
curl "{{.BaseURL}}/stats"</pre>
    <p><code>/stats</code> 包含 <code>weight_path</code>、<code>weight_source</code>、<code>weight_sha256</code>、量化模式、平均延迟、队列等待、内存和模型维度。</p>
  </section>
</main>
</body>
</html>`

const adminHTML = `<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PaddleOCR-VL 管理后台</title>
<style>
:root{--bg:#f4f6f8;--panel:#fff;--text:#17202a;--muted:#667085;--line:#d8dee6;--brand:#0f766e;--brand2:#155e75;--danger:#b42318;--warn:#b54708;--ok:#027a48;--shadow:0 12px 28px rgba(16,24,40,.08)}
*{box-sizing:border-box}body{margin:0;font-family:Inter,Segoe UI,Arial,sans-serif;background:var(--bg);color:var(--text);letter-spacing:0}button,input{font:inherit}button{cursor:pointer;border:0;border-radius:6px;padding:10px 14px;background:var(--brand);color:white;font-weight:650}button.secondary{background:#eef4f3;color:#134e4a}button.ghost{background:transparent;color:var(--muted)}button.danger{background:var(--danger)}input,select{width:100%;border:1px solid var(--line);border-radius:6px;padding:10px 11px;background:#fff;color:var(--text)}label{display:block;font-size:12px;font-weight:700;color:#344054;margin:0 0 7px}.shell{display:grid;grid-template-columns:248px 1fr;min-height:100vh}.side{background:#111827;color:#d1d5db;padding:22px 18px;display:flex;flex-direction:column;gap:20px}.brand{color:#fff;font-size:18px;font-weight:800}.brand span{display:block;color:#9ca3af;font-size:12px;font-weight:600;margin-top:5px}.nav{display:grid;gap:6px}.nav button{text-align:left;background:transparent;color:#cbd5e1;padding:10px 12px}.nav button.active{background:#1f2937;color:#fff}.sidefoot{margin-top:auto;font-size:12px;color:#9ca3af;line-height:1.6}.main{padding:26px 30px}.top{display:flex;align-items:center;justify-content:space-between;margin-bottom:20px}.title h1{margin:0;font-size:24px}.title p{margin:6px 0 0;color:var(--muted)}.status{display:flex;align-items:center;gap:9px;background:#fff;border:1px solid var(--line);border-radius:8px;padding:9px 12px}.dot{width:9px;height:9px;border-radius:99px;background:var(--ok)}.dot.bad{background:var(--danger)}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px}.grid.two{grid-template-columns:1fr 1fr}.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;box-shadow:var(--shadow);padding:16px}.metric .k{font-size:12px;color:var(--muted);font-weight:700}.metric .v{font-size:24px;font-weight:800;margin-top:8px}.section{margin-top:16px}.row{display:grid;grid-template-columns:180px 1fr;gap:12px;padding:10px 0;border-bottom:1px solid #eef2f6}.row:last-child{border-bottom:0}.key{color:var(--muted);font-size:13px}.val{font-family:Consolas,Menlo,monospace;font-size:13px;word-break:break-all}.actions{display:flex;gap:10px;align-items:center;margin-top:14px}.formgrid{display:grid;grid-template-columns:1fr 1fr;gap:14px}.full{grid-column:1/-1}.notice{border-radius:7px;padding:10px 12px;background:#fff7ed;color:var(--warn);border:1px solid #fed7aa}.oktxt{color:var(--ok)}.badtxt{color:var(--danger)}.auth{min-height:100vh;display:grid;place-items:center;background:linear-gradient(180deg,#eef4f3,#f7fafc)}.authbox{width:min(440px,calc(100vw - 32px));background:#fff;border:1px solid var(--line);border-radius:8px;box-shadow:var(--shadow);padding:24px}.authbox h1{margin:0;font-size:24px}.authbox p{color:var(--muted);line-height:1.55}.stack{display:grid;gap:14px}.hide{display:none}.toast{position:fixed;right:22px;bottom:22px;background:#111827;color:#fff;border-radius:7px;padding:11px 14px;box-shadow:var(--shadow);opacity:0;transform:translateY(8px);transition:.2s}.toast.show{opacity:1;transform:none}@media(max-width:900px){.shell{grid-template-columns:1fr}.side{position:static}.grid,.grid.two,.formgrid{grid-template-columns:1fr}.main{padding:18px}.row{grid-template-columns:1fr}.top{align-items:flex-start;gap:12px;flex-direction:column}}
</style>
</head>
<body>
<div id="auth" class="auth">
  <div class="authbox">
    <h1 id="authTitle">PaddleOCR-VL 管理后台</h1>
    <p id="authHint">正在检查初始化状态</p>
    <div id="initForm" class="stack hide">
      <div><label>管理员账号</label><input id="initUser" value="admin"></div>
      <div><label>管理员密码</label><input id="initPassword" type="password" autocomplete="new-password"></div>
      <div><label>本机模型路径</label><input id="initModelDir"></div>
      <div><label>下载后处理路径</label><input id="initPostDir" placeholder="可选"></div>
      <button onclick="initAdmin()">完成初始化</button>
    </div>
    <div id="loginForm" class="stack hide">
      <div><label>管理员账号</label><input id="loginUser" value="admin"></div>
      <div><label>管理员密码</label><input id="loginPassword" type="password" autocomplete="current-password"></div>
      <button onclick="login()">登录</button>
    </div>
  </div>
</div>
<div id="app" class="shell hide">
  <aside class="side">
    <div class="brand">PaddleOCR-VL<span>Inference Admin Console</span></div>
    <nav class="nav">
      <button id="navDashboard" class="active" onclick="showTab('dashboard')">总览</button>
      <button id="navSettings" onclick="showTab('settings')">设置</button>
      <button id="navKeys" onclick="showTab('keys')">凭证</button>
      <button id="navAudit" onclick="showTab('audit')">审计</button>
      <button id="navApi" onclick="showTab('api')">API 文档</button>
    </nav>
    <div class="sidefoot">
      <div>本机推理服务</div>
      <div id="sideHost"></div>
      <button class="ghost" onclick="logout()">退出登录</button>
    </div>
  </aside>
  <main class="main">
    <div class="top">
      <div class="title"><h1 id="pageTitle">服务总览</h1><p id="pageSub">模型、API 与运行状态</p></div>
      <div class="status"><span id="readyDot" class="dot"></span><span id="readyText">ready</span></div>
    </div>
    <section id="dashboard">
      <div class="grid">
        <div class="card metric"><div class="k">请求成功</div><div id="mOk" class="v">0</div></div>
        <div class="card metric"><div class="k">请求失败</div><div id="mFail" class="v">0</div></div>
        <div class="card metric"><div class="k">平均耗时 ms</div><div id="mLatency" class="v">0</div></div>
        <div class="card metric"><div class="k">可用槽位</div><div id="mSlots" class="v">0</div></div>
      </div>
      <div class="section grid two">
        <div class="card"><h3>模型状态</h3><div id="modelRows"></div></div>
        <div class="card"><h3>运行配置</h3><div id="runtimeRows"></div></div>
      </div>
    </section>
    <section id="settings" class="hide">
      <div class="card">
        <h3>管理员与模型设置</h3>
        <div id="restartNotice" class="notice hide">模型路径已变更，需要重启服务后生效。</div>
        <div class="formgrid section">
          <div><label>管理员账号</label><input id="cfgUser"></div>
          <div><label>新管理员密码</label><input id="cfgPassword" type="password" placeholder="留空不修改"></div>
          <div class="full"><label>本机模型路径</label><input id="cfgModelDir"></div>
          <div class="full"><label>下载后处理路径</label><input id="cfgPostDir"></div>
        </div>
        <div class="actions"><button onclick="saveConfig()">保存设置</button><button class="secondary" onclick="validateModel()">校验模型路径</button><button class="secondary" onclick="exportConfig()">导出备份</button><input id="restoreFile" type="file" accept="application/json" onchange="restoreConfig()" style="max-width:260px"></div>
      </div>
    </section>
    <section id="keys" class="hide">
      <div class="card">
        <h3>API Key 与配额</h3>
        <div class="formgrid section">
          <div><label>名称</label><input id="newKeyName" placeholder="例如 ai-agent-prod"></div>
          <div><label>请求额度</label><input id="newKeyQuota" type="number" min="0" value="0"></div>
          <div><label>每分钟限制</label><input id="newKeyRate" type="number" min="0" value="0"></div>
        </div>
        <div class="actions"><button onclick="createKey()">生成 API Key</button><span class="key">额度和每分钟限制为 0 表示不限；明文只显示一次。</span></div>
        <div id="newKeyBox" class="notice section hide"></div>
        <div id="keyRows" class="section"></div>
      </div>
    </section>
    <section id="audit" class="hide">
      <div class="card">
        <h3>调用审计</h3>
        <div class="actions"><button class="secondary" onclick="loadAudit()">刷新</button><button class="secondary" onclick="exportAudit()">导出 CSV</button><button class="danger" onclick="clearAudit()">清空审计</button></div>
        <div id="auditRows" class="section"></div>
      </div>
    </section>
    <section id="api" class="hide">
      <div class="card">
        <h3>API URL</h3>
        <div id="apiRows"></div>
        <div class="actions"><button onclick="copyCurl()">复制 OCR curl 示例</button><a href="/doc" target="_blank">打开完整文档</a></div>
      </div>
    </section>
  </main>
</div>
<div id="toast" class="toast"></div>
<script>
let overview=null,overviewTimer=null;
const $=id=>document.getElementById(id);
async function api(path,opts={}){const r=await fetch(path,{headers:{'Content-Type':'application/json'},...opts});const t=await r.text();let j={};try{j=t?JSON.parse(t):{};}catch(e){}if(!r.ok)throw new Error(j.error||r.statusText);return j;}
function toast(msg){const el=$('toast');el.textContent=msg;el.classList.add('show');setTimeout(()=>el.classList.remove('show'),2600);}
function stopOverviewTimer(){if(overviewTimer){clearInterval(overviewTimer);overviewTimer=null;}}
function showAuth(init){stopOverviewTimer();$('auth').classList.remove('hide');$('app').classList.add('hide');$('initForm').classList.toggle('hide',!init);$('loginForm').classList.toggle('hide',init);$('authTitle').textContent=init?'首次初始化':'PaddleOCR-VL 管理后台';$('authHint').textContent=init?'创建管理员、API Key，并确认本机模型路径。':'使用管理员账号登录后台。';}
async function boot(){try{const s=await api('/admin/api/session');if(!s.initialized){showAuth(true);return;}if(!s.authenticated){showAuth(false);return;}$('auth').classList.add('hide');$('app').classList.remove('hide');await loadOverview();if(!overviewTimer)overviewTimer=setInterval(loadOverview,5000);}catch(e){showAuth(false);}}
async function initAdmin(){try{const j=await api('/admin/api/init',{method:'POST',body:JSON.stringify({admin_user:$('initUser').value,password:$('initPassword').value,model_dir:$('initModelDir').value,post_process_dir:$('initPostDir').value})});toast('初始化完成，默认 API Key 已生成');if(j.api_key){$('newKeyBox').textContent='默认 API Key：'+j.api_key;$('newKeyBox').classList.remove('hide');navigator.clipboard?.writeText(j.api_key);}boot();}catch(e){toast(e.message);}}
async function login(){try{await api('/admin/api/login',{method:'POST',body:JSON.stringify({admin_user:$('loginUser').value,password:$('loginPassword').value})});boot();}catch(e){toast(e.message);}}
async function logout(){await api('/admin/api/logout',{method:'POST'});showAuth(false);}
function showTab(tab){for(const id of ['dashboard','settings','keys','audit','api'])$(id).classList.toggle('hide',id!==tab);for(const id of ['Dashboard','Settings','Keys','Audit','Api'])$('nav'+id).classList.remove('active');$('nav'+tab[0].toUpperCase()+tab.slice(1)).classList.add('active');$('pageTitle').textContent=tab==='dashboard'?'服务总览':tab==='settings'?'系统设置':tab==='keys'?'能力凭证':tab==='audit'?'调用审计':'API 文档';$('pageSub').textContent=tab==='dashboard'?'模型、API 与运行状态':tab==='settings'?'管理员与模型路径':tab==='keys'?'签发、额度、启停与删除':tab==='audit'?'最近请求、来源、状态与耗时':'服务端点与调用示例';if(tab==='audit')loadAudit();}
async function loadOverview(){overview=await api('/admin/api/overview');const st=overview.stats,ad=overview.admin;$('sideHost').textContent=overview.current_host;$('readyText').textContent=overview.ready.status;$('readyDot').classList.toggle('bad',overview.ready.status!=='ready');$('mOk').textContent=st.requests.succeeded;$('mFail').textContent=st.requests.failed;$('mLatency').textContent=st.requests.avg_latency_ms;$('mSlots').textContent=st.available_slots+'/'+st.concurrency;$('modelRows').innerHTML=rows([['本机模型路径',overview.model_dir],['当前权重',st.weight_path],['权重来源',st.weight_source],['SHA256',st.weight_sha256],['量化',st.quantization],['后端',st.backend]]);$('runtimeRows').innerHTML=rows([['API Base URL',overview.api_base_url],['API 文档',overview.api_base_url+'/doc'],['OpenAPI JSON',overview.api_base_url+'/doc/openapi.json'],['AI 接入说明',overview.api_base_url+'/doc/llms.txt'],['Max New Tokens',st.max_new_limit],['Batch 上限',st.max_batch_size||'不限'],['内存',formatBytes(st.memory.heap_alloc)],['运行时间',st.uptime_seconds+' 秒'],['最后错误',st.requests.last_error||'无']]);$('cfgUser').value=ad.admin_user;$('cfgModelDir').value=ad.model_dir;$('cfgPostDir').value=ad.post_process_dir;$('restartNotice').classList.toggle('hide',!ad.restart_required);$('apiRows').innerHTML=rows([['完整文档',overview.api_base_url+'/doc'],['OpenAPI JSON',overview.api_base_url+'/doc/openapi.json'],['AI 接入说明',overview.api_base_url+'/doc/llms.txt'],['Health',overview.api_base_url+'/health'],['Ready',overview.api_base_url+'/ready'],['Stats',overview.api_base_url+'/stats'],['OCR',overview.api_base_url+'/v1/ocr'],['Generate',overview.api_base_url+'/v1/generate'],['Batch',overview.api_base_url+'/v1/batch']]);await loadKeys();await loadAudit();}
function rows(items){return items.map(([k,v])=>'<div class="row"><div class="key">'+esc(k)+'</div><div class="val">'+esc(String(v??''))+'</div></div>').join('');}
function esc(s){return s.replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function formatBytes(n){if(!n)return '0 B';const u=['B','KB','MB','GB'];let i=0;while(n>=1024&&i<u.length-1){n/=1024;i++;}return n.toFixed(i?1:0)+' '+u[i];}
async function saveConfig(){try{await api('/admin/api/config',{method:'POST',body:JSON.stringify({admin_user:$('cfgUser').value,password:$('cfgPassword').value,model_dir:$('cfgModelDir').value,post_process_dir:$('cfgPostDir').value})});$('cfgPassword').value='';toast('设置已保存');await loadOverview();}catch(e){toast(e.message);}}
function exportConfig(){location.href='/admin/api/config/backup';}
async function restoreConfig(){const f=$('restoreFile').files[0];if(!f)return;if(!confirm('确认从备份恢复后台配置？当前管理员、API Key 和路径设置会被覆盖。')){$('restoreFile').value='';return;}try{const text=await f.text();await api('/admin/api/config/restore',{method:'POST',body:text});$('restoreFile').value='';toast('配置已恢复');await loadOverview();}catch(e){toast(e.message);}}
async function loadKeys(){const j=await api('/admin/api/keys');$('keyRows').innerHTML=j.keys.length?j.keys.map(keyCard).join(''):'<div class="notice">暂无 API Key，请生成后交给接入方使用。</div>';}
function keyField(prefix,id){return prefix+String(id).replace(/[^A-Za-z0-9_-]/g,'_');}
function jsArg(v){return escAttr(JSON.stringify(String(v)));}
function keyCard(k){const remain=k.remaining<0?'不限':k.remaining;const rate=k.rate_per_minute||0;const last=k.last_used_at?('最近 '+k.last_used_at+(k.last_used_ip?' · '+k.last_used_ip:'')):'尚未调用';const id=jsArg(k.id);const nameID=escAttr(keyField('name_',k.id));const quotaID=escAttr(keyField('quota_',k.id));const rateID=escAttr(keyField('rate_',k.id));return '<div class="row"><div><input id="'+nameID+'" value="'+escAttr(k.name)+'" title="名称" style="max-width:220px"><div class="key">'+esc(k.preview)+' · 已用 '+k.used+' · 剩余 '+remain+' · 每分钟 '+(rate||'不限')+'</div><div class="key">'+esc(last)+'</div></div><div class="actions"><input id="'+quotaID+'" type="number" min="0" value="'+k.quota+'" title="请求额度" style="max-width:112px"><input id="'+rateID+'" type="number" min="0" value="'+rate+'" title="每分钟限制" style="max-width:112px"><button class="secondary" onclick="updateKey('+id+','+!k.disabled+')">'+(k.disabled?'启用':'禁用')+'</button><button class="secondary" onclick="saveKeyQuota('+id+','+k.disabled+')">保存</button><button class="secondary" onclick="rotateKey('+id+')">轮换密钥</button><button class="secondary" onclick="resetKeyUsage('+id+')">清零用量</button><button class="danger" onclick="deleteKey('+id+')">删除</button></div></div>';}
async function createKey(){try{const j=await api('/admin/api/keys',{method:'POST',body:JSON.stringify({name:$('newKeyName').value,quota:Number($('newKeyQuota').value||0),rate_per_minute:Number($('newKeyRate').value||0)})});$('newKeyBox').textContent='新 API Key：'+j.api_key;$('newKeyBox').classList.remove('hide');navigator.clipboard?.writeText(j.api_key);toast('API Key 已生成，明文已复制');await loadKeys();}catch(e){toast(e.message);}}
async function updateKey(id,disabled){await api('/admin/api/keys/update',{method:'POST',body:JSON.stringify({id,name:$(keyField('name_',id)).value,quota:Number($(keyField('quota_',id)).value||0),rate_per_minute:Number($(keyField('rate_',id)).value||0),disabled})});await loadKeys();}
async function saveKeyQuota(id,disabled){await updateKey(id,disabled);toast('凭证已保存');}
async function rotateKey(id){if(!confirm('确认轮换这个 API Key？旧密钥会立即失效。'))return;const j=await api('/admin/api/keys/rotate',{method:'POST',body:JSON.stringify({id})});$('newKeyBox').textContent='轮换后的新 API Key：'+j.api_key;$('newKeyBox').classList.remove('hide');navigator.clipboard?.writeText(j.api_key);toast('新密钥已生成，明文已复制');await loadKeys();}
async function resetKeyUsage(id){if(!confirm('确认清零这个 API Key 的已用次数？'))return;await api('/admin/api/keys/reset',{method:'POST',body:JSON.stringify({id})});toast('用量已清零');await loadKeys();}
async function deleteKey(id){if(!confirm('确认删除这个 API Key？'))return;await api('/admin/api/keys/delete',{method:'POST',body:JSON.stringify({id})});await loadKeys();}
async function loadAudit(){const el=$('auditRows');if(!el)return;const j=await api('/admin/api/audit');el.innerHTML=j.items.length?j.items.map(auditCard).join(''):'<div class="notice">暂无调用记录。</div>';}
function auditCard(e){const bad=e.status>=400?' badtxt':' oktxt';const key=e.key_name?e.key_name+' '+(e.key_preview||''):'未识别 Key';return '<div class="row"><div><b class="'+bad+'">'+e.status+'</b><div class="key">'+esc(e.time)+' · '+esc(e.duration_ms+' ms')+'</div></div><div><div class="val">'+esc(e.method+' '+e.path)+'</div><div class="key">'+esc(key)+' · '+esc(e.client_ip||'')+(e.error?' · '+esc(e.error):'')+'</div></div></div>';}
function exportAudit(){location.href='/admin/api/audit.csv';}
async function clearAudit(){if(!confirm('确认清空最近调用审计？'))return;await api('/admin/api/audit/clear',{method:'POST'});toast('审计已清空');await loadAudit();}
function escAttr(s){return esc(String(s));}
async function validateModel(){try{const j=await api('/admin/api/validate-model-dir',{method:'POST',body:JSON.stringify({model_dir:$('cfgModelDir').value})});toast(j.ok?'模型路径有效':j.error);}catch(e){toast(e.message);}}
function copyCurl(){const base=overview?.api_base_url||location.origin;const cmd='curl -X POST '+base+'/v1/ocr -H "Authorization: Bearer <API_KEY>" -F "image=@page.png" -F "task=ocr"';navigator.clipboard.writeText(cmd);toast('curl 已复制');}
boot();
</script>
</body>
</html>`
