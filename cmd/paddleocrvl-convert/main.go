package main

import (
	"encoding/json"
	"flag"
	"log"
	"os"
	"path/filepath"
	"time"

	"paddleocrvl-go/internal/backend"
	"paddleocrvl-go/internal/fileutil"
	"paddleocrvl-go/internal/gguf"
)

func main() {
	modelDir := flag.String("model-dir", ".", "directory containing model.safetensors and config.json")
	out := flag.String("out", "", "output GGUF path; default is <model-dir>/model.gguf")
	quant := flag.String("quant", "f32", "output quantization: f32, q8, q6, or q4")
	progress := flag.Bool("progress", true, "log conversion progress")
	outputJSON := flag.Bool("json", false, "emit conversion summary as JSON")
	gomaxprocs := flag.Int("gomaxprocs", 0, "set Go GOMAXPROCS; 0 keeps current value")
	gcPercent := flag.Int("gc-percent", 0, "set Go GC percent; 0 keeps current value, -1 disables GC")
	flag.Parse()
	if _, err := backend.SetGOMAXPROCS(*gomaxprocs); err != nil {
		log.Fatal(err)
	}
	backend.SetGCPercent(*gcPercent)
	normalized := normalizedQuant(*quant)
	dst := *out
	if dst == "" {
		dst = defaultOutputPath(*modelDir, normalized)
	}
	if err := ensureOutputDir(dst); err != nil {
		log.Fatal(err)
	}
	start := time.Now()
	opts := gguf.ConvertOptions{Quantization: normalized}
	if *progress {
		last := time.Now()
		opts.Progress = func(done, total int, name string, typ string) {
			if done == total {
				log.Printf("converted %d/%d tensors", done, total)
				return
			}
			now := time.Now()
			if done == 0 || now.Sub(last) >= 2*time.Second {
				log.Printf("converting %d/%d %s %s", done+1, total, typ, name)
				last = now
			}
		}
	}
	if err := gguf.ConvertSafetensorsWithOptions(*modelDir, dst, *modelDir, opts); err != nil {
		log.Fatal(err)
	}
	size := int64(0)
	if st, err := os.Stat(dst); err == nil {
		size = st.Size()
	}
	outPath := fileutil.Abs(dst)
	sha := fileutil.SHA256(dst)
	meta := readConvertMetadata(dst)
	elapsed := time.Since(start)
	if *outputJSON {
		_ = json.NewEncoder(os.Stdout).Encode(convertSummary{
			Output:           outPath,
			Quant:            normalized,
			Bytes:            size,
			SHA256:           sha,
			ElapsedMS:        elapsed.Milliseconds(),
			ModelDir:         fileutil.Abs(*modelDir),
			ConfigPath:       fileutil.Abs(*modelDir),
			SourceTensors:    meta.SourceTensors,
			F32Tensors:       meta.F32Tensors,
			QuantizedTensors: meta.QuantizedTensors,
		})
		return
	}
	log.Printf("wrote %s quant=%s bytes=%d sha256=%s tensors=%d f32=%d quantized=%d elapsed=%s", outPath, normalized, size, sha, meta.SourceTensors, meta.F32Tensors, meta.QuantizedTensors, elapsed.Round(time.Millisecond))
}

type convertSummary struct {
	Output           string `json:"output"`
	Quant            string `json:"quant"`
	Bytes            int64  `json:"bytes"`
	SHA256           string `json:"sha256"`
	ElapsedMS        int64  `json:"elapsed_ms"`
	ModelDir         string `json:"model_dir"`
	ConfigPath       string `json:"config_path"`
	SourceTensors    uint64 `json:"source_tensors"`
	F32Tensors       uint64 `json:"f32_tensors"`
	QuantizedTensors uint64 `json:"quantized_tensors"`
}

type convertMetadata struct {
	SourceTensors    uint64
	F32Tensors       uint64
	QuantizedTensors uint64
}

func readConvertMetadata(path string) convertMetadata {
	gf, err := gguf.Open(path)
	if err != nil {
		return convertMetadata{}
	}
	defer gf.Close()
	return convertMetadata{
		SourceTensors:    metadataUint64(gf.Metadata, "paddleocrvl.source_tensors"),
		F32Tensors:       metadataUint64(gf.Metadata, "paddleocrvl.f32_tensors"),
		QuantizedTensors: metadataUint64(gf.Metadata, "paddleocrvl.quantized_tensors"),
	}
}

func metadataUint64(metadata map[string]any, key string) uint64 {
	if v, ok := metadata[key].(uint64); ok {
		return v
	}
	return 0
}

func ensureOutputDir(path string) error {
	dir := filepath.Dir(path)
	if dir == "." || dir == "" {
		return nil
	}
	return os.MkdirAll(dir, 0o755)
}

func defaultOutputPath(modelDir, quant string) string {
	switch normalizedQuant(quant) {
	case "q8":
		return filepath.Join(modelDir, "model-q8.gguf")
	case "q6":
		return filepath.Join(modelDir, "model-q6.gguf")
	case "q4":
		return filepath.Join(modelDir, "model-q4.gguf")
	default:
		return filepath.Join(modelDir, "model.gguf")
	}
}

func normalizedQuant(quant string) string {
	quant = lowerASCII(trimASCIIWhitespace(quant))
	if quant == "" {
		return "f32"
	}
	return quant
}

func trimASCIIWhitespace(s string) string {
	if len(s) == 0 || (!isASCIIWhitespace(s[0]) && !isASCIIWhitespace(s[len(s)-1])) {
		return s
	}
	start, end := 0, len(s)
	for start < end && isASCIIWhitespace(s[start]) {
		start++
	}
	for end > start && isASCIIWhitespace(s[end-1]) {
		end--
	}
	return s[start:end]
}

func isASCIIWhitespace(c byte) bool {
	return c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == '\v' || c == '\f'
}

func lowerASCII(s string) string {
	firstUpper := -1
	for i := 0; i < len(s); i++ {
		c := s[i]
		if 'A' <= c && c <= 'Z' {
			firstUpper = i
			break
		}
	}
	if firstUpper < 0 {
		return s
	}
	out := []byte(s)
	for i := firstUpper; i < len(out); i++ {
		c := out[i]
		if 'A' <= c && c <= 'Z' {
			out[i] = c + ('a' - 'A')
		}
	}
	return string(out)
}
