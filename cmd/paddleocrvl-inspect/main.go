package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"paddleocrvl-go/internal/config"
	"paddleocrvl-go/internal/fileutil"
	"paddleocrvl-go/internal/gguf"
	"paddleocrvl-go/internal/safetensors"
)

type tensorMeta struct {
	DType string
	Shape []int64
	Bytes int64
}

type weightInfo struct {
	format       string
	quantization string
	path         string
	fileBytes    int64
	files        []weightFile
	tensors      map[string]tensorMeta
	metadata     map[string]any
}

type weightFile struct {
	Path   string `json:"path"`
	Bytes  int64  `json:"bytes"`
	SHA256 string `json:"sha256"`
}

const maxInspectWeightFiles = 4096

type inspectSummary struct {
	ModelType    string                `json:"model_type"`
	WeightFormat string                `json:"weight_format"`
	Quantization string                `json:"quantization"`
	WeightPath   string                `json:"weight_path"`
	WeightBytes  int64                 `json:"weight_file_bytes"`
	WeightFiles  []weightFile          `json:"weight_files"`
	VocabSize    int                   `json:"vocab_size"`
	Text         map[string]int        `json:"text"`
	Vision       map[string]int        `json:"vision"`
	Tensors      int                   `json:"tensors"`
	Bytes        int64                 `json:"bytes"`
	DTypes       map[string]int        `json:"dtypes"`
	Metadata     map[string]any        `json:"metadata,omitempty"`
	FirstTensors map[string]tensorMeta `json:"first_tensors"`
	Largest      map[string]tensorMeta `json:"largest_tensors"`
}

func main() {
	jsonOut := flag.Bool("json", false, "print machine-readable JSON")
	flag.Parse()
	dir := "."
	if flag.NArg() > 0 {
		dir = flag.Arg(0)
	}
	cfg, err := config.Load(dir)
	if err != nil {
		log.Fatal(err)
	}
	info, err := readTensors(dir)
	if err != nil {
		log.Fatal(err)
	}

	var totalBytes int64
	dtypes := make(map[string]int, 4)
	keys := make([]string, 0, len(info.tensors))
	for name, meta := range info.tensors {
		keys = append(keys, name)
		dtypes[meta.DType]++
		totalBytes += meta.Bytes
	}
	sort.Strings(keys)
	first := firstTensorMap(keys, info.tensors, 16)
	largest := largestTensorMap(info.tensors, 8)
	if *jsonOut {
		summary := inspectSummary{
			ModelType:    "paddleocr_vl",
			WeightFormat: info.format,
			Quantization: info.quantization,
			WeightPath:   info.path,
			WeightBytes:  info.fileBytes,
			WeightFiles:  info.files,
			VocabSize:    cfg.VocabSize,
			Text: map[string]int{
				"layers":   cfg.NumHiddenLayers,
				"hidden":   cfg.HiddenSize,
				"heads":    cfg.NumAttentionHeads,
				"kv_heads": cfg.NumKeyValueHeads,
				"head_dim": cfg.HeadDim,
			},
			Vision: map[string]int{
				"layers": cfg.VisionConfig.NumHiddenLayers,
				"hidden": cfg.VisionConfig.HiddenSize,
				"heads":  cfg.VisionConfig.NumAttentionHeads,
				"patch":  cfg.VisionConfig.PatchSize,
				"image":  cfg.VisionConfig.ImageSize,
			},
			Tensors:      len(info.tensors),
			Bytes:        totalBytes,
			DTypes:       dtypes,
			Metadata:     filteredMetadata(info.metadata),
			FirstTensors: first,
			Largest:      largest,
		}
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(summary); err != nil {
			log.Fatal(err)
		}
		return
	}
	fmt.Printf("model_type: paddleocr_vl\n")
	fmt.Printf("weight_format: %s\n", info.format)
	fmt.Printf("quantization: %s\n", info.quantization)
	fmt.Printf("weight_path: %s\n", info.path)
	fmt.Printf("weight_file_bytes: %d\n", info.fileBytes)
	for _, item := range info.files {
		fmt.Printf("weight_file: %s bytes=%d sha256=%s\n", item.Path, item.Bytes, item.SHA256)
	}
	fmt.Printf("vocab_size: %d\n", cfg.VocabSize)
	fmt.Printf("text_layers: %d hidden=%d heads=%d kv_heads=%d head_dim=%d\n",
		cfg.NumHiddenLayers, cfg.HiddenSize, cfg.NumAttentionHeads, cfg.NumKeyValueHeads, cfg.HeadDim)
	fmt.Printf("vision_layers: %d hidden=%d heads=%d patch=%d image=%d\n",
		cfg.VisionConfig.NumHiddenLayers, cfg.VisionConfig.HiddenSize, cfg.VisionConfig.NumAttentionHeads,
		cfg.VisionConfig.PatchSize, cfg.VisionConfig.ImageSize)
	fmt.Printf("tensors: %d bytes=%d\n", len(info.tensors), totalBytes)
	fmt.Printf("dtypes:")
	for dtype, n := range dtypes {
		fmt.Printf(" %s=%d", dtype, n)
	}
	fmt.Println()
	if len(info.metadata) > 0 {
		fmt.Printf("metadata:")
		meta := filteredMetadata(info.metadata)
		metaKeys := make([]string, 0, len(meta))
		for k := range meta {
			metaKeys = append(metaKeys, k)
		}
		sort.Strings(metaKeys)
		for _, k := range metaKeys {
			fmt.Printf(" %s=%v", k, meta[k])
		}
		fmt.Println()
	}
	fmt.Println("first_tensors:")
	for _, name := range keys[:min(16, len(keys))] {
		meta := first[name]
		fmt.Printf("  %s %s %v\n", name, meta.DType, meta.Shape)
	}
	fmt.Println("largest_tensors:")
	largestKeys := make([]string, 0, len(largest))
	for name := range largest {
		largestKeys = append(largestKeys, name)
	}
	sort.Slice(largestKeys, func(i, j int) bool {
		return largest[largestKeys[i]].Bytes > largest[largestKeys[j]].Bytes
	})
	for _, name := range largestKeys {
		meta := largest[name]
		fmt.Printf("  %s %s bytes=%d %v\n", name, meta.DType, meta.Bytes, meta.Shape)
	}
}

func filteredMetadata(metadata map[string]any) map[string]any {
	if len(metadata) == 0 {
		return nil
	}
	out := make(map[string]any, min(len(metadata), 8))
	for k, v := range metadata {
		if strings.HasPrefix(k, "paddleocrvl.") || k == "general.file_type" || k == "general.name" || k == "general.architecture" {
			out[k] = v
		}
	}
	return out
}

func firstTensorMap(keys []string, tensors map[string]tensorMeta, limit int) map[string]tensorMeta {
	n := min(limit, len(keys))
	out := make(map[string]tensorMeta, n)
	for _, name := range keys[:n] {
		out[name] = tensors[name]
	}
	return out
}

func largestTensorMap(tensors map[string]tensorMeta, limit int) map[string]tensorMeta {
	keys := make([]string, 0, len(tensors))
	for name := range tensors {
		keys = append(keys, name)
	}
	sort.Slice(keys, func(i, j int) bool {
		li, lj := tensors[keys[i]].Bytes, tensors[keys[j]].Bytes
		if li == lj {
			return keys[i] < keys[j]
		}
		return li > lj
	})
	n := min(limit, len(keys))
	out := make(map[string]tensorMeta, n)
	for _, name := range keys[:n] {
		out[name] = tensors[name]
	}
	return out
}

func readTensors(dir string) (weightInfo, error) {
	for _, file := range []string{"model-q4.gguf", "model-q6.gguf", "model-q8.gguf", "model.gguf"} {
		ggufPath := filepath.Join(dir, file)
		if st, err := os.Stat(ggufPath); err == nil {
			gf, err := gguf.Open(ggufPath)
			if err != nil {
				return weightInfo{}, err
			}
			defer gf.Close()
			out := make(map[string]tensorMeta, len(gf.Tensors))
			for name, meta := range gf.Tensors {
				out[name] = tensorMeta{DType: gguf.TensorTypeName(meta.Type), Shape: meta.Shape, Bytes: gguf.TensorBytes(meta)}
			}
			files := weightFiles([]string{ggufPath})
			return weightInfo{format: file, quantization: metadataString(gf.Metadata, "paddleocrvl.quantization", quantFromGGUFFile(file)), path: fileutil.Abs(ggufPath), fileBytes: st.Size(), files: files, tensors: out, metadata: gf.Metadata}, nil
		}
	}
	sf, err := safetensors.OpenModel(dir)
	if err != nil {
		return weightInfo{}, err
	}
	defer sf.Close()
	out := make(map[string]tensorMeta, len(sf.Tensors))
	for name, meta := range sf.Tensors {
		out[name] = tensorMeta{DType: meta.DType, Shape: meta.Shape, Bytes: meta.DataOffsets[1] - meta.DataOffsets[0]}
	}
	files := weightFiles(safetensorsFiles(dir))
	return weightInfo{format: "safetensors", quantization: "source", path: safetensorsPath(dir), fileBytes: sumWeightFiles(files), files: files, tensors: out}, nil
}

func safetensorsPath(dir string) string {
	index := filepath.Join(dir, "model.safetensors.index.json")
	if _, err := os.Stat(index); err == nil {
		return fileutil.Abs(index)
	}
	return fileutil.Abs(filepath.Join(dir, "model.safetensors"))
}

func safetensorsFileBytes(dir string) int64 {
	return sumWeightFiles(weightFiles(safetensorsFiles(dir)))
}

func safetensorsFiles(dir string) []string {
	seen := make(map[string]bool, 4)
	var out []string
	for _, pattern := range []string{"model.safetensors", "model.safetensors.index.json", "*.safetensors"} {
		matches, _ := filepath.Glob(filepath.Join(dir, pattern))
		for _, path := range matches {
			if seen[path] {
				continue
			}
			st, err := os.Stat(path)
			if err != nil || !st.Mode().IsRegular() {
				continue
			}
			seen[path] = true
			out = append(out, path)
			if len(out) >= maxInspectWeightFiles {
				sort.Strings(out)
				return out
			}
		}
	}
	sort.Strings(out)
	return out
}

func weightFiles(paths []string) []weightFile {
	out := make([]weightFile, 0, len(paths))
	for _, path := range paths {
		st, err := os.Stat(path)
		if err != nil {
			continue
		}
		if !st.Mode().IsRegular() {
			continue
		}
		out = append(out, weightFile{Path: fileutil.Abs(path), Bytes: st.Size(), SHA256: fileutil.SHA256(path)})
	}
	return out
}

func sumWeightFiles(files []weightFile) int64 {
	var total int64
	for _, item := range files {
		total += item.Bytes
	}
	return total
}

func metadataString(metadata map[string]any, key, fallback string) string {
	if v, ok := metadata[key].(string); ok && v != "" {
		return v
	}
	return fallback
}

func quantFromGGUFFile(file string) string {
	switch file {
	case "model-q8.gguf":
		return "q8"
	case "model-q6.gguf":
		return "q6"
	case "model-q4.gguf":
		return "q4"
	default:
		return "f32"
	}
}
