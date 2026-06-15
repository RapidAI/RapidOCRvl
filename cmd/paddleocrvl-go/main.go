package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"
	"time"

	"paddleocrvl-go/internal/backend"
	"paddleocrvl-go/internal/model"
	"paddleocrvl-go/internal/tokenizer"
)

func main() {
	modelDir := flag.String("model-dir", ".", "directory containing config.json and model.safetensors")
	quant := flag.String("quant", "f32", "weight quantization: f32, q8, q6, q4, auto, auto-fast, or auto-quality")
	backendName := flag.String("backend", "cpu", "compute backend: cpu, vulkan, or auto")
	gomaxprocs := flag.Int("gomaxprocs", 0, "set Go GOMAXPROCS; 0 keeps current value")
	gcPercent := flag.Int("gc-percent", 0, "set Go GC percent; 0 keeps current value, -1 disables GC")
	tokenCSV := flag.String("tokens", "", "comma-separated input token ids")
	prompt := flag.String("prompt", "", "text prompt to encode with tokenizer.json")
	task := flag.String("task", "", "document task prompt: ocr, table, formula, chart")
	imagePath := flag.String("image", "", "optional PNG/JPEG image path; replaces image placeholder token ids")
	maxNew := flag.Int("max-new-tokens", 16, "number of tokens to generate")
	temperature := flag.Float64("temperature", 0, "sampling temperature; 0 means greedy")
	topK := flag.Int("top-k", 0, "sample from top K tokens when temperature > 0; 0 means full vocab")
	seed := flag.Int64("seed", 0, "sampling seed; 0 uses current time")
	eosCSV := flag.String("eos", "2", "comma-separated EOS token ids")
	decode := flag.Bool("decode", true, "decode output ids with tokenizer.json when available")
	decodeGeneratedOnly := flag.Bool("decode-generated-only", false, "decode only newly generated tokens")
	skipSpecial := flag.Bool("skip-special", false, "skip special tokens during decode")
	statsOnly := flag.Bool("stats-only", false, "load model, print weight stats, then exit")
	verifyOnly := flag.Bool("verify-only", false, "load model and exit after verifying weights")
	verifyVision := flag.Bool("verify-vision", false, "also load vision weights during -verify-only or -stats-only")
	outputJSON := flag.Bool("json", false, "emit machine-readable JSON")
	flag.Parse()
	if _, err := backend.SetGOMAXPROCS(*gomaxprocs); err != nil {
		log.Fatal(err)
	}
	backend.SetGCPercent(*gcPercent)

	var ids []int
	var tok *tokenizer.Tokenizer
	var err error
	if *statsOnly || *verifyOnly {
		// Tokenizer and input parsing are unnecessary for load/convert checks.
	} else if *task != "" {
		if *imagePath == "" {
			log.Fatal("-task requires -image")
		}
		tok, err = tokenizer.Load(*modelDir)
		if err != nil {
			log.Fatal(err)
		}
		p, err := taskPrompt(*task)
		if err != nil {
			log.Fatal(err)
		}
		ids = tok.EncodeReadOnly(chatPrompt(p, true))
	} else if *prompt != "" {
		tok, err = tokenizer.Load(*modelDir)
		if err != nil {
			log.Fatal(err)
		}
		ids = tok.EncodeReadOnly(*prompt)
	} else {
		if trimASCIIWhitespace(*tokenCSV) == "" {
			log.Fatal("-tokens or -prompt is required")
		}
		ids, err = parseTokens(*tokenCSV)
		if err != nil {
			log.Fatal(err)
		}
	}

	backendSel, err := backend.Select(*backendName)
	if err != nil {
		log.Fatal(err)
	}
	rt, err := model.LoadWithOptions(*modelDir, model.LoadOptions{Quantization: *quant, Backend: backendSel.Active, Progress: model.ProgressLogger("loader")})
	if err != nil {
		log.Fatal(err)
	}
	defer rt.Close()
	if *verifyVision {
		if err := rt.PreloadVision(); err != nil {
			log.Fatal(err)
		}
	}
	if *statsOnly {
		cpu := backend.CPU()
		if *outputJSON {
			cmdPlan := rt.VulkanCommandPlan()
			cmdPlanErr := backend.ValidateVulkanCommandPlan(cmdPlan)
			writeJSON(cliStats{
				RequestedQuantization: rt.RequestedQuantization(),
				Quantization:          rt.Quantization(),
				WeightPath:            rt.WeightPath(),
				WeightSource:          rt.WeightSource(),
				Backend:               rt.Backend(),
				Weights:               rt.WeightStats(),
				LoadStats:             rt.LoadStats(),
				CacheStats:            rt.CacheStats(),
				CPU:                   cpu,
				Memory:                backend.Memory(),
				Backends:              backendSel,
				VulkanPlans:           rt.VulkanPlans(),
				VulkanPlanSummary:     rt.VulkanPlanSummary(),
				VulkanExecutionGraph:  rt.VulkanExecutionGraph(),
				VulkanPipelinePlan:    rt.VulkanPipelinePlan(),
				VulkanCommandPlan:     cmdPlan,
				VulkanCommandPlanOK:   cmdPlanErr == "",
				VulkanCommandPlanErr:  cmdPlanErr,
				VisionLoaded:          rt.VisionLoaded(),
			})
			return
		}
		fmt.Printf("requested_quantization=%s quantization=%s weight_path=%s weight_source=%s backend=%s load_stats=%+v weights=%+v cpu=%+v backends=%+v\n", rt.RequestedQuantization(), rt.Quantization(), rt.WeightPath(), rt.WeightSource(), rt.Backend(), rt.LoadStats(), rt.WeightStats(), cpu, backendSel)
		return
	}
	if *verifyOnly {
		if *outputJSON {
			writeJSON(cliVerify{
				OK:                    true,
				RequestedQuantization: rt.RequestedQuantization(),
				Quantization:          rt.Quantization(),
				WeightPath:            rt.WeightPath(),
				WeightSource:          rt.WeightSource(),
				Backend:               rt.Backend(),
				VisionLoaded:          rt.VisionLoaded(),
			})
			return
		}
		fmt.Printf("ok requested_quantization=%s quantization=%s weight_path=%s weight_source=%s backend=%s vision=%t\n", rt.RequestedQuantization(), rt.Quantization(), rt.WeightPath(), rt.WeightSource(), rt.Backend(), *verifyVision)
		return
	}

	eosIDs, err := parseTokens(*eosCSV)
	if err != nil {
		log.Fatal(err)
	}
	if *seed == 0 && needsSamplingSeed(*maxNew, *temperature, *topK) {
		*seed = time.Now().UnixNano()
	}
	opts := model.GenerateOptions{
		MaxNewTokens: *maxNew,
		Temperature:  *temperature,
		TopK:         *topK,
		Seed:         *seed,
		EOSTokenIDs:  eosIDs,
	}
	if err := model.ValidateGenerateOptions(opts); err != nil {
		log.Fatal(err)
	}
	var res model.GenerateResult
	if *imagePath != "" {
		res, err = rt.GenerateWithImageOptions(context.Background(), ids, *imagePath, opts)
	} else {
		res, err = rt.GenerateWithOptions(context.Background(), ids, opts)
	}
	if err != nil {
		log.Fatal(err)
	}
	out := res.Tokens
	text := ""
	if *decode {
		toDecode := out
		if *decodeGeneratedOnly {
			toDecode = out[res.PromptTokens:]
		}
		if tok == nil {
			tok, err = tokenizer.Load(*modelDir)
		}
		if err == nil {
			text = tok.Decode(toDecode, *skipSpecial)
		}
	}
	if *outputJSON {
		writeJSON(cliGenerate{
			Tokens:          out,
			PromptTokens:    res.PromptTokens,
			GeneratedTokens: max(0, len(out)-res.PromptTokens),
			Text:            text,
			Seed:            *seed,
			Quantization:    rt.Quantization(),
			WeightPath:      rt.WeightPath(),
			WeightSource:    rt.WeightSource(),
			Backend:         rt.Backend(),
		})
		return
	}
	for i, id := range out {
		if i > 0 {
			fmt.Print(",")
		}
		fmt.Print(id)
	}
	fmt.Println()
	if *decode && text != "" {
		fmt.Println(text)
	}
}

type cliStats struct {
	RequestedQuantization string                       `json:"requested_quantization"`
	Quantization          string                       `json:"quantization"`
	WeightPath            string                       `json:"weight_path"`
	WeightSource          string                       `json:"weight_source"`
	Backend               string                       `json:"backend"`
	Weights               model.WeightStats            `json:"weights"`
	LoadStats             model.LoadStats              `json:"load_stats"`
	CacheStats            model.CacheStats             `json:"cache_stats"`
	CPU                   backend.CPUInfo              `json:"cpu"`
	Memory                backend.MemoryInfo           `json:"memory"`
	Backends              backend.Selection            `json:"backends"`
	VulkanPlans           []backend.VulkanPlan         `json:"vulkan_plans,omitempty"`
	VulkanPlanSummary     backend.VulkanPlanSummary    `json:"vulkan_plan_summary,omitempty"`
	VulkanExecutionGraph  backend.VulkanExecutionGraph `json:"vulkan_execution_graph,omitempty"`
	VulkanPipelinePlan    []backend.VulkanPipelinePlan `json:"vulkan_pipeline_plan,omitempty"`
	VulkanCommandPlan     backend.VulkanCommandPlan    `json:"vulkan_command_plan,omitempty"`
	VulkanCommandPlanOK   bool                         `json:"vulkan_command_plan_valid"`
	VulkanCommandPlanErr  string                       `json:"vulkan_command_plan_error,omitempty"`
	VisionLoaded          bool                         `json:"vision_loaded"`
}

type cliVerify struct {
	OK                    bool   `json:"ok"`
	RequestedQuantization string `json:"requested_quantization"`
	Quantization          string `json:"quantization"`
	WeightPath            string `json:"weight_path"`
	WeightSource          string `json:"weight_source"`
	Backend               string `json:"backend"`
	VisionLoaded          bool   `json:"vision_loaded"`
}

type cliGenerate struct {
	Tokens          []int  `json:"tokens"`
	PromptTokens    int    `json:"prompt_tokens"`
	GeneratedTokens int    `json:"generated_tokens"`
	Text            string `json:"text,omitempty"`
	Seed            int64  `json:"seed"`
	Quantization    string `json:"quantization"`
	WeightPath      string `json:"weight_path"`
	WeightSource    string `json:"weight_source"`
	Backend         string `json:"backend"`
}

func writeJSON(v any) {
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		log.Fatal(err)
	}
}

func needsSamplingSeed(maxNew int, temperature float64, topK int) bool {
	return maxNew > 0 && temperature > 0 && topK != 1
}

func parseTokens(s string) ([]int, error) {
	parts := strings.Split(s, ",")
	ids := make([]int, 0, len(parts))
	for _, p := range parts {
		p = trimASCIIWhitespace(p)
		if p == "" {
			continue
		}
		id, err := strconv.Atoi(p)
		if err != nil {
			return nil, fmt.Errorf("bad token %q: %w", p, err)
		}
		ids = append(ids, id)
	}
	if len(ids) == 0 {
		return nil, fmt.Errorf("no token ids parsed")
	}
	return ids, nil
}

func taskPrompt(task string) (string, error) {
	_, p, ok := canonicalTask(task)
	if ok {
		return p, nil
	}
	return "", fmt.Errorf("unknown task %q; use ocr, table, formula, or chart", task)
}

func canonicalTask(task string) (key, prompt string, ok bool) {
	task = trimASCIIWhitespace(task)
	switch len(task) {
	case 3:
		if asciiEqualFold(task, "ocr") {
			return "ocr", "OCR:", true
		}
	case 5:
		if asciiEqualFold(task, "table") {
			return "table", "Table Recognition:", true
		}
		if asciiEqualFold(task, "chart") {
			return "chart", "Chart Recognition:", true
		}
	case 7:
		if asciiEqualFold(task, "formula") {
			return "formula", "Formula Recognition:", true
		}
	}
	return "", "", false
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

func asciiEqualFold(s, lower string) bool {
	if len(s) != len(lower) {
		return false
	}
	for i := 0; i < len(s); i++ {
		c := s[i]
		if 'A' <= c && c <= 'Z' {
			c += 'a' - 'A'
		}
		if c != lower[i] {
			return false
		}
	}
	return true
}

func chatPrompt(userText string, withImage bool) string {
	const prefix = "<|begin_of_sentence|>User: "
	const image = "<|IMAGE_START|><|IMAGE_PLACEHOLDER|><|IMAGE_END|>"
	const suffix = "\nAssistant: "
	if withImage {
		switch userText {
		case "OCR:":
			return prefix + image + "OCR:" + suffix
		case "Table Recognition:":
			return prefix + image + "Table Recognition:" + suffix
		case "Formula Recognition:":
			return prefix + image + "Formula Recognition:" + suffix
		case "Chart Recognition:":
			return prefix + image + "Chart Recognition:" + suffix
		}
		return prefix + image + userText + suffix
	}
	switch userText {
	case "OCR:":
		return prefix + "OCR:" + suffix
	case "Table Recognition:":
		return prefix + "Table Recognition:" + suffix
	case "Formula Recognition:":
		return prefix + "Formula Recognition:" + suffix
	case "Chart Recognition:":
		return prefix + "Chart Recognition:" + suffix
	}
	return prefix + userText + suffix
}
