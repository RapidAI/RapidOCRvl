package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"sort"

	"paddleocrvl-go/internal/fileutil"
)

const baseURL = "https://huggingface.co/PaddlePaddle/PaddleOCR-VL/resolve/main/"

var files = []string{
	"config.json",
	"generation_config.json",
	"preprocessor_config.json",
	"processor_config.json",
	"special_tokens_map.json",
	"tokenizer_config.json",
	"tokenizer.json",
}

type modelIndex struct {
	WeightMap map[string]string `json:"weight_map"`
}

type downloadSummary struct {
	Output string         `json:"output"`
	Files  []downloadFile `json:"files"`
	Bytes  int64          `json:"bytes"`
}

type downloadFile struct {
	Name   string `json:"name"`
	Bytes  int64  `json:"bytes"`
	SHA256 string `json:"sha256"`
	Status string `json:"status"`
}

func main() {
	outFlag := flag.String("out", ".", "output directory")
	base := flag.String("base-url", baseURL, "model base URL")
	timeout := flag.Duration("timeout", 0, "HTTP timeout; 0 disables")
	outputJSON := flag.Bool("json", false, "emit download summary as JSON")
	flag.Parse()
	out := *outFlag
	if flag.NArg() > 0 {
		out = flag.Arg(0)
	}
	if err := os.MkdirAll(out, 0o755); err != nil {
		log.Fatal(err)
	}
	client := &http.Client{Timeout: *timeout}
	summary := downloadSummary{Output: fileutil.Abs(out)}
	for _, name := range files {
		item, err := download(client, filepath.Join(out, name), joinURL(*base, name))
		if err != nil {
			log.Fatal(err)
		}
		summary.add(item)
	}
	weightFiles, err := downloadWeights(client, out, *base)
	if err != nil {
		log.Fatal(err)
	}
	for _, item := range weightFiles {
		summary.add(item)
	}
	if *outputJSON {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(summary); err != nil {
			log.Fatal(err)
		}
	}
}

func (s *downloadSummary) add(item downloadFile) {
	s.Files = append(s.Files, item)
	s.Bytes += item.Bytes
}

func downloadWeights(client *http.Client, out, base string) ([]downloadFile, error) {
	var downloaded []downloadFile
	indexPath := filepath.Join(out, "model.safetensors.index.json")
	item, err := download(client, indexPath, joinURL(base, "model.safetensors.index.json"))
	if err == nil {
		downloaded = append(downloaded, item)
		b, err := os.ReadFile(indexPath)
		if err != nil {
			return nil, err
		}
		var idx modelIndex
		if err := json.Unmarshal(b, &idx); err != nil {
			return nil, err
		}
		for _, shard := range uniqueShards(idx.WeightMap) {
			item, err := download(client, filepath.Join(out, shard), joinURL(base, shard))
			if err != nil {
				return nil, err
			}
			downloaded = append(downloaded, item)
		}
		return downloaded, nil
	}
	item, err = download(client, filepath.Join(out, "model.safetensors"), joinURL(base, "model.safetensors"))
	if err != nil {
		return nil, err
	}
	return []downloadFile{item}, nil
}

func download(client *http.Client, dst, url string) (downloadFile, error) {
	if st, err := os.Stat(dst); err == nil && st.Size() > 0 {
		fmt.Printf("skip %s (%d bytes)\n", filepath.Base(dst), st.Size())
		return downloadFile{Name: filepath.Base(dst), Bytes: st.Size(), SHA256: fileutil.SHA256(dst), Status: "skipped"}, nil
	}
	tmp := dst + ".part"
	fmt.Printf("download %s\n", filepath.Base(dst))
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return downloadFile{}, err
	}
	req.Header.Set("User-Agent", "paddleocrvl-go")
	resp, err := client.Do(req)
	if err != nil {
		return downloadFile{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return downloadFile{}, fmt.Errorf("%s: %s", url, resp.Status)
	}
	f, err := os.Create(tmp)
	if err != nil {
		return downloadFile{}, err
	}
	defer f.Close()
	n, err := io.Copy(f, resp.Body)
	if err != nil {
		return downloadFile{}, err
	}
	if err := f.Close(); err != nil {
		return downloadFile{}, err
	}
	if err := replaceDownloadFile(tmp, dst); err != nil {
		return downloadFile{}, err
	}
	return downloadFile{Name: filepath.Base(dst), Bytes: n, SHA256: fileutil.SHA256(dst), Status: "downloaded"}, nil
}

func replaceDownloadFile(tmp, dst string) error {
	if err := os.Remove(dst); err != nil && !os.IsNotExist(err) {
		return err
	}
	return os.Rename(tmp, dst)
}

func uniqueShards(weightMap map[string]string) []string {
	seen := map[string]bool{}
	out := make([]string, 0, len(weightMap))
	for _, shard := range weightMap {
		if shard == "" || seen[shard] {
			continue
		}
		seen[shard] = true
		out = append(out, shard)
	}
	sort.Strings(out)
	return out
}

func joinURL(base, name string) string {
	if base == "" {
		base = baseURL
	}
	if base[len(base)-1] != '/' {
		base += "/"
	}
	return base + name
}
