package gguf

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"
	"unsafe"

	"paddleocrvl-go/internal/config"
	"paddleocrvl-go/internal/safetensors"
	"paddleocrvl-go/internal/tensor"
)

type kv struct {
	key string
	typ uint32
	val any
}

type tensorInfo struct {
	name   string
	shape  []int64
	typ    uint32
	offset uint64
	size   uint64
}

func ConvertSafetensors(src, dst, configPath string) error {
	return ConvertSafetensorsWithOptions(src, dst, configPath, ConvertOptions{})
}

type ConvertOptions struct {
	Quantization string
	Progress     func(done, total int, name string, typ string)
}

func ConvertSafetensorsWithOptions(src, dst, configPath string, opts ConvertOptions) error {
	opts.Quantization = normalizeQuantization(opts.Quantization)
	switch opts.Quantization {
	case "", "f32", "q8", "q6", "q4":
	default:
		return fmt.Errorf("unsupported quantization %q", opts.Quantization)
	}
	sf, err := safetensors.OpenModel(src)
	if err != nil {
		return err
	}
	defer sf.Close()
	cfgDir := "."
	if configPath != "" {
		cfgDir = configPath
	}
	cfg, _ := config.Load(cfgDir)

	names := make([]string, 0, len(sf.Tensors))
	for name := range sf.Tensors {
		names = append(names, name)
	}
	sort.Strings(names)
	infos := make([]tensorInfo, len(names))
	var off uint64
	var f32Tensors, quantizedTensors int
	for i, name := range names {
		meta := sf.Tensors[name]
		typ := typeF32
		size := uint64(elemCount(meta.Shape) * 4)
		quantizable := opts.Quantization != "f32" && isQuantizedTextTensor(name, meta.Shape, cfg)
		if opts.Quantization == "q8" && quantizable {
			typ = typeQ8Row
			rows := uint64(meta.Shape[0])
			cols := uint64(meta.Shape[1])
			size = rows*4 + rows*cols
		} else if opts.Quantization == "q6" && quantizable {
			typ = typeQ6Row
			rows := uint64(meta.Shape[0])
			cols := uint64(meta.Shape[1])
			size = rows*4 + rows*((cols*6+7)/8)
		} else if opts.Quantization == "q4" && quantizable {
			typ = typeQ4Row
			rows := uint64(meta.Shape[0])
			cols := uint64(meta.Shape[1])
			size = rows*4 + rows*((cols+1)/2)
		}
		if typ == typeF32 {
			f32Tensors++
		} else {
			quantizedTensors++
		}
		off = align(int64(off), uint64(Alignment))
		infos[i] = tensorInfo{name: name, shape: meta.Shape, typ: typ, offset: off, size: size}
		off += size
	}

	tmp := dst + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	ok := false
	defer func() {
		f.Close()
		if !ok {
			_ = os.Remove(tmp)
		}
	}()
	kvs := metadata(cfg, opts.Quantization, len(sf.Tensors), sf.ShardCount(), f32Tensors, quantizedTensors)
	if err := writeHeader(f, kvs, infos); err != nil {
		return err
	}
	if err := padToAlignment(f); err != nil {
		return err
	}
	var current uint64
	floatBuf := make([]byte, 64*1024*4)
	var rowFloatBuf []float32
	var decodeRawBuf []byte
	var rowBuf []byte
	var scaleBuf []float32
	var qRowBuf []byte
	for i, info := range infos {
		if opts.Progress != nil {
			opts.Progress(i, len(infos), info.name, TensorTypeName(info.typ))
		}
		if current < info.offset {
			if err := writeZeros(f, int64(info.offset-current)); err != nil {
				return err
			}
			current = info.offset
		}
		if info.typ == typeF32 {
			meta := sf.Tensors[info.name]
			if meta.DType == "F32" {
				if _, err := sf.WriteRawTo(info.name, f, floatBuf); err != nil {
					return err
				}
				current += info.size
				continue
			}
			needFloats := min(elemCount(info.shape), 256*1024)
			if cap(rowFloatBuf) < needFloats {
				rowFloatBuf = make([]float32, needFloats)
			} else {
				rowFloatBuf = rowFloatBuf[:needFloats]
			}
			needRaw := needFloats * 2
			if cap(decodeRawBuf) < needRaw {
				decodeRawBuf = make([]byte, needRaw)
			} else {
				decodeRawBuf = decodeRawBuf[:needRaw]
			}
			if _, err := sf.WriteFloat32To(info.name, f, decodeRawBuf, rowFloatBuf); err != nil {
				return err
			}
			current += info.size
			continue
		}
		if info.typ == typeQ8Row || info.typ == typeQ4Row || info.typ == typeQ6Row {
			meta := sf.Tensors[info.name]
			needFloats := 256 * 1024
			if meta.DType != "F32" && len(info.shape) == 2 {
				needFloats = int(info.shape[1])
			}
			if cap(rowFloatBuf) < needFloats {
				rowFloatBuf = make([]float32, needFloats)
			} else {
				rowFloatBuf = rowFloatBuf[:needFloats]
			}
			if rowBuf == nil {
				rowBuf = make([]byte, 256*1024)
			}
			if meta.DType != "F32" && len(decodeRawBuf) == 0 {
				decodeRawBuf = make([]byte, 1<<20)
			}
			scaleBuf, qRowBuf, err = writeQuantizedRows(f, sf, info, floatBuf, rowFloatBuf, decodeRawBuf, rowBuf, scaleBuf, qRowBuf)
			if err != nil {
				return err
			}
			current += info.size
			continue
		}
	}
	if opts.Progress != nil {
		opts.Progress(len(infos), len(infos), "", "")
	}
	if err := f.Close(); err != nil {
		return err
	}
	if err := replaceFile(tmp, dst); err != nil {
		return err
	}
	ok = true
	return nil
}

func replaceFile(tmp, dst string) error {
	if err := os.Remove(dst); err != nil && !os.IsNotExist(err) {
		return err
	}
	return os.Rename(tmp, dst)
}

func writeQuantizedRows(f *os.File, sf *safetensors.Set, info tensorInfo, floatBuf []byte, rowFloatBuf []float32, decodeRawBuf []byte, rowBuf []byte, scaleBuf []float32, qRowBuf []byte) ([]float32, []byte, error) {
	rows := int(info.shape[0])
	cols := int(info.shape[1])
	if cap(scaleBuf) < rows {
		scaleBuf = make([]float32, rows)
	} else {
		scaleBuf = scaleBuf[:rows]
	}
	var q8Bytes, q6Row, q4Row []byte
	switch info.typ {
	case typeQ8Row:
		q8Bytes, qRowBuf = quantRowBuffer(qRowBuf, cols)
	case typeQ6Row:
		q6Row, qRowBuf = quantRowBuffer(qRowBuf, tensor.PackedQ6Cols(cols))
	case typeQ4Row:
		q4Row, qRowBuf = quantRowBuffer(qRowBuf, (cols+1)/2)
	default:
		return scaleBuf, qRowBuf, fmt.Errorf("unsupported quantized tensor type %d", info.typ)
	}
	start, err := f.Seek(0, io.SeekCurrent)
	if err != nil {
		return scaleBuf, qRowBuf, err
	}
	if err := writeZeros(f, int64(rows*4)); err != nil {
		return scaleBuf, qRowBuf, err
	}
	pending := rowBuf[:0]
	flushRows := func() error {
		if len(pending) == 0 {
			return nil
		}
		_, err := f.Write(pending)
		pending = pending[:0]
		return err
	}
	writeRow := func(row []byte) error {
		if len(row) > cap(rowBuf) {
			if err := flushRows(); err != nil {
				return err
			}
			_, err := f.Write(row)
			return err
		}
		if len(pending)+len(row) > cap(rowBuf) {
			if err := flushRows(); err != nil {
				return err
			}
		}
		pending = append(pending, row...)
		return nil
	}
	_, err = sf.Float32RowsBuffer(info.name, rowFloatBuf, decodeRawBuf, func(row int, values []float32) error {
		switch info.typ {
		case typeQ8Row:
			scaleBuf[row] = tensor.QuantizeQ8RowBytesInto(values, q8Bytes)
			return writeRow(q8Bytes)
		case typeQ6Row:
			scaleBuf[row] = tensor.QuantizeQ6RowInto(values, q6Row)
			return writeRow(q6Row)
		case typeQ4Row:
			scaleBuf[row] = tensor.QuantizeQ4RowInto(values, q4Row)
			return writeRow(q4Row)
		default:
			return fmt.Errorf("unsupported quantized tensor type %d", info.typ)
		}
	})
	if err != nil {
		return scaleBuf, qRowBuf, err
	}
	if err := flushRows(); err != nil {
		return scaleBuf, qRowBuf, err
	}
	end, err := f.Seek(0, io.SeekCurrent)
	if err != nil {
		return scaleBuf, qRowBuf, err
	}
	if _, err := f.Seek(start, io.SeekStart); err != nil {
		return scaleBuf, qRowBuf, err
	}
	if err := writeFloat32sBuf(f, scaleBuf, floatBuf); err != nil {
		return scaleBuf, qRowBuf, err
	}
	_, err = f.Seek(end, io.SeekStart)
	return scaleBuf, qRowBuf, err
}

func quantRowBuffer(buf []byte, size int) ([]byte, []byte) {
	if cap(buf) < size {
		buf = make([]byte, size)
	} else {
		buf = buf[:size]
	}
	return buf, buf
}

func metadata(cfg *config.Config, quantization string, tensorCount, shardCount, f32Tensors, quantizedTensors int) []kv {
	fileType := uint32(0)
	if quantization == "q8" {
		fileType = 7
	} else if quantization == "q6" {
		fileType = 6
	} else if quantization == "q4" {
		fileType = 3
	}
	out := []kv{
		{"general.architecture", metaString, "paddleocrvl"},
		{"general.name", metaString, "PaddleOCR-VL-0.9B"},
		{"general.alignment", metaUint32, Alignment},
		{"general.file_type", metaUint32, fileType},
		{"paddleocrvl.gguf_version", metaUint32, Version},
		{"paddleocrvl.quantization", metaString, quantizationName(quantization)},
		{"paddleocrvl.source_tensors", metaUint64, uint64(tensorCount)},
		{"paddleocrvl.source_shards", metaUint64, uint64(shardCount)},
		{"paddleocrvl.f32_tensors", metaUint64, uint64(f32Tensors)},
		{"paddleocrvl.quantized_tensors", metaUint64, uint64(quantizedTensors)},
	}
	if cfg != nil {
		out = append(out,
			kv{"paddleocrvl.vocab_size", metaUint64, uint64(cfg.VocabSize)},
			kv{"paddleocrvl.text_layers", metaUint64, uint64(cfg.NumHiddenLayers)},
			kv{"paddleocrvl.hidden_size", metaUint64, uint64(cfg.HiddenSize)},
			kv{"paddleocrvl.vision_layers", metaUint64, uint64(cfg.VisionConfig.NumHiddenLayers)},
		)
	}
	return out
}

func quantizationName(q string) string {
	q = normalizeQuantization(q)
	if q == "" {
		return "f32"
	}
	return q
}

func normalizeQuantization(q string) string {
	q = trimASCIIWhitespace(q)
	if q == "" {
		return "f32"
	}
	return lowerASCII(q)
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
	if len(s) <= 64 {
		var small [64]byte
		copy(small[:], s[:firstUpper])
		for i := firstUpper; i < len(s); i++ {
			c := s[i]
			if 'A' <= c && c <= 'Z' {
				c += 'a' - 'A'
			}
			small[i] = c
		}
		return string(small[:len(s)])
	}
	out := []byte(s)
	for i := firstUpper; i < len(out); i++ {
		c := out[i]
		if 'A' <= c && c <= 'Z' {
			out[i] = c + ('a' - 'A')
		}
	}
	return unsafe.String(&out[0], len(out))
}

func isQuantizedTextTensor(name string, shape []int64, cfg *config.Config) bool {
	if cfg == nil || len(shape) != 2 {
		return false
	}
	if name == "lm_head.weight" {
		return shape[0] == int64(cfg.VocabSize) && shape[1] == int64(cfg.HiddenSize)
	}
	rest, ok := strings.CutPrefix(name, "model.layers.")
	if !ok {
		return false
	}
	layerText, suffix, ok := strings.Cut(rest, ".")
	if !ok {
		return false
	}
	layer, err := strconv.Atoi(layerText)
	if err != nil || layer < 0 || layer >= cfg.NumHiddenLayers {
		return false
	}
	switch suffix {
	case "self_attn.q_proj.weight":
		return shape[0] == int64(cfg.NumAttentionHeads*cfg.HeadDim) && shape[1] == int64(cfg.HiddenSize)
	case "self_attn.k_proj.weight", "self_attn.v_proj.weight":
		return shape[0] == int64(cfg.NumKeyValueHeads*cfg.HeadDim) && shape[1] == int64(cfg.HiddenSize)
	case "self_attn.o_proj.weight":
		return shape[0] == int64(cfg.HiddenSize) && shape[1] == int64(cfg.NumAttentionHeads*cfg.HeadDim)
	case "mlp.gate_proj.weight", "mlp.up_proj.weight":
		return shape[0] == int64(cfg.IntermediateSize) && shape[1] == int64(cfg.HiddenSize)
	case "mlp.down_proj.weight":
		return shape[0] == int64(cfg.HiddenSize) && shape[1] == int64(cfg.IntermediateSize)
	}
	return false
}

func writeHeader(w io.Writer, kvs []kv, infos []tensorInfo) error {
	if _, err := w.Write([]byte(Magic)); err != nil {
		return err
	}
	if err := writeU32(w, Version); err != nil {
		return err
	}
	if err := writeU64(w, uint64(len(infos))); err != nil {
		return err
	}
	if err := writeU64(w, uint64(len(kvs))); err != nil {
		return err
	}
	for _, item := range kvs {
		if err := writeString(w, item.key); err != nil {
			return err
		}
		if err := writeU32(w, item.typ); err != nil {
			return err
		}
		switch item.typ {
		case metaString:
			if err := writeString(w, item.val.(string)); err != nil {
				return err
			}
		case metaUint32:
			if err := writeU32(w, item.val.(uint32)); err != nil {
				return err
			}
		case metaUint64:
			if err := writeU64(w, item.val.(uint64)); err != nil {
				return err
			}
		default:
			return fmt.Errorf("unsupported metadata type %d", item.typ)
		}
	}
	for _, info := range infos {
		if err := writeString(w, info.name); err != nil {
			return err
		}
		if err := writeU32(w, uint32(len(info.shape))); err != nil {
			return err
		}
		for _, d := range info.shape {
			if err := writeU64(w, uint64(d)); err != nil {
				return err
			}
		}
		if err := writeU32(w, info.typ); err != nil {
			return err
		}
		if err := writeU64(w, info.offset); err != nil {
			return err
		}
	}
	return nil
}

func padToAlignment(f *os.File) error {
	pos, err := f.Seek(0, io.SeekCurrent)
	if err != nil {
		return err
	}
	next := align(pos, uint64(Alignment))
	return writeZeros(f, int64(next)-pos)
}

func writeZeros(w io.Writer, n int64) error {
	if n <= 0 {
		return nil
	}
	for n > 0 {
		chunk := min(n, int64(len(zeroBlock)))
		if _, err := w.Write(zeroBlock[:chunk]); err != nil {
			return err
		}
		n -= chunk
	}
	return nil
}

var zeroBlock [32 * 1024]byte

func writeFloat32s(w io.Writer, values []float32) error {
	return writeFloat32sBuf(w, values, nil)
}

func writeFloat32sBuf(w io.Writer, values []float32, buf []byte) error {
	const chunkFloats = 64 * 1024
	if hostLittleEndian {
		for len(values) > 0 {
			n := min(len(values), chunkFloats)
			chunk := values[:n]
			raw := unsafe.Slice((*byte)(unsafe.Pointer(&chunk[0])), n*4)
			if _, err := w.Write(raw); err != nil {
				return err
			}
			values = values[n:]
		}
		return nil
	}
	if cap(buf) < chunkFloats*4 {
		buf = make([]byte, chunkFloats*4)
	}
	buf = buf[:chunkFloats*4]
	for len(values) > 0 {
		n := min(len(values), chunkFloats)
		chunk := values[:n]
		i := 0
		for ; i+7 < n; i += 8 {
			j := i * 4
			binary.LittleEndian.PutUint32(buf[j:], math.Float32bits(chunk[i]))
			binary.LittleEndian.PutUint32(buf[j+4:], math.Float32bits(chunk[i+1]))
			binary.LittleEndian.PutUint32(buf[j+8:], math.Float32bits(chunk[i+2]))
			binary.LittleEndian.PutUint32(buf[j+12:], math.Float32bits(chunk[i+3]))
			binary.LittleEndian.PutUint32(buf[j+16:], math.Float32bits(chunk[i+4]))
			binary.LittleEndian.PutUint32(buf[j+20:], math.Float32bits(chunk[i+5]))
			binary.LittleEndian.PutUint32(buf[j+24:], math.Float32bits(chunk[i+6]))
			binary.LittleEndian.PutUint32(buf[j+28:], math.Float32bits(chunk[i+7]))
		}
		for ; i < n; i++ {
			binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(chunk[i]))
		}
		if _, err := w.Write(buf[:n*4]); err != nil {
			return err
		}
		values = values[n:]
	}
	return nil
}

func writeString(w io.Writer, s string) error {
	if err := writeU64(w, uint64(len(s))); err != nil {
		return err
	}
	_, err := io.WriteString(w, s)
	return err
}

func writeU32(w io.Writer, v uint32) error {
	var buf [4]byte
	binary.LittleEndian.PutUint32(buf[:], v)
	_, err := w.Write(buf[:])
	return err
}

func writeU64(w io.Writer, v uint64) error {
	var buf [8]byte
	binary.LittleEndian.PutUint64(buf[:], v)
	_, err := w.Write(buf[:])
	return err
}
