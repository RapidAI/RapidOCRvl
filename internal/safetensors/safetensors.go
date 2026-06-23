package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"unsafe"

	"paddleocrvl-go/internal/jsonutil"
)

const maxHeaderBytes = 256 << 20
const maxIndexBytes = 16 << 20
const maxTensorCount = 1 << 20
const maxShapeDims = 8

type TensorMeta struct {
	DType       string   `json:"dtype"`
	Shape       []int64  `json:"shape"`
	DataOffsets [2]int64 `json:"data_offsets"`
}

type File struct {
	f         *os.File
	dataStart int64
	Tensors   map[string]TensorMeta
}

type Set struct {
	files      map[string]*File
	tensorFile map[string]string
	Tensors    map[string]TensorMeta
}

var float32RowsBufferPool sync.Pool
var encodedRawBufferPool sync.Pool

type indexFile struct {
	WeightMap map[string]string `json:"weight_map"`
}

func OpenModel(dir string) (*Set, error) {
	if st, err := os.Stat(dir); err == nil && !st.IsDir() {
		return OpenSetFile(dir)
	}
	single := filepath.Join(dir, "model.safetensors")
	if sf, err := OpenSetFile(single); err == nil {
		return sf, nil
	} else if !os.IsNotExist(err) {
		return nil, err
	}
	index := filepath.Join(dir, "model.safetensors.index.json")
	if sf, err := OpenSetIndex(index); err == nil {
		return sf, nil
	} else if !os.IsNotExist(err) {
		return nil, err
	}
	return nil, fmt.Errorf("no model.safetensors or model.safetensors.index.json in %s", dir)
}

func OpenSetFile(path string) (*Set, error) {
	f, err := Open(path)
	if err != nil {
		return nil, err
	}
	base := filepath.Base(path)
	files := map[string]*File{base: f}
	tensorFile := make(map[string]string, len(f.Tensors))
	tensors := make(map[string]TensorMeta, len(f.Tensors))
	for name, meta := range f.Tensors {
		tensorFile[name] = base
		tensors[name] = meta
	}
	return &Set{files: files, tensorFile: tensorFile, Tensors: tensors}, nil
}

func OpenSetIndex(path string) (*Set, error) {
	st, err := os.Stat(path)
	if err != nil {
		return nil, err
	}
	if st.Size() > maxIndexBytes {
		return nil, fmt.Errorf("safetensors index too large: %d bytes > %d", st.Size(), maxIndexBytes)
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if err := jsonutil.RejectDuplicateKeys(b, path); err != nil {
		return nil, err
	}
	var idx indexFile
	if err := json.Unmarshal(b, &idx); err != nil {
		return nil, err
	}
	if len(idx.WeightMap) == 0 {
		return nil, fmt.Errorf("%s has empty weight_map", path)
	}
	if err := validateTensorCount(len(idx.WeightMap)); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	dir := filepath.Dir(path)
	files := make(map[string]*File, 4)
	tensorFile := make(map[string]string, len(idx.WeightMap))
	tensors := make(map[string]TensorMeta, len(idx.WeightMap))
	for tensorName, shard := range idx.WeightMap {
		if err := validateTensorName(tensorName); err != nil {
			for _, f := range files {
				_ = f.Close()
			}
			return nil, fmt.Errorf("%s: %w", path, err)
		}
		sf := files[shard]
		if sf == nil {
			shardPath, err := safeShardPath(dir, shard)
			if err != nil {
				for _, f := range files {
					_ = f.Close()
				}
				return nil, err
			}
			sf, err = Open(shardPath)
			if err != nil {
				for _, f := range files {
					_ = f.Close()
				}
				return nil, err
			}
			files[shard] = sf
		}
		meta, ok := sf.Tensors[tensorName]
		if !ok {
			for _, f := range files {
				_ = f.Close()
			}
			return nil, fmt.Errorf("%s listed in index but missing from %s", tensorName, shard)
		}
		tensorFile[tensorName] = shard
		tensors[tensorName] = meta
	}
	return &Set{files: files, tensorFile: tensorFile, Tensors: tensors}, nil
}

func safeShardPath(dir, shard string) (string, error) {
	if shard == "" || strings.ContainsAny(shard, `/\`) || !safeShardFileName(shard) {
		return "", fmt.Errorf("unsafe shard path %q", shard)
	}
	sep := string(filepath.Separator)
	if strings.HasSuffix(dir, sep) {
		return dir + shard, nil
	}
	return dir + sep + shard, nil
}

func safeShardFileName(name string) bool {
	if name == "" || name == "." || name == ".." {
		return false
	}
	for i := 0; i < len(name); i++ {
		c := name[i]
		if c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || c == '.' || c == '_' || c == '-' {
			continue
		}
		return false
	}
	return true
}

func Open(path string) (*File, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	st, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, err
	}
	var nbuf [8]byte
	if _, err := io.ReadFull(f, nbuf[:]); err != nil {
		f.Close()
		return nil, err
	}
	headerLen := int64(binary.LittleEndian.Uint64(nbuf[:]))
	if headerLen < 0 || headerLen > maxHeaderBytes {
		f.Close()
		return nil, fmt.Errorf("safetensors header too large: %d bytes", headerLen)
	}
	header := make([]byte, headerLen)
	if _, err := io.ReadFull(f, header); err != nil {
		f.Close()
		return nil, err
	}
	if err := jsonutil.RejectDuplicateKeys(header, path); err != nil {
		f.Close()
		return nil, err
	}
	raw := map[string]json.RawMessage{}
	if err := json.Unmarshal(header, &raw); err != nil {
		f.Close()
		return nil, err
	}
	tensorCount := len(raw)
	if _, ok := raw["__metadata__"]; ok {
		tensorCount--
	}
	if err := validateTensorCount(tensorCount); err != nil {
		f.Close()
		return nil, err
	}
	tensors := make(map[string]TensorMeta, len(raw))
	for k, v := range raw {
		if k == "__metadata__" {
			continue
		}
		if err := validateTensorName(k); err != nil {
			f.Close()
			return nil, err
		}
		var tm TensorMeta
		if err := json.Unmarshal(v, &tm); err != nil {
			f.Close()
			return nil, fmt.Errorf("%s: %w", k, err)
		}
		if err := validateTensorMeta(k, tm, st.Size()-(8+headerLen)); err != nil {
			f.Close()
			return nil, err
		}
		tensors[k] = tm
	}
	return &File{f: f, dataStart: 8 + headerLen, Tensors: tensors}, nil
}

func validateTensorCount(count int) error {
	if count > maxTensorCount {
		return fmt.Errorf("safetensors tensor count too large: %d > %d", count, maxTensorCount)
	}
	return nil
}

func validateTensorName(name string) error {
	if name == "" {
		return fmt.Errorf("safetensors tensor name must not be empty")
	}
	for i := 0; i < len(name); i++ {
		if name[i] < 0x20 || name[i] == 0x7f {
			return fmt.Errorf("safetensors tensor name contains control character")
		}
	}
	return nil
}

func validateTensorMeta(name string, meta TensorMeta, dataBytes int64) error {
	size, err := tensorDataBytes(meta)
	if err != nil {
		return fmt.Errorf("%s: %w", name, err)
	}
	start, end := meta.DataOffsets[0], meta.DataOffsets[1]
	if start < 0 || end < start {
		return fmt.Errorf("%s: invalid data_offsets %v", name, meta.DataOffsets)
	}
	if end-start != size {
		return fmt.Errorf("%s: data_offsets size %d does not match tensor size %d", name, end-start, size)
	}
	if end > dataBytes {
		return fmt.Errorf("%s: data_offsets %v exceed file data size %d", name, meta.DataOffsets, dataBytes)
	}
	return nil
}

func tensorDataBytes(meta TensorMeta) (int64, error) {
	elemSize, err := dtypeSize(meta.DType)
	if err != nil {
		return 0, err
	}
	countInt, err := checkedElemCount(meta.Shape)
	if err != nil {
		return 0, err
	}
	count := int64(countInt)
	if count > math.MaxInt64/int64(elemSize) {
		return 0, fmt.Errorf("tensor byte size overflows int64")
	}
	return count * int64(elemSize), nil
}

func (s *Set) Close() error {
	var first error
	for _, f := range s.files {
		if err := f.Close(); err != nil && first == nil {
			first = err
		}
	}
	return first
}

func (s *Set) Float32(name string) ([]float32, []int64, error) {
	shard, ok := s.tensorFile[name]
	if !ok {
		return nil, nil, fmt.Errorf("missing tensor %s", name)
	}
	return s.files[shard].Float32(name)
}

func (s *Set) Float32Rows(name string, fn func(row int, values []float32) error) ([]int64, error) {
	return s.Float32RowsBuffer(name, nil, nil, fn)
}

func (s *Set) Float32RowsBuffer(name string, floatBuf []float32, rawBuf []byte, fn func(row int, values []float32) error) ([]int64, error) {
	shard, ok := s.tensorFile[name]
	if !ok {
		return nil, fmt.Errorf("missing tensor %s", name)
	}
	return s.files[shard].Float32RowsBuffer(name, floatBuf, rawBuf, fn)
}

func (s *Set) ShardCount() int {
	return len(s.files)
}

func (s *Set) Shape(name string) ([]int64, error) {
	meta, ok := s.Tensors[name]
	if !ok {
		return nil, fmt.Errorf("missing tensor %s", name)
	}
	return meta.Shape, nil
}

func (s *Set) DType(name string) (string, bool) {
	meta, ok := s.Tensors[name]
	if !ok {
		return "", false
	}
	return meta.DType, true
}

func (s *Set) Raw(name string) ([]byte, TensorMeta, error) {
	shard, ok := s.tensorFile[name]
	if !ok {
		return nil, TensorMeta{}, fmt.Errorf("missing tensor %s", name)
	}
	return s.files[shard].Raw(name)
}

func (s *Set) WriteRawTo(name string, w io.Writer, buf []byte) (TensorMeta, error) {
	shard, ok := s.tensorFile[name]
	if !ok {
		return TensorMeta{}, fmt.Errorf("missing tensor %s", name)
	}
	return s.files[shard].WriteRawTo(name, w, buf)
}

func (s *Set) WriteFloat32To(name string, w io.Writer, rawBuf []byte, floatBuf []float32) (TensorMeta, error) {
	shard, ok := s.tensorFile[name]
	if !ok {
		return TensorMeta{}, fmt.Errorf("missing tensor %s", name)
	}
	return s.files[shard].WriteFloat32To(name, w, rawBuf, floatBuf)
}

func (sf *File) Close() error {
	if sf.f == nil {
		return nil
	}
	return sf.f.Close()
}

func (sf *File) Float32(name string) ([]float32, []int64, error) {
	meta, ok := sf.Tensors[name]
	if !ok {
		return nil, nil, fmt.Errorf("missing tensor %s", name)
	}
	count := elemCount(meta.Shape)
	out := make([]float32, count)
	if meta.DType == "F32" {
		if err := sf.readFloat32At(out, sf.dataStart+meta.DataOffsets[0]); err != nil {
			return nil, nil, err
		}
		return out, meta.Shape, nil
	}
	switch meta.DType {
	case "BF16":
		if err := sf.readEncodedFloat32At(out, sf.dataStart+meta.DataOffsets[0], "BF16"); err != nil {
			return nil, nil, err
		}
	case "F16":
		if err := sf.readEncodedFloat32At(out, sf.dataStart+meta.DataOffsets[0], "F16"); err != nil {
			return nil, nil, err
		}
	default:
		return nil, nil, fmt.Errorf("unsupported dtype %s for %s", meta.DType, name)
	}
	return out, meta.Shape, nil
}

func (sf *File) readEncodedFloat32At(out []float32, off int64, dtype string) error {
	const targetBytes = 1 << 20
	if len(out) == 0 {
		return nil
	}
	blockElems := min(len(out), targetBytes/2)
	needRaw := blockElems * 2
	raw, rawHandle := getEncodedRawBuffer(needRaw)
	defer putEncodedRawBuffer(rawHandle, raw)
	done := 0
	for done < len(out) {
		n := min(len(out)-done, blockElems)
		chunk := raw[:n*2]
		if _, err := sf.f.ReadAt(chunk, off+int64(done*2)); err != nil {
			return err
		}
		switch dtype {
		case "BF16":
			decodeBF16(out[done:done+n], chunk)
		case "F16":
			decodeF16(out[done:done+n], chunk)
		}
		done += n
	}
	return nil
}

func getEncodedRawBuffer(n int) ([]byte, *[]byte) {
	if n <= 0 {
		return nil, nil
	}
	if v := encodedRawBufferPool.Get(); v != nil {
		p := v.(*[]byte)
		if cap(*p) >= n {
			return (*p)[:n], p
		}
	}
	p := new([]byte)
	*p = make([]byte, n)
	return *p, p
}

func putEncodedRawBuffer(p *[]byte, buf []byte) {
	const maxEncodedRawBuffer = 1 << 20
	if p == nil || cap(buf) == 0 || cap(buf) > maxEncodedRawBuffer {
		return
	}
	*p = buf[:0]
	encodedRawBufferPool.Put(p)
}

func (sf *File) Float32Rows(name string, fn func(row int, values []float32) error) ([]int64, error) {
	return sf.Float32RowsBuffer(name, nil, nil, fn)
}

func (sf *File) Float32RowsBuffer(name string, floatBuf []float32, rawBuf []byte, fn func(row int, values []float32) error) ([]int64, error) {
	meta, ok := sf.Tensors[name]
	if !ok {
		return nil, fmt.Errorf("missing tensor %s", name)
	}
	if len(meta.Shape) != 2 {
		return nil, fmt.Errorf("tensor %s must be 2D for row streaming", name)
	}
	rows := int(meta.Shape[0])
	cols := int(meta.Shape[1])
	elemSize, err := dtypeSize(meta.DType)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", name, err)
	}
	rowBytes := cols * elemSize
	if meta.DType == "F32" && hostLittleEndian {
		return sf.float32RowsF32Block(meta, meta.Shape, cols, rowBytes, floatBuf, fn)
	}
	if meta.DType == "BF16" || meta.DType == "F16" {
		return sf.float32RowsEncodedBlock(meta, meta.Shape, cols, rowBytes, floatBuf, rawBuf, fn)
	}
	raw := rawBuf
	if len(raw) < rowBytes {
		raw = make([]byte, rowBytes)
	} else {
		raw = raw[:rowBytes]
	}
	values := make([]float32, cols)
	base := sf.dataStart + meta.DataOffsets[0]
	for r := 0; r < rows; r++ {
		if meta.DType == "F32" {
			if err := sf.readFloat32At(values, base+int64(r*rowBytes)); err != nil {
				return nil, err
			}
		} else {
			if _, err := sf.f.ReadAt(raw, base+int64(r*rowBytes)); err != nil {
				return nil, err
			}
			decodeFloat32Row(values, raw, meta.DType)
		}
		if err := fn(r, values); err != nil {
			return nil, err
		}
	}
	return meta.Shape, nil
}

func (sf *File) float32RowsF32Block(meta TensorMeta, shape []int64, cols, rowBytes int, buf []float32, fn func(row int, values []float32) error) ([]int64, error) {
	rows := int(shape[0])
	rowsPerBlock := float32RowsPerBlock(rows, rowBytes)
	need := rowsPerBlock * cols
	usePool := buf == nil
	var poolHandle *[]float32
	if buf == nil {
		buf, poolHandle = getFloat32RowsBuffer(need)
	}
	if len(buf) < need {
		buf = make([]float32, need)
	} else {
		buf = buf[:need]
	}
	if usePool {
		defer putFloat32RowsBuffer(poolHandle, buf)
	}
	base := sf.dataStart + meta.DataOffsets[0]
	for r := 0; r < rows; {
		n := rowsPerBlock
		if rows-r < n {
			n = rows - r
		}
		block := buf[:n*cols]
		if err := sf.readFloat32At(block, base+int64(r*rowBytes)); err != nil {
			return nil, err
		}
		for i := 0; i < n; i++ {
			row := r + i
			if err := fn(row, block[i*cols:(i+1)*cols]); err != nil {
				return nil, err
			}
		}
		r += n
	}
	return shape, nil
}

func (sf *File) float32RowsEncodedBlock(meta TensorMeta, shape []int64, cols, rowBytes int, floatBuf []float32, raw []byte, fn func(row int, values []float32) error) ([]int64, error) {
	rows := int(shape[0])
	rowsPerBlock := float32RowsPerBlock(rows, rowBytes)
	needRaw := rowsPerBlock * rowBytes
	useRawPool := raw == nil
	var rawHandle *[]byte
	if raw == nil {
		raw, rawHandle = getEncodedRawBuffer(needRaw)
	} else if len(raw) < needRaw {
		raw = make([]byte, needRaw)
	} else {
		raw = raw[:needRaw]
	}
	if useRawPool {
		defer putEncodedRawBuffer(rawHandle, raw)
	}
	values := floatBuf
	usePool := values == nil
	var valueHandle *[]float32
	if values == nil {
		values, valueHandle = getFloat32RowsBuffer(cols)
	}
	if len(values) < cols {
		values = make([]float32, cols)
	} else {
		values = values[:cols]
	}
	if usePool {
		defer putFloat32RowsBuffer(valueHandle, values)
	}
	base := sf.dataStart + meta.DataOffsets[0]
	for r := 0; r < rows; {
		n := rowsPerBlock
		if rows-r < n {
			n = rows - r
		}
		block := raw[:n*rowBytes]
		if _, err := sf.f.ReadAt(block, base+int64(r*rowBytes)); err != nil {
			return nil, err
		}
		for i := 0; i < n; i++ {
			decodeFloat32Row(values, block[i*rowBytes:(i+1)*rowBytes], meta.DType)
			if err := fn(r+i, values); err != nil {
				return nil, err
			}
		}
		r += n
	}
	return shape, nil
}

func getFloat32RowsBuffer(n int) ([]float32, *[]float32) {
	if n <= 0 {
		return nil, nil
	}
	if v := float32RowsBufferPool.Get(); v != nil {
		p := v.(*[]float32)
		if cap(*p) >= n {
			return (*p)[:n], p
		}
	}
	p := new([]float32)
	*p = make([]float32, n)
	return *p, p
}

func putFloat32RowsBuffer(p *[]float32, buf []float32) {
	const maxPooledFloat32RowsBuffer = 1 << 20
	if p == nil || cap(buf) == 0 || cap(buf) > maxPooledFloat32RowsBuffer {
		return
	}
	*p = buf[:0]
	float32RowsBufferPool.Put(p)
}

func float32RowsPerBlock(rows, rowBytes int) int {
	const targetBlockBytes = 1 << 20
	rowsPerBlock := targetBlockBytes / rowBytes
	if rowsPerBlock < 1 {
		rowsPerBlock = 1
	}
	if rowsPerBlock > rows {
		rowsPerBlock = rows
	}
	return rowsPerBlock
}

func (sf *File) Raw(name string) ([]byte, TensorMeta, error) {
	meta, ok := sf.Tensors[name]
	if !ok {
		return nil, TensorMeta{}, fmt.Errorf("missing tensor %s", name)
	}
	size := meta.DataOffsets[1] - meta.DataOffsets[0]
	buf := make([]byte, size)
	if _, err := sf.f.ReadAt(buf, sf.dataStart+meta.DataOffsets[0]); err != nil {
		return nil, TensorMeta{}, err
	}
	return buf, meta, nil
}

func (sf *File) WriteRawTo(name string, w io.Writer, buf []byte) (TensorMeta, error) {
	meta, ok := sf.Tensors[name]
	if !ok {
		return TensorMeta{}, fmt.Errorf("missing tensor %s", name)
	}
	size := meta.DataOffsets[1] - meta.DataOffsets[0]
	if size < 0 {
		return TensorMeta{}, fmt.Errorf("bad offsets for %s", name)
	}
	if len(buf) == 0 {
		buf = make([]byte, 1<<20)
	}
	off := sf.dataStart + meta.DataOffsets[0]
	for size > 0 {
		n := len(buf)
		if size < int64(n) {
			n = int(size)
		}
		chunk := buf[:n]
		if _, err := sf.f.ReadAt(chunk, off); err != nil {
			return TensorMeta{}, err
		}
		if _, err := w.Write(chunk); err != nil {
			return TensorMeta{}, err
		}
		off += int64(n)
		size -= int64(n)
	}
	return meta, nil
}

func (sf *File) WriteFloat32To(name string, w io.Writer, rawBuf []byte, floatBuf []float32) (TensorMeta, error) {
	meta, ok := sf.Tensors[name]
	if !ok {
		return TensorMeta{}, fmt.Errorf("missing tensor %s", name)
	}
	if meta.DType == "F32" {
		return sf.WriteRawTo(name, w, rawBuf)
	}
	elemSize, err := dtypeSize(meta.DType)
	if err != nil {
		return TensorMeta{}, fmt.Errorf("%s: %w", name, err)
	}
	count := elemCount(meta.Shape)
	if count == 0 {
		return meta, nil
	}
	const targetElems = 256 * 1024
	blockElems := min(count, targetElems)
	needRaw := blockElems * elemSize
	if cap(rawBuf) < needRaw {
		rawBuf = make([]byte, needRaw)
	} else {
		rawBuf = rawBuf[:needRaw]
	}
	if cap(floatBuf) < blockElems {
		floatBuf = make([]float32, blockElems)
	} else {
		floatBuf = floatBuf[:blockElems]
	}
	base := sf.dataStart + meta.DataOffsets[0]
	for done := 0; done < count; {
		n := min(count-done, blockElems)
		raw := rawBuf[:n*elemSize]
		if _, err := sf.f.ReadAt(raw, base+int64(done*elemSize)); err != nil {
			return TensorMeta{}, err
		}
		values := floatBuf[:n]
		decodeFloat32Row(values, raw, meta.DType)
		if err := writeFloat32LittleEndian(w, values); err != nil {
			return TensorMeta{}, err
		}
		done += n
	}
	return meta, nil
}

func dtypeSize(dtype string) (int, error) {
	switch dtype {
	case "F32":
		return 4, nil
	case "BF16", "F16":
		return 2, nil
	default:
		return 0, fmt.Errorf("unsupported dtype %s", dtype)
	}
}

func decodeFloat32Row(out []float32, raw []byte, dtype string) {
	switch dtype {
	case "F32":
		decodeF32(out, raw)
	case "BF16":
		decodeBF16(out, raw)
	case "F16":
		decodeF16(out, raw)
	}
}

func writeFloat32LittleEndian(w io.Writer, values []float32) error {
	if len(values) == 0 {
		return nil
	}
	if hostLittleEndian {
		raw := unsafe.Slice((*byte)(unsafe.Pointer(&values[0])), len(values)*4)
		_, err := w.Write(raw)
		return err
	}
	buf := make([]byte, len(values)*4)
	for i, v := range values {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
	}
	_, err := w.Write(buf)
	return err
}

func decodeF32(out []float32, raw []byte) {
	i := 0
	for ; i+7 < len(out); i += 8 {
		j := i * 4
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[j:]))
		out[i+1] = math.Float32frombits(binary.LittleEndian.Uint32(raw[j+4:]))
		out[i+2] = math.Float32frombits(binary.LittleEndian.Uint32(raw[j+8:]))
		out[i+3] = math.Float32frombits(binary.LittleEndian.Uint32(raw[j+12:]))
		out[i+4] = math.Float32frombits(binary.LittleEndian.Uint32(raw[j+16:]))
		out[i+5] = math.Float32frombits(binary.LittleEndian.Uint32(raw[j+20:]))
		out[i+6] = math.Float32frombits(binary.LittleEndian.Uint32(raw[j+24:]))
		out[i+7] = math.Float32frombits(binary.LittleEndian.Uint32(raw[j+28:]))
	}
	for ; i < len(out); i++ {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
}

func decodeBF16(out []float32, raw []byte) {
	if hostLittleEndian && len(out) > 0 {
		in := unsafe.Slice((*uint16)(unsafe.Pointer(&raw[0])), len(out))
		i := 0
		for ; i+7 < len(out); i += 8 {
			out[i] = math.Float32frombits(uint32(in[i]) << 16)
			out[i+1] = math.Float32frombits(uint32(in[i+1]) << 16)
			out[i+2] = math.Float32frombits(uint32(in[i+2]) << 16)
			out[i+3] = math.Float32frombits(uint32(in[i+3]) << 16)
			out[i+4] = math.Float32frombits(uint32(in[i+4]) << 16)
			out[i+5] = math.Float32frombits(uint32(in[i+5]) << 16)
			out[i+6] = math.Float32frombits(uint32(in[i+6]) << 16)
			out[i+7] = math.Float32frombits(uint32(in[i+7]) << 16)
		}
		for ; i < len(out); i++ {
			out[i] = math.Float32frombits(uint32(in[i]) << 16)
		}
		return
	}
	for i := range out {
		out[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(raw[i*2:])) << 16)
	}
}

func decodeF16(out []float32, raw []byte) {
	if hostLittleEndian && len(out) > 0 {
		in := unsafe.Slice((*uint16)(unsafe.Pointer(&raw[0])), len(out))
		i := 0
		for ; i+15 < len(out); i += 16 {
			out[i] = f16ValueTable[in[i]]
			out[i+1] = f16ValueTable[in[i+1]]
			out[i+2] = f16ValueTable[in[i+2]]
			out[i+3] = f16ValueTable[in[i+3]]
			out[i+4] = f16ValueTable[in[i+4]]
			out[i+5] = f16ValueTable[in[i+5]]
			out[i+6] = f16ValueTable[in[i+6]]
			out[i+7] = f16ValueTable[in[i+7]]
			out[i+8] = f16ValueTable[in[i+8]]
			out[i+9] = f16ValueTable[in[i+9]]
			out[i+10] = f16ValueTable[in[i+10]]
			out[i+11] = f16ValueTable[in[i+11]]
			out[i+12] = f16ValueTable[in[i+12]]
			out[i+13] = f16ValueTable[in[i+13]]
			out[i+14] = f16ValueTable[in[i+14]]
			out[i+15] = f16ValueTable[in[i+15]]
		}
		for ; i+7 < len(out); i += 8 {
			out[i] = f16ValueTable[in[i]]
			out[i+1] = f16ValueTable[in[i+1]]
			out[i+2] = f16ValueTable[in[i+2]]
			out[i+3] = f16ValueTable[in[i+3]]
			out[i+4] = f16ValueTable[in[i+4]]
			out[i+5] = f16ValueTable[in[i+5]]
			out[i+6] = f16ValueTable[in[i+6]]
			out[i+7] = f16ValueTable[in[i+7]]
		}
		for ; i < len(out); i++ {
			out[i] = f16ValueTable[in[i]]
		}
		return
	}
	i := 0
	for ; i+7 < len(out); i += 8 {
		j := i * 2
		out[i] = f16ValueTable[binary.LittleEndian.Uint16(raw[j:])]
		out[i+1] = f16ValueTable[binary.LittleEndian.Uint16(raw[j+2:])]
		out[i+2] = f16ValueTable[binary.LittleEndian.Uint16(raw[j+4:])]
		out[i+3] = f16ValueTable[binary.LittleEndian.Uint16(raw[j+6:])]
		out[i+4] = f16ValueTable[binary.LittleEndian.Uint16(raw[j+8:])]
		out[i+5] = f16ValueTable[binary.LittleEndian.Uint16(raw[j+10:])]
		out[i+6] = f16ValueTable[binary.LittleEndian.Uint16(raw[j+12:])]
		out[i+7] = f16ValueTable[binary.LittleEndian.Uint16(raw[j+14:])]
	}
	for ; i < len(out); i++ {
		out[i] = f16ValueTable[binary.LittleEndian.Uint16(raw[i*2:])]
	}
}

func elemCount(shape []int64) int {
	n, err := checkedElemCount64(shape)
	if err != nil || n > int64(maxInt()) {
		return 0
	}
	return int(n)
}

func checkedElemCount64(shape []int64) (int64, error) {
	if len(shape) > maxShapeDims {
		return 0, fmt.Errorf("shape has too many dimensions: %d", len(shape))
	}
	n := int64(1)
	for _, d := range shape {
		if d < 0 {
			return 0, fmt.Errorf("shape contains negative dimension %d", d)
		}
		if d != 0 && n > math.MaxInt64/d {
			return 0, fmt.Errorf("shape element count overflows int64")
		}
		n *= d
	}
	return n, nil
}

func checkedElemCount(shape []int64) (int, error) {
	n, err := checkedElemCount64(shape)
	if err != nil {
		return 0, err
	}
	if n > int64(maxInt()) {
		return 0, fmt.Errorf("shape element count overflows int")
	}
	return int(n), nil
}

func maxInt() int {
	return int(^uint(0) >> 1)
}

var f16ValueTable = func() [1 << 16]float32 {
	var t [1 << 16]float32
	for i := range t {
		t[i] = f16Slow(uint16(i))
	}
	return t
}()

func f16Slow(h uint16) float32 {
	sign := uint32(h>>15) & 1
	exp := uint32(h>>10) & 0x1f
	frac := uint32(h & 0x03ff)
	var bits uint32
	switch exp {
	case 0:
		if frac == 0 {
			bits = sign << 31
		} else {
			exp32 := int32(127 - 15 + 1)
			for frac&0x0400 == 0 {
				frac <<= 1
				exp32--
			}
			frac &= 0x03ff
			bits = (sign << 31) | (uint32(exp32) << 23) | (frac << 13)
		}
	case 0x1f:
		bits = (sign << 31) | 0x7f800000 | (frac << 13)
	default:
		bits = (sign << 31) | ((exp + 112) << 23) | (frac << 13)
	}
	return math.Float32frombits(bits)
}
