package gguf

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"sync"
	"unsafe"
)

const (
	Magic     = "GGUF"
	Version   = uint32(3)
	Alignment = uint32(32)

	typeF32   = uint32(0)
	typeQ8Row = uint32(1000)
	typeQ4Row = uint32(1001)
	typeQ6Row = uint32(1002)

	metaUint32 = uint32(4)
	metaUint64 = uint32(10)
	metaString = uint32(8)
)

type TensorMeta struct {
	Shape  []int64
	Type   uint32
	Offset uint64
}

type File struct {
	f         *os.File
	dataStart int64
	Tensors   map[string]TensorMeta
	Metadata  map[string]any
}

var float32RowsBufferPool sync.Pool

type headerReader struct {
	r    *bufio.Reader
	n    int64
	pool []byte
	tmp  [8]byte
}

func (r *headerReader) Read(p []byte) (int, error) {
	n, err := r.r.Read(p)
	r.n += int64(n)
	return n, err
}

func (r *headerReader) readString() (string, error) {
	n, err := r.readU64()
	if err != nil {
		return "", err
	}
	if n == 0 {
		return "", nil
	}
	const maxInt = int(^uint(0) >> 1)
	if n > uint64(maxInt-len(r.pool)) {
		return "", fmt.Errorf("GGUF string too large")
	}
	start := len(r.pool)
	end := start + int(n)
	if cap(r.pool) < end {
		next := make([]byte, start, max(end, cap(r.pool)*2))
		copy(next, r.pool)
		r.pool = next
	}
	r.pool = r.pool[:end]
	if _, err := io.ReadFull(r.r, r.pool[start:end]); err != nil {
		return "", err
	}
	r.n += int64(n)
	return unsafe.String(unsafe.SliceData(r.pool[start:end]), int(n)), nil
}

func (r *headerReader) readU32() (uint32, error) {
	if _, err := io.ReadFull(r.r, r.tmp[:4]); err != nil {
		return 0, err
	}
	r.n += 4
	return binary.LittleEndian.Uint32(r.tmp[:4]), nil
}

func (r *headerReader) readU64() (uint64, error) {
	if _, err := io.ReadFull(r.r, r.tmp[:8]); err != nil {
		return 0, err
	}
	r.n += 8
	return binary.LittleEndian.Uint64(r.tmp[:8]), nil
}

func Open(path string) (*File, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	cr := &headerReader{r: bufio.NewReaderSize(f, 64<<10), pool: make([]byte, 0, 8<<10)}
	var magic [4]byte
	if _, err := io.ReadFull(cr, magic[:]); err != nil {
		f.Close()
		return nil, err
	}
	if string(magic[:]) != Magic {
		f.Close()
		return nil, fmt.Errorf("bad GGUF magic")
	}
	version, err := cr.readU32()
	if err != nil {
		f.Close()
		return nil, err
	}
	if version != Version {
		f.Close()
		return nil, fmt.Errorf("unsupported GGUF version %d", version)
	}
	tensorCount, err := cr.readU64()
	if err != nil {
		f.Close()
		return nil, err
	}
	kvCount, err := cr.readU64()
	if err != nil {
		f.Close()
		return nil, err
	}
	metadata := make(map[string]any, kvCount)
	for i := uint64(0); i < kvCount; i++ {
		key, val, err := readHeaderKV(cr)
		if err != nil {
			f.Close()
			return nil, err
		}
		metadata[key] = val
	}
	tensors := make(map[string]TensorMeta, tensorCount)
	shapeBuf := make([]int64, 0, tensorCount*2)
	for i := uint64(0); i < tensorCount; i++ {
		name, err := cr.readString()
		if err != nil {
			f.Close()
			return nil, err
		}
		nd, err := cr.readU32()
		if err != nil {
			f.Close()
			return nil, err
		}
		start := len(shapeBuf)
		end := start + int(nd)
		if cap(shapeBuf) < end {
			next := make([]int64, end, max(end, cap(shapeBuf)*2))
			copy(next, shapeBuf)
			shapeBuf = next
		} else {
			shapeBuf = shapeBuf[:end]
		}
		shape := shapeBuf[start:end]
		for j := 0; j < len(shape); j++ {
			d, err := cr.readU64()
			if err != nil {
				f.Close()
				return nil, err
			}
			shape[j] = int64(d)
		}
		typ, err := cr.readU32()
		if err != nil {
			f.Close()
			return nil, err
		}
		off, err := cr.readU64()
		if err != nil {
			f.Close()
			return nil, err
		}
		tensors[name] = TensorMeta{Shape: shape, Type: typ, Offset: off}
	}
	dataStart := align(cr.n, uint64(Alignment))
	return &File{f: f, dataStart: int64(dataStart), Tensors: tensors, Metadata: metadata}, nil
}

func (gf *File) Close() error {
	if gf.f == nil {
		return nil
	}
	return gf.f.Close()
}

func (gf *File) Shape(name string) ([]int64, error) {
	meta, ok := gf.Tensors[name]
	if !ok {
		return nil, fmt.Errorf("missing tensor %s", name)
	}
	return meta.Shape, nil
}

func (gf *File) Float32(name string) ([]float32, []int64, error) {
	meta, ok := gf.Tensors[name]
	if !ok {
		return nil, nil, fmt.Errorf("missing tensor %s", name)
	}
	if meta.Type != typeF32 {
		return nil, nil, fmt.Errorf("unsupported GGUF tensor type %d for %s", meta.Type, name)
	}
	count := elemCount(meta.Shape)
	out := make([]float32, count)
	if err := gf.readFloat32At(out, gf.dataStart+int64(meta.Offset)); err != nil {
		return nil, nil, err
	}
	return out, meta.Shape, nil
}

func (gf *File) Float32Rows(name string, fn func(row int, values []float32) error) ([]int64, error) {
	return gf.Float32RowsBuffer(name, nil, fn)
}

func (gf *File) Float32RowsBuffer(name string, buf []float32, fn func(row int, values []float32) error) ([]int64, error) {
	meta, ok := gf.Tensors[name]
	if !ok {
		return nil, fmt.Errorf("missing tensor %s", name)
	}
	if meta.Type != typeF32 {
		return nil, fmt.Errorf("unsupported GGUF tensor type %d for %s", meta.Type, name)
	}
	if len(meta.Shape) != 2 {
		return nil, fmt.Errorf("tensor %s must be 2D for row streaming", name)
	}
	rows := int(meta.Shape[0])
	cols := int(meta.Shape[1])
	rowsPerBlock := float32RowsPerBlock(rows, cols*4)
	need := rowsPerBlock * cols
	usePool := buf == nil
	if buf == nil {
		if v := float32RowsBufferPool.Get(); v != nil {
			p := v.(*[]float32)
			if cap(*p) >= need {
				buf = (*p)[:need]
			}
		}
	}
	if len(buf) < need {
		buf = make([]float32, need)
	} else {
		buf = buf[:need]
	}
	if usePool {
		defer putFloat32RowsBuffer(buf)
	}
	base := gf.dataStart + int64(meta.Offset)
	for r := 0; r < rows; {
		n := rowsPerBlock
		if rows-r < n {
			n = rows - r
		}
		block := buf[:n*cols]
		if err := gf.readFloat32At(block, base+int64(r*cols*4)); err != nil {
			return nil, err
		}
		for i := 0; i < n; i++ {
			if err := fn(r+i, block[i*cols:(i+1)*cols]); err != nil {
				return nil, err
			}
		}
		r += n
	}
	return meta.Shape, nil
}

func putFloat32RowsBuffer(buf []float32) {
	const maxPooledFloat32RowsBuffer = 1 << 20
	if cap(buf) == 0 || cap(buf) > maxPooledFloat32RowsBuffer {
		return
	}
	buf = buf[:0]
	float32RowsBufferPool.Put(&buf)
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

func (gf *File) Q8Row(name string) ([]int8, []float32, []int64, error) {
	meta, ok := gf.Tensors[name]
	if !ok {
		return nil, nil, nil, fmt.Errorf("missing tensor %s", name)
	}
	if meta.Type != typeQ8Row {
		return nil, nil, nil, fmt.Errorf("unsupported GGUF tensor type %d for %s", meta.Type, name)
	}
	if len(meta.Shape) != 2 {
		return nil, nil, nil, fmt.Errorf("Q8_ROW tensor %s must be 2D", name)
	}
	rows := int(meta.Shape[0])
	cols := int(meta.Shape[1])
	scaleBytes := rows * 4
	dataBytes := rows * cols
	base := gf.dataStart + int64(meta.Offset)
	if hostLittleEndian {
		raw := make([]byte, scaleBytes+dataBytes)
		if _, err := gf.f.ReadAt(raw, base); err != nil {
			return nil, nil, nil, err
		}
		var scales []float32
		if rows > 0 {
			scales = unsafe.Slice((*float32)(unsafe.Pointer(&raw[0])), rows)
		}
		var data []int8
		if dataBytes > 0 {
			data = unsafe.Slice((*int8)(unsafe.Pointer(&raw[scaleBytes])), dataBytes)
		}
		return data, scales, meta.Shape, nil
	}
	scales := make([]float32, rows)
	if err := gf.readFloat32At(scales, base); err != nil {
		return nil, nil, nil, err
	}
	data := make([]int8, dataBytes)
	if err := gf.readInt8At(data, base+int64(scaleBytes)); err != nil {
		return nil, nil, nil, err
	}
	return data, scales, meta.Shape, nil
}

func bytesToInt8(dst []int8, src []byte) {
	i := 0
	n := len(src)
	for ; i+7 < n; i += 8 {
		dst[i] = int8(src[i])
		dst[i+1] = int8(src[i+1])
		dst[i+2] = int8(src[i+2])
		dst[i+3] = int8(src[i+3])
		dst[i+4] = int8(src[i+4])
		dst[i+5] = int8(src[i+5])
		dst[i+6] = int8(src[i+6])
		dst[i+7] = int8(src[i+7])
	}
	for ; i < n; i++ {
		dst[i] = int8(src[i])
	}
}

func (gf *File) Q4Row(name string) ([]byte, []float32, []int64, error) {
	meta, ok := gf.Tensors[name]
	if !ok {
		return nil, nil, nil, fmt.Errorf("missing tensor %s", name)
	}
	if meta.Type != typeQ4Row {
		return nil, nil, nil, fmt.Errorf("unsupported GGUF tensor type %d for %s", meta.Type, name)
	}
	if len(meta.Shape) != 2 {
		return nil, nil, nil, fmt.Errorf("Q4_ROW tensor %s must be 2D", name)
	}
	rows := int(meta.Shape[0])
	cols := int(meta.Shape[1])
	scaleBytes := rows * 4
	dataBytes := rows * ((cols + 1) / 2)
	base := gf.dataStart + int64(meta.Offset)
	if hostLittleEndian {
		raw := make([]byte, scaleBytes+dataBytes)
		if _, err := gf.f.ReadAt(raw, base); err != nil {
			return nil, nil, nil, err
		}
		var scales []float32
		if rows > 0 {
			scales = unsafe.Slice((*float32)(unsafe.Pointer(&raw[0])), rows)
		}
		return raw[scaleBytes:], scales, meta.Shape, nil
	}
	scales := make([]float32, rows)
	if err := gf.readFloat32At(scales, base); err != nil {
		return nil, nil, nil, err
	}
	data := make([]byte, dataBytes)
	if _, err := gf.f.ReadAt(data, base+int64(scaleBytes)); err != nil {
		return nil, nil, nil, err
	}
	return data, scales, meta.Shape, nil
}

func (gf *File) Q6Row(name string) ([]byte, []float32, []int64, error) {
	meta, ok := gf.Tensors[name]
	if !ok {
		return nil, nil, nil, fmt.Errorf("missing tensor %s", name)
	}
	if meta.Type != typeQ6Row {
		return nil, nil, nil, fmt.Errorf("unsupported GGUF tensor type %d for %s", meta.Type, name)
	}
	if len(meta.Shape) != 2 {
		return nil, nil, nil, fmt.Errorf("Q6_ROW tensor %s must be 2D", name)
	}
	rows := int(meta.Shape[0])
	cols := int(meta.Shape[1])
	scaleBytes := rows * 4
	dataBytes := rows * ((cols*6 + 7) / 8)
	base := gf.dataStart + int64(meta.Offset)
	if hostLittleEndian {
		raw := make([]byte, scaleBytes+dataBytes)
		if _, err := gf.f.ReadAt(raw, base); err != nil {
			return nil, nil, nil, err
		}
		var scales []float32
		if rows > 0 {
			scales = unsafe.Slice((*float32)(unsafe.Pointer(&raw[0])), rows)
		}
		return raw[scaleBytes:], scales, meta.Shape, nil
	}
	scales := make([]float32, rows)
	if err := gf.readFloat32At(scales, base); err != nil {
		return nil, nil, nil, err
	}
	data := make([]byte, dataBytes)
	if _, err := gf.f.ReadAt(data, base+int64(scaleBytes)); err != nil {
		return nil, nil, nil, err
	}
	return data, scales, meta.Shape, nil
}

func TensorTypeName(typ uint32) string {
	switch typ {
	case typeF32:
		return "F32"
	case typeQ8Row:
		return "Q8_ROW"
	case typeQ4Row:
		return "Q4_ROW"
	case typeQ6Row:
		return "Q6_ROW"
	default:
		return fmt.Sprintf("GGUF_%d", typ)
	}
}

func TensorBytes(meta TensorMeta) int64 {
	switch meta.Type {
	case typeF32:
		return int64(elemCount(meta.Shape) * 4)
	case typeQ8Row:
		if len(meta.Shape) != 2 {
			return 0
		}
		rows, cols := meta.Shape[0], meta.Shape[1]
		return rows*4 + rows*cols
	case typeQ4Row:
		if len(meta.Shape) != 2 {
			return 0
		}
		rows, cols := meta.Shape[0], meta.Shape[1]
		return rows*4 + rows*((cols+1)/2)
	case typeQ6Row:
		if len(meta.Shape) != 2 {
			return 0
		}
		rows, cols := meta.Shape[0], meta.Shape[1]
		return rows*4 + rows*((cols*6+7)/8)
	default:
		return 0
	}
}

func elemCount(shape []int64) int {
	n := int64(1)
	for _, d := range shape {
		n *= d
	}
	return int(n)
}

func readHeaderKV(r *headerReader) (string, any, error) {
	key, err := r.readString()
	if err != nil {
		return "", nil, err
	}
	typ, err := r.readU32()
	if err != nil {
		return "", nil, err
	}
	switch typ {
	case metaUint32:
		v, err := r.readU32()
		return key, v, err
	case metaUint64:
		v, err := r.readU64()
		return key, v, err
	case metaString:
		v, err := r.readString()
		return key, v, err
	default:
		return "", nil, fmt.Errorf("unsupported GGUF metadata type %d", typ)
	}
}

func readString(r io.Reader) (string, error) {
	if sr, ok := r.(interface{ readString() (string, error) }); ok {
		return sr.readString()
	}
	n, err := readU64(r)
	if err != nil {
		return "", err
	}
	buf := make([]byte, n)
	if _, err := io.ReadFull(r, buf); err != nil {
		return "", err
	}
	if len(buf) == 0 {
		return "", nil
	}
	return unsafe.String(unsafe.SliceData(buf), len(buf)), nil
}

func readU32(r io.Reader) (uint32, error) {
	if rr, ok := r.(interface{ readU32() (uint32, error) }); ok {
		return rr.readU32()
	}
	var buf [4]byte
	if _, err := io.ReadFull(r, buf[:]); err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint32(buf[:]), nil
}

func readU64(r io.Reader) (uint64, error) {
	if rr, ok := r.(interface{ readU64() (uint64, error) }); ok {
		return rr.readU64()
	}
	var buf [8]byte
	if _, err := io.ReadFull(r, buf[:]); err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint64(buf[:]), nil
}

func align(v int64, a uint64) uint64 {
	u := uint64(v)
	return u + (a-(u%a))%a
}
