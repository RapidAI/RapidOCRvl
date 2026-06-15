package safetensors

import "unsafe"

var hostLittleEndian = func() bool {
	var x uint16 = 1
	return *(*byte)(unsafe.Pointer(&x)) == 1
}()

func (sf *File) readFloat32At(dst []float32, off int64) error {
	if len(dst) == 0 {
		return nil
	}
	if !hostLittleEndian {
		buf := make([]byte, len(dst)*4)
		if _, err := sf.f.ReadAt(buf, off); err != nil {
			return err
		}
		decodeF32(dst, buf)
		return nil
	}
	buf := unsafe.Slice((*byte)(unsafe.Pointer(&dst[0])), len(dst)*4)
	_, err := sf.f.ReadAt(buf, off)
	return err
}
