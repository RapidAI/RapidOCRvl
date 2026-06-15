package gguf

import (
	"unsafe"
)

func (gf *File) readInt8At(dst []int8, off int64) error {
	if len(dst) == 0 {
		return nil
	}
	buf := unsafe.Slice((*byte)(unsafe.Pointer(&dst[0])), len(dst))
	_, err := gf.f.ReadAt(buf, off)
	return err
}
