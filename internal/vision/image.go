package vision

import (
	"fmt"
	"image"
	"image/color"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"math"
	"os"
	"runtime"
	"sync"
)

type Grid struct {
	T int
	H int
	W int
}

type Preprocessed struct {
	Patches [][]float32
	Grid    Grid
	Width   int
	Height  int
}

type bilinearIndex struct {
	I0 int
	I1 int
	W0 float32
	W1 float32
}

const (
	patchSize = 14
	mergeSize = 2
	minPixels = 28 * 28 * 130
	maxPixels = 28 * 28 * 1280
)

var clipMean = [3]float32{0.48145466, 0.4578275, 0.40821073}
var clipStd = [3]float32{0.26862954, 0.26130258, 0.27577711}
var clipScale = [3]float32{1 / (255 * 0.26862954), 1 / (255 * 0.26130258), 1 / (255 * 0.27577711)}
var clipBias = [3]float32{-0.48145466 / 0.26862954, -0.4578275 / 0.26130258, -0.40821073 / 0.27577711}

const maxPooledBilinearIndexes = 1 << 15

var bilinearIndexPool sync.Pool

func getBilinearIndexes(n int) ([]bilinearIndex, *[]bilinearIndex) {
	if v := bilinearIndexPool.Get(); v != nil {
		p := v.(*[]bilinearIndex)
		if cap(*p) >= n {
			return (*p)[:n], p
		}
	}
	idx := make([]bilinearIndex, n)
	return idx, &idx
}

func putBilinearIndexes(idx []bilinearIndex, p *[]bilinearIndex) {
	if cap(idx) <= maxPooledBilinearIndexes {
		*p = idx[:0]
		bilinearIndexPool.Put(p)
	}
}

func LoadImage(path string) (*Preprocessed, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		return nil, err
	}
	return PreprocessImage(img)
}

func LoadImageReader(r io.Reader) (*Preprocessed, error) {
	img, _, err := image.Decode(r)
	if err != nil {
		return nil, err
	}
	return PreprocessImage(img)
}

func PreprocessImage(img image.Image) (*Preprocessed, error) {
	b := img.Bounds()
	h, w := b.Dy(), b.Dx()
	nh, nw, err := smartResize(h, w, patchSize*mergeSize, minPixels, maxPixels)
	if err != nil {
		return nil, err
	}
	gridH, gridW := nh/patchSize, nw/patchSize
	patchCount := gridH * gridW
	patchDim := 3 * patchSize * patchSize
	patches := make([][]float32, patchCount)
	patchData := make([]float32, patchCount*patchDim)
	rs, gs, bs := clipScale[0], clipScale[1], clipScale[2]
	rb, gb, bb := clipBias[0], clipBias[1], clipBias[2]
	if rgba, ok := img.(*image.RGBA); ok && rgba.Bounds().Min.X == 0 && rgba.Bounds().Min.Y == 0 && rgba.Bounds().Dx() == nw && rgba.Bounds().Dy() == nh {
		extractPatchesRGBAMaybeParallel(patches, patchData, rgba, gridW, patchCount, patchDim, rs, gs, bs, rb, gb, bb)
		return &Preprocessed{Patches: patches, Grid: Grid{T: 1, H: gridH, W: gridW}, Width: nw, Height: nh}, nil
	}
	if rgba, ok := img.(*image.RGBA); ok && rgba.Bounds().Min.X >= 0 && rgba.Bounds().Min.Y >= 0 && nw > 1 && nh > 1 {
		extractResizedPatchesRGBA(patches, patchData, rgba, nw, nh, gridW, rs, gs, bs, rb, gb, bb)
		return &Preprocessed{Patches: patches, Grid: Grid{T: 1, H: gridH, W: gridW}, Width: nw, Height: nh}, nil
	}
	resized := resizeBilinear(img, nw, nh)
	extractPatchesRGBAMaybeParallel(patches, patchData, resized, gridW, patchCount, patchDim, rs, gs, bs, rb, gb, bb)
	return &Preprocessed{Patches: patches, Grid: Grid{T: 1, H: gridH, W: gridW}, Width: nw, Height: nh}, nil
}

func extractPatchesRGBAMaybeParallel(patches [][]float32, patchData []float32, resized *image.RGBA, gridW, patchCount, patchDim int, rs, gs, bs, rb, gb, bb float32) {
	work := patchCount * patchDim
	if work >= 1<<18 && patchCount >= 4 {
		workers := min(runtime.GOMAXPROCS(0), patchCount)
		var wg sync.WaitGroup
		for worker := 1; worker < workers; worker++ {
			start := worker * patchCount / workers
			end := (worker + 1) * patchCount / workers
			wg.Add(1)
			go func() {
				extractPatchesRGBA(patches, patchData, resized, gridW, start, end, rs, gs, bs, rb, gb, bb)
				wg.Done()
			}()
		}
		extractPatchesRGBA(patches, patchData, resized, gridW, 0, patchCount/workers, rs, gs, bs, rb, gb, bb)
		wg.Wait()
	} else {
		extractPatchesRGBA(patches, patchData, resized, gridW, 0, patchCount, rs, gs, bs, rb, gb, bb)
	}
}

func extractResizedPatchesRGBA(patches [][]float32, patchData []float32, src *image.RGBA, w, h, gridW int, rs, gs, bs, rb, gb, bb float32) {
	sb := src.Bounds()
	xScale := float64(sb.Dx()-1) / float64(w-1)
	yScale := float64(sb.Dy()-1) / float64(h-1)
	idx, idxPtr := getBilinearIndexes(w + h)
	defer putBilinearIndexes(idx, idxPtr)
	xs := idx[:w]
	for x := 0; x < w; x++ {
		fx := float64(x)*xScale + float64(sb.Min.X)
		x0 := int(fx)
		x1 := min(x0+1, sb.Max.X-1)
		wx := float32(fx - float64(x0))
		xs[x] = bilinearIndex{I0: (x0 - src.Rect.Min.X) * 4, I1: (x1 - src.Rect.Min.X) * 4, W0: 1 - wx, W1: wx}
	}
	ys := idx[w:]
	for y := 0; y < h; y++ {
		fy := float64(y)*yScale + float64(sb.Min.Y)
		y0 := int(fy)
		y1 := min(y0+1, sb.Max.Y-1)
		wy := float32(fy - float64(y0))
		ys[y] = bilinearIndex{I0: (y0 - src.Rect.Min.Y) * src.Stride, I1: (y1 - src.Rect.Min.Y) * src.Stride, W0: 1 - wy, W1: wy}
	}
	patchCount := len(patches)
	work := patchCount * 3 * patchSize * patchSize
	if work >= 1<<18 && patchCount >= 4 {
		workers := min(runtime.GOMAXPROCS(0), patchCount)
		var wg sync.WaitGroup
		for worker := 1; worker < workers; worker++ {
			start := worker * patchCount / workers
			end := (worker + 1) * patchCount / workers
			wg.Add(1)
			go func() {
				extractResizedPatchesRGBARange(patches, patchData, src.Pix, gridW, xs, ys, start, end, rs, gs, bs, rb, gb, bb)
				wg.Done()
			}()
		}
		extractResizedPatchesRGBARange(patches, patchData, src.Pix, gridW, xs, ys, 0, patchCount/workers, rs, gs, bs, rb, gb, bb)
		wg.Wait()
		return
	}
	extractResizedPatchesRGBARange(patches, patchData, src.Pix, gridW, xs, ys, 0, patchCount, rs, gs, bs, rb, gb, bb)
}

func extractResizedPatchesRGBARange(patches [][]float32, patchData []float32, pix []byte, gridW int, xs, ys []bilinearIndex, start, end int, rs, gs, bs, rb, gb, bb float32) {
	patchDim := 3 * patchSize * patchSize
	channel := patchSize * patchSize
	for patchIdx := start; patchIdx < end; patchIdx++ {
		gy, gx := patchIdx/gridW, patchIdx%gridW
		p := patchData[patchIdx*patchDim : (patchIdx+1)*patchDim]
		pr := p[:channel]
		pg := p[channel : 2*channel]
		pb := p[2*channel:]
		for py := 0; py < patchSize; py++ {
			y := gy*patchSize + py
			yi := ys[y]
			row0 := yi.I0
			row1 := yi.I1
			yw0 := yi.W0
			yw1 := yi.W1
			idx := py * patchSize
			px := 0
			for ; px+1 < patchSize; px += 2 {
				x := gx*patchSize + px
				xi := xs[x]
				xw0 := xi.W0
				xw1 := xi.W1
				col0 := xi.I0
				col1 := xi.I1
				o00 := row0 + col0
				o10 := row0 + col1
				o01 := row1 + col0
				o11 := row1 + col1
				r := (float32(pix[o00])*xw0+float32(pix[o10])*xw1)*yw0 + (float32(pix[o01])*xw0+float32(pix[o11])*xw1)*yw1
				g := (float32(pix[o00+1])*xw0+float32(pix[o10+1])*xw1)*yw0 + (float32(pix[o01+1])*xw0+float32(pix[o11+1])*xw1)*yw1
				b := (float32(pix[o00+2])*xw0+float32(pix[o10+2])*xw1)*yw0 + (float32(pix[o01+2])*xw0+float32(pix[o11+2])*xw1)*yw1
				j := idx + px
				pr[j] = roundByteFloat(r)*rs + rb
				pg[j] = roundByteFloat(g)*gs + gb
				pb[j] = roundByteFloat(b)*bs + bb

				x++
				xi = xs[x]
				xw0 = xi.W0
				xw1 = xi.W1
				col0 = xi.I0
				col1 = xi.I1
				o00 = row0 + col0
				o10 = row0 + col1
				o01 = row1 + col0
				o11 = row1 + col1
				r = (float32(pix[o00])*xw0+float32(pix[o10])*xw1)*yw0 + (float32(pix[o01])*xw0+float32(pix[o11])*xw1)*yw1
				g = (float32(pix[o00+1])*xw0+float32(pix[o10+1])*xw1)*yw0 + (float32(pix[o01+1])*xw0+float32(pix[o11+1])*xw1)*yw1
				b = (float32(pix[o00+2])*xw0+float32(pix[o10+2])*xw1)*yw0 + (float32(pix[o01+2])*xw0+float32(pix[o11+2])*xw1)*yw1
				pr[j+1] = roundByteFloat(r)*rs + rb
				pg[j+1] = roundByteFloat(g)*gs + gb
				pb[j+1] = roundByteFloat(b)*bs + bb
			}
			for ; px < patchSize; px++ {
				x := gx*patchSize + px
				xi := xs[x]
				xw0 := xi.W0
				xw1 := xi.W1
				col0 := xi.I0
				col1 := xi.I1
				o00 := row0 + col0
				o10 := row0 + col1
				o01 := row1 + col0
				o11 := row1 + col1
				r := (float32(pix[o00])*xw0+float32(pix[o10])*xw1)*yw0 + (float32(pix[o01])*xw0+float32(pix[o11])*xw1)*yw1
				g := (float32(pix[o00+1])*xw0+float32(pix[o10+1])*xw1)*yw0 + (float32(pix[o01+1])*xw0+float32(pix[o11+1])*xw1)*yw1
				b := (float32(pix[o00+2])*xw0+float32(pix[o10+2])*xw1)*yw0 + (float32(pix[o01+2])*xw0+float32(pix[o11+2])*xw1)*yw1
				j := idx + px
				pr[j] = roundByteFloat(r)*rs + rb
				pg[j] = roundByteFloat(g)*gs + gb
				pb[j] = roundByteFloat(b)*bs + bb
			}
		}
		patches[patchIdx] = p
	}
}

func roundByteFloat(v float32) float32 {
	return float32(int(v + 0.5))
}

func extractPatchesRGBA(patches [][]float32, patchData []float32, resized *image.RGBA, gridW, start, end int, rs, gs, bs, rb, gb, bb float32) {
	patchDim := 3 * patchSize * patchSize
	channel := patchSize * patchSize
	for patchIdx := start; patchIdx < end; patchIdx++ {
		gy, gx := patchIdx/gridW, patchIdx%gridW
		p := patchData[patchIdx*patchDim : (patchIdx+1)*patchDim]
		pr := p[:channel]
		pg := p[channel : 2*channel]
		pb := p[2*channel:]
		for py := 0; py < patchSize; py++ {
			y := gy*patchSize + py
			off := y*resized.Stride + gx*patchSize*4
			idx := py * patchSize
			px := 0
			for ; px+3 < patchSize; px += 4 {
				r := float32(resized.Pix[off])
				g := float32(resized.Pix[off+1])
				b := float32(resized.Pix[off+2])
				r1 := float32(resized.Pix[off+4])
				g1 := float32(resized.Pix[off+5])
				b1 := float32(resized.Pix[off+6])
				r2 := float32(resized.Pix[off+8])
				g2 := float32(resized.Pix[off+9])
				b2 := float32(resized.Pix[off+10])
				r3 := float32(resized.Pix[off+12])
				g3 := float32(resized.Pix[off+13])
				b3 := float32(resized.Pix[off+14])
				j := idx + px
				pr[j] = r*rs + rb
				pg[j] = g*gs + gb
				pb[j] = b*bs + bb
				pr[j+1] = r1*rs + rb
				pg[j+1] = g1*gs + gb
				pb[j+1] = b1*bs + bb
				pr[j+2] = r2*rs + rb
				pg[j+2] = g2*gs + gb
				pb[j+2] = b2*bs + bb
				pr[j+3] = r3*rs + rb
				pg[j+3] = g3*gs + gb
				pb[j+3] = b3*bs + bb
				off += 16
			}
			for ; px < patchSize; px++ {
				j := idx + px
				pr[j] = float32(resized.Pix[off])*rs + rb
				pg[j] = float32(resized.Pix[off+1])*gs + gb
				pb[j] = float32(resized.Pix[off+2])*bs + bb
				off += 4
			}
		}
		patches[patchIdx] = p
	}
}

func smartResize(height, width, factor, minPix, maxPix int) (int, int, error) {
	if height < factor {
		width = int(math.Round(float64(width*factor) / float64(height)))
		height = factor
	}
	if width < factor {
		height = int(math.Round(float64(height*factor) / float64(width)))
		width = factor
	}
	ratio := float64(max(height, width)) / float64(min(height, width))
	if ratio > 200 {
		return 0, 0, fmt.Errorf("absolute aspect ratio must be smaller than 200, got %.3f", ratio)
	}
	hBar := int(math.Round(float64(height)/float64(factor))) * factor
	wBar := int(math.Round(float64(width)/float64(factor))) * factor
	if hBar*wBar > maxPix {
		beta := math.Sqrt(float64(height*width) / float64(maxPix))
		hBar = int(math.Floor(float64(height)/beta/float64(factor))) * factor
		wBar = int(math.Floor(float64(width)/beta/float64(factor))) * factor
	} else if hBar*wBar < minPix {
		beta := math.Sqrt(float64(minPix) / float64(height*width))
		hBar = int(math.Ceil(float64(height)*beta/float64(factor))) * factor
		wBar = int(math.Ceil(float64(width)*beta/float64(factor))) * factor
	}
	return hBar, wBar, nil
}

func resizeBilinear(src image.Image, w, h int) *image.RGBA {
	dst := image.NewRGBA(image.Rect(0, 0, w, h))
	sb := src.Bounds()
	if w == 1 || h == 1 {
		return dst
	}
	if rgba, ok := src.(*image.RGBA); ok {
		resizeBilinearRGBA(dst, rgba, w, h)
		return dst
	}
	xScale := float64(sb.Dx()-1) / float64(w-1)
	yScale := float64(sb.Dy()-1) / float64(h-1)
	for y := 0; y < h; y++ {
		fy := float64(y)*yScale + float64(sb.Min.Y)
		y0 := int(math.Floor(fy))
		y1 := min(y0+1, sb.Max.Y-1)
		wy := float32(fy - float64(y0))
		for x := 0; x < w; x++ {
			fx := float64(x)*xScale + float64(sb.Min.X)
			x0 := int(math.Floor(fx))
			x1 := min(x0+1, sb.Max.X-1)
			wx := float32(fx - float64(x0))
			r00, g00, b00 := rgb(src.At(x0, y0))
			r10, g10, b10 := rgb(src.At(x1, y0))
			r01, g01, b01 := rgb(src.At(x0, y1))
			r11, g11, b11 := rgb(src.At(x1, y1))
			r := bilerp(r00, r10, r01, r11, wx, wy)
			g := bilerp(g00, g10, g01, g11, wx, wy)
			b := bilerp(b00, b10, b01, b11, wx, wy)
			dst.SetRGBA(x, y, color.RGBA{uint8(clamp(r)), uint8(clamp(g)), uint8(clamp(b)), 255})
		}
	}
	return dst
}

func resizeBilinearRGBA(dst, src *image.RGBA, w, h int) {
	sb := src.Bounds()
	if sb.Min.X >= 0 && sb.Min.Y >= 0 {
		resizeBilinearRGBAFast(dst, src, w, h, sb)
		return
	}
	xScale := float64(sb.Dx()-1) / float64(w-1)
	yScale := float64(sb.Dy()-1) / float64(h-1)
	for y := 0; y < h; y++ {
		fy := float64(y)*yScale + float64(sb.Min.Y)
		y0 := int(math.Floor(fy))
		y1 := min(y0+1, sb.Max.Y-1)
		wy := float32(fy - float64(y0))
		row0 := (y0 - src.Rect.Min.Y) * src.Stride
		row1 := (y1 - src.Rect.Min.Y) * src.Stride
		for x := 0; x < w; x++ {
			fx := float64(x)*xScale + float64(sb.Min.X)
			x0 := int(math.Floor(fx))
			x1 := min(x0+1, sb.Max.X-1)
			wx := float32(fx - float64(x0))
			col0 := (x0 - src.Rect.Min.X) * 4
			col1 := (x1 - src.Rect.Min.X) * 4
			o00 := row0 + col0
			o10 := row0 + col1
			o01 := row1 + col0
			o11 := row1 + col1
			r := bilerp(float32(src.Pix[o00]), float32(src.Pix[o10]), float32(src.Pix[o01]), float32(src.Pix[o11]), wx, wy)
			g := bilerp(float32(src.Pix[o00+1]), float32(src.Pix[o10+1]), float32(src.Pix[o01+1]), float32(src.Pix[o11+1]), wx, wy)
			b := bilerp(float32(src.Pix[o00+2]), float32(src.Pix[o10+2]), float32(src.Pix[o01+2]), float32(src.Pix[o11+2]), wx, wy)
			do := y*dst.Stride + x*4
			dst.Pix[do] = uint8(clamp(r))
			dst.Pix[do+1] = uint8(clamp(g))
			dst.Pix[do+2] = uint8(clamp(b))
			dst.Pix[do+3] = 255
		}
	}
}

func resizeBilinearRGBAFast(dst, src *image.RGBA, w, h int, sb image.Rectangle) {
	xScale := float64(sb.Dx()-1) / float64(w-1)
	yScale := float64(sb.Dy()-1) / float64(h-1)
	spix := src.Pix
	dpix := dst.Pix
	idx, idxPtr := getBilinearIndexes(w + h)
	defer putBilinearIndexes(idx, idxPtr)
	xs := idx[:w]
	for x := 0; x < w; x++ {
		fx := float64(x)*xScale + float64(sb.Min.X)
		x0 := int(fx)
		x1 := min(x0+1, sb.Max.X-1)
		wx := float32(fx - float64(x0))
		xs[x] = bilinearIndex{I0: (x0 - src.Rect.Min.X) * 4, I1: (x1 - src.Rect.Min.X) * 4, W0: 1 - wx, W1: wx}
	}
	ys := idx[w:]
	for y := 0; y < h; y++ {
		fy := float64(y)*yScale + float64(sb.Min.Y)
		y0 := int(fy)
		y1 := min(y0+1, sb.Max.Y-1)
		wy := float32(fy - float64(y0))
		ys[y] = bilinearIndex{I0: (y0 - src.Rect.Min.Y) * src.Stride, I1: (y1 - src.Rect.Min.Y) * src.Stride, W0: 1 - wy, W1: wy}
	}
	pixels := w * h
	if pixels >= 1<<16 && h >= 4 {
		workers := min(runtime.GOMAXPROCS(0), h)
		var wg sync.WaitGroup
		for worker := 1; worker < workers; worker++ {
			start := worker * h / workers
			end := (worker + 1) * h / workers
			wg.Add(1)
			go func() {
				resizeBilinearRGBARows(dpix, spix, dst.Stride, w, xs, ys, start, end)
				wg.Done()
			}()
		}
		resizeBilinearRGBARows(dpix, spix, dst.Stride, w, xs, ys, 0, h/workers)
		wg.Wait()
		return
	}
	resizeBilinearRGBARows(dpix, spix, dst.Stride, w, xs, ys, 0, h)
}

func resizeBilinearRGBARows(dpix, spix []byte, dstStride, w int, xs, ys []bilinearIndex, start, end int) {
	for y := start; y < end; y++ {
		yi := ys[y]
		yw0 := yi.W0
		yw1 := yi.W1
		row0 := yi.I0
		row1 := yi.I1
		drow := y * dstStride
		for x := 0; x < w; x++ {
			xi := xs[x]
			xw0 := xi.W0
			xw1 := xi.W1
			col0 := xi.I0
			col1 := xi.I1
			o00 := row0 + col0
			o10 := row0 + col1
			o01 := row1 + col0
			o11 := row1 + col1
			r := (float32(spix[o00])*xw0+float32(spix[o10])*xw1)*yw0 + (float32(spix[o01])*xw0+float32(spix[o11])*xw1)*yw1
			g := (float32(spix[o00+1])*xw0+float32(spix[o10+1])*xw1)*yw0 + (float32(spix[o01+1])*xw0+float32(spix[o11+1])*xw1)*yw1
			b := (float32(spix[o00+2])*xw0+float32(spix[o10+2])*xw1)*yw0 + (float32(spix[o01+2])*xw0+float32(spix[o11+2])*xw1)*yw1
			do := drow + x*4
			dpix[do] = uint8(clamp(r))
			dpix[do+1] = uint8(clamp(g))
			dpix[do+2] = uint8(clamp(b))
			dpix[do+3] = 255
		}
	}
}

func rgb(c color.Color) (float32, float32, float32) {
	r, g, b, _ := c.RGBA()
	return float32(r >> 8), float32(g >> 8), float32(b >> 8)
}

func bilerp(a, b, c, d, wx, wy float32) float32 {
	top := a*(1-wx) + b*wx
	bot := c*(1-wx) + d*wx
	return top*(1-wy) + bot*wy
}

func clamp(v float32) int {
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return int(v + 0.5)
}
