package vision

import (
	"bytes"
	"encoding/binary"
	"hash/crc32"
	"image"
	"image/color"
	"math"
	"runtime"
	"strings"
	"testing"
)

type wrappedRGBA struct {
	img *image.RGBA
}

func (w wrappedRGBA) ColorModel() color.Model { return w.img.ColorModel() }
func (w wrappedRGBA) Bounds() image.Rectangle { return w.img.Bounds() }
func (w wrappedRGBA) At(x, y int) color.Color { return w.img.At(x, y) }

type wrappedNRGBA struct {
	img *image.NRGBA
}

func (w wrappedNRGBA) ColorModel() color.Model { return w.img.ColorModel() }
func (w wrappedNRGBA) Bounds() image.Rectangle { return w.img.Bounds() }
func (w wrappedNRGBA) At(x, y int) color.Color { return w.img.At(x, y) }

type boundsOnlyImage struct {
	rect image.Rectangle
}

func (b boundsOnlyImage) ColorModel() color.Model { return color.RGBAModel }
func (b boundsOnlyImage) Bounds() image.Rectangle { return b.rect }
func (b boundsOnlyImage) At(x, y int) color.Color { return color.RGBA{} }

func TestPreprocessImageChannelOrder(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 28, 28))
	for y := 0; y < 28; y++ {
		for x := 0; x < 28; x++ {
			img.SetRGBA(x, y, color.RGBA{R: 255, G: 128, B: 0, A: 255})
		}
	}
	pp, err := PreprocessImage(img)
	if err != nil {
		t.Fatal(err)
	}
	if len(pp.Patches) == 0 {
		t.Fatal("no patches")
	}
	p := pp.Patches[0]
	wantR := (float32(255)/255 - clipMean[0]) / clipStd[0]
	wantG := (float32(128)/255 - clipMean[1]) / clipStd[1]
	wantB := (float32(0)/255 - clipMean[2]) / clipStd[2]
	check := func(name string, got, want float32) {
		t.Helper()
		if math.Abs(float64(got-want)) > 1e-5 {
			t.Fatalf("%s got %f want %f", name, got, want)
		}
	}
	check("r", p[0], wantR)
	check("g", p[patchSize*patchSize], wantG)
	check("b", p[2*patchSize*patchSize], wantB)
}

func TestPreprocessImageRGBAFastPathMatchesGeneric(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 37, 31))
	for y := 0; y < 31; y++ {
		for x := 0; x < 37; x++ {
			img.SetRGBA(x, y, color.RGBA{
				R: uint8((x*7 + y*3) & 255),
				G: uint8((x*5 + y*11) & 255),
				B: uint8((x*13 + y*17) & 255),
				A: 255,
			})
		}
	}
	fast, err := PreprocessImage(img)
	if err != nil {
		t.Fatal(err)
	}
	generic, err := PreprocessImage(wrappedRGBA{img: img})
	if err != nil {
		t.Fatal(err)
	}
	if fast.Grid != generic.Grid || fast.Width != generic.Width || fast.Height != generic.Height {
		t.Fatalf("shape mismatch fast=%+v %dx%d generic=%+v %dx%d", fast.Grid, fast.Width, fast.Height, generic.Grid, generic.Width, generic.Height)
	}
	if len(fast.Patches) != len(generic.Patches) {
		t.Fatalf("patch count got %d want %d", len(fast.Patches), len(generic.Patches))
	}
	for i := range fast.Patches {
		for j := range fast.Patches[i] {
			if math.Abs(float64(fast.Patches[i][j]-generic.Patches[i][j])) > 1e-5 {
				t.Fatalf("patch[%d][%d] got %f want %f", i, j, fast.Patches[i][j], generic.Patches[i][j])
			}
		}
	}
}

func TestPreprocessImageRGBAExactSizeFastPathMatchesGeneric(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 364, 364))
	for y := 0; y < 364; y++ {
		for x := 0; x < 364; x++ {
			img.SetRGBA(x, y, color.RGBA{
				R: uint8((x*7 + y*3) & 255),
				G: uint8((x*5 + y*11) & 255),
				B: uint8((x*13 + y*17) & 255),
				A: 255,
			})
		}
	}
	fast, err := PreprocessImage(img)
	if err != nil {
		t.Fatal(err)
	}
	generic, err := PreprocessImage(wrappedRGBA{img: img})
	if err != nil {
		t.Fatal(err)
	}
	if fast.Grid != generic.Grid || fast.Width != generic.Width || fast.Height != generic.Height {
		t.Fatalf("shape mismatch fast=%+v %dx%d generic=%+v %dx%d", fast.Grid, fast.Width, fast.Height, generic.Grid, generic.Width, generic.Height)
	}
	for i := range fast.Patches {
		for j := range fast.Patches[i] {
			if math.Abs(float64(fast.Patches[i][j]-generic.Patches[i][j])) > 1e-5 {
				t.Fatalf("patch[%d][%d] got %f want %f", i, j, fast.Patches[i][j], generic.Patches[i][j])
			}
		}
	}
}

func TestPreprocessImageNRGBAFastPathMatchesGeneric(t *testing.T) {
	img := image.NewNRGBA(image.Rect(0, 0, 37, 31))
	for y := 0; y < 31; y++ {
		for x := 0; x < 37; x++ {
			img.SetNRGBA(x, y, color.NRGBA{
				R: uint8((x*7 + y*3) & 255),
				G: uint8((x*5 + y*11) & 255),
				B: uint8((x*13 + y*17) & 255),
				A: 255,
			})
		}
	}
	fast, err := PreprocessImage(img)
	if err != nil {
		t.Fatal(err)
	}
	generic, err := PreprocessImage(wrappedNRGBA{img: img})
	if err != nil {
		t.Fatal(err)
	}
	if fast.Grid != generic.Grid || fast.Width != generic.Width || fast.Height != generic.Height {
		t.Fatalf("shape mismatch fast=%+v %dx%d generic=%+v %dx%d", fast.Grid, fast.Width, fast.Height, generic.Grid, generic.Width, generic.Height)
	}
	if len(fast.Patches) != len(generic.Patches) {
		t.Fatalf("patch count got %d want %d", len(fast.Patches), len(generic.Patches))
	}
	for i := range fast.Patches {
		for j := range fast.Patches[i] {
			if math.Abs(float64(fast.Patches[i][j]-generic.Patches[i][j])) > 1e-5 {
				t.Fatalf("patch[%d][%d] got %f want %f", i, j, fast.Patches[i][j], generic.Patches[i][j])
			}
		}
	}
}

func TestPreprocessImageNRGBAAlphaFastPathMatchesGeneric(t *testing.T) {
	img := image.NewNRGBA(image.Rect(0, 0, 37, 31))
	for y := 0; y < 31; y++ {
		for x := 0; x < 37; x++ {
			img.SetNRGBA(x, y, color.NRGBA{
				R: uint8((x*7 + y*3) & 255),
				G: uint8((x*5 + y*11) & 255),
				B: uint8((x*13 + y*17) & 255),
				A: uint8(64 + ((x*19 + y*23) & 127)),
			})
		}
	}
	fast, err := PreprocessImage(img)
	if err != nil {
		t.Fatal(err)
	}
	generic, err := PreprocessImage(wrappedNRGBA{img: img})
	if err != nil {
		t.Fatal(err)
	}
	if fast.Grid != generic.Grid || fast.Width != generic.Width || fast.Height != generic.Height {
		t.Fatalf("shape mismatch fast=%+v %dx%d generic=%+v %dx%d", fast.Grid, fast.Width, fast.Height, generic.Grid, generic.Width, generic.Height)
	}
	if len(fast.Patches) != len(generic.Patches) {
		t.Fatalf("patch count got %d want %d", len(fast.Patches), len(generic.Patches))
	}
	for i := range fast.Patches {
		for j := range fast.Patches[i] {
			if math.Abs(float64(fast.Patches[i][j]-generic.Patches[i][j])) > 1e-5 {
				t.Fatalf("patch[%d][%d] got %f want %f", i, j, fast.Patches[i][j], generic.Patches[i][j])
			}
		}
	}
}

func TestResizeBilinearNRGBAAlphaFastPathMatchesGeneric(t *testing.T) {
	img := image.NewNRGBA(image.Rect(0, 0, 37, 31))
	for y := 0; y < 31; y++ {
		for x := 0; x < 37; x++ {
			img.SetNRGBA(x, y, color.NRGBA{
				R: uint8((x*7 + y*3) & 255),
				G: uint8((x*5 + y*11) & 255),
				B: uint8((x*13 + y*17) & 255),
				A: uint8(64 + ((x*19 + y*23) & 127)),
			})
		}
	}
	fast := resizeBilinear(img, 56, 84)
	generic := resizeBilinear(wrappedNRGBA{img: img}, 56, 84)
	if !fast.Bounds().Eq(generic.Bounds()) {
		t.Fatalf("bounds got %v want %v", fast.Bounds(), generic.Bounds())
	}
	for y := 0; y < fast.Bounds().Dy(); y++ {
		for x := 0; x < fast.Bounds().Dx(); x++ {
			fo := y*fast.Stride + x*4
			goff := y*generic.Stride + x*4
			for c := 0; c < 4; c++ {
				if fast.Pix[fo+c] != generic.Pix[goff+c] {
					t.Fatalf("pixel(%d,%d)[%d] got %d want %d", x, y, c, fast.Pix[fo+c], generic.Pix[goff+c])
				}
			}
		}
	}
}

func TestResizeBilinearNRGBAAlphaNegativeBoundsMatchesGeneric(t *testing.T) {
	img := image.NewNRGBA(image.Rect(-3, -2, 34, 29))
	for y := img.Bounds().Min.Y; y < img.Bounds().Max.Y; y++ {
		for x := img.Bounds().Min.X; x < img.Bounds().Max.X; x++ {
			img.SetNRGBA(x, y, color.NRGBA{
				R: uint8((x*7 + y*3) & 255),
				G: uint8((x*5 + y*11) & 255),
				B: uint8((x*13 + y*17) & 255),
				A: uint8(64 + ((x*19 + y*23) & 127)),
			})
		}
	}
	fast := resizeBilinear(img, 56, 84)
	generic := resizeBilinear(wrappedNRGBA{img: img}, 56, 84)
	if !fast.Bounds().Eq(generic.Bounds()) {
		t.Fatalf("bounds got %v want %v", fast.Bounds(), generic.Bounds())
	}
	for y := 0; y < fast.Bounds().Dy(); y++ {
		for x := 0; x < fast.Bounds().Dx(); x++ {
			fo := y*fast.Stride + x*4
			goff := y*generic.Stride + x*4
			for c := 0; c < 4; c++ {
				if fast.Pix[fo+c] != generic.Pix[goff+c] {
					t.Fatalf("pixel(%d,%d)[%d] got %d want %d", x, y, c, fast.Pix[fo+c], generic.Pix[goff+c])
				}
			}
		}
	}
}

func TestLoadImageReaderRejectsHugeDimensionsBeforeDecode(t *testing.T) {
	raw := pngHeaderForTest(maxInputDimension+1, 64)
	if _, err := LoadImageReader(bytes.NewReader(raw)); err == nil || !strings.Contains(err.Error(), "dimensions too large") {
		t.Fatalf("err=%v", err)
	}
	raw = pngHeaderForTest(maxInputPixels/2, 3)
	if _, err := LoadImageReader(bytes.NewReader(raw)); err == nil || !strings.Contains(err.Error(), "exceeds") {
		t.Fatalf("err=%v", err)
	}
}

func TestLoadImageReaderRejectsHugeDimensionsFromNonSeeker(t *testing.T) {
	raw := pngHeaderForTest(maxInputDimension+1, 64)
	if _, err := LoadImageReader(bytes.NewBuffer(raw)); err == nil || !strings.Contains(err.Error(), "dimensions too large") {
		t.Fatalf("err=%v", err)
	}
}

func TestPreprocessImageRejectsHugeBounds(t *testing.T) {
	img := boundsOnlyImage{rect: image.Rect(0, 0, maxInputDimension+1, 64)}
	if _, err := PreprocessImage(img); err == nil || !strings.Contains(err.Error(), "dimensions too large") {
		t.Fatalf("err=%v", err)
	}
}

func pngHeaderForTest(width, height int) []byte {
	var out []byte
	out = append(out, 0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n')
	var ihdr [13]byte
	binary.BigEndian.PutUint32(ihdr[0:4], uint32(width))
	binary.BigEndian.PutUint32(ihdr[4:8], uint32(height))
	ihdr[8] = 8
	ihdr[9] = 2
	out = appendPNGChunkForTest(out, "IHDR", ihdr[:])
	return out
}

func appendPNGChunkForTest(out []byte, typ string, data []byte) []byte {
	out = binary.BigEndian.AppendUint32(out, uint32(len(data)))
	start := len(out)
	out = append(out, typ...)
	out = append(out, data...)
	crc := crc32.ChecksumIEEE(out[start:])
	out = binary.BigEndian.AppendUint32(out, crc)
	return out
}

func BenchmarkPreprocessImageRGBA(b *testing.B) {
	img := image.NewRGBA(image.Rect(0, 0, 224, 224))
	for y := 0; y < 224; y++ {
		for x := 0; x < 224; x++ {
			img.SetRGBA(x, y, color.RGBA{R: uint8(x), G: uint8(y), B: uint8(x + y), A: 255})
		}
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := PreprocessImage(img); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkPreprocessImageNRGBA(b *testing.B) {
	img := image.NewNRGBA(image.Rect(0, 0, 224, 224))
	for y := 0; y < 224; y++ {
		for x := 0; x < 224; x++ {
			img.SetNRGBA(x, y, color.NRGBA{R: uint8(x), G: uint8(y), B: uint8(x + y), A: 255})
		}
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := PreprocessImage(img); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkPreprocessImageNRGBAAlpha(b *testing.B) {
	img := image.NewNRGBA(image.Rect(0, 0, 224, 224))
	for y := 0; y < 224; y++ {
		for x := 0; x < 224; x++ {
			img.SetNRGBA(x, y, color.NRGBA{
				R: uint8(x),
				G: uint8(y),
				B: uint8(x + y),
				A: uint8(64 + ((x*19 + y*23) & 127)),
			})
		}
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := PreprocessImage(img); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkPreprocessImageNRGBAGeneric(b *testing.B) {
	img := image.NewNRGBA(image.Rect(0, 0, 224, 224))
	for y := 0; y < 224; y++ {
		for x := 0; x < 224; x++ {
			img.SetNRGBA(x, y, color.NRGBA{R: uint8(x), G: uint8(y), B: uint8(x + y), A: 255})
		}
	}
	wrapped := wrappedNRGBA{img: img}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := PreprocessImage(wrapped); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkPreprocessImageRGBASingleProc(b *testing.B) {
	prev := runtime.GOMAXPROCS(1)
	defer runtime.GOMAXPROCS(prev)
	BenchmarkPreprocessImageRGBA(b)
}

func BenchmarkPreprocessImageRGBAExactSize(b *testing.B) {
	img := image.NewRGBA(image.Rect(0, 0, 364, 364))
	for y := 0; y < 364; y++ {
		for x := 0; x < 364; x++ {
			img.SetRGBA(x, y, color.RGBA{R: uint8(x), G: uint8(y), B: uint8(x + y), A: 255})
		}
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := PreprocessImage(img); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkPreprocessImageNRGBAExactSize(b *testing.B) {
	img := image.NewNRGBA(image.Rect(0, 0, 364, 364))
	for y := 0; y < 364; y++ {
		for x := 0; x < 364; x++ {
			img.SetNRGBA(x, y, color.NRGBA{R: uint8(x), G: uint8(y), B: uint8(x + y), A: 255})
		}
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := PreprocessImage(img); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkPreprocessImageNRGBAAlphaExactSize(b *testing.B) {
	img := image.NewNRGBA(image.Rect(0, 0, 364, 364))
	for y := 0; y < 364; y++ {
		for x := 0; x < 364; x++ {
			img.SetNRGBA(x, y, color.NRGBA{
				R: uint8(x),
				G: uint8(y),
				B: uint8(x + y),
				A: uint8(64 + ((x*19 + y*23) & 127)),
			})
		}
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := PreprocessImage(img); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkPreprocessImageNRGBAExactSizeGeneric(b *testing.B) {
	img := image.NewNRGBA(image.Rect(0, 0, 364, 364))
	for y := 0; y < 364; y++ {
		for x := 0; x < 364; x++ {
			img.SetNRGBA(x, y, color.NRGBA{R: uint8(x), G: uint8(y), B: uint8(x + y), A: 255})
		}
	}
	wrapped := wrappedNRGBA{img: img}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := PreprocessImage(wrapped); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkPreprocessImageRGBAExactSizeSingleProc(b *testing.B) {
	prev := runtime.GOMAXPROCS(1)
	defer runtime.GOMAXPROCS(prev)
	BenchmarkPreprocessImageRGBAExactSize(b)
}

func BenchmarkExtractPatchesRGBAExactSize(b *testing.B) {
	img := image.NewRGBA(image.Rect(0, 0, 364, 364))
	for y := 0; y < 364; y++ {
		for x := 0; x < 364; x++ {
			img.SetRGBA(x, y, color.RGBA{R: uint8(x), G: uint8(y), B: uint8(x + y), A: 255})
		}
	}
	gridW := img.Bounds().Dx() / patchSize
	patchCount := gridW * (img.Bounds().Dy() / patchSize)
	patchDim := 3 * patchSize * patchSize
	patches := make([][]float32, patchCount)
	patchData := make([]float32, patchCount*patchDim)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		extractPatchesRGBA(patches, patchData, img, gridW, 0, patchCount, clipScale[0], clipScale[1], clipScale[2], clipBias[0], clipBias[1], clipBias[2])
	}
}

func BenchmarkExtractResizedPatchesRGBA(b *testing.B) {
	img := image.NewRGBA(image.Rect(0, 0, 224, 224))
	for y := 0; y < 224; y++ {
		for x := 0; x < 224; x++ {
			img.SetRGBA(x, y, color.RGBA{R: uint8(x), G: uint8(y), B: uint8(x + y), A: 255})
		}
	}
	nh, nw, err := smartResize(224, 224, patchSize*mergeSize, minPixels, maxPixels)
	if err != nil {
		b.Fatal(err)
	}
	gridW := nw / patchSize
	patchCount := (nh / patchSize) * gridW
	patchDim := 3 * patchSize * patchSize
	patches := make([][]float32, patchCount)
	patchData := make([]float32, patchCount*patchDim)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		extractResizedPatchesRGBA(patches, patchData, img, nw, nh, gridW, clipScale[0], clipScale[1], clipScale[2], clipBias[0], clipBias[1], clipBias[2])
	}
}

func BenchmarkExtractResizedPatchesNRGBA(b *testing.B) {
	img := image.NewNRGBA(image.Rect(0, 0, 224, 224))
	for y := 0; y < 224; y++ {
		for x := 0; x < 224; x++ {
			img.SetNRGBA(x, y, color.NRGBA{R: uint8(x), G: uint8(y), B: uint8(x + y), A: 255})
		}
	}
	nh, nw, err := smartResize(224, 224, patchSize*mergeSize, minPixels, maxPixels)
	if err != nil {
		b.Fatal(err)
	}
	gridW := nw / patchSize
	patchCount := (nh / patchSize) * gridW
	patchDim := 3 * patchSize * patchSize
	patches := make([][]float32, patchCount)
	patchData := make([]float32, patchCount*patchDim)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		extractResizedPatchesNRGBA(patches, patchData, img, nw, nh, gridW, clipScale[0], clipScale[1], clipScale[2], clipBias[0], clipBias[1], clipBias[2])
	}
}

func BenchmarkExtractResizedPatchesNRGBAAlpha(b *testing.B) {
	img := image.NewNRGBA(image.Rect(0, 0, 224, 224))
	for y := 0; y < 224; y++ {
		for x := 0; x < 224; x++ {
			img.SetNRGBA(x, y, color.NRGBA{
				R: uint8(x),
				G: uint8(y),
				B: uint8(x + y),
				A: uint8(64 + ((x*19 + y*23) & 127)),
			})
		}
	}
	nh, nw, err := smartResize(224, 224, patchSize*mergeSize, minPixels, maxPixels)
	if err != nil {
		b.Fatal(err)
	}
	gridW := nw / patchSize
	patchCount := (nh / patchSize) * gridW
	patchDim := 3 * patchSize * patchSize
	patches := make([][]float32, patchCount)
	patchData := make([]float32, patchCount*patchDim)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		extractResizedPatchesNRGBA(patches, patchData, img, nw, nh, gridW, clipScale[0], clipScale[1], clipScale[2], clipBias[0], clipBias[1], clipBias[2])
	}
}

func BenchmarkResizeBilinearRGBA(b *testing.B) {
	img := image.NewRGBA(image.Rect(0, 0, 224, 224))
	for y := 0; y < 224; y++ {
		for x := 0; x < 224; x++ {
			img.SetRGBA(x, y, color.RGBA{R: uint8(x), G: uint8(y), B: uint8(x + y), A: 255})
		}
	}
	nh, nw, err := smartResize(224, 224, patchSize*mergeSize, minPixels, maxPixels)
	if err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = resizeBilinear(img, nw, nh)
	}
}

func BenchmarkResizeBilinearNRGBA(b *testing.B) {
	img := image.NewNRGBA(image.Rect(0, 0, 224, 224))
	for y := 0; y < 224; y++ {
		for x := 0; x < 224; x++ {
			img.SetNRGBA(x, y, color.NRGBA{R: uint8(x), G: uint8(y), B: uint8(x + y), A: 255})
		}
	}
	nh, nw, err := smartResize(224, 224, patchSize*mergeSize, minPixels, maxPixels)
	if err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = resizeBilinear(img, nw, nh)
	}
}

func BenchmarkResizeBilinearNRGBAGeneric(b *testing.B) {
	img := image.NewNRGBA(image.Rect(0, 0, 224, 224))
	for y := 0; y < 224; y++ {
		for x := 0; x < 224; x++ {
			img.SetNRGBA(x, y, color.NRGBA{R: uint8(x), G: uint8(y), B: uint8(x + y), A: 255})
		}
	}
	nh, nw, err := smartResize(224, 224, patchSize*mergeSize, minPixels, maxPixels)
	if err != nil {
		b.Fatal(err)
	}
	wrapped := wrappedNRGBA{img: img}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = resizeBilinear(wrapped, nw, nh)
	}
}

func BenchmarkResizeBilinearRGBARows(b *testing.B) {
	src := image.NewRGBA(image.Rect(0, 0, 224, 224))
	for y := 0; y < 224; y++ {
		for x := 0; x < 224; x++ {
			src.SetRGBA(x, y, color.RGBA{R: uint8(x), G: uint8(y), B: uint8(x + y), A: 255})
		}
	}
	nh, nw, err := smartResize(224, 224, patchSize*mergeSize, minPixels, maxPixels)
	if err != nil {
		b.Fatal(err)
	}
	dst := image.NewRGBA(image.Rect(0, 0, nw, nh))
	sb := src.Bounds()
	idx, idxPtr := getBilinearIndexes(nw + nh)
	defer putBilinearIndexes(idx, idxPtr)
	xs := idx[:nw]
	xScale := float64(sb.Dx()-1) / float64(nw-1)
	for x := 0; x < nw; x++ {
		fx := float64(x) * xScale
		x0 := int(fx)
		x1 := min(x0+1, sb.Max.X-1)
		wx := float32(fx - float64(x0))
		xs[x] = bilinearIndex{I0: x0 * 4, I1: x1 * 4, W0: 1 - wx, W1: wx}
	}
	ys := idx[nw:]
	yScale := float64(sb.Dy()-1) / float64(nh-1)
	for y := 0; y < nh; y++ {
		fy := float64(y) * yScale
		y0 := int(fy)
		y1 := min(y0+1, sb.Max.Y-1)
		wy := float32(fy - float64(y0))
		ys[y] = bilinearIndex{I0: y0 * src.Stride, I1: y1 * src.Stride, W0: 1 - wy, W1: wy}
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		resizeBilinearRGBARows(dst.Pix, src.Pix, dst.Stride, nw, xs, ys, 0, nh)
	}
}

func BenchmarkResizeBilinearRGBASingleProc(b *testing.B) {
	prev := runtime.GOMAXPROCS(1)
	defer runtime.GOMAXPROCS(prev)
	BenchmarkResizeBilinearRGBA(b)
}
