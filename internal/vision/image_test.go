package vision

import (
	"image"
	"image/color"
	"math"
	"testing"
)

type wrappedRGBA struct {
	img *image.RGBA
}

func (w wrappedRGBA) ColorModel() color.Model { return w.img.ColorModel() }
func (w wrappedRGBA) Bounds() image.Rectangle { return w.img.Bounds() }
func (w wrappedRGBA) At(x, y int) color.Color { return w.img.At(x, y) }

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
