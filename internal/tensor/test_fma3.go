package tensor

import "fmt"

func testFMA3(a []float32) float32

func init() {
    a := make([]float32, 24)
    for i := 0; i < 8; i++ { a[i] = 1 }
    for i := 8; i < 16; i++ { a[i] = 2 }
    for i := 16; i < 24; i++ { a[i] = 3 }
    r := testFMA3(a)
    // 231: dst = src1*src2 + dst => Y0 = Y1*Y2 + Y0 = 2*3+1 = 7 per element
    // Sum of 8 elements = 56
    fmt.Printf("FMA3 result: %f (expected 56)\n", r)
}
