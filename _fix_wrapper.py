with open('internal/tensor/dot_amd64.go', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace dotQ8VNNI to use asm helper
old_dot = '''func dotQ8VNNI(a []int8, xq []uint8, scaleX, scaleW float32, rowSumW int32) float32 {
	var dot int32
	n := len(a)
	if len(xq) < n {
		n = len(xq)
	}
	for i := 0; i < n; i++ {
		dot += int32(a[i]) * (int32(xq[i]) - 128)
	}
	return float32(dot-128*rowSumW) * scaleX * scaleW
}'''

new_dot = '''func dotQ8VNNI(a []int8, xq []uint8, scaleX, scaleW float32, rowSumW int32) float32 {
	n := len(a)
	if len(xq) < n {
		n = len(xq)
	}
	if n == 0 {
		return float32(-128*rowSumW) * scaleX * scaleW
	}
	var dot int32
	if useVNNI {
		dot = dotQ8VNNICore(&a[0], &xq[0], n)
	} else {
		for i := 0; i < n; i++ {
			dot += int32(a[i]) * (int32(xq[i]) - 128)
		}
	}
	return float32(dot-128*rowSumW) * scaleX * scaleW
}'''

if old_dot in content:
    content = content.replace(old_dot, new_dot)
    print('fixed dotQ8VNNI')
else:
    print('dotQ8VNNI NOT FOUND')

# Replace dotQ8PairVNNI
old_pair = '''func dotQ8PairVNNI(a, b []int8, xq []uint8, scaleX float32, rowSumWA, rowSumWB int32, scaleWA, scaleWB float32) (float32, float32) {
	var dotA, dotB int32
	n := len(a)
	if len(xq) < n {
		n = len(xq)
	}
	for i := 0; i < n; i++ {
		dotA += int32(a[i]) * (int32(xq[i]) - 128)
		dotB += int32(b[i]) * (int32(xq[i]) - 128)
	}
	return float32(dotA-128*rowSumWA) * scaleX * scaleWA, float32(dotB-128*rowSumWB) * scaleX * scaleWB
}'''

new_pair = '''func dotQ8PairVNNI(a, b []int8, xq []uint8, scaleX float32, rowSumWA, rowSumWB int32, scaleWA, scaleWB float32) (float32, float32) {
	n := len(a)
	if len(xq) < n {
		n = len(xq)
	}
	if n == 0 {
		return float32(-128*rowSumWA) * scaleX * scaleWA, float32(-128*rowSumWB) * scaleX * scaleWB
	}
	var dotA, dotB int32
	if useVNNI {
		dotA, dotB = dotQ8PairVNNICore(&a[0], &b[0], &xq[0], n)
	} else {
		for i := 0; i < n; i++ {
			dotA += int32(a[i]) * (int32(xq[i]) - 128)
			dotB += int32(b[i]) * (int32(xq[i]) - 128)
		}
	}
	return float32(dotA-128*rowSumWA) * scaleX * scaleWA, float32(dotB-128*rowSumWB) * scaleX * scaleWB
}'''

if old_pair in content:
    content = content.replace(old_pair, new_pair)
    print('fixed dotQ8PairVNNI')
else:
    print('dotQ8PairVNNI NOT FOUND')

# Replace dotQ8TripletVNNI
old_trip = '''func dotQ8TripletVNNI(a, b, c []int8, xq []uint8, scaleX float32, rowSumWA, rowSumWB, rowSumWC int32, scaleWA, scaleWB, scaleWC float32) (float32, float32, float32) {
	var dotA, dotB, dotC int32
	n := len(a)
	if len(xq) < n {
		n = len(xq)
	}
	for i := 0; i < n; i++ {
		dotA += int32(a[i]) * (int32(xq[i]) - 128)
		dotB += int32(b[i]) * (int32(xq[i]) - 128)
		dotC += int32(c[i]) * (int32(xq[i]) - 128)
	}
	return float32(dotA-128*rowSumWA) * scaleX * scaleWA, float32(dotB-128*rowSumWB) * scaleX * scaleWB, float32(dotC-128*rowSumWC) * scaleX * scaleWC
}'''

new_trip = '''func dotQ8TripletVNNI(a, b, c []int8, xq []uint8, scaleX float32, rowSumWA, rowSumWB, rowSumWC int32, scaleWA, scaleWB, scaleWC float32) (float32, float32, float32) {
	n := len(a)
	if len(xq) < n {
		n = len(xq)
	}
	if n == 0 {
		return float32(-128*rowSumWA) * scaleX * scaleWA,
			float32(-128*rowSumWB) * scaleX * scaleWB,
			float32(-128*rowSumWC) * scaleX * scaleWC
	}
	var dotA, dotB, dotC int32
	if useVNNI {
		dotA, dotB, dotC = dotQ8TripletVNNICore(&a[0], &b[0], &c[0], &xq[0], n)
	} else {
		for i := 0; i < n; i++ {
			dotA += int32(a[i]) * (int32(xq[i]) - 128)
			dotB += int32(b[i]) * (int32(xq[i]) - 128)
			dotC += int32(c[i]) * (int32(xq[i]) - 128)
		}
	}
	return float32(dotA-128*rowSumWA) * scaleX * scaleWA,
		float32(dotB-128*rowSumWB) * scaleX * scaleWB,
		float32(dotC-128*rowSumWC) * scaleX * scaleWC
}'''

if old_trip in content:
    content = content.replace(old_trip, new_trip)
    print('fixed dotQ8TripletVNNI')
else:
    print('dotQ8TripletVNNI NOT FOUND')

# Replace rowSumQ8AVX2
old_rowsum = '''func rowSumQ8AVX2(a []int8) int32 {
	var s int32
	for _, v := range a {
		s += int32(v)
	}
	return s
}'''

new_rowsum = '''func rowSumQ8AVX2(a []int8) int32 {
	if len(a) == 0 {
		return 0
	}
	if useVNNI {
		return rowSumQ8Asm(&a[0], len(a))
	}
	var s int32
	for _, v := range a {
		s += int32(v)
	}
	return s
}'''

if old_rowsum in content:
    content = content.replace(old_rowsum, new_rowsum)
    print('fixed rowSumQ8AVX2')
else:
    print('rowSumQ8AVX2 NOT FOUND')

with open('internal/tensor/dot_amd64.go', 'w', encoding='utf-8') as f:
    f.write(content)
