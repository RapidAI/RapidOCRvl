with open('internal/tensor/dot_amd64.go', 'r', encoding='utf-8') as f:
    content = f.read()

if 'dotQ8VNNICore' not in content:
    marker = '// VNNI-accelerated Q8 dot product using VPDPBUSD.'
    helpers = '// Assembly helper declarations for VNNI kernels.\n'
    helpers += 'func dotQ8VNNICore(a *int8, xq *uint8, n int) int32\n'
    helpers += 'func dotQ8PairVNNICore(a, b *int8, xq *uint8, n int) (int32, int32)\n'
    helpers += 'func dotQ8TripletVNNICore(a, b, c *int8, xq *uint8, n int) (int32, int32, int32)\n'
    helpers += 'func rowSumQ8Asm(a *int8, n int) int32\n\n'
    content = content.replace(marker, helpers + marker)
    with open('internal/tensor/dot_amd64.go', 'w', encoding='utf-8') as f:
        f.write(content)
    print('added')
else:
    print('already present')
