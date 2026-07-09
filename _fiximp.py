with open("internal/tensor/vnni_debug_test.go", "r", encoding="utf-8") as f:
    c = f.read()
c = c.replace('import (\n\t"math"\n\t"testing"\n)', 'import "testing"')
with open("internal/tensor/vnni_debug_test.go", "w", encoding="utf-8") as f:
    f.write(c)
print("done")
