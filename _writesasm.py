content = """//go:build amd64

#include "textflag.h"

TEXT \u00b7dotQ8VNNICore(SB), NOSPLIT, $0-32
\tMOVQ a+0(FP), SI
\tMOVQ xq+8(FP), DI
\tMOVQ n+16(FP), CX
\tVPXORD Y0, Y0, Y0
\tVPXORD Y1, Y1, Y1
\tVPXORD Y2, Y2, Y2
\tVPXORD Y3, Y3, Y3
\tVPXORD Y4, Y4, Y4
\tVPXORD Y5, Y5, Y5
\tVPXORD Y6, Y6, Y6
\tVPXORD Y7, Y7, Y7
\tCMPQ CX, $8
\tJB vnniCoreTail
vnniCoreLoop:
\tCMPQ CX, $64
\tJB vnniCoreLoop32
\tVMOVDQU (DI), Y8
\tVPDPBUSD (SI), Y8, Y0
\tVMOVDQU 32(DI), Y9
\tVPDPBUSD 32(SI), Y9, Y1
\tVMOVDQU 64(DI), Y10
\tVPDPBUSD 64(SI), Y10, Y2
\tVMOVDQU 96(DI), Y11
\tVPDPBUSD 96(SI), Y11, Y3
\tVMOVDQU 128(DI), Y12
\tVPDPBUSD 128(SI), Y12, Y4
\tVMOVDQU 160(DI), Y13
\tVPDPBUSD 160(SI), Y13, Y5
\tVMOVDQU 192(DI), Y14
\tVPDPBUSD 192(SI), Y14, Y6
\tVMOVDQU 224(DI), Y15
\tVPDPBUSD 224(SI), Y15, Y7
\tADDQ $256, SI
\tADDQ $256, DI
\tSUBQ $64, CX
\tJMP vnniCoreLoop
vnniCoreLoop32:
\tCMPQ CX, $32
\tJB vnniCoreLoop16
\tVMOVDQU (DI), Y8
\tVPDPBUSD (SI), Y8, Y0
\tVMOVDQU 32(DI), Y9
\tVPDPBUSD 32(SI), Y9, Y1
\tVMOVDQU 64(DI), Y10
\tVPDPBUSD 64(SI), Y10, Y2
\tVMOVDQU 96(DI), Y11
\tVPDPBUSD 96(SI), Y11, Y3
\tADDQ $128, SI
\tADDQ $128, DI
\tSUBQ $32, CX
\tJMP vnniCoreLoop
vnniCoreLoop16:
\tCMPQ CX, $16
\tJB vnniCoreLoop8
\tVMOVDQU (DI), Y8
\tVPDPBUSD (SI), Y8, Y0
\tVMOVDQU 32(DI), Y9
\tVPDPBUSD 32(SI), Y9, Y1
\tADDQ $64, SI
\tADDQ $64, DI
\tSUBQ $16, CX
\tJMP vnniCoreLoop
vnniCoreLoop8:
\tCMPQ CX, $8
\tJB vnniCoreReduce
\tVMOVDQU (DI), Y8
\tVPDPBUSD (SI), Y8, Y0
\tADDQ $32, SI
\tADDQ $32, DI
\tSUBQ $8, CX
\tJMP vnniCoreLoop8
vnniCoreReduce:
\tVPADDD Y1, Y0, Y0
\tVPADDD Y3, Y2, Y2
\tVPADDD Y5, Y4, Y4
\tVPADDD Y7, Y6, Y6
\tVPADDD Y2, Y0, Y0
\tVPADDD Y6, Y4, Y4
\tVPADDD Y4, Y0, Y0
\tVEXTRACTI128 $1, Y0, X2
\tVPADDD X2, X0, X0
\tVPSHUFD $0x4E, X0, X1
\tVPADDD X1, X0, X0
\tVPSHUFD $0xB1, X0, X1
\tVPADDD X1, X0, X0
\tVMOVD X0, AX
\tVZEROUPPER
\tMOVL AX, ret+24(FP)
\tRET
vnniCoreTail:
\tXORL AX, AX
vnniCoreTailLoop:
\tCMPQ CX, $0
\tJE vnniCoreTailDone
\tMOVBQSX (SI), R8
\tMOVBQZX (DI), R9
\tIMULL R8, R9
\tADDL R9, AX
\tINCQ SI
\tINCQ DI
\tDECQ CX
\tJMP vnniCoreTailLoop
vnniCoreTailDone:
\tMOVL AX, ret+24(FP)
\tRET

TEXT \u00b7rowSumQ8Asm(SB), NOSPLIT, $0-24
\tMOVQ a+0(FP), SI
\tMOVQ n+8(FP), CX
\tXORL AX, AX
rowSumLoop:
\tCMPQ CX, $0
\tJE rowSumDone
\tMOVBQSX (SI), R8
\tADDL R8, AX
\tINCQ SI
\tDECQ CX
\tJMP rowSumLoop
rowSumDone:
\tMOVL AX, ret+16(FP)
\tRET
"""
with open("internal/tensor/dot_vnni_amd64.s", "w", encoding="utf-8") as f:
    f.write(content)
print("asm file written")
