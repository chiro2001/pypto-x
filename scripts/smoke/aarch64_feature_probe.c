/*
 * Minimal no-model AArch64 capability probe for PyPTO-X smoke tests.
 * It intentionally uses only Linux auxv/prctl interfaces and does not
 * execute a model or allocate a large buffer.
 */
#include <stdio.h>
#include <sys/auxv.h>
#include <sys/prctl.h>

#include <asm/hwcap.h>

#ifndef HWCAP_SVE
#define HWCAP_SVE (1UL << 22)
#endif

#ifndef HWCAP2_SVE2
#define HWCAP2_SVE2 (1UL << 1)
#endif

#ifndef PR_SVE_GET_VL
#define PR_SVE_GET_VL 51
#endif

#ifndef PR_SVE_VL_LEN_MASK
#define PR_SVE_VL_LEN_MASK 0xffff
#endif

int main(void)
{
    unsigned long hwcap = getauxval(AT_HWCAP);
    unsigned long hwcap2 = getauxval(AT_HWCAP2);
    int raw_vl = prctl(PR_SVE_GET_VL);
    int vl_bytes = raw_vl < 0 ? -1 : (raw_vl & PR_SVE_VL_LEN_MASK);

    printf("hwcap=0x%lx hwcap2=0x%lx sve=%d sve2=%d vl_bytes=%d\n",
           hwcap,
           hwcap2,
           (hwcap & HWCAP_SVE) != 0,
           (hwcap2 & HWCAP2_SVE2) != 0,
           vl_bytes);
    return 0;
}
