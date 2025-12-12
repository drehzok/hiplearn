# Basic kernels in HIP

## `vecadd.cpp`

Vector addition kernel

dim3 constructor

HIP syntax


## `grayscale.cpp`

OpenCV read image

Grayscaling kernel

Inline/noinline analysis `grayscale_hipcc_tests.cpp`


## `blur.cpp`

Image blurring with 3 dimensional grid






## Notes on hipcc testing

`grayscale_hipcc_tests.cpp` is meant to test out how no decoration, `__noinline__`, `__forceinline__` affects the resulting program behaviour.

Results are the following:
1. no decoration (only `__device__`) results in operations being inlined in the kernel
2. `__noinline__` the kernel calls device function, i.e. overhead
3. `__forceinline__` expectedly, behaves as 1


### Assembly code level proof:
noinline:
```
	s_load_b128 s[0:3], s[0:1], 0x0
	v_mad_u64_u32 v[3:4], null, v1, s4, v[0:1]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v0, v3, 1, v3
	v_ashrrev_i32_e32 v1, 31, v0
	s_waitcnt lgkmcnt(0)
	v_add_co_u32 v0, vcc_lo, s2, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e32 v1, vcc_lo, s3, v1, vcc_lo
	s_getpc_b64 s[2:3]
	s_add_u32 s2, s2, _Z13makegray_noinhhh@rel32@lo+4
	s_addc_u32 s3, s3, _Z13makegray_noinhhh@rel32@hi+12
	s_clause 0x1
	global_load_u16 v4, v[0:1], off
	global_load_u8 v2, v[0:1], off offset:2
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v0, 0xff, v4
	v_lshrrev_b32_e32 v1, 8, v4
	s_swappc_b64 s[30:31], s[2:3]
	v_ashrrev_i32_e32 v2, 31, v3
	v_add_co_u32 v1, vcc_lo, s0, v3
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e32 v2, vcc_lo, s1, v2, vcc_lo
	global_store_b8 v[1:2], v0, off
```
pay attention to `s_getpc_b64`, `s_add_u32 s2, s2, _Z13makegray_noinhhh@rel32@lo+4`, and `s_swappc_b64`, which indicates that indeed a separate kernel for `makegray_noin` was created and being called.

inline,default:
```
	s_load_b128 s[0:3], s[0:1], 0x0
	v_mad_u64_u32 v[2:3], null, v1, s5, v[0:1]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v0, v2, 1, v2
	v_ashrrev_i32_e32 v1, 31, v0
	s_waitcnt lgkmcnt(0)
	v_add_co_u32 v0, vcc_lo, s2, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e32 v1, vcc_lo, s3, v1, vcc_lo
	s_clause 0x1
	global_load_u16 v3, v[0:1], off
	global_load_u8 v0, v[0:1], off offset:2
	s_waitcnt vmcnt(1)
	v_cvt_f32_ubyte1_e32 v1, v3
	v_cvt_f32_ubyte0_e32 v3, v3
	s_waitcnt vmcnt(0)
	v_cvt_f32_ubyte0_e32 v0, v0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v1, 0x3f35c28f, v1
	v_fmamk_f32 v1, v3, 0x3e570a3d, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fmamk_f32 v0, v0, 0x3d8f5c29, v1
	v_ashrrev_i32_e32 v1, 31, v2
	v_cvt_i32_f32_e32 v3, v0
	v_add_co_u32 v0, vcc_lo, s0, v2
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_ci_u32_e32 v1, vcc_lo, s1, v1, vcc_lo
	global_store_b8 v[0:1], v3, off
```
the function calls existed in noinline doesn't exist and it directly computes, i.e. inlined.
