/*
Copyright (c) 2019 SChernykh

This file is part of RandomX OpenCL.

RandomX OpenCL is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

RandomX OpenCL is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with RandomX OpenCL. If not, see <http://www.gnu.org/licenses/>.
*/

.amdcl2
.gpu GFX900
.64bit
.arch_minor 0
.arch_stepping 0
.driver_version 223600
.kernel randomx_run
	.config
		.dims x
		.cws 64, 1, 1
		.sgprsnum 32
		.vgprsnum 40
		.localsize 1024
		.floatmode 0xc0
		.pgmrsrc1 0x00ac0089
		.pgmrsrc2 0x00000090
		.dx10clamp
		.ieeemode
		.useargs
		.priority 0
		.arg _.global_offset_0, "size_t", long
		.arg _.global_offset_1, "size_t", long
		.arg _.global_offset_2, "size_t", long
		.arg _.printf_buffer, "size_t", void*, global, , rdonly
		.arg _.vqueue_pointer, "size_t", long
		.arg _.aqlwrap_pointer, "size_t", long
		.arg dataset, "uchar*", uchar*, global, const, rdonly
		.arg scratchpad, "uchar*", uchar*, global, 
		.arg registers, "ulong*", ulong*, global, 
		.arg rounding_modes, "uint*", uint*, global, , rdonly
		.arg programs, "uint*", uint*, global, 
		.arg batch_size, "uint", uint
	.text
		s_icache_inv
		v_lshl_add_u32  v1, s8, 6, v0
		s_load_dwordx2  s[0:1], s[4:5], 0x0
		s_load_dwordx2  s[2:3], s[4:5], 0x40
		s_waitcnt       lgkmcnt(0)
		v_add_u32       v5, s0, v1
		v_lshrrev_b32   v6, 4, v5
		v_lshlrev_b32   v1, 5, v6
		v_and_b32       v18, 15, v5
		v_mov_b32       v2, 0
		v_lshlrev_b64   v[1:2], 3, v[1:2]
		v_lshlrev_b32   v8, 4, v18
		v_add_co_u32    v19, vcc, s2, v1
		v_mov_b32       v1, s3
		v_addc_co_u32   v20, vcc, v1, v2, vcc
		v_add_co_u32    v1, vcc, v19, v8
		v_addc_co_u32   v2, vcc, v20, 0, vcc
		global_load_dwordx4 v[1:4], v[1:2], off
		v_and_b32       v0, 0xf0, v0
		v_bfi_b32       v5, 15, v5, v0
		v_lshlrev_b32   v5, 4, v5
		s_waitcnt       vmcnt(0)
		ds_write2_b64   v5, v[1:2], v[3:4] offset1:1
		s_waitcnt       lgkmcnt(0)
		s_mov_b64       s[0:1], exec
		v_cmpx_le_u32   s[2:3], v18, 7
		v_lshlrev_b32   v10, 6, v6
		v_lshlrev_b32   v12, 4, v0

		# v39 = R
		v_mov_b32       v39, v12

		s_cbranch_execz program_end
		v_cmp_lt_u32    s[2:3], v18, 4
		ds_read_b32     v11, v12 offset:152
		ds_read2_b64    v[31:34], v12 offset0:18 offset1:16
		ds_read_b64     v[4:5], v12 offset:136
		s_movk_i32      s9, 0x0
		s_mov_b64       s[6:7], exec
		s_andn2_b64     exec, s[6:7], s[2:3]
		ds_read_b64     v[6:7], v12 offset:160
		s_andn2_b64     exec, s[6:7], exec
		v_mov_b32       v6, 0
		v_mov_b32       v7, 0
		s_mov_b64       exec, s[6:7]
		s_lshl_b64      s[6:7], s[8:9], 16
		v_add3_u32      v21, v12, v8, 64
		s_mov_b64       s[8:9], exec
		s_andn2_b64     exec, s[8:9], s[2:3]
		ds_read_b64     v[8:9], v12 offset:168
		s_andn2_b64     exec, s[8:9], exec
		v_mov_b32       v8, 0
		v_mov_b32       v9, 0
		s_mov_b64       exec, s[8:9]
		s_load_dwordx4  s[8:11], s[4:5], 0x30
		s_load_dwordx2  s[12:13], s[4:5], 0x50
		v_lshlrev_b32   v22, 3, v18
		v_add_u32       v23, v12, v22
		s_waitcnt       lgkmcnt(0)
		v_add_co_u32    v24, vcc, s10, v10
		v_mov_b32       v10, s11
		v_addc_co_u32   v25, vcc, v10, 0, vcc
		v_mov_b32       v13, 0xffffff
		v_add_co_u32    v26, vcc, s8, v11
		v_mov_b32       v10, s9
		v_addc_co_u32   v27, vcc, v10, 0, vcc
		ds_read_b64     v[35:36], v23
		s_load_dword    s4, s[4:5], 0x58
		s_add_u32       s6, s12, s6
		s_addc_u32      s7, s13, s7
		v_cndmask_b32   v28, v13, -1, s[2:3]
		v_lshl_add_u32  v29, v32, 3, v12
		v_lshl_add_u32  v30, v31, 3, v12
		v_lshl_add_u32  v31, v5, 3, v12
		v_lshl_add_u32  v32, v4, 3, v12
		v_mov_b32       v10, v33
		v_mov_b32       v11, v34
		s_movk_i32      s2, 0x7ff

main_loop:
		ds_read_b64     v[2:3], v32
		ds_read_b64     v[4:5], v31
		s_waitcnt       lgkmcnt(0)
		v_xor_b32       v3, v5, v3
		v_xor_b32       v2, v4, v2
		v_xor_b32       v3, v3, v10
		v_xor_b32       v2, v2, v11
		v_and_b32       v3, 0x1fffc0, v3
		v_and_b32       v2, 0x1fffc0, v2
		v_mad_u32_u24   v3, v3, s4, v22
		v_mad_u32_u24   v2, v2, s4, v22
		v_add_co_u32    v37, vcc, v24, v3
		v_addc_co_u32   v38, vcc, v25, 0, vcc
		v_add_co_u32    v10, vcc, v24, v2
		v_addc_co_u32   v11, vcc, v25, 0, vcc

		# load from spAddr1
		global_load_dwordx2 v[4:5], v[37:38], off

		# load from spAddr0
		global_load_dwordx2 v[12:13], v[10:11], off
		s_waitcnt       vmcnt(1)

		v_cvt_f64_i32   v[14:15], v4
		v_cvt_f64_i32   v[4:5], v5
		s_waitcnt       vmcnt(0)

		# R[sub] ^= *p0;
		v_xor_b32       v0, v35, v12
		v_xor_b32       v1, v36, v13

		v_add_co_u32    v12, vcc, v26, v33
		v_addc_co_u32   v13, vcc, v27, 0, vcc
		v_or_b32        v2, v14, v6
		v_and_or_b32    v3, v15, v28, v7
		v_or_b32        v4, v4, v8
		v_and_or_b32    v5, v5, v28, v9
		v_add_co_u32    v12, vcc, v12, v22
		v_addc_co_u32   v13, vcc, v13, 0, vcc
		ds_write2_b64   v21, v[2:3], v[4:5] offset1:1
		s_waitcnt       lgkmcnt(0)

		# Program 0
		s_mov_b64 exec, 1

		# load VM integer registers
		v_readlane_b32	s16, v0, 0
		v_readlane_b32	s17, v1, 0
		v_readlane_b32	s18, v0, 1
		v_readlane_b32	s19, v1, 1
		v_readlane_b32	s20, v0, 2
		v_readlane_b32	s21, v1, 2
		v_readlane_b32	s22, v0, 3
		v_readlane_b32	s23, v1, 3
		v_readlane_b32	s24, v0, 4
		v_readlane_b32	s25, v1, 4
		v_readlane_b32	s26, v0, 5
		v_readlane_b32	s27, v1, 5
		v_readlane_b32	s28, v0, 6
		v_readlane_b32	s29, v1, 6
		v_readlane_b32	s30, v0, 7
		v_readlane_b32	s31, v1, 7

		# call JIT code
		s_swappc_b64    s[12:13], s[6:7]

		# store VM integer registers
		v_writelane_b32 v0, s16, 0
		v_writelane_b32 v1, s17, 0
		v_writelane_b32 v0, s18, 1
		v_writelane_b32 v1, s19, 1
		v_writelane_b32 v0, s20, 2
		v_writelane_b32 v1, s21, 2
		v_writelane_b32 v0, s22, 3
		v_writelane_b32 v1, s23, 3
		v_writelane_b32 v0, s24, 4
		v_writelane_b32 v1, s25, 4
		v_writelane_b32 v0, s26, 5
		v_writelane_b32 v1, s27, 5
		v_writelane_b32 v0, s28, 6
		v_writelane_b32 v1, s29, 6
		v_writelane_b32 v0, s30, 7
		v_writelane_b32 v1, s31, 7

		# Program 1
		s_lshl_b64 exec, exec, 16

		# load VM integer registers
		v_readlane_b32	s16, v0, 16 + 0
		v_readlane_b32	s17, v1, 16 + 0
		v_readlane_b32	s18, v0, 16 + 1
		v_readlane_b32	s19, v1, 16 + 1
		v_readlane_b32	s20, v0, 16 + 2
		v_readlane_b32	s21, v1, 16 + 2
		v_readlane_b32	s22, v0, 16 + 3
		v_readlane_b32	s23, v1, 16 + 3
		v_readlane_b32	s24, v0, 16 + 4
		v_readlane_b32	s25, v1, 16 + 4
		v_readlane_b32	s26, v0, 16 + 5
		v_readlane_b32	s27, v1, 16 + 5
		v_readlane_b32	s28, v0, 16 + 6
		v_readlane_b32	s29, v1, 16 + 6
		v_readlane_b32	s30, v0, 16 + 7
		v_readlane_b32	s31, v1, 16 + 7

		# call JIT code
		s_swappc_b64    s[12:13], s[14:15]

		# store VM integer registers
		v_writelane_b32 v0, s16, 16 + 0
		v_writelane_b32 v1, s17, 16 + 0
		v_writelane_b32 v0, s18, 16 + 1
		v_writelane_b32 v1, s19, 16 + 1
		v_writelane_b32 v0, s20, 16 + 2
		v_writelane_b32 v1, s21, 16 + 2
		v_writelane_b32 v0, s22, 16 + 3
		v_writelane_b32 v1, s23, 16 + 3
		v_writelane_b32 v0, s24, 16 + 4
		v_writelane_b32 v1, s25, 16 + 4
		v_writelane_b32 v0, s26, 16 + 5
		v_writelane_b32 v1, s27, 16 + 5
		v_writelane_b32 v0, s28, 16 + 6
		v_writelane_b32 v1, s29, 16 + 6
		v_writelane_b32 v0, s30, 16 + 7
		v_writelane_b32 v1, s31, 16 + 7

		# Program 2
		s_lshl_b64 exec, exec, 16

		# load VM integer registers
		v_readlane_b32	s16, v0, 32 + 0
		v_readlane_b32	s17, v1, 32 + 0
		v_readlane_b32	s18, v0, 32 + 1
		v_readlane_b32	s19, v1, 32 + 1
		v_readlane_b32	s20, v0, 32 + 2
		v_readlane_b32	s21, v1, 32 + 2
		v_readlane_b32	s22, v0, 32 + 3
		v_readlane_b32	s23, v1, 32 + 3
		v_readlane_b32	s24, v0, 32 + 4
		v_readlane_b32	s25, v1, 32 + 4
		v_readlane_b32	s26, v0, 32 + 5
		v_readlane_b32	s27, v1, 32 + 5
		v_readlane_b32	s28, v0, 32 + 6
		v_readlane_b32	s29, v1, 32 + 6
		v_readlane_b32	s30, v0, 32 + 7
		v_readlane_b32	s31, v1, 32 + 7

		# call JIT code
		s_swappc_b64    s[12:13], s[14:15]

		# store VM integer registers
		v_writelane_b32 v0, s16, 32 + 0
		v_writelane_b32 v1, s17, 32 + 0
		v_writelane_b32 v0, s18, 32 + 1
		v_writelane_b32 v1, s19, 32 + 1
		v_writelane_b32 v0, s20, 32 + 2
		v_writelane_b32 v1, s21, 32 + 2
		v_writelane_b32 v0, s22, 32 + 3
		v_writelane_b32 v1, s23, 32 + 3
		v_writelane_b32 v0, s24, 32 + 4
		v_writelane_b32 v1, s25, 32 + 4
		v_writelane_b32 v0, s26, 32 + 5
		v_writelane_b32 v1, s27, 32 + 5
		v_writelane_b32 v0, s28, 32 + 6
		v_writelane_b32 v1, s29, 32 + 6
		v_writelane_b32 v0, s30, 32 + 7
		v_writelane_b32 v1, s31, 32 + 7

		# Program 3
		s_lshl_b64 exec, exec, 16

		# load VM integer registers
		v_readlane_b32	s16, v0, 48 + 0
		v_readlane_b32	s17, v1, 48 + 0
		v_readlane_b32	s18, v0, 48 + 1
		v_readlane_b32	s19, v1, 48 + 1
		v_readlane_b32	s20, v0, 48 + 2
		v_readlane_b32	s21, v1, 48 + 2
		v_readlane_b32	s22, v0, 48 + 3
		v_readlane_b32	s23, v1, 48 + 3
		v_readlane_b32	s24, v0, 48 + 4
		v_readlane_b32	s25, v1, 48 + 4
		v_readlane_b32	s26, v0, 48 + 5
		v_readlane_b32	s27, v1, 48 + 5
		v_readlane_b32	s28, v0, 48 + 6
		v_readlane_b32	s29, v1, 48 + 6
		v_readlane_b32	s30, v0, 48 + 7
		v_readlane_b32	s31, v1, 48 + 7

		# call JIT code
		s_swappc_b64    s[12:13], s[14:15]

		# store VM integer registers
		v_writelane_b32 v0, s16, 48 + 0
		v_writelane_b32 v1, s17, 48 + 0
		v_writelane_b32 v0, s18, 48 + 1
		v_writelane_b32 v1, s19, 48 + 1
		v_writelane_b32 v0, s20, 48 + 2
		v_writelane_b32 v1, s21, 48 + 2
		v_writelane_b32 v0, s22, 48 + 3
		v_writelane_b32 v1, s23, 48 + 3
		v_writelane_b32 v0, s24, 48 + 4
		v_writelane_b32 v1, s25, 48 + 4
		v_writelane_b32 v0, s26, 48 + 5
		v_writelane_b32 v1, s27, 48 + 5
		v_writelane_b32 v0, s28, 48 + 6
		v_writelane_b32 v1, s29, 48 + 6
		v_writelane_b32 v0, s30, 48 + 7
		v_writelane_b32 v1, s31, 48 + 7

		# Restore execution mask
		s_mov_b32       s14, 0xff00ff
		s_mov_b32       s15, s14
		s_mov_b64       exec, s[14:15]

		# Write out VM integer registers
		ds_write_b64    v23, v[0:1]

		global_load_dwordx2 v[4:5], v[12:13], off
		s_waitcnt       vmcnt(0) & lgkmcnt(0)
		v_xor_b32       v35, v4, v0
		v_xor_b32       v36, v5, v1
		ds_read_b64     v[0:1], v30
		ds_read_b64     v[4:5], v29
		ds_write_b64    v23, v[35:36]
		ds_read2_b64    v[14:17], v23 offset0:8 offset1:16
		s_waitcnt       lgkmcnt(3)
		v_xor_b32       v0, v0, v34
		s_waitcnt       lgkmcnt(0)
		v_xor_b32       v12, v16, v14
		v_xor_b32       v13, v17, v15
		v_xor_b32       v0, v0, v4
		global_store_dwordx2 v[37:38], v[35:36], off
		v_and_b32       v2, 0x7fffffc0, v0
		global_store_dwordx2 v[10:11], v[12:13], off
		s_cmp_eq_u32    s2, 0
		s_cbranch_scc1  main_loop_end
		s_sub_i32       s2, s2, 1
		v_mov_b32       v10, 0
		v_mov_b32       v34, v33
		v_mov_b32       v11, 0
		v_mov_b32       v33, v2
		s_branch        main_loop

main_loop_end:
		v_add_co_u32    v2, vcc, v19, v22
		v_addc_co_u32   v3, vcc, v20, 0, vcc
		global_store_dwordx2 v[2:3], v[35:36], off
		global_store_dwordx2 v[2:3], v[12:13], off inst_offset:64
		v_or_b32        v0, 16, v18
		v_lshlrev_b32   v0, 3, v0
		v_add_co_u32    v0, vcc, v19, v0
		v_addc_co_u32   v1, vcc, v20, 0, vcc
		global_store_dwordx2 v[0:1], v[16:17], off

program_end:
		s_endpgm
