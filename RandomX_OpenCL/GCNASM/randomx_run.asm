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
        .sgprsnum 16
        .vgprsnum 39
        .localsize 1024
        .floatmode 0xc0
        .pgmrsrc1 0x00ac0049
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
        s_cbranch_execz .L808_0
        v_cmp_lt_u32    s[2:3], v18, 4
        ds_read_b32     v11, v12 offset:152
        ds_read2_b64    v[31:34], v12 offset0:18 offset1:16
        ds_read_b64     v[4:5], v12 offset:136
        s_mov_b64       s[6:7], exec
        s_andn2_b64     exec, s[6:7], s[2:3]
        s_cbranch_execz .L204_0
        ds_read_b64     v[6:7], v12 offset:160
.L204_0:
        s_andn2_b64     exec, s[6:7], exec
        v_mov_b32       v6, 0
        v_mov_b32       v7, 0
        s_mov_b64       exec, s[6:7]
        v_add3_u32      v21, v12, v8, 64
        s_andn2_b64     exec, s[6:7], s[2:3]
        s_cbranch_execz .L244_0
        ds_read_b64     v[8:9], v12 offset:168
.L244_0:
        s_andn2_b64     exec, s[6:7], exec
        v_mov_b32       v8, 0
        v_mov_b32       v9, 0
        s_mov_b64       exec, s[6:7]
        s_load_dwordx4  s[8:11], s[4:5], 0x30
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
        s_load_dwordx2  s[6:7], s[4:5], 0x50
        s_load_dword    s4, s[4:5], 0x58
        v_cndmask_b32   v28, v13, -1, s[2:3]
        v_lshl_add_u32  v29, v32, 3, v12
        v_lshl_add_u32  v30, v31, 3, v12
        v_lshl_add_u32  v31, v5, 3, v12
        v_lshl_add_u32  v32, v4, 3, v12
        v_mov_b32       v10, v34
        v_mov_b32       v11, v33
        s_movk_i32      s2, 0x7ff
.L396_0:
        ds_read_b64     v[2:3], v32
        ds_read_b64     v[4:5], v31
        s_waitcnt       lgkmcnt(0)
        v_xor_b32       v3, v5, v3
        v_xor_b32       v2, v4, v2
        v_xor_b32       v3, v3, v11
        v_xor_b32       v2, v2, v10
        v_and_b32       v3, 0x1fffc0, v3
        v_and_b32       v2, 0x1fffc0, v2
        v_mad_u32_u24   v3, v3, s4, v22
        v_mad_u32_u24   v2, v2, s4, v22
        v_add_co_u32    v37, vcc, v24, v3
        v_addc_co_u32   v38, vcc, v25, 0, vcc
        v_add_co_u32    v10, vcc, v24, v2
        v_addc_co_u32   v11, vcc, v25, 0, vcc
        global_load_dwordx2 v[4:5], v[37:38], off
        global_load_dwordx2 v[12:13], v[10:11], off
        s_waitcnt       vmcnt(1)
        v_cvt_f64_i32   v[14:15], v4
        v_cvt_f64_i32   v[4:5], v5
        s_waitcnt       vmcnt(0)
        v_xor_b32       v0, v35, v12
        v_xor_b32       v1, v36, v13
        v_add_co_u32    v12, vcc, v26, v33
        v_addc_co_u32   v13, vcc, v27, 0, vcc
        v_or_b32        v2, v14, v6
        v_and_or_b32    v3, v15, v28, v7
        v_or_b32        v4, v4, v8
        v_and_or_b32    v5, v5, v28, v9
        v_mov_b32       v16, 0
        v_mov_b32       v17, 1
        v_add_co_u32    v12, vcc, v12, v22
        v_addc_co_u32   v13, vcc, v13, 0, vcc
        ds_write_b64    v23, v[0:1]
        ds_write2_b64   v21, v[2:3], v[4:5] offset1:1
        s_waitcnt       lgkmcnt(0)

        # call JIT code
        s_swappc_b64    s[12:13], s[6:7]

        global_load_dwordx2 v[0:1], v[12:13], off
        ds_read_b64     v[4:5], v23
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
        s_cbranch_scc1  .L752_0
        s_sub_i32       s2, s2, 1
        v_mov_b32       v34, v33
        v_mov_b32       v10, 0
        v_mov_b32       v11, 0
        v_mov_b32       v33, v2
        s_branch        .L396_0
.L752_0:
        v_add_co_u32    v2, vcc, v19, v22
        v_addc_co_u32   v3, vcc, v20, 0, vcc
        global_store_dwordx2 v[2:3], v[35:36], off
        global_store_dwordx2 v[2:3], v[12:13], off inst_offset:64
        v_or_b32        v0, 16, v18
        v_lshlrev_b32   v0, 3, v0
        v_add_co_u32    v0, vcc, v19, v0
        v_addc_co_u32   v1, vcc, v20, 0, vcc
        global_store_dwordx2 v[0:1], v[16:17], off
.L808_0:
        s_endpgm
