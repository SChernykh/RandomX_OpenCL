/*
Copyright (c) 2019-2020 SChernykh

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

.rocm
.gpu GFX1010
.arch_minor 1
.arch_stepping 0
.eflags 53
.llvm10binfmt
.metadatav3
.md_version 1, 0
.globaldata
    .fill 64, 1, 0
.kernel randomx_run
    .config
        .dims x
        .sgprsnum 96
        .vgprsnum 128
        .shared_vgprs 0
        .dx10clamp
        .ieeemode
        .floatmode 0xf0
        .priority 0
        .exceptions 0x60
        .userdatanum 6
        .pgmrsrc1 0x60af0105
        .pgmrsrc2 0x0000008c
        .pgmrsrc3 0x00000000
        .group_segment_fixed_size 256
        .private_segment_fixed_size 0
        .kernel_code_entry_offset 0x10c0
        .use_private_segment_buffer
        .use_kernarg_segment_ptr
        .use_wave32
    .config
        .md_symname "randomx_run.kd"
        .md_language "OpenCL C", 1, 2
        .reqd_work_group_size 32, 1, 1
        .md_kernarg_segment_size 104
        .md_kernarg_segment_align 8
        .md_group_segment_fixed_size 256
        .md_private_segment_fixed_size 0
        .md_wavefront_size 32
        .md_sgprsnum 96
        .md_vgprsnum 128
        .spilledsgprs 0
        .spilledvgprs 0
        .max_flat_work_group_size 32
        .arg dataset, "uchar*", 8, 0, globalbuf, u8, global, default const
        .arg scratchpad, "uchar*", 8, 8, globalbuf, u8, global, default
        .arg registers, "ulong*", 8, 16, globalbuf, u64, global, default
        .arg rounding_modes, "uint*", 8, 24, globalbuf, u32, global, default
        .arg programs, "uint*", 8, 32, globalbuf, u32, global, default
        .arg batch_size, "uint", 4, 40, value, u32
        .arg rx_parameters, "uint", 4, 44, value, u32
        .arg , "", 8, 48, gox, i64
        .arg , "", 8, 56, goy, i64
        .arg , "", 8, 64, goz, i64
        .arg , "", 8, 72, none, i8
        .arg , "", 8, 80, none, i8
        .arg , "", 8, 88, none, i8
        .arg , "", 8, 96, multigridsyncarg, i8
.text
randomx_run:
    s_mov_b32       m0, 0x10000
    s_dcache_wb
    s_waitcnt       vmcnt(0) & lgkmcnt(0)
    s_icache_inv
    s_branch begin

begin:
    s_load_dwordx4  s[0:3], s[4:5], 0x10
    s_mov_b32       s9, 0
    s_lshl_b32      s8, s6, 5
    v_lshlrev_b32   v39, 3, v0
    s_mov_b32       s12, s7
    v_cmp_gt_u32    vcc_lo, 8, v0
    s_waitcnt       lgkmcnt(0)
    s_lshl_b64      s[2:3], s[8:9], 3
    s_mov_b32       s32, s12
    s_add_u32       s0, s0, s2
    s_addc_u32      s1, s1, s3
    v_add_co_u32    v1, s0, s0, v39
    v_add_co_ci_u32 v2, s0, s1, 0, s0
    global_load_dwordx2 v[4:5], v[1:2], off
    s_waitcnt       vmcnt(0)
    ds_write_b64    v39, v[4:5]
    s_waitcnt       vmcnt(0) & lgkmcnt(0)
    s_waitcnt_vscnt null, 0x0
    s_waitcnt       vmcnt(0) & lgkmcnt(0)
    s_waitcnt_vscnt null, 0x0
    buffer_gl0_inv
    s_and_saveexec_b32 s0, vcc_lo
    s_cbranch_execz program_end
    s_load_dwordx4  s[8:11], s[4:5], 0x0
    s_load_dwordx2  s[0:1], s[4:5], 0x20

    # rx_parameters
    s_load_dword    s20, s[4:5], 0x2c

    v_mov_b32       v5, 0
    v_mov_b32       v10, 0
    s_waitcnt_vscnt null, 0x0
    ds_read_b64     v[8:9], v39
    v_cmp_gt_u32    vcc_lo, 4, v0
    v_lshlrev_b32   v0, 3, v0
    ds_read2_b64    v[25:28], v5 offset0:16 offset1:17
    ds_read_b32     v11, v5 offset:152
    ds_read_b64     v[17:18], v5 offset:168
    ds_read2_b64    v[20:23], v5 offset0:18 offset1:20
    v_cndmask_b32   v4, 0xffffff, -1, vcc_lo
    v_add_nc_u32    v5, v39, v0
    s_waitcnt       lgkmcnt(0)
    v_mov_b32       v13, s11
    v_mov_b32       v7, s1
    v_mov_b32       v6, s0

    # Scratchpad L1 size
    s_bfe_u32       s21, s20, 0x050000
    s_lshl_b32      s21, 1, s21

    # Scratchpad L2 size
    s_bfe_u32       s22, s20, 0x050005
    s_lshl_b32      s22, 1, s22

    # Scratchpad L3 size
    s_bfe_u32       s0, s20, 0x05000A
    s_lshl_b32      s23, 1, s0

    # program iterations
    s_bfe_u32       s24, s20, 0x04000F
    s_lshl_b32      s24, 1, s24

    v_mov_b32       v12, s10
    v_mad_u64_u32   v[6:7], s2, 10048, s6, v[6:7]

    v_readlane_b32  s4, v6, 0
    v_readlane_b32  s5, v7, 0

    s_lshl_b32      s2, 1, s0
    v_add_co_u32    v15, s0, s8, v11
    v_cndmask_b32   v16, v18, 0, vcc_lo
    v_cndmask_b32   v14, v23, 0, vcc_lo
    v_cndmask_b32   v38, v22, 0, vcc_lo
    s_add_i32       s3, s2, 64
    v_add_co_ci_u32 v29, s0, s9, v10, s0
    s_sub_i32       s0, s2, 64
    v_cndmask_b32   v17, v17, 0, vcc_lo
    v_add_co_u32    v22, vcc_lo, v15, v0
    v_mad_u64_u32   v[12:13], s2, s3, s6, v[12:13]
    v_mov_b32       v10, v26
    v_mov_b32       v11, v25
    v_lshlrev_b32   v18, 3, v27
    v_lshlrev_b32   v19, 3, v28
    v_lshlrev_b32   v20, 3, v20
    v_lshlrev_b32   v21, 3, v21
    v_add_co_ci_u32 v23, vcc_lo, 0, v29, vcc_lo

    # loop counter
    s_sub_u32       s2, s24, 1

    # ScratchpadL3Mask64
    s_sub_u32       s86, s23, 64

main_loop:
    s_waitcnt_vscnt null, 0x0
    ds_read_b64     v[27:28], v19
    ds_read_b64     v[29:30], v18
    s_waitcnt       lgkmcnt(0)
    v_xor_b32       v28, v28, v30
    v_xor_b32       v25, v28, v25
    v_and_b32       v25, s0, v25
    v_add_nc_u32    v25, v25, v0
    v_add_co_u32    v46, vcc_lo, v12, v25
    v_xor_b32       v25, v27, v29
    v_mov_b32       v29, v11
    v_add_co_ci_u32 v47, vcc_lo, 0, v13, vcc_lo
    v_xor_b32       v25, v25, v26
    global_load_dwordx2 v[27:28], v[46:47], off
    v_and_b32       v25, s0, v25
    v_add_nc_u32    v25, v25, v0
    v_add_co_u32    v33, vcc_lo, v12, v25
    v_add_co_ci_u32 v34, vcc_lo, 0, v13, vcc_lo
    v_add_co_u32    v29, vcc_lo, v22, v29
    global_load_dwordx2 v[25:26], v[33:34], off
    v_add_co_ci_u32 v30, vcc_lo, 0, v23, vcc_lo
    v_mov_b32       v35, v11
    s_and_b32       vcc_lo, exec_lo, vcc_lo
    s_waitcnt       vmcnt(1)
    v_cvt_f64_i32   v[42:43], v28
    v_cvt_f64_i32   v[40:41], v27
    v_or_b32        v42, v42, v17
    s_waitcnt       vmcnt(0)
    v_xor_b32       v9, v26, v9
    v_xor_b32       v8, v25, v8
    v_and_b32       v26, v4, v43
    ds_write_b64    v39, v[8:9]
    v_and_b32       v9, v4, v41
    v_or_b32        v43, v26, v16
    v_or_b32        v8, v40, v38
    v_mov_b32       v26, 0
    v_or_b32        v9, v9, v14
    v_mov_b32       v25, v26
    ds_write2_b64   v5, v[8:9], v[42:43] offset0:8 offset1:9
    s_waitcnt       vmcnt(0) & lgkmcnt(0)
    s_waitcnt_vscnt null, 0x0
    s_waitcnt       vmcnt(0) & lgkmcnt(0)
    s_waitcnt_vscnt null, 0x0

    # call JIT code
    s_swappc_b64    s[12:13], s[4:5]

    global_load_dwordx2 v[8:9], v[29:30], off
    ds_read_b32     v11, v21
    ds_read_b32     v36, v20
    ds_read2_b64    v[27:30], v39 offset1:8
    s_waitcnt       lgkmcnt(1)
    v_xor_b32       v11, v11, v36
    v_xor_b32       v10, v10, v11
    v_and_b32       v36, 0x7fffffc0, v10
    v_mov_b32       v10, v35
    v_mov_b32       v11, v36
    s_waitcnt       vmcnt(0) & lgkmcnt(0)
    v_xor_b32       v9, v28, v9
    v_xor_b32       v8, v27, v8
    global_store_dwordx2 v[46:47], v[8:9], off
    ds_read_b64     v[27:28], v39 offset:128
    ds_write_b64    v39, v[8:9]
    s_waitcnt       lgkmcnt(1)
    v_xor_b32       v32, v28, v30
    v_xor_b32       v31, v27, v29
    global_store_dwordx2 v[33:34], v[31:32], off
    s_cmp_eq_u32    s2, 0
    s_cbranch_scc1  main_loop_end
    s_sub_i32       s2, s2, 1
    s_branch       main_loop
main_loop_end:

    global_store_dwordx2 v[1:2], v[8:9], off
    global_store_dwordx2 v[1:2], v[31:32], off inst_offset:64
    global_store_dwordx2 v[1:2], v[27:28], off inst_offset:128

program_end:
    s_endpgm
