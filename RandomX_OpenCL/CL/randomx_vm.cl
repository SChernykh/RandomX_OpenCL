#pragma OPENCL EXTENSION cl_khr_fp64 : enable

//Dataset base size in bytes. Must be a power of 2.
#define RANDOMX_DATASET_BASE_SIZE  2147483648

//Dataset extra size. Must be divisible by 64.
#define RANDOMX_DATASET_EXTRA_SIZE 33554368

//Scratchpad L3 size in bytes. Must be a power of 2.
#define RANDOMX_SCRATCHPAD_L3      2097152

//Scratchpad L2 size in bytes. Must be a power of two and less than or equal to RANDOMX_SCRATCHPAD_L3.
#define RANDOMX_SCRATCHPAD_L2      262144

//Scratchpad L1 size in bytes. Must be a power of two (minimum 64) and less than or equal to RANDOMX_SCRATCHPAD_L2.
#define RANDOMX_SCRATCHPAD_L1      16384

//Jump condition mask size in bits.
#define RANDOMX_JUMP_BITS          8

//Jump condition mask offset in bits. The sum of RANDOMX_JUMP_BITS and RANDOMX_JUMP_OFFSET must not exceed 16.
#define RANDOMX_JUMP_OFFSET        8

//Integer instructions
#define RANDOMX_FREQ_IADD_RS       25
#define RANDOMX_FREQ_IADD_M         7
#define RANDOMX_FREQ_ISUB_R        16
#define RANDOMX_FREQ_ISUB_M         7
#define RANDOMX_FREQ_IMUL_R        16
#define RANDOMX_FREQ_IMUL_M         4
#define RANDOMX_FREQ_IMULH_R        4
#define RANDOMX_FREQ_IMULH_M        1
#define RANDOMX_FREQ_ISMULH_R       4
#define RANDOMX_FREQ_ISMULH_M       1
#define RANDOMX_FREQ_IMUL_RCP       8
#define RANDOMX_FREQ_INEG_R         2
#define RANDOMX_FREQ_IXOR_R        15
#define RANDOMX_FREQ_IXOR_M         5
#define RANDOMX_FREQ_IROR_R         8
#define RANDOMX_FREQ_IROL_R         2
#define RANDOMX_FREQ_ISWAP_R        4

//Floating point instructions
#define RANDOMX_FREQ_FSWAP_R        4
#define RANDOMX_FREQ_FADD_R        16
#define RANDOMX_FREQ_FADD_M         5
#define RANDOMX_FREQ_FSUB_R        16
#define RANDOMX_FREQ_FSUB_M         5
#define RANDOMX_FREQ_FSCAL_R        6
#define RANDOMX_FREQ_FMUL_R        32
#define RANDOMX_FREQ_FDIV_M         4
#define RANDOMX_FREQ_FSQRT_R        6

//Control instructions
#define RANDOMX_FREQ_CBRANCH       16
#define RANDOMX_FREQ_CFROUND        1

//Store instruction
#define RANDOMX_FREQ_ISTORE        16

//No-op instruction
#define RANDOMX_FREQ_NOP            0

#define RANDOMX_DATASET_ITEM_SIZE 64

#define RANDOMX_PROGRAM_SIZE 256
#define WORKERS_PER_HASH 8
#define HASH_SIZE 64
#define ENTROPY_SIZE (128 + RANDOMX_PROGRAM_SIZE * 8)
#define REGISTERS_SIZE 256
#define IMM_BUF_SIZE (RANDOMX_PROGRAM_SIZE * 4 - REGISTERS_SIZE)
#define IMM_INDEX_COUNT ((IMM_BUF_SIZE / 4) - 2)
#define VM_STATE_SIZE (REGISTERS_SIZE + IMM_BUF_SIZE + RANDOMX_PROGRAM_SIZE * 4)

#define CacheLineSize 64
#define ScratchpadL3Mask64 (RANDOMX_SCRATCHPAD_L3 - CacheLineSize)
#define CacheLineAlignMask ((RANDOMX_DATASET_BASE_SIZE - 1) & ~(CacheLineSize - 1))

#define mantissaSize 52
#define exponentSize 11
#define mantissaMask ((1UL << mantissaSize) - 1)
#define exponentMask ((1UL << exponentSize) - 1)
#define exponentBias 1023
#define constExponentBits 0x300
#define dynamicExponentBits 4
#define staticExponentBits 4

#define RegistersCount 8
#define RegisterCountFlt (RegistersCount / 2)
#define ConditionOffset RANDOMX_JUMP_OFFSET
#define StoreL3Condition 14
#define DatasetExtraItems (RANDOMX_DATASET_EXTRA_SIZE / RANDOMX_DATASET_ITEM_SIZE)

#define RegisterNeedsDisplacement 5

//
// VM state:
//
// Bytes 0-255: registers
// Bytes 256-1023: imm32 values (up to 192 values can be stored). IMUL_RCP and CBRANCH use 2 consecutive imm32 values.
// Bytes 1024-2047: up to 256 instructions
//
// Instruction encoding:
//
// Bits 0-2: dst (0-7)
// Bits 3-5: src (0-7)
// Bits 6-13: imm32/64 offset (in DWORDs, 0-191)
// Bit 14: src location (register, scratchpad)
// Bits 15-16: src shift (0-3), ADD/MUL switch for FMA instruction
// Bit 17: src=imm32
// Bit 18: src=imm64
// Bit 19: src = -src
// Bits 20-23: opcode (add_rs, add, mul, umul_hi, imul_hi, neg, xor, ror, swap, cbranch, store, fswap, fma, fsqrt, fdiv, cfround)
// Bits 24-27: how many parallel instructions to run starting with this one (1-16)
// Bits 28-31: how many of them are FP instructions (0-8)
//

#define DST_OFFSET			0
#define SRC_OFFSET			3
#define IMM_OFFSET			6
#define LOC_OFFSET			14
#define SHIFT_OFFSET		15
#define SRC_IS_IMM32_OFFSET	17
#define SRC_IS_IMM64_OFFSET	18
#define NEGATIVE_SRC_OFFSET	19
#define OPCODE_OFFSET		20
#define NUM_INSTS_OFFSET	24
#define NUM_FP_INSTS_OFFSET	28

// ISWAP r0, r0
#define INST_NOP			(8 << OPCODE_OFFSET)

#define LOC_L1 (32 - 14)
#define LOC_L2 (32 - 18)
#define LOC_L3 (32 - 21)

typedef uchar uint8_t;
typedef ushort uint16_t;
typedef uint uint32_t;
typedef ulong uint64_t;

typedef int int32_t;

double getSmallPositiveFloatBits(uint64_t entropy)
{
	uint64_t exponent = entropy >> 59; //0..31
	uint64_t mantissa = entropy & mantissaMask;
	exponent += exponentBias;
	exponent &= exponentMask;
	exponent <<= mantissaSize;
	return as_double(exponent | mantissa);
}

uint64_t getStaticExponent(uint64_t entropy)
{
	uint64_t exponent = constExponentBits;
	exponent |= (entropy >> (64 - staticExponentBits)) << dynamicExponentBits;
	exponent <<= mantissaSize;
	return exponent;
}

uint64_t getFloatMask(uint64_t entropy)
{
	const uint64_t mask22bit = (1UL << 22) - 1;
	return (entropy & mask22bit) | getStaticExponent(entropy);
}

void set_buffer(__local uint32_t *dst_buf, uint32_t N, const uint32_t value)
{
	uint32_t i = get_local_id(0) * sizeof(uint32_t);
	const uint32_t step = get_local_size(0) * sizeof(uint32_t);
	__local uint8_t* dst = ((__local uint8_t*)dst_buf) + i;
	while (i < sizeof(uint32_t) * N)
	{
		*(__local uint32_t*)(dst) = value;
		dst += step;
		i += step;
	}
}

uint64_t imul_rcp_value(uint32_t divisor)
{
	if ((divisor & (divisor - 1)) == 0)
	{
		return 1UL;
	}

	const uint64_t p2exp63 = 1UL << 63;

	uint64_t quotient = p2exp63 / divisor;
	uint64_t remainder = p2exp63 % divisor;

	const uint32_t bsr = 31 - clz(divisor);

	for (uint32_t shift = 0; shift <= bsr; ++shift)
	{
		const bool b = (remainder >= divisor - remainder);
		quotient = (quotient << 1) | (b ? 1 : 0);
		remainder = (remainder << 1) - (b ? divisor : 0);
	}

	return quotient;
}

#define set_byte(a, position, value) do { ((uint8_t*)&(a))[(position)] = (value); } while (0)
uint32_t get_byte(uint64_t a, uint32_t position) { return (a >> (position << 3)) & 0xFF; }
#define update_max(value, next_value) do { if ((value) < (next_value)) (value) = (next_value); } while (0)

__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void init_vm(__global const void* entropy_data, __global void* vm_states)
{
#if RANDOMX_PROGRAM_SIZE <= 256
	typedef uint8_t exec_t;
#else
	typedef uint16_t exec_t;
#endif

	__local uint32_t execution_plan_buf[RANDOMX_PROGRAM_SIZE * WORKERS_PER_HASH * (64 / 8) * sizeof(exec_t) / sizeof(uint32_t)];

	set_buffer(execution_plan_buf, sizeof(execution_plan_buf) / sizeof(uint32_t), 0);
	barrier(CLK_LOCAL_MEM_FENCE);

	const uint32_t global_index = get_global_id(0);
	const uint32_t idx = global_index / 8;
	const uint32_t sub = global_index % 8;

	__local exec_t* execution_plan = (__local exec_t*)(execution_plan_buf + (get_local_id(0) / 8) * RANDOMX_PROGRAM_SIZE * WORKERS_PER_HASH * sizeof(exec_t) / sizeof(uint32_t));

	__global uint64_t* R = ((__global uint64_t*)vm_states) + idx * VM_STATE_SIZE / sizeof(uint64_t);
	R[sub] = 0;

	const __global uint64_t* entropy = ((const __global uint64_t*)entropy_data) + idx * ENTROPY_SIZE / sizeof(uint64_t);

	__global double* A = (__global double*)(R + 24);
	A[sub] = getSmallPositiveFloatBits(entropy[sub]);

	if (sub == 0)
	{
		__global uint2* src_program = (__global uint2*)(entropy + 128 / sizeof(uint64_t));

#if RANDOMX_PROGRAM_SIZE <= 256
		uint64_t registerLastChanged = 0;
		uint64_t registerWasChanged = 0;
#else
		int32_t registerLastChanged[8] = { -1, -1, -1, -1, -1, -1, -1, -1 };
#endif

		// Initialize CBRANCH instructions
		for (uint32_t i = 0; i < RANDOMX_PROGRAM_SIZE; ++i)
		{
			// Clear all src flags (branch target, FP, branch)
			*(__global uint32_t*)(src_program + i) &= ~(0xF8U << 8);

			const uint2 src_inst = src_program[i];
			uint2 inst = src_inst;

			uint32_t opcode = inst.x & 0xff;
			const uint32_t dst = (inst.x >> 8) & 7;
			const uint32_t src = (inst.x >> 16) & 7;

			if (opcode < RANDOMX_FREQ_IADD_RS + RANDOMX_FREQ_IADD_M + RANDOMX_FREQ_ISUB_R + RANDOMX_FREQ_ISUB_M + RANDOMX_FREQ_IMUL_R + RANDOMX_FREQ_IMUL_M + RANDOMX_FREQ_IMULH_R + RANDOMX_FREQ_IMULH_M + RANDOMX_FREQ_ISMULH_R + RANDOMX_FREQ_ISMULH_M)
			{
#if RANDOMX_PROGRAM_SIZE <= 256
				set_byte(registerLastChanged, dst, i);
				set_byte(registerWasChanged, dst, 1);
#else
				registerLastChanged[dst] = i;
#endif
				continue;
			}
			opcode -= RANDOMX_FREQ_IADD_RS + RANDOMX_FREQ_IADD_M + RANDOMX_FREQ_ISUB_R + RANDOMX_FREQ_ISUB_M + RANDOMX_FREQ_IMUL_R + RANDOMX_FREQ_IMUL_M + RANDOMX_FREQ_IMULH_R + RANDOMX_FREQ_IMULH_M + RANDOMX_FREQ_ISMULH_R + RANDOMX_FREQ_ISMULH_M;

			if (opcode < RANDOMX_FREQ_IMUL_RCP)
			{
				if (inst.y & (inst.y - 1))
				{
#if RANDOMX_PROGRAM_SIZE <= 256
					set_byte(registerLastChanged, dst, i);
					set_byte(registerWasChanged, dst, 1);
#else
					registerLastChanged[dst] = i;
#endif
				}
				continue;
			}
			opcode -= RANDOMX_FREQ_IMUL_RCP;

			if (opcode < RANDOMX_FREQ_INEG_R + RANDOMX_FREQ_IXOR_R + RANDOMX_FREQ_IXOR_M + RANDOMX_FREQ_IROR_R + RANDOMX_FREQ_IROL_R)
			{
#if RANDOMX_PROGRAM_SIZE <= 256
				set_byte(registerLastChanged, dst, i);
				set_byte(registerWasChanged, dst, 1);
#else
				registerLastChanged[dst] = i;
#endif
				continue;
			}
			opcode -= RANDOMX_FREQ_INEG_R + RANDOMX_FREQ_IXOR_R + RANDOMX_FREQ_IXOR_M + RANDOMX_FREQ_IROR_R + RANDOMX_FREQ_IROL_R;

			if (opcode < RANDOMX_FREQ_ISWAP_R)
			{
				if (src != dst)
				{
#if RANDOMX_PROGRAM_SIZE <= 256
					set_byte(registerLastChanged, dst, i);
					set_byte(registerWasChanged, dst, 1);
					set_byte(registerLastChanged, src, i);
					set_byte(registerWasChanged, src, 1);
#else
					registerLastChanged[dst] = i;
					registerLastChanged[src] = i;
#endif
				}
				continue;
			}
			opcode -= RANDOMX_FREQ_ISWAP_R;

			if (opcode < RANDOMX_FREQ_FSWAP_R + RANDOMX_FREQ_FADD_R + RANDOMX_FREQ_FADD_M + RANDOMX_FREQ_FSUB_R + RANDOMX_FREQ_FSUB_M + RANDOMX_FREQ_FSCAL_R + RANDOMX_FREQ_FMUL_R + RANDOMX_FREQ_FDIV_M + RANDOMX_FREQ_FSQRT_R)
			{
				// Mark FP instruction (src |= 0x20)
				*(__global uint32_t*)(src_program + i) |= 0x20 << 8;
				continue;
			}
			opcode -= RANDOMX_FREQ_FSWAP_R + RANDOMX_FREQ_FADD_R + RANDOMX_FREQ_FADD_M + RANDOMX_FREQ_FSUB_R + RANDOMX_FREQ_FSUB_M + RANDOMX_FREQ_FSCAL_R + RANDOMX_FREQ_FMUL_R + RANDOMX_FREQ_FDIV_M + RANDOMX_FREQ_FSQRT_R;

			if (opcode < RANDOMX_FREQ_CBRANCH)
			{
				const uint32_t creg = dst;
#if RANDOMX_PROGRAM_SIZE <= 256
				const uint32_t change = get_byte(registerLastChanged, dst);
				const int32_t lastChanged = (get_byte(registerWasChanged, dst) == 0) ? -1 : (int32_t)(change);

				// Store condition register and branch target in CBRANCH instruction
				*(__global uint32_t*)(src_program + i) = (src_inst.x & 0xFF0000FFU) | ((creg | ((lastChanged == -1) ? 0x90 : 0x10)) << 8) | (((uint32_t)(lastChanged) & 0xFF) << 16);
#else
				const int32_t lastChanged = registerLastChanged[dst];

				// Store condition register in CBRANCH instruction
				*(__global uint32_t*)(src_program + i) = (src_inst.x & 0xFF0000FFU) | ((creg | 0x10) << 8);
#endif

				// Mark branch target instruction (src |= 0x40)
				*(__global uint32_t*)(src_program + lastChanged + 1) |= 0x40 << 8;

#if RANDOMX_PROGRAM_SIZE <= 256
				uint32_t tmp = i | (i << 8);
				registerLastChanged = tmp | (tmp << 16);
				registerLastChanged = registerLastChanged | (registerLastChanged << 32);

				registerWasChanged = 0x0101010101010101UL;
#else
				registerLastChanged[0] = i;
				registerLastChanged[1] = i;
				registerLastChanged[2] = i;
				registerLastChanged[3] = i;
				registerLastChanged[4] = i;
				registerLastChanged[5] = i;
				registerLastChanged[6] = i;
				registerLastChanged[7] = i;
#endif
			}
		}

		uint64_t registerLatency = 0;
		uint64_t registerReadCycle = 0;
		uint64_t registerLatencyFP = 0;
		uint64_t registerReadCycleFP = 0;
		uint32_t ScratchpadHighLatency = 0;
		uint32_t ScratchpadLatency = 0;

		int32_t first_available_slot = 0;
		int32_t first_allowed_slot_cfround = 0;
		int32_t last_used_slot = -1;
		int32_t last_memory_op_slot = -1;

		uint32_t num_slots_used = 0;
		uint32_t num_instructions = 0;

		int32_t first_instruction_slot = -1;
		bool first_instruction_fp = false;

		//if (global_index == 0)
		//{
		//	for (int j = 0; j < RANDOMX_PROGRAM_SIZE; ++j)
		//	{
		//		print_inst(src_program[j]);
		//		printf("\n");
		//	}
		//	printf("\n");
		//}

		// Schedule instructions
		bool update_branch_target_mark = false;
		bool first_available_slot_is_branch_target = false;
		for (uint32_t i = 0; i < RANDOMX_PROGRAM_SIZE; ++i)
		{
			const uint2 inst = src_program[i];

			uint32_t opcode = inst.x & 0xff;
			uint32_t dst = (inst.x >> 8) & 7;
			const uint32_t src = (inst.x >> 16) & 7;
			const uint32_t mod = (inst.x >> 24);

			bool is_branch_target = (inst.x & (0x40 << 8)) != 0;
			if (is_branch_target)
			{
				// If an instruction is a branch target, we can't move it before any previous instructions
				first_available_slot = last_used_slot + 1;

				// Mark this slot as a branch target
				// Whatever instruction takes this slot will receive branch target flag
				first_available_slot_is_branch_target = true;
			}

			const uint32_t dst_latency = get_byte(registerLatency, dst);
			const uint32_t src_latency = get_byte(registerLatency, src);
			const uint32_t reg_read_latency = (dst_latency > src_latency) ? dst_latency : src_latency;
			const uint32_t mem_read_latency = ((dst == src) && ((inst.y & ScratchpadL3Mask64) >= RANDOMX_SCRATCHPAD_L2)) ? ScratchpadHighLatency : ScratchpadLatency;

			uint32_t full_read_latency = mem_read_latency;
			update_max(full_read_latency, reg_read_latency);

			uint32_t latency = 0;
			bool is_memory_op = false;
			bool is_memory_store = false;
			bool is_nop = false;
			bool is_branch = false;
			bool is_swap = false;
			bool is_src_read = true;
			bool is_fp = false;
			bool is_cfround = false;

			do {
				if (opcode < RANDOMX_FREQ_IADD_RS)
				{
					latency = reg_read_latency;
					break;
				}
				opcode -= RANDOMX_FREQ_IADD_RS;

				if (opcode < RANDOMX_FREQ_IADD_M)
				{
					latency = full_read_latency;
					is_memory_op = true;
					break;
				}
				opcode -= RANDOMX_FREQ_IADD_M;

				if (opcode < RANDOMX_FREQ_ISUB_R)
				{
					latency = reg_read_latency;
					break;
				}
				opcode -= RANDOMX_FREQ_ISUB_R;

				if (opcode < RANDOMX_FREQ_ISUB_M)
				{
					latency = full_read_latency;
					is_memory_op = true;
					break;
				}
				opcode -= RANDOMX_FREQ_ISUB_M;

				if (opcode < RANDOMX_FREQ_IMUL_R)
				{
					latency = reg_read_latency;
					break;
				}
				opcode -= RANDOMX_FREQ_IMUL_R;

				if (opcode < RANDOMX_FREQ_IMUL_M)
				{
					latency = full_read_latency;
					is_memory_op = true;
					break;
				}
				opcode -= RANDOMX_FREQ_IMUL_M;

				if (opcode < RANDOMX_FREQ_IMULH_R)
				{
					latency = reg_read_latency;
					break;
				}
				opcode -= RANDOMX_FREQ_IMULH_R;

				if (opcode < RANDOMX_FREQ_IMULH_M)
				{
					latency = full_read_latency;
					is_memory_op = true;
					break;
				}
				opcode -= RANDOMX_FREQ_IMULH_M;

				if (opcode < RANDOMX_FREQ_ISMULH_R)
				{
					latency = reg_read_latency;
					break;
				}
				opcode -= RANDOMX_FREQ_ISMULH_R;

				if (opcode < RANDOMX_FREQ_ISMULH_M)
				{
					latency = full_read_latency;
					is_memory_op = true;
					break;
				}
				opcode -= RANDOMX_FREQ_ISMULH_M;

				if (opcode < RANDOMX_FREQ_IMUL_RCP)
				{
					is_src_read = false;
					if (inst.y & (inst.y - 1))
						latency = dst_latency;
					else
						is_nop = true;
					break;
				}
				opcode -= RANDOMX_FREQ_IMUL_RCP;

				if (opcode < RANDOMX_FREQ_INEG_R)
				{
					is_src_read = false;
					latency = dst_latency;
					break;
				}
				opcode -= RANDOMX_FREQ_INEG_R;

				if (opcode < RANDOMX_FREQ_IXOR_R)
				{
					latency = reg_read_latency;
					break;
				}
				opcode -= RANDOMX_FREQ_IXOR_R;

				if (opcode < RANDOMX_FREQ_IXOR_M)
				{
					latency = full_read_latency;
					is_memory_op = true;
					break;
				}
				opcode -= RANDOMX_FREQ_IXOR_M;

				if (opcode < RANDOMX_FREQ_IROR_R + RANDOMX_FREQ_IROL_R)
				{
					latency = reg_read_latency;
					break;
				}
				opcode -= RANDOMX_FREQ_IROR_R + RANDOMX_FREQ_IROL_R;

				if (opcode < RANDOMX_FREQ_ISWAP_R)
				{
					is_swap = true;
					if (dst != src)
						latency = reg_read_latency;
					else
						is_nop = true;
					break;
				}
				opcode -= RANDOMX_FREQ_ISWAP_R;

				if (opcode < RANDOMX_FREQ_FSWAP_R)
				{
					latency = get_byte(registerLatencyFP, dst);
					is_fp = true;
					is_src_read = false;
					break;
				}
				opcode -= RANDOMX_FREQ_FSWAP_R;

				if (opcode < RANDOMX_FREQ_FADD_R)
				{
					dst %= RegisterCountFlt;
					latency = get_byte(registerLatencyFP, dst);
					is_fp = true;
					is_src_read = false;
					break;
				}
				opcode -= RANDOMX_FREQ_FADD_R;

				if (opcode < RANDOMX_FREQ_FADD_M)
				{
					dst %= RegisterCountFlt;
					latency = get_byte(registerLatencyFP, dst);
					update_max(latency, src_latency);
					update_max(latency, ScratchpadLatency);
					is_fp = true;
					is_memory_op = true;
					break;
				}
				opcode -= RANDOMX_FREQ_FADD_M;

				if (opcode < RANDOMX_FREQ_FSUB_R)
				{
					dst %= RegisterCountFlt;
					latency = get_byte(registerLatencyFP, dst);
					is_fp = true;
					is_src_read = false;
					break;
				}
				opcode -= RANDOMX_FREQ_FSUB_R;

				if (opcode < RANDOMX_FREQ_FSUB_M)
				{
					dst %= RegisterCountFlt;
					latency = get_byte(registerLatencyFP, dst);
					update_max(latency, src_latency);
					update_max(latency, ScratchpadLatency);
					is_fp = true;
					is_memory_op = true;
					break;
				}
				opcode -= RANDOMX_FREQ_FSUB_M;

				if (opcode < RANDOMX_FREQ_FSCAL_R)
				{
					dst %= RegisterCountFlt;
					latency = get_byte(registerLatencyFP, dst);
					is_fp = true;
					is_src_read = false;
					break;
				}
				opcode -= RANDOMX_FREQ_FSCAL_R;

				if (opcode < RANDOMX_FREQ_FMUL_R)
				{
					dst = (dst % RegisterCountFlt) + RegisterCountFlt;
					latency = get_byte(registerLatencyFP, dst);
					is_fp = true;
					is_src_read = false;
					break;
				}
				opcode -= RANDOMX_FREQ_FMUL_R;

				if (opcode < RANDOMX_FREQ_FDIV_M)
				{
					dst = (dst % RegisterCountFlt) + RegisterCountFlt;
					latency = get_byte(registerLatencyFP, dst);
					update_max(latency, src_latency);
					update_max(latency, ScratchpadLatency);
					is_fp = true;
					is_memory_op = true;
					break;
				}
				opcode -= RANDOMX_FREQ_FDIV_M;

				if (opcode < RANDOMX_FREQ_FSQRT_R)
				{
					dst = (dst % RegisterCountFlt) + RegisterCountFlt;
					latency = get_byte(registerLatencyFP, dst);
					is_fp = true;
					is_src_read = false;
					break;
				}
				opcode -= RANDOMX_FREQ_FSQRT_R;

				if (opcode < RANDOMX_FREQ_CBRANCH)
				{
					is_src_read = false;
					is_branch = true;
					latency = dst_latency;

					// We can't move CBRANCH before any previous instructions
					first_available_slot = last_used_slot + 1;
					break;
				}
				opcode -= RANDOMX_FREQ_CBRANCH;

				if (opcode < RANDOMX_FREQ_CFROUND)
				{
					latency = src_latency;
					is_cfround = true;
					break;
				}
				opcode -= RANDOMX_FREQ_CFROUND;

				if (opcode < RANDOMX_FREQ_ISTORE)
				{
					latency = reg_read_latency;
					update_max(latency, (last_memory_op_slot + WORKERS_PER_HASH) / WORKERS_PER_HASH);
					is_memory_op = true;
					is_memory_store = true;
					break;
				}
				opcode -= RANDOMX_FREQ_ISTORE;

				is_nop = true;
			} while (false);

			if (is_nop)
			{
				if (is_branch_target)
				{
					// Mark next non-NOP instruction as the branch target instead of this NOP
					update_branch_target_mark = true;
				}
				continue;
			}

			if (update_branch_target_mark)
			{
				*(__global uint32_t*)(src_program + i) |= 0x40 << 8;
				update_branch_target_mark = false;
				is_branch_target = true;
			}

			int32_t first_allowed_slot = first_available_slot;
			update_max(first_allowed_slot, latency * WORKERS_PER_HASH);
			if (is_cfround)
				update_max(first_allowed_slot, first_allowed_slot_cfround);
			else
				update_max(first_allowed_slot, get_byte(is_fp ? registerReadCycleFP : registerReadCycle, dst) * WORKERS_PER_HASH);

			if (is_swap)
				update_max(first_allowed_slot, get_byte(registerReadCycle, src) * WORKERS_PER_HASH);

			int32_t slot_to_use = last_used_slot + 1;
			update_max(slot_to_use, first_allowed_slot);

			if (is_fp)
			{
				slot_to_use = -1;
				for (int32_t j = first_allowed_slot; slot_to_use < 0; ++j)
				{
					if ((execution_plan[j] == 0) && (execution_plan[j + 1] == 0) && ((j + 1) % WORKERS_PER_HASH))
					{
						bool blocked = false;
						for (int32_t k = (j / WORKERS_PER_HASH) * WORKERS_PER_HASH; k < j; ++k)
						{
							if (execution_plan[k] || (k == first_instruction_slot))
							{
								const uint32_t inst = src_program[execution_plan[k]].x;

								// If there is an integer instruction which is a branch target or a branch, or this FP instruction is a branch target itself, we can't reorder it to add more FP instructions to this cycle
								if (((inst & (0x20 << 8)) == 0) && (((inst & (0x50 << 8)) != 0) || is_branch_target))
								{
									blocked = true;
									continue;
								}
							}
						}

						if (!blocked)
						{
							for (int32_t k = (j / WORKERS_PER_HASH) * WORKERS_PER_HASH; k < j; ++k)
							{
								if (execution_plan[k] || (k == first_instruction_slot))
								{
									const uint32_t inst = src_program[execution_plan[k]].x;
									if ((inst & (0x20 << 8)) == 0)
									{
										execution_plan[j] = execution_plan[k];
										execution_plan[j + 1] = execution_plan[k + 1];
										if (first_instruction_slot == k) first_instruction_slot = j;
										if (first_instruction_slot == k + 1) first_instruction_slot = j + 1;
										slot_to_use = k;
										break;
									}
								}
							}

							if (slot_to_use < 0)
							{
								slot_to_use = j;
							}

							break;
						}
					}
				}
			}
			else
			{
				for (int32_t j = first_allowed_slot; j <= last_used_slot; ++j)
				{
					if (execution_plan[j] == 0)
					{
						slot_to_use = j;
						break;
					}
				}
			}

			if (i == 0)
			{
				first_instruction_slot = slot_to_use;
				first_instruction_fp = is_fp;
			}

			if (is_cfround)
			{
				first_allowed_slot_cfround = slot_to_use - (slot_to_use % WORKERS_PER_HASH) + WORKERS_PER_HASH;
			}

			++num_instructions;

			execution_plan[slot_to_use] = i;
			++num_slots_used;

			if (is_fp)
			{
				execution_plan[slot_to_use + 1] = i;
				++num_slots_used;
			}

			const uint32_t next_latency = (slot_to_use / WORKERS_PER_HASH) + 1;

			if (is_src_read)
			{
				int32_t value = get_byte(registerReadCycle, src);
				update_max(value, slot_to_use / WORKERS_PER_HASH);
				set_byte(registerReadCycle, src, value);
			}

			if (is_memory_op)
			{
				update_max(last_memory_op_slot, slot_to_use);
			}

			if (is_cfround)
			{
				const uint32_t t = next_latency | (next_latency << 8);
				registerLatencyFP = t | (t << 16);
				registerLatencyFP = registerLatencyFP | (registerLatencyFP << 32);
			}
			else if (is_fp)
			{
				set_byte(registerLatencyFP, dst, next_latency);

				int32_t value = get_byte(registerReadCycleFP, dst);
				update_max(value, slot_to_use / WORKERS_PER_HASH);
				set_byte(registerReadCycleFP, dst, value);
			}
			else
			{
				if (!is_memory_store && !is_nop)
				{
					set_byte(registerLatency, dst, next_latency);
					if (is_swap)
						set_byte(registerLatency, src, next_latency);

					int32_t value = get_byte(registerReadCycle, dst);
					update_max(value, slot_to_use / WORKERS_PER_HASH);
					set_byte(registerReadCycle, dst, value);
				}

				if (is_branch)
				{
					const uint32_t t = next_latency | (next_latency << 8);
					registerLatency = t | (t << 16);
					registerLatency = registerLatency | (registerLatency << 32);
				}

				if (is_memory_store)
				{
					int32_t value = get_byte(registerReadCycle, dst);
					update_max(value, slot_to_use / WORKERS_PER_HASH);
					set_byte(registerReadCycle, dst, value);
					ScratchpadLatency = slot_to_use / WORKERS_PER_HASH;
					if ((mod >> 4) >= StoreL3Condition)
						ScratchpadHighLatency = slot_to_use / WORKERS_PER_HASH;
				}
			}

			if (execution_plan[first_available_slot] || (first_available_slot == first_instruction_slot))
			{
				if (first_available_slot_is_branch_target)
				{
					src_program[i].x |= 0x40 << 8;
					first_available_slot_is_branch_target = false;
				}

				if (is_fp)
					++first_available_slot;

				do {
					++first_available_slot;
				} while ((first_available_slot < RANDOMX_PROGRAM_SIZE * WORKERS_PER_HASH) && (execution_plan[first_available_slot] != 0));
			}

			if (is_branch_target)
			{
				update_max(first_available_slot, is_fp ? (slot_to_use + 2) : (slot_to_use + 1));
			}

			update_max(last_used_slot, is_fp ? (slot_to_use + 1) : slot_to_use);
			while (execution_plan[last_used_slot] || (last_used_slot == first_instruction_slot) || ((last_used_slot == first_instruction_slot + 1) && first_instruction_fp))
			{
				++last_used_slot;
			}
			--last_used_slot;

			if (is_fp && (last_used_slot >= first_allowed_slot_cfround))
				first_allowed_slot_cfround = last_used_slot + 1;

			//if (global_index == 0)
			//{
			//	printf("slot_to_use = %d, first_available_slot = %d, last_used_slot = %d\n", slot_to_use, first_available_slot, last_used_slot);
			//	for (int j = 0; j <= last_used_slot; ++j)
			//	{
			//		if (execution_plan[j] || (j == first_instruction_slot) || ((j == first_instruction_slot + 1) && first_instruction_fp))
			//		{
			//			print_inst(src_program[execution_plan[j]]);
			//			printf(" | ");
			//		}
			//		else
			//		{
			//			printf("                      | ");
			//		}
			//		if (((j + 1) % WORKERS_PER_HASH) == 0) printf("\n");
			//	}
			//	printf("\n\n");
			//}
		}

		//if (global_index == 0)
		//{
		//	printf("IPC = %.3f, WPC = %.3f, num_instructions = %u, num_slots_used = %u, first_instruction_slot = %d, last_used_slot = %d, registerLatency = %016llx, registerLatencyFP = %016llx \n",
		//		num_instructions / static_cast<double>(last_used_slot / WORKERS_PER_HASH + 1),
		//		num_slots_used / static_cast<double>(last_used_slot / WORKERS_PER_HASH + 1),
		//		num_instructions,
		//		num_slots_used,
		//		first_instruction_slot,
		//		last_used_slot,
		//		registerLatency,
		//		registerLatencyFP
		//	);

		//	//for (int j = 0; j < RANDOMX_PROGRAM_SIZE; ++j)
		//	//{
		//	//	print_inst(src_program[j]);
		//	//	printf("\n");
		//	//}
		//	//printf("\n");

		//	for (int j = 0; j <= last_used_slot; ++j)
		//	{
		//		if (execution_plan[j] || (j == first_instruction_slot) || ((j == first_instruction_slot + 1) && first_instruction_fp))
		//		{
		//			print_inst(src_program[execution_plan[j]]);
		//			printf(" | ");
		//		}
		//		else
		//		{
		//			printf("                      | ");
		//		}
		//		if (((j + 1) % WORKERS_PER_HASH) == 0) printf("\n");
		//	}
		//	printf("\n\n");
		//}

		//atomicAdd((uint32_t*)num_vm_cycles, (last_used_slot / WORKERS_PER_HASH) + 1);
		//atomicAdd((uint32_t*)(num_vm_cycles) + 1, num_slots_used);

		uint32_t ma = (uint32_t)(entropy[8]) & CacheLineAlignMask;
		uint32_t mx = (uint32_t)(entropy[10]) & CacheLineAlignMask;

		uint32_t addressRegisters = (uint32_t)(entropy[12]);
		addressRegisters = ((addressRegisters & 1) | (((addressRegisters & 2) ? 3U : 2U) << 8) | (((addressRegisters & 4) ? 5U : 4U) << 16) | (((addressRegisters & 8) ? 7U : 6U) << 24)) * sizeof(uint64_t);

		uint32_t datasetOffset = (entropy[13] & DatasetExtraItems) * CacheLineSize;

		ulong2 eMask = *(__global ulong2*)(entropy + 14);
		eMask.x = getFloatMask(eMask.x);
		eMask.y = getFloatMask(eMask.y);

		((__global uint32_t*)(R + 16))[0] = ma;
		((__global uint32_t*)(R + 16))[1] = mx;
		((__global uint32_t*)(R + 16))[2] = addressRegisters;
		((__global uint32_t*)(R + 16))[3] = datasetOffset;
		((__global ulong2*)(R + 18))[0] = eMask;

		__global uint32_t* imm_buf = (__global uint32_t*)(R + REGISTERS_SIZE / sizeof(uint64_t));
		uint32_t imm_index = 0;
		int32_t imm_index_fscal_r = -1;
		__global uint32_t* compiled_program = (__global uint32_t*)(R + (REGISTERS_SIZE + IMM_BUF_SIZE) / sizeof(uint64_t));

		// Generate opcodes for execute_vm
		int32_t branch_target_slot = -1;
		int32_t k = -1;
		for (int32_t i = 0; i <= last_used_slot; ++i)
		{
			if (!(execution_plan[i] || (i == first_instruction_slot) || ((i == first_instruction_slot + 1) && first_instruction_fp)))
				continue;

			uint32_t num_workers = 1;
			uint32_t num_fp_insts = 0;
			while ((i + num_workers <= last_used_slot) && ((i + num_workers) % WORKERS_PER_HASH) && (execution_plan[i + num_workers] || (i + num_workers == first_instruction_slot) || ((i + num_workers == first_instruction_slot + 1) && first_instruction_fp)))
			{
				if ((num_workers & 1) && ((src_program[execution_plan[i + num_workers]].x & (0x20 << 8)) != 0))
					++num_fp_insts;
				++num_workers;
			}

			//if (global_index == 0)
			//	printf("i = %d, num_workers = %u, num_fp_insts = %u\n", i, num_workers, num_fp_insts);

			num_workers = ((num_workers - 1) << NUM_INSTS_OFFSET) | (num_fp_insts << NUM_FP_INSTS_OFFSET);

			const uint2 src_inst = src_program[execution_plan[i]];
			uint2 inst = src_inst;

			uint32_t opcode = inst.x & 0xff;
			const uint32_t dst = (inst.x >> 8) & 7;
			const uint32_t src = (inst.x >> 16) & 7;
			const uint32_t mod = (inst.x >> 24);

			const bool is_fp = (src_inst.x & (0x20 << 8)) != 0;
			if (is_fp && ((i & 1) == 0))
				++i;

			const bool is_branch_target = (src_inst.x & (0x40 << 8)) != 0;
			if (is_branch_target && (branch_target_slot < 0))
				branch_target_slot = k;

			++k;

			inst.x = INST_NOP;

			if (opcode < RANDOMX_FREQ_IADD_RS)
			{
				const uint32_t shift = (mod >> 2) % 4;

				inst.x = (dst << DST_OFFSET) | (src << SRC_OFFSET) | (shift << SHIFT_OFFSET);

				if (dst != RegisterNeedsDisplacement)
				{
					// Encode regular ADD (opcode 1)
					inst.x |= (1 << OPCODE_OFFSET);
				}
				else
				{
					// Encode ADD with src and imm32 (opcode 0)
					inst.x |= imm_index << IMM_OFFSET;
					if (imm_index < IMM_INDEX_COUNT)
						imm_buf[imm_index++] = inst.y;
				}

				*(compiled_program++) = inst.x | num_workers;
				continue;
			}
			opcode -= RANDOMX_FREQ_IADD_RS;

			if (opcode < RANDOMX_FREQ_IADD_M)
			{
				const uint32_t location = (src == dst) ? 3 : ((mod % 4) ? 1 : 2);
				inst.x = (dst << DST_OFFSET) | (src << SRC_OFFSET) | (1 << LOC_OFFSET) | (1 << OPCODE_OFFSET);
				inst.x |= imm_index << IMM_OFFSET;
				if (imm_index < IMM_INDEX_COUNT)
					imm_buf[imm_index++] = (inst.y & 0xFC1FFFFFU) | (((location == 1) ? LOC_L1 : ((location == 2) ? LOC_L2 : LOC_L3)) << 21);
				else
					inst.x = INST_NOP;

				*(compiled_program++) = inst.x | num_workers;
				continue;
			}
			opcode -= RANDOMX_FREQ_IADD_M;

			if (opcode < RANDOMX_FREQ_ISUB_R)
			{
				inst.x = (dst << DST_OFFSET) | (src << SRC_OFFSET) | (1 << OPCODE_OFFSET) | (1 << NEGATIVE_SRC_OFFSET);
				if (src == dst)
				{
					inst.x |= (imm_index << IMM_OFFSET) | (1 << SRC_IS_IMM32_OFFSET);
					if (imm_index < IMM_INDEX_COUNT)
						imm_buf[imm_index++] = inst.y;
				}

				*(compiled_program++) = inst.x | num_workers;
				continue;
			}
			opcode -= RANDOMX_FREQ_ISUB_R;

			if (opcode < RANDOMX_FREQ_ISUB_M)
			{
				const uint32_t location = (src == dst) ? 3 : ((mod % 4) ? 1 : 2);
				inst.x = (dst << DST_OFFSET) | (src << SRC_OFFSET) | (1 << LOC_OFFSET) | (1 << OPCODE_OFFSET) | (1 << NEGATIVE_SRC_OFFSET);
				inst.x |= imm_index << IMM_OFFSET;
				if (imm_index < IMM_INDEX_COUNT)
					imm_buf[imm_index++] = (inst.y & 0xFC1FFFFFU) | (((location == 1) ? LOC_L1 : ((location == 2) ? LOC_L2 : LOC_L3)) << 21);
				else
					inst.x = INST_NOP;

				*(compiled_program++) = inst.x | num_workers;
				continue;
			}
			opcode -= RANDOMX_FREQ_ISUB_M;

			if (opcode < RANDOMX_FREQ_IMUL_R)
			{
				inst.x = (dst << DST_OFFSET) | (src << SRC_OFFSET) | (2 << OPCODE_OFFSET);
				if (src == dst)
				{
					inst.x |= (imm_index << IMM_OFFSET) | (1 << SRC_IS_IMM32_OFFSET);
					if (imm_index < IMM_INDEX_COUNT)
						imm_buf[imm_index++] = inst.y;
				}

				*(compiled_program++) = inst.x | num_workers;
				continue;
			}
			opcode -= RANDOMX_FREQ_IMUL_R;

			if (opcode < RANDOMX_FREQ_IMUL_M)
			{
				const uint32_t location = (src == dst) ? 3 : ((mod % 4) ? 1 : 2);
				inst.x = (dst << DST_OFFSET) | (src << SRC_OFFSET) | (1 << LOC_OFFSET) | (2 << OPCODE_OFFSET);
				inst.x |= imm_index << IMM_OFFSET;
				if (imm_index < IMM_INDEX_COUNT)
					imm_buf[imm_index++] = (inst.y & 0xFC1FFFFFU) | (((location == 1) ? LOC_L1 : ((location == 2) ? LOC_L2 : LOC_L3)) << 21);
				else
					inst.x = INST_NOP;

				*(compiled_program++) = inst.x | num_workers;
				continue;
			}
			opcode -= RANDOMX_FREQ_IMUL_M;

			if (opcode < RANDOMX_FREQ_IMULH_R)
			{
				inst.x = (dst << DST_OFFSET) | (src << SRC_OFFSET) | (6 << OPCODE_OFFSET);

				*(compiled_program++) = inst.x | num_workers;
				continue;
			}
			opcode -= RANDOMX_FREQ_IMULH_R;

			if (opcode < RANDOMX_FREQ_IMULH_M)
			{
				const uint32_t location = (src == dst) ? 3 : ((mod % 4) ? 1 : 2);
				inst.x = (dst << DST_OFFSET) | (src << SRC_OFFSET) | (1 << LOC_OFFSET) | (6 << OPCODE_OFFSET);
				inst.x |= imm_index << IMM_OFFSET;
				if (imm_index < IMM_INDEX_COUNT)
					imm_buf[imm_index++] = (inst.y & 0xFC1FFFFFU) | (((location == 1) ? LOC_L1 : ((location == 2) ? LOC_L2 : LOC_L3)) << 21);
				else
					inst.x = INST_NOP;

				*(compiled_program++) = inst.x | num_workers;
				continue;
			}
			opcode -= RANDOMX_FREQ_IMULH_M;

			if (opcode < RANDOMX_FREQ_ISMULH_R)
			{
				inst.x = (dst << DST_OFFSET) | (src << SRC_OFFSET) | (4 << OPCODE_OFFSET);

				*(compiled_program++) = inst.x | num_workers;
				continue;
			}
			opcode -= RANDOMX_FREQ_ISMULH_R;

			if (opcode < RANDOMX_FREQ_ISMULH_M)
			{
				const uint32_t location = (src == dst) ? 3 : ((mod % 4) ? 1 : 2);
				inst.x = (dst << DST_OFFSET) | (src << SRC_OFFSET) | (1 << LOC_OFFSET) | (4 << OPCODE_OFFSET);
				inst.x |= imm_index << IMM_OFFSET;
				if (imm_index < IMM_INDEX_COUNT)
					imm_buf[imm_index++] = (inst.y & 0xFC1FFFFFU) | (((location == 1) ? LOC_L1 : ((location == 2) ? LOC_L2 : LOC_L3)) << 21);
				else
					inst.x = INST_NOP;

				*(compiled_program++) = inst.x | num_workers;
				continue;
			}
			opcode -= RANDOMX_FREQ_ISMULH_M;

			if (opcode < RANDOMX_FREQ_IMUL_RCP)
			{
				const uint64_t r = imul_rcp_value(inst.y);
				if (r == 1)
				{
					*(compiled_program++) = INST_NOP | num_workers;
					continue;
				}

				inst.x = (dst << DST_OFFSET) | (src << SRC_OFFSET) | (2 << OPCODE_OFFSET);
				inst.x |= (imm_index << IMM_OFFSET) | (1 << SRC_IS_IMM64_OFFSET);

				if (imm_index < IMM_INDEX_COUNT - 1)
				{
					imm_buf[imm_index] = ((const uint32_t*)&r)[0];
					imm_buf[imm_index + 1] = ((const uint32_t*)&r)[1];
					imm_index += 2;
				}

				*(compiled_program++) = inst.x | num_workers;
				continue;
			}
			opcode -= RANDOMX_FREQ_IMUL_RCP;

			if (opcode < RANDOMX_FREQ_INEG_R)
			{
				inst.x = (dst << DST_OFFSET) | (5 << OPCODE_OFFSET);

				*(compiled_program++) = inst.x | num_workers;
				continue;
			}
			opcode -= RANDOMX_FREQ_INEG_R;

			if (opcode < RANDOMX_FREQ_IXOR_R)
			{
				inst.x = (dst << DST_OFFSET) | (src << SRC_OFFSET) | (3 << OPCODE_OFFSET);
				if (src == dst)
				{
					inst.x |= (imm_index << IMM_OFFSET) | (1 << SRC_IS_IMM32_OFFSET);
					if (imm_index < IMM_INDEX_COUNT)
						imm_buf[imm_index++] = inst.y;
				}

				*(compiled_program++) = inst.x | num_workers;
				continue;
			}
			opcode -= RANDOMX_FREQ_IXOR_R;

			if (opcode < RANDOMX_FREQ_IXOR_M)
			{
				const uint32_t location = (src == dst) ? 3 : ((mod % 4) ? 1 : 2);
				inst.x = (dst << DST_OFFSET) | (src << SRC_OFFSET) | (1 << LOC_OFFSET) | (3 << OPCODE_OFFSET);
				inst.x |= imm_index << IMM_OFFSET;
				if (imm_index < IMM_INDEX_COUNT)
					imm_buf[imm_index++] = (inst.y & 0xFC1FFFFFU) | (((location == 1) ? LOC_L1 : ((location == 2) ? LOC_L2 : LOC_L3)) << 21);
				else
					inst.x = INST_NOP;

				*(compiled_program++) = inst.x | num_workers;
				continue;
			}
			opcode -= RANDOMX_FREQ_IXOR_M;

			if (opcode < RANDOMX_FREQ_IROR_R + RANDOMX_FREQ_IROL_R)
			{
				inst.x = (dst << DST_OFFSET) | (src << SRC_OFFSET) | (7 << OPCODE_OFFSET);
				if (src == dst)
				{
					inst.x |= (imm_index << IMM_OFFSET) | (1 << SRC_IS_IMM32_OFFSET);
					if (imm_index < IMM_INDEX_COUNT)
						imm_buf[imm_index++] = inst.y;
				}
				if (opcode >= RANDOMX_FREQ_IROR_R)
				{
					inst.x |= (1 << NEGATIVE_SRC_OFFSET);
				}

				*(compiled_program++) = inst.x | num_workers;
				continue;
			}
			opcode -= RANDOMX_FREQ_IROR_R + RANDOMX_FREQ_IROL_R;

			if (opcode < RANDOMX_FREQ_ISWAP_R)
			{
				inst.x = (dst << DST_OFFSET) | (src << SRC_OFFSET) | (8 << OPCODE_OFFSET);

				*(compiled_program++) = ((src != dst) ? inst.x : INST_NOP) | num_workers;
				continue;
			}
			opcode -= RANDOMX_FREQ_ISWAP_R;

			if (opcode < RANDOMX_FREQ_FSWAP_R)
			{
				inst.x = (dst << DST_OFFSET) | (11 << OPCODE_OFFSET);

				*(compiled_program++) = inst.x | num_workers;
				continue;
			}
			opcode -= RANDOMX_FREQ_FSWAP_R;

			if (opcode < RANDOMX_FREQ_FADD_R)
			{
				inst.x = ((dst % RegisterCountFlt) << DST_OFFSET) | ((src % RegisterCountFlt) << (SRC_OFFSET + 1)) | (12 << OPCODE_OFFSET);

				*(compiled_program++) = inst.x | num_workers;
				continue;
			}
			opcode -= RANDOMX_FREQ_FADD_R;

			if (opcode < RANDOMX_FREQ_FADD_M)
			{
				const uint32_t location = (mod % 4) ? 1 : 2;
				inst.x = ((dst % RegisterCountFlt) << DST_OFFSET) | (src << SRC_OFFSET) | (1 << LOC_OFFSET) | (12 << OPCODE_OFFSET);
				inst.x |= imm_index << IMM_OFFSET;
				if (imm_index < IMM_INDEX_COUNT)
					imm_buf[imm_index++] = (inst.y & 0xFC1FFFFFU) | (((location == 1) ? LOC_L1 : ((location == 2) ? LOC_L2 : LOC_L3)) << 21);
				else
					inst.x = INST_NOP;

				*(compiled_program++) = inst.x | num_workers;
				continue;
			}
			opcode -= RANDOMX_FREQ_FADD_M;

			if (opcode < RANDOMX_FREQ_FSUB_R)
			{
				inst.x = ((dst % RegisterCountFlt) << DST_OFFSET) | ((src % RegisterCountFlt) << (SRC_OFFSET + 1)) | (12 << OPCODE_OFFSET) | (1 << NEGATIVE_SRC_OFFSET);

				*(compiled_program++) = inst.x | num_workers;
				continue;
			}
			opcode -= RANDOMX_FREQ_FSUB_R;

			if (opcode < RANDOMX_FREQ_FSUB_M)
			{
				const uint32_t location = (mod % 4) ? 1 : 2;
				inst.x = ((dst % RegisterCountFlt) << DST_OFFSET) | (src << SRC_OFFSET) | (1 << LOC_OFFSET) | (12 << OPCODE_OFFSET) | (1 << NEGATIVE_SRC_OFFSET);
				inst.x |= imm_index << IMM_OFFSET;
				if (imm_index < IMM_INDEX_COUNT)
					imm_buf[imm_index++] = (inst.y & 0xFC1FFFFFU) | (((location == 1) ? LOC_L1 : ((location == 2) ? LOC_L2 : LOC_L3)) << 21);
				else
					inst.x = INST_NOP;

				*(compiled_program++) = inst.x | num_workers;
				continue;
			}
			opcode -= RANDOMX_FREQ_FSUB_M;

			if (opcode < RANDOMX_FREQ_FSCAL_R)
			{
				inst.x = ((dst % RegisterCountFlt) << DST_OFFSET) | (1 << SRC_IS_IMM64_OFFSET) | (3 << OPCODE_OFFSET);
				if (imm_index_fscal_r >= 0)
				{
					inst.x |= (imm_index_fscal_r << IMM_OFFSET);
				}
				else
				{
					imm_index_fscal_r = imm_index;
					inst.x |= (imm_index << IMM_OFFSET);

					if (imm_index < IMM_INDEX_COUNT - 1)
					{
						imm_buf[imm_index] = 0;
						imm_buf[imm_index + 1] = 0x80F00000UL;
						imm_index += 2;
					}
				}

				*(compiled_program++) = inst.x | num_workers;
				continue;
			}
			opcode -= RANDOMX_FREQ_FSCAL_R;

			if (opcode < RANDOMX_FREQ_FMUL_R)
			{
				inst.x = (((dst % RegisterCountFlt) + RegisterCountFlt) << DST_OFFSET) | ((src % RegisterCountFlt) << (SRC_OFFSET + 1)) | (1 << SHIFT_OFFSET) | (12 << OPCODE_OFFSET);

				*(compiled_program++) = inst.x | num_workers;
				continue;
			}
			opcode -= RANDOMX_FREQ_FMUL_R;

			if (opcode < RANDOMX_FREQ_FDIV_M)
			{
				const uint32_t location = (mod % 4) ? 1 : 2;
				inst.x = (((dst % RegisterCountFlt) + RegisterCountFlt) << DST_OFFSET) | (src << SRC_OFFSET) | (1 << LOC_OFFSET) | (15 << OPCODE_OFFSET);
				inst.x |= imm_index << IMM_OFFSET;
				if (imm_index < IMM_INDEX_COUNT)
					imm_buf[imm_index++] = (inst.y & 0xFC1FFFFFU) | (((location == 1) ? LOC_L1 : ((location == 2) ? LOC_L2 : LOC_L3)) << 21);
				else
					inst.x = INST_NOP;

				*(compiled_program++) = inst.x | num_workers;
				continue;
			}
			opcode -= RANDOMX_FREQ_FDIV_M;

			if (opcode < RANDOMX_FREQ_FSQRT_R)
			{
				inst.x = (((dst % RegisterCountFlt) + RegisterCountFlt) << DST_OFFSET) | (14 << OPCODE_OFFSET);

				*(compiled_program++) = inst.x | num_workers;
				continue;
			}
			opcode -= RANDOMX_FREQ_FSQRT_R;

			if (opcode < RANDOMX_FREQ_CBRANCH)
			{
				inst.x = (dst << DST_OFFSET) | (9 << OPCODE_OFFSET);
				inst.x |= (imm_index << IMM_OFFSET);

				const uint32_t cshift = (mod >> 4) + ConditionOffset;

				uint32_t imm = inst.y | (1U << cshift);
				if (cshift > 0)
					imm &= ~(1U << (cshift - 1));

				if (imm_index < IMM_INDEX_COUNT - 1)
				{
					imm_buf[imm_index] = imm;
					imm_buf[imm_index + 1] = cshift | ((uint32_t)(branch_target_slot) << 5);
					imm_index += 2;
				}
				else
				{
					// Data doesn't fit, skip it
					inst.x = INST_NOP;
				}

				branch_target_slot = -1;

				*(compiled_program++) = inst.x | num_workers;
				continue;
			}
			opcode -= RANDOMX_FREQ_CBRANCH;

			if (opcode < RANDOMX_FREQ_CFROUND)
			{
				inst.x = (src << SRC_OFFSET) | (13 << OPCODE_OFFSET) | ((inst.y & 63) << IMM_OFFSET);

				*(compiled_program++) = inst.x | num_workers;
				continue;
			}
			opcode -= RANDOMX_FREQ_CFROUND;

			if (opcode < RANDOMX_FREQ_ISTORE)
			{
				const uint32_t location = ((mod >> 4) >= StoreL3Condition) ? 3 : ((mod % 4) ? 1 : 2);
				inst.x = (dst << DST_OFFSET) | (src << SRC_OFFSET) | (1 << LOC_OFFSET) | (10 << OPCODE_OFFSET);
				inst.x |= imm_index << IMM_OFFSET;
				if (imm_index < IMM_INDEX_COUNT)
					imm_buf[imm_index++] = (inst.y & 0xFC1FFFFFU) | (((location == 1) ? LOC_L1 : ((location == 2) ? LOC_L2 : LOC_L3)) << 21);
				else
					inst.x = INST_NOP;
				*(compiled_program++) = inst.x | num_workers;
				continue;
			}
			opcode -= RANDOMX_FREQ_ISTORE;

			*(compiled_program++) = inst.x | num_workers;
		}

		((__global uint32_t*)(R + 20))[0] = (uint32_t)(compiled_program - (__global uint32_t*)(R + (REGISTERS_SIZE + IMM_BUF_SIZE) / sizeof(uint64_t)));
	}
}
