/*
Copyright (c) 2019 SChernykh
Portions Copyright (c) 2018-2019 tevador

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

#define mantissaSize 52
#define exponentSize 11
#define mantissaMask ((1UL << mantissaSize) - 1)
#define exponentMask ((1UL << exponentSize) - 1)
#define exponentBias 1023

#define dynamicExponentBits 4
#define staticExponentBits 4
#define constExponentBits 0x300
#define dynamicMantissaMask ((1UL << (mantissaSize + dynamicExponentBits)) - 1)

#define CacheLineSize 64U
#define CacheLineAlignMask ((1U << 31) - 1) & ~(CacheLineSize - 1)
#define DatasetExtraItems 524287U

#define RANDOMX_PROGRAM_SIZE 256
#define ENTROPY_SIZE (128 + 2048)
#define COMPILED_PROGRAM_SIZE 65536
#define HASHES_PER_GROUP 4

#define RANDOMX_FREQ_IADD_RS       25

ulong getSmallPositiveFloatBits(const ulong entropy)
{
	ulong exponent = entropy >> 59;
	ulong mantissa = entropy & mantissaMask;
	exponent += exponentBias;
	exponent &= exponentMask;
	exponent <<= mantissaSize;
	return exponent | mantissa;
}

ulong getStaticExponent(const ulong entropy)
{
	ulong exponent = constExponentBits;
	exponent |= (entropy >> (64 - staticExponentBits)) << dynamicExponentBits;
	exponent <<= mantissaSize;
	return exponent;
}

ulong getFloatMask(const ulong entropy)
{
	const uint mask22bit = (1U << 22) - 1;
	return (entropy & mask22bit) | getStaticExponent(entropy);
}

__global uint* generate_jit_code(__global const uint2* e, __global uint* p, uint index)
{
	for (uint i = 0; i < RANDOMX_PROGRAM_SIZE; ++i)
	{
		const uint2 inst = e[i];
		uint opcode = inst.x & 0xFF;
		const uint dst = (inst.x >> 8) & 7;
		const uint src = (inst.x >> 16) & 7;
		const uint mod = inst.x >> 24;

		if (opcode < RANDOMX_FREQ_IADD_RS)
		{
			const uint shift = (mod >> 2) % 4;
			if (shift > 0)
			{
				// s_lshl_b64 s[14:15], s[(16 + src * 2):(17 + src * 2)], shift
				*(p++) = 0x8e8e8010u | (src << 1) | (shift << 8);

				// s_add_u32 s(16 + dst * 2), s(16 + dst * 2), s14
				*(p++) = 0x80100e10u | (dst << 1) | (dst << 17);

				// s_addc_u32 s(17 + dst * 2), s(17 + dst * 2), s15
				*(p++) = 0x82110f11u | (dst << 1) | (dst << 17);
			}
			else
			{
				// s_add_u32 s(16 + dst * 2), s(16 + dst * 2), s(16 + src * 2)
				*(p++) = 0x80101010u | (dst << 1) | (dst << 17) | (src << 9);

				// s_addc_u32 s(17 + dst * 2), s(17 + dst * 2), s(17 + src * 2)
				*(p++) = 0x82111111u | (dst << 1) | (dst << 17) | (src << 9);
			}

			if (dst == 5)
			{
				// s_add_u32 s(16 + dst * 2), s(16 + dst * 2), imm32
				*(p++) = 0x8010ff10u | (dst << 1) | (dst << 17);
				*(p++) = inst.y;

				// s_addc_u32 s(17 + dst * 2), s(17 + dst * 2), ((inst.y < 0) ? -1 : 0)
				*(p++) = 0x82110011 | (dst << 1) | (dst << 17) | (((as_int(inst.y) < 0) ? 0xc1 : 0x80) << 8);
			}
			continue;
		}
		opcode -= RANDOMX_FREQ_IADD_RS;
	}

	// Jump back to randomx_run kernel
	*(p++) = 0xbe8e1e0cu; // s_swappc_b64 s[14:15], s[12:13]

	return p;
}

__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void randomx_init(__global const ulong* entropy, __global ulong* registers, __global uint* programs, uint batch_size)
{
	const uint global_index = get_global_id(0);
	if ((global_index % HASHES_PER_GROUP) == 0)
	{
		__global uint* p = programs + (global_index / HASHES_PER_GROUP) * (COMPILED_PROGRAM_SIZE / sizeof(uint));
		__global const uint2* e = (__global const uint2*)(entropy + (global_index / HASHES_PER_GROUP) * HASHES_PER_GROUP * (ENTROPY_SIZE / sizeof(ulong)) + (128 / sizeof(ulong)));

		#pragma unroll(1)
		for (uint i = 0; i < HASHES_PER_GROUP; ++i, e += (ENTROPY_SIZE / sizeof(uint2)))
			p = generate_jit_code(e, p, i);
	}

	__global ulong* R = registers + global_index * 32;
	entropy += global_index * (ENTROPY_SIZE / sizeof(ulong));

	// Group R registers
	R[0] = 0;
	R[1] = 0;
	R[2] = 0;
	R[3] = 0;
	R[4] = 0;
	R[5] = 0;
	R[6] = 0;
	R[7] = 0;

	// Group A registers
	R[24] = getSmallPositiveFloatBits(entropy[0]);
	R[25] = getSmallPositiveFloatBits(entropy[1]);
	R[26] = getSmallPositiveFloatBits(entropy[2]);
	R[27] = getSmallPositiveFloatBits(entropy[3]);
	R[28] = getSmallPositiveFloatBits(entropy[4]);
	R[29] = getSmallPositiveFloatBits(entropy[5]);
	R[30] = getSmallPositiveFloatBits(entropy[6]);
	R[31] = getSmallPositiveFloatBits(entropy[7]);

	// ma, mx
	((__global uint*)(R + 16))[0] = entropy[8] & CacheLineAlignMask;
	((__global uint*)(R + 16))[1] = entropy[10];

	// address registers
	uint addressRegisters = entropy[12];
	((__global uint*)(R + 17))[0] = 0 + (addressRegisters & 1);
	addressRegisters >>= 1;
	((__global uint*)(R + 17))[1] = 2 + (addressRegisters & 1);
	addressRegisters >>= 1;
	((__global uint*)(R + 17))[2] = 4 + (addressRegisters & 1);
	addressRegisters >>= 1;
	((__global uint*)(R + 17))[3] = 6 + (addressRegisters & 1);

	// dataset offset
	((__global uint*)(R + 19))[0] = (entropy[13] & DatasetExtraItems) * CacheLineSize;

	// eMask
	R[20] = getFloatMask(entropy[14]);
	R[21] = getFloatMask(entropy[15]);
}
