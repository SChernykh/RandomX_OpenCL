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

#include <vector>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include "opencl_helpers.h"
#include "tests.h"

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4804)
#endif

#include "../RandomX/src/blake2/blake2.h"
#include "../RandomX/src/aes_hash.hpp"

#ifdef _MSC_VER
#pragma warning(pop)
#endif

static constexpr int SCRATCHPAD_SIZE = 1 << 21;
static constexpr int ENTROPY_SIZE = 128 + 2048;
static constexpr int REGISTERS_SIZE = 256;
static constexpr int INITIAL_HASH_SIZE = 64;

static const std::string AES_CL = "CL/aes.cl";
static const std::string CL_FILLAES1RX4_SCRATCHPAD = "fillAes1Rx4_scratchpad";
static const std::string CL_FILLAES1RX4_ENTROPY = "fillAes1Rx4_entropy";
static const std::string CL_HASHAES1RX4 = "hashAes1Rx4";

static const std::string BLAKE2B_CL = "CL/blake2b.cl";
static const std::string CL_BLAKE2B_INITIAL_HASH = "blake2b_initial_hash";
static const std::string CL_BLAKE2B_HASH_REGISTERS_32 = "blake2b_hash_registers_32";
static const std::string CL_BLAKE2B_HASH_REGISTERS_64 = "blake2b_hash_registers_64";
static const std::string CL_BLAKE2B_512_SINGLE_BLOCK_BENCH = "blake2b_512_single_block_bench";
static const std::string CL_BLAKE2B_512_DOUBLE_BLOCK_BENCH = "blake2b_512_double_block_bench";

static uint8_t blockTemplate[] = {
		0x07, 0x07, 0xf7, 0xa4, 0xf0, 0xd6, 0x05, 0xb3, 0x03, 0x26, 0x08, 0x16, 0xba, 0x3f, 0x10, 0x90, 0x2e, 0x1a, 0x14,
		0x5a, 0xc5, 0xfa, 0xd3, 0xaa, 0x3a, 0xf6, 0xea, 0x44, 0xc1, 0x18, 0x69, 0xdc, 0x4f, 0x85, 0x3f, 0x00, 0x2b, 0x2e,
		0xea, 0x00, 0x00, 0x00, 0x00, 0x77, 0xb2, 0x06, 0xa0, 0x2c, 0xa5, 0xb1, 0xd4, 0xce, 0x6b, 0xbf, 0xdf, 0x0a, 0xca,
		0xc3, 0x8b, 0xde, 0xd3, 0x4d, 0x2d, 0xcd, 0xee, 0xf9, 0x5c, 0xd2, 0x0c, 0xef, 0xc1, 0x2f, 0x61, 0xd5, 0x61, 0x09
};

constexpr size_t BLAKE2B_STEP = 1 << 28;

using namespace std::chrono;

int tests(uint32_t platform_id, uint32_t device_id, size_t intensity)
{
	std::cout << "Initializing GPU #" << device_id << " on OpenCL platform #" << platform_id << std::endl << std::endl;

	OpenCLContext ctx;
	if (!ctx.Init(platform_id, device_id,
		{
			AES_CL,
			BLAKE2B_CL
		},
		{
			CL_FILLAES1RX4_SCRATCHPAD,
			CL_FILLAES1RX4_ENTROPY,
			CL_HASHAES1RX4,
			CL_BLAKE2B_INITIAL_HASH,
			CL_BLAKE2B_HASH_REGISTERS_32,
			CL_BLAKE2B_HASH_REGISTERS_64,
			CL_BLAKE2B_512_SINGLE_BLOCK_BENCH,
			CL_BLAKE2B_512_DOUBLE_BLOCK_BENCH
		}))
	{
		return 1;
	}

	if (!intensity)
		intensity = std::min(ctx.device_max_alloc_size, ctx.device_global_mem_size) / SCRATCHPAD_SIZE;

	intensity -= (intensity & 63);

	DevicePtr scratchpads_gpu(ctx, intensity * SCRATCHPAD_SIZE);
	if (!scratchpads_gpu)
	{
		return 1;
	}
	std::cout << "Allocated " << intensity << " scratchpads" << std::endl << std::endl;

	DevicePtr entropy_gpu(ctx, intensity * ENTROPY_SIZE);
	if (!entropy_gpu)
	{
		return 1;
	}

	DevicePtr registers_gpu(ctx, intensity * REGISTERS_SIZE);
	if (!registers_gpu)
	{
		return 1;
	}

	uint32_t zero = 0;
	cl_int err;
	CL_CHECKED_CALL(clEnqueueFillBuffer, ctx.queue, registers_gpu, &zero, sizeof(zero), 0, intensity * REGISTERS_SIZE, 0, nullptr, nullptr);

	DevicePtr hash_gpu(ctx, intensity * INITIAL_HASH_SIZE);
	if (!hash_gpu)
	{
		return 1;
	}

	DevicePtr blockTemplate_gpu(ctx, sizeof(blockTemplate));
	if (!blockTemplate_gpu)
	{
		return 1;
	}

	CL_CHECKED_CALL(clEnqueueWriteBuffer, ctx.queue, blockTemplate_gpu, CL_FALSE, 0, sizeof(blockTemplate), blockTemplate, 0, nullptr, nullptr);

	DevicePtr nonce_gpu(ctx, sizeof(uint64_t));
	if (!nonce_gpu) {
		return 1;
	}

	cl_kernel kernel = ctx.kernels[CL_BLAKE2B_INITIAL_HASH];
	if (!clSetKernelArgs(kernel, hash_gpu, blockTemplate_gpu, 0))
	{
		return 1;
	}

	size_t global_work_size = intensity;
	size_t local_work_size = 64;
	CL_CHECKED_CALL(clEnqueueNDRangeKernel, ctx.queue, kernel, 1, nullptr, &global_work_size, &local_work_size, 0, nullptr, nullptr);
	CL_CHECKED_CALL(clFinish, ctx.queue);

	std::vector<uint8_t> hashes;
	hashes.resize(intensity * INITIAL_HASH_SIZE);
	CL_CHECKED_CALL(clEnqueueReadBuffer, ctx.queue, hash_gpu, CL_TRUE, 0, intensity * INITIAL_HASH_SIZE, hashes.data(), 0, nullptr, nullptr);

	std::vector<uint8_t> hashes2;
	hashes2.resize(intensity * INITIAL_HASH_SIZE);
	for (uint32_t i = 0; i < intensity; ++i)
	{
		*(uint32_t*)(blockTemplate + 39) = i;
		blake2b(hashes2.data() + static_cast<size_t>(i) * INITIAL_HASH_SIZE, INITIAL_HASH_SIZE, blockTemplate, sizeof(blockTemplate), nullptr, 0);
	}
	*(uint32_t*)(blockTemplate + 39) = 0;

	if (hashes != hashes2)
	{
		std::cerr << "blake2b_initial_hash test failed!" << std::endl;
		return 1;
	}

	std::cout << "blake2b_initial_hash test passed" << std::endl;

	kernel = ctx.kernels[CL_FILLAES1RX4_SCRATCHPAD];
	if (!clSetKernelArgs(kernel, hash_gpu, scratchpads_gpu, static_cast<uint32_t>(intensity)))
	{
		return 1;
	}

	global_work_size = intensity * 4;
	local_work_size = 64;
	CL_CHECKED_CALL(clEnqueueNDRangeKernel, ctx.queue, kernel, 1, nullptr, &global_work_size, &local_work_size, 0, nullptr, nullptr);
	CL_CHECKED_CALL(clFinish, ctx.queue);

	struct Dummy
	{
		Dummy() {}

		uint64_t k;
	};
	std::vector<Dummy> scratchpads_buf(SCRATCHPAD_SIZE * (intensity + 1) / sizeof(Dummy));
	uint8_t* scratchpads = reinterpret_cast<uint8_t*>(scratchpads_buf.data());

	CL_CHECKED_CALL(clEnqueueReadBuffer, ctx.queue, hash_gpu, CL_TRUE, 0, intensity * INITIAL_HASH_SIZE, hashes.data(), 0, nullptr, nullptr);
	CL_CHECKED_CALL(clEnqueueReadBuffer, ctx.queue, scratchpads_gpu, CL_TRUE, 0, intensity * SCRATCHPAD_SIZE, scratchpads, 0, nullptr, nullptr);

	for (size_t i = 0; i < intensity; ++i)
	{
		fillAes1Rx4<false>(hashes2.data() + i * INITIAL_HASH_SIZE, SCRATCHPAD_SIZE, scratchpads + SCRATCHPAD_SIZE * intensity);

		if (memcmp(hashes.data() + i * INITIAL_HASH_SIZE, hashes2.data() + i * INITIAL_HASH_SIZE, INITIAL_HASH_SIZE) != 0)
		{
			std::cerr << "fillAes1Rx4_scratchpad test (hash) failed!" << std::endl;
			return 1;
		}

		const uint8_t* p1 = scratchpads + i * 64;
		const uint8_t* p2 = scratchpads + SCRATCHPAD_SIZE * intensity;
		for (int j = 0; j < SCRATCHPAD_SIZE; j += 64)
		{
			if (memcmp(p1 + j * intensity, p2 + j, 64) != 0)
			{
				std::cerr << "fillAes1Rx4_scratchpad test (scratchpad) failed!" << std::endl;
				return 1;
			}
		}
	}

	std::cout << "fillAes1Rx4_scratchpad test passed" << std::endl;

	kernel = ctx.kernels[CL_FILLAES1RX4_ENTROPY];
	if (!clSetKernelArgs(kernel, hash_gpu, entropy_gpu, static_cast<uint32_t>(intensity)))
	{
		return 1;
	}

	global_work_size = intensity * 4;
	local_work_size = 64;
	CL_CHECKED_CALL(clEnqueueNDRangeKernel, ctx.queue, kernel, 1, nullptr, &global_work_size, &local_work_size, 0, nullptr, nullptr);
	CL_CHECKED_CALL(clFinish, ctx.queue);

	std::vector<uint8_t> entropy(ENTROPY_SIZE * (intensity + 1));

	CL_CHECKED_CALL(clEnqueueReadBuffer, ctx.queue, hash_gpu, CL_TRUE, 0, intensity * INITIAL_HASH_SIZE, hashes.data(), 0, nullptr, nullptr);
	CL_CHECKED_CALL(clEnqueueReadBuffer, ctx.queue, entropy_gpu, CL_TRUE, 0, intensity * ENTROPY_SIZE, entropy.data(), 0, nullptr, nullptr);

	for (size_t i = 0; i < intensity; ++i)
	{
		fillAes1Rx4<false>(hashes2.data() + i * INITIAL_HASH_SIZE, ENTROPY_SIZE, entropy.data() + ENTROPY_SIZE * intensity);

		if (memcmp(hashes.data() + i * INITIAL_HASH_SIZE, hashes2.data() + i * INITIAL_HASH_SIZE, INITIAL_HASH_SIZE) != 0)
		{
			std::cerr << "fillAes1Rx4_entropy test (hash) failed!" << std::endl;
			return 1;
		}

		if (memcmp(entropy.data() + i * ENTROPY_SIZE, entropy.data() + ENTROPY_SIZE * intensity, ENTROPY_SIZE) != 0)
		{
			std::cerr << "fillAes1Rx4_entropy test (entropy) failed!" << std::endl;
			return 1;
		}
	}

	std::cout << "fillAes1Rx4_entropy test passed" << std::endl;

	kernel = ctx.kernels[CL_HASHAES1RX4];
	if (!clSetKernelArgs(kernel, scratchpads_gpu, registers_gpu, static_cast<uint32_t>(intensity)))
	{
		return 1;
	}

	global_work_size = intensity * 4;
	local_work_size = 64;
	CL_CHECKED_CALL(clEnqueueNDRangeKernel, ctx.queue, kernel, 1, nullptr, &global_work_size, &local_work_size, 0, nullptr, nullptr);
	CL_CHECKED_CALL(clFinish, ctx.queue);

	std::vector<uint8_t> registers(REGISTERS_SIZE * (intensity + 1));

	CL_CHECKED_CALL(clEnqueueReadBuffer, ctx.queue, registers_gpu, CL_TRUE, 0, intensity * REGISTERS_SIZE, registers.data(), 0, nullptr, nullptr);

	for (size_t i = 0; i < intensity; ++i)
	{
		memset(registers.data() + REGISTERS_SIZE * intensity, 0, REGISTERS_SIZE);
		uint8_t* src = scratchpads + i * 64;
		uint8_t* dst = scratchpads + SCRATCHPAD_SIZE * intensity;
		for (size_t j = 0; j < SCRATCHPAD_SIZE; j += 64)
		{
			memcpy(dst, src, 64);
			dst += 64;
			src += intensity * 64;
		}

		hashAes1Rx4<false>(scratchpads + SCRATCHPAD_SIZE * intensity, SCRATCHPAD_SIZE, registers.data() + intensity * REGISTERS_SIZE + 192);

		if (memcmp(registers.data() + i * REGISTERS_SIZE, registers.data() + intensity * REGISTERS_SIZE, REGISTERS_SIZE) != 0)
		{
			std::cerr << "hashAes1Rx4 test failed!" << std::endl;
			return 1;
		}
	}

	std::cout << "hashAes1Rx4 test passed" << std::endl;

	kernel = ctx.kernels[CL_BLAKE2B_HASH_REGISTERS_32];
	if (!clSetKernelArgs(kernel, hash_gpu, registers_gpu))
	{
		return 1;
	}

	global_work_size = intensity;
	local_work_size = 64;
	CL_CHECKED_CALL(clEnqueueNDRangeKernel, ctx.queue, kernel, 1, nullptr, &global_work_size, &local_work_size, 0, nullptr, nullptr);
	CL_CHECKED_CALL(clFinish, ctx.queue);

	CL_CHECKED_CALL(clEnqueueReadBuffer, ctx.queue, hash_gpu, CL_TRUE, 0, intensity * 32, hashes.data(), 0, nullptr, nullptr);

	for (size_t i = 0; i < intensity; ++i)
	{
		blake2b(hashes2.data() + i * 32, 32, registers.data() + i * REGISTERS_SIZE, REGISTERS_SIZE, nullptr, 0);
	}

	if (memcmp(hashes.data(), hashes2.data(), intensity * 32) != 0)
	{
		std::cerr << "blake2b_hash_registers (32 byte hash) test failed!" << std::endl;
		return 1;
	}

	std::cout << "blake2b_hash_registers (32 byte hash) test passed" << std::endl;

	kernel = ctx.kernels[CL_BLAKE2B_HASH_REGISTERS_64];
	if (!clSetKernelArgs(kernel, hash_gpu, registers_gpu))
	{
		return 1;
	}

	global_work_size = intensity;
	local_work_size = 64;
	CL_CHECKED_CALL(clEnqueueNDRangeKernel, ctx.queue, kernel, 1, nullptr, &global_work_size, &local_work_size, 0, nullptr, nullptr);
	CL_CHECKED_CALL(clFinish, ctx.queue);

	CL_CHECKED_CALL(clEnqueueReadBuffer, ctx.queue, hash_gpu, CL_TRUE, 0, intensity * 64, hashes.data(), 0, nullptr, nullptr);

	for (size_t i = 0; i < intensity; ++i)
	{
		blake2b(hashes2.data() + i * 64, 64, registers.data() + i * REGISTERS_SIZE, REGISTERS_SIZE, nullptr, 0);
	}

	if (memcmp(hashes.data(), hashes2.data(), intensity * 64) != 0)
	{
		std::cerr << "blake2b_hash_registers (64 byte hash) test failed!" << std::endl;
		return 1;
	}

	std::cout << "blake2b_hash_registers (64 byte hash) test passed" << std::endl;

	auto start_time = high_resolution_clock::now();

	kernel = ctx.kernels[CL_FILLAES1RX4_SCRATCHPAD];
	for (int i = 0; i < 100; ++i)
	{
		std::cout << "Benchmarking fillAes1Rx4 " << (i + 1) << "/100";
		if (i > 0)
		{
			const double dt = duration_cast<nanoseconds>(high_resolution_clock::now() - start_time).count() / 1e9;
			std::cout << ", " << ((i * intensity * 10) / dt) << " scratchpads/s  ";
		}
		std::cout << "\r";

		global_work_size = intensity * 4;
		local_work_size = 64;
		for (int j = 0; j < 10; ++j)
			CL_CHECKED_CALL(clEnqueueNDRangeKernel, ctx.queue, kernel, 1, nullptr, &global_work_size, &local_work_size, 0, nullptr, nullptr);

		CL_CHECKED_CALL(clFinish, ctx.queue);
	}
	std::cout << std::endl;

	start_time = high_resolution_clock::now();

	kernel = ctx.kernels[CL_HASHAES1RX4];
	for (int i = 0; i < 100; ++i)
	{
		std::cout << "Benchmarking hashAes1Rx4 " << (i + 1) << "/100";
		if (i > 0)
		{
			const double dt = duration_cast<nanoseconds>(high_resolution_clock::now() - start_time).count() / 1e9;
			std::cout << ", " << ((i * intensity * 10) / dt) << " scratchpads/s  ";
		}
		std::cout << "\r";

		global_work_size = intensity * 4;
		local_work_size = 64;
		for (int j = 0; j < 10; ++j)
			CL_CHECKED_CALL(clEnqueueNDRangeKernel, ctx.queue, kernel, 1, nullptr, &global_work_size, &local_work_size, 0, nullptr, nullptr);

		CL_CHECKED_CALL(clFinish, ctx.queue);
	}
	std::cout << std::endl;

	CL_CHECKED_CALL(clEnqueueWriteBuffer, ctx.queue, blockTemplate_gpu, CL_FALSE, 0, sizeof(blockTemplate), blockTemplate, 0, nullptr, nullptr);

	kernel = ctx.kernels[CL_BLAKE2B_512_SINGLE_BLOCK_BENCH];
	if (!clSetKernelArgs(kernel, nonce_gpu, blockTemplate_gpu, 0ULL))
	{
		return 1;
	}

	start_time = high_resolution_clock::now();

	for (uint64_t start_nonce = 0; start_nonce < BLAKE2B_STEP * 100; start_nonce += BLAKE2B_STEP)
	{
		std::cout << "Benchmarking blake2b_512_single_block " << ((start_nonce + BLAKE2B_STEP) / BLAKE2B_STEP) << "/100";
		if (start_nonce > 0)
		{
			const double dt = duration_cast<nanoseconds>(high_resolution_clock::now() - start_time).count() / 1e9;
			std::cout << ", " << start_nonce / dt / 1e6 << " MH/s   ";
		}
		std::cout << "\r";

		CL_CHECKED_CALL(clEnqueueFillBuffer, ctx.queue, nonce_gpu, &zero, sizeof(zero), 0, sizeof(uint64_t), 0, nullptr, nullptr);

		CL_CHECKED_CALL(clSetKernelArg, kernel, 2, sizeof(start_nonce), &start_nonce);

		global_work_size = BLAKE2B_STEP;
		local_work_size = 64;
		CL_CHECKED_CALL(clEnqueueNDRangeKernel, ctx.queue, kernel, 1, nullptr, &global_work_size, &local_work_size, 0, nullptr, nullptr);
		CL_CHECKED_CALL(clFinish, ctx.queue);

		uint64_t nonce;
		CL_CHECKED_CALL(clEnqueueReadBuffer, ctx.queue, nonce_gpu, CL_TRUE, 0, sizeof(uint64_t), &nonce, 0, nullptr, nullptr);

		if (nonce)
		{
			*(uint64_t*)(blockTemplate) = nonce;
			uint64_t hash[INITIAL_HASH_SIZE / sizeof(uint64_t)];
			blake2b(hash, INITIAL_HASH_SIZE, blockTemplate, sizeof(blockTemplate), nullptr, 0);
			std::cout << "nonce = " << nonce << ", hash[7] = " << std::hex << std::setw(16) << std::setfill('0') << hash[7] << "                  " << std::endl;
			std::cout << std::dec;
		}
	}
	std::cout << std::endl;

	kernel = ctx.kernels[CL_BLAKE2B_512_DOUBLE_BLOCK_BENCH];
	if (!clSetKernelArgs(kernel, nonce_gpu, registers_gpu, 0ULL))
	{
		return 1;
	}

	CL_CHECKED_CALL(clEnqueueFillBuffer, ctx.queue, registers_gpu, &zero, sizeof(zero), 0, REGISTERS_SIZE, 0, nullptr, nullptr);

	start_time = high_resolution_clock::now();

	for (uint64_t start_nonce = 0; start_nonce < BLAKE2B_STEP * 100; start_nonce += BLAKE2B_STEP)
	{
		std::cout << "Benchmarking blake2b_512_double_block " << ((start_nonce + BLAKE2B_STEP) / BLAKE2B_STEP) << "/100";
		if (start_nonce > 0)
		{
			const double dt = duration_cast<nanoseconds>(high_resolution_clock::now() - start_time).count() / 1e9;
			std::cout << ", " << start_nonce / dt / 1e6 << " MH/s   ";
		}
		std::cout << "\r";

		CL_CHECKED_CALL(clEnqueueFillBuffer, ctx.queue, nonce_gpu, &zero, sizeof(zero), 0, sizeof(uint64_t), 0, nullptr, nullptr);

		CL_CHECKED_CALL(clSetKernelArg, kernel, 2, sizeof(start_nonce), &start_nonce);

		global_work_size = BLAKE2B_STEP;
		local_work_size = 64;
		CL_CHECKED_CALL(clEnqueueNDRangeKernel, ctx.queue, kernel, 1, nullptr, &global_work_size, &local_work_size, 0, nullptr, nullptr);
		CL_CHECKED_CALL(clFinish, ctx.queue);

		uint64_t nonce;
		CL_CHECKED_CALL(clEnqueueReadBuffer, ctx.queue, nonce_gpu, CL_TRUE, 0, sizeof(uint64_t), &nonce, 0, nullptr, nullptr);

		if (nonce)
		{
			memset(registers.data(), 0, REGISTERS_SIZE);
			*(uint64_t*)(registers.data()) = nonce;
			uint64_t hash[8];
			blake2b(hash, 64, registers.data(), REGISTERS_SIZE, nullptr, 0);
			std::cout << "nonce = " << nonce << ", hash[7] = " << std::hex << std::setw(16) << std::setfill('0') << hash[7] << "                  " << std::endl;
			std::cout << std::dec;
		}
	}
	std::cout << std::endl;

	return 0;
}
