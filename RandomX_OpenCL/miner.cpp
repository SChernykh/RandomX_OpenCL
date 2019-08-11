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

#include <algorithm>
#include <chrono>
#include <atomic>
#include <sstream>
#include <cctype>
#include "miner.h"
#include "opencl_helpers.h"
#include "definitions.h"

#include "../RandomX/src/randomx.h"
#include "../RandomX/src/configuration.h"
#include "../RandomX/src/common.hpp"

using namespace std::chrono;

bool test_mining(uint32_t platform_id, uint32_t device_id, size_t intensity, uint32_t start_nonce, uint32_t workers_per_hash, uint32_t bfactor, bool portable, bool dataset_host_allocated, bool validate)
{
	std::cout << "Initializing GPU #" << device_id << " on OpenCL platform #" << platform_id << std::endl << std::endl;

	OpenCLContext ctx;
	if (!ctx.Init(platform_id, device_id))
	{
		return false;
	}

	if (!ctx.Compile("base_kernels.bin",
		{
			AES_CL,
			BLAKE2B_CL
		},
		{
			CL_FILLAES1RX4_SCRATCHPAD,
			CL_FILLAES4RX4_ENTROPY,
			CL_HASHAES1RX4,
			CL_BLAKE2B_INITIAL_HASH,
			CL_BLAKE2B_HASH_REGISTERS_32,
			CL_BLAKE2B_HASH_REGISTERS_64,
			CL_BLAKE2B_512_SINGLE_BLOCK_BENCH,
			CL_BLAKE2B_512_DOUBLE_BLOCK_BENCH
		},
		"", COMPILE_CACHE_BINARY))
	{
		return false;
	}

	if (portable)
	{
		switch (workers_per_hash)
		{
		case 2:
		case 4:
		case 8:
		case 16:
			break;

		default:
			workers_per_hash = 8;
			break;
		}

		if (bfactor > 10)
			bfactor = 10;

		std::stringstream options;
		options << "-D WORKERS_PER_HASH=" << workers_per_hash << " -Werror";
		if (!ctx.Compile("randomx_vm.bin", { RANDOMX_VM_CL }, { CL_INIT_VM, CL_EXECUTE_VM }, options.str(), COMPILE_CACHE_BINARY))
		{
			return false;
		}
	}
	else
	{
		const char* gcn_binary = "randomx_run_gfx803.bin";
		int gcn_version = 12;

		std::vector<char> t;
		std::transform(ctx.device_name.begin(), ctx.device_name.end(), std::back_inserter(t), [](char c) { return static_cast<char>(std::toupper(c)); });
		if (strcmp(t.data(), "GFX900") == 0)
		{
			gcn_binary = "randomx_run_gfx900.bin";
			gcn_version = 14;
		}

		std::stringstream options;
		options << "-D GCN_VERSION=" << gcn_version;
		if (!ctx.Compile("randomx_init.bin", { RANDOMX_INIT_CL }, { CL_RANDOMX_INIT }, options.str(), ALWAYS_COMPILE))
		{
			return false;
		}

		options.str("");
		options << "-D RANDOMX_PROGRAM_ITERATIONS=" << RANDOMX_PROGRAM_ITERATIONS;
		if (!ctx.Compile(gcn_binary, { RANDOMX_RUN_CL }, { CL_RANDOMX_RUN }, options.str(), ALWAYS_USE_BINARY, ctx.elf_binary_flags))
		{
			return false;
		}
	}

	if (!intensity)
		intensity = std::min(ctx.device_max_alloc_size, ctx.device_global_mem_size) / RANDOMX_SCRATCHPAD_L3;

	intensity -= (intensity & 63);

	const size_t dataset_size = randomx_dataset_item_count() * RANDOMX_DATASET_ITEM_SIZE;
	cl_int err;
	cl_mem dataset_gpu = nullptr;
	if (!dataset_host_allocated)
	{
		dataset_gpu = clCreateBuffer(ctx.context, CL_MEM_READ_ONLY, dataset_size, nullptr, &err);
		CL_CHECK_RESULT(clCreateBuffer);
		std::cout << "Allocated " << (dataset_size / 1048576.0) << " MB dataset on GPU" << std::endl;
	}
	std::cout << "Initializing dataset...";

	randomx_dataset *myDataset;
	bool large_pages_available = true;
	{
		auto t1 = high_resolution_clock::now();

		myDataset = randomx_alloc_dataset(RANDOMX_FLAG_LARGE_PAGES);
		if (!myDataset)
		{
			std::cout << "\nCouldn't allocate dataset using large pages" << std::endl;
			myDataset = randomx_alloc_dataset(RANDOMX_FLAG_DEFAULT);
			large_pages_available = false;
		}

		char* dataset_memory = reinterpret_cast<char*>(randomx_get_dataset_memory(myDataset));
		bool read_ok = false;

		FILE* fp = fopen("dataset.bin", "rb");
		if (fp)
		{
			read_ok = (fread(dataset_memory, 1, randomx::DatasetSize, fp) == randomx::DatasetSize);
			fclose(fp);
		}

		if (!read_ok)
		{
			randomx_cache *myCache = randomx_alloc_cache((randomx_flags)(RANDOMX_FLAG_JIT | (large_pages_available ? RANDOMX_FLAG_LARGE_PAGES : 0)));
			if (!myCache)
			{
				std::cout << "\nCouldn't allocate cache using large pages" << std::endl;
				myCache = randomx_alloc_cache(RANDOMX_FLAG_JIT);
				large_pages_available = false;
			}

			const char mySeed[] = "RandomX example seed";
			randomx_init_cache(myCache, mySeed, sizeof(mySeed));

			std::vector<SThread> threads;
			for (uint32_t i = 0, n = std::thread::hardware_concurrency(); i < n; ++i)
				threads.emplace_back([myDataset, myCache, i, n]() { randomx_init_dataset(myDataset, myCache, (i * randomx_dataset_item_count()) / n, ((i + 1) * randomx_dataset_item_count()) / n - (i * randomx_dataset_item_count()) / n); });

			for (auto& t : threads)
				t.join();

			randomx_release_cache(myCache);

			fp = fopen("dataset.bin", "wb");
			if (fp)
			{
				fwrite(dataset_memory, 1, randomx::DatasetSize, fp);
				fclose(fp);
			}
		}

		if (!dataset_host_allocated)
		{
			CL_CHECKED_CALL(clEnqueueWriteBuffer, ctx.queue, dataset_gpu, CL_TRUE, 0, dataset_size, randomx_get_dataset_memory(myDataset), 0, nullptr, nullptr);
		}

		std::cout << "done in " << (duration_cast<nanoseconds>(high_resolution_clock::now() - t1).count() / 1e9) << " seconds" << std::endl;
	}

	if (dataset_host_allocated)
	{
		dataset_gpu = clCreateBuffer(ctx.context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, dataset_size, randomx_get_dataset_memory(myDataset), &err);
		CL_CHECK_RESULT(clCreateBuffer);
		std::cout << "Using host-allocated " << (dataset_size / 1048576.0) << " MB dataset" << std::endl;
	}

	ALLOCATE_DEVICE_MEMORY(scratchpads_gpu, ctx, intensity * (RANDOMX_SCRATCHPAD_L3 + 64));
	std::cout << "Allocated " << intensity << " scratchpads\n" << std::endl;

	ALLOCATE_DEVICE_MEMORY(hashes_gpu, ctx, intensity * INITIAL_HASH_SIZE);
	ALLOCATE_DEVICE_MEMORY(entropy_gpu, ctx, intensity * ENTROPY_SIZE);
	ALLOCATE_DEVICE_MEMORY(vm_states_gpu, ctx, portable ? (intensity * VM_STATE_SIZE) : (intensity * REGISTERS_SIZE));
	ALLOCATE_DEVICE_MEMORY(rounding_gpu, ctx, intensity * sizeof(uint32_t));
	ALLOCATE_DEVICE_MEMORY(blocktemplate_gpu, ctx, intensity * sizeof(blockTemplate));
	ALLOCATE_DEVICE_MEMORY(intermediate_programs_gpu, ctx, portable ? 0 : (intensity * INTERMEDIATE_PROGRAM_SIZE));
	ALLOCATE_DEVICE_MEMORY(compiled_programs_gpu, ctx, portable ? 0 : ((intensity / HASHES_PER_GROUP) * COMPILED_PROGRAM_SIZE));

	CL_CHECKED_CALL(clEnqueueWriteBuffer, ctx.queue, blocktemplate_gpu, CL_TRUE, 0, sizeof(blockTemplate), blockTemplate, 0, nullptr, nullptr);

	auto prev_time = high_resolution_clock::now();

	std::vector<uint8_t> hashes, hashes_check;
	hashes.resize(intensity * 32);
	hashes_check.resize(intensity * 32);

	std::vector<SThread> threads;
	std::atomic<uint32_t> nonce_counter;
	bool cpu_limited = false;

	uint32_t failed_nonces = 0;

	cl_kernel kernel_blake2b_initial_hash = ctx.kernels[CL_BLAKE2B_INITIAL_HASH];
	if (!clSetKernelArgs(kernel_blake2b_initial_hash, hashes_gpu, blocktemplate_gpu, 0U))
	{
		return false;
	}

	cl_kernel kernel_fillaes1rx4_scratchpad = ctx.kernels[CL_FILLAES1RX4_SCRATCHPAD];
	if (!clSetKernelArgs(kernel_fillaes1rx4_scratchpad, hashes_gpu, scratchpads_gpu, static_cast<uint32_t>(intensity)))
	{
		return false;
	}

	cl_kernel kernel_fillaes1rx4_entropy = ctx.kernels[CL_FILLAES4RX4_ENTROPY];
	if (!clSetKernelArgs(kernel_fillaes1rx4_entropy, hashes_gpu, entropy_gpu, static_cast<uint32_t>(intensity)))
	{
		return false;
	}

	cl_kernel kernel_randomx_init, kernel_randomx_run;
	if (portable)
	{
		kernel_randomx_init = ctx.kernels[CL_INIT_VM];
		if (!clSetKernelArgs(kernel_randomx_init, entropy_gpu, vm_states_gpu))
		{
			return false;
		}

		kernel_randomx_run = ctx.kernels[CL_EXECUTE_VM];
		if (!clSetKernelArgs(kernel_randomx_run, vm_states_gpu, rounding_gpu, scratchpads_gpu, dataset_gpu, static_cast<uint32_t>(intensity), static_cast<uint32_t>(RANDOMX_PROGRAM_ITERATIONS >> bfactor), 1U, 1U))
		{
			return false;
		}
	}
	else
	{
		kernel_randomx_init = ctx.kernels[CL_RANDOMX_INIT];
		if (!clSetKernelArgs(kernel_randomx_init, entropy_gpu, vm_states_gpu, intermediate_programs_gpu, compiled_programs_gpu, static_cast<uint32_t>(intensity)))
		{
			return false;
		}

		kernel_randomx_run = ctx.kernels[CL_RANDOMX_RUN];

		constexpr uint32_t rx_parameters =
			(PowerOf2(RANDOMX_SCRATCHPAD_L1) << 0) |
			(PowerOf2(RANDOMX_SCRATCHPAD_L2) << 5) |
			(PowerOf2(RANDOMX_SCRATCHPAD_L3) << 10) |
			(PowerOf2(RANDOMX_PROGRAM_ITERATIONS) << 15);

		if (!clSetKernelArgs(kernel_randomx_run, dataset_gpu, scratchpads_gpu, vm_states_gpu, rounding_gpu, compiled_programs_gpu, static_cast<uint32_t>(intensity), rx_parameters))
		{
			return false;
		}
	}

	cl_kernel kernel_hashaes1rx4 = ctx.kernels[CL_HASHAES1RX4];
	if (!clSetKernelArgs(kernel_hashaes1rx4, scratchpads_gpu, vm_states_gpu, 192U, static_cast<uint32_t>(portable ? VM_STATE_SIZE : REGISTERS_SIZE), static_cast<uint32_t>(intensity)))
	{
		return false;
	}

	cl_kernel kernel_blake2b_hash_registers_32 = ctx.kernels[CL_BLAKE2B_HASH_REGISTERS_32];
	if (!clSetKernelArgs(kernel_blake2b_hash_registers_32, hashes_gpu, vm_states_gpu, static_cast<uint32_t>(portable ? VM_STATE_SIZE : REGISTERS_SIZE)))
	{
		return false;
	}

	cl_kernel kernel_blake2b_hash_registers_64 = ctx.kernels[CL_BLAKE2B_HASH_REGISTERS_64];
	if (!clSetKernelArgs(kernel_blake2b_hash_registers_64, hashes_gpu, vm_states_gpu, static_cast<uint32_t>(portable ? VM_STATE_SIZE : REGISTERS_SIZE)))
	{
		return false;
	}

	const size_t global_work_size = intensity;
	const size_t global_work_size4 = intensity * 4;
	const size_t global_work_size8 = intensity * 8;
	const size_t global_work_size16 = intensity * 16;
	const size_t global_work_size64 = intensity * 64;
	const size_t local_work_size = 64;
	const size_t local_work_size32 = 32;
	const size_t local_work_size16 = 16;
	const uint32_t zero = 0;

	for (size_t nonce = start_nonce, k = 0; nonce < 0xFFFFFFFFUL; nonce += intensity, ++k)
	{
		auto validation_thread = [&nonce_counter, myDataset, &hashes_check, intensity, nonce, &large_pages_available]() {
			const randomx_flags flags = (randomx_flags)(RANDOMX_FLAG_FULL_MEM | RANDOMX_FLAG_JIT | RANDOMX_FLAG_HARD_AES);
			randomx_vm *myMachine = randomx_create_vm((randomx_flags)(flags | (large_pages_available ? RANDOMX_FLAG_LARGE_PAGES : 0)), nullptr, myDataset);

			if (!myMachine && large_pages_available)
			{
				large_pages_available = false;
				myMachine = randomx_create_vm(flags, nullptr, myDataset);
			}

			uint8_t buf[sizeof(blockTemplate)];
			memcpy(buf, blockTemplate, sizeof(buf));

			for (;;)
			{
				const uint32_t i = nonce_counter.fetch_add(1);
				if (i >= intensity)
					break;

				*(uint32_t*)(buf + 39) = static_cast<uint32_t>(nonce + i);

				randomx_calculate_hash(myMachine, buf, sizeof(buf), (hashes_check.data() + i * 32));
			}
			randomx_destroy_vm(myMachine);
		};

		if (validate)
		{
			nonce_counter = 0;

			const uint32_t n = std::max(std::thread::hardware_concurrency() / 2, 1U);

			threads.clear();
			for (uint32_t i = 0; i < n; ++i)
				threads.emplace_back(validation_thread);
		}

		auto cur_time = high_resolution_clock::now();
		if (k > 0)
		{
			const double dt = duration_cast<nanoseconds>(cur_time - prev_time).count() / 1e9;

			if (validate)
			{
				const size_t n = nonce - start_nonce;
				printf("%zu (%.3f%%) hashes validated successfully, %u (%.3f%%) hashes failed, %.0f h/s%s\n",
					n - failed_nonces,
					static_cast<double>(n - failed_nonces) / n * 100.0,
					failed_nonces,
					static_cast<double>(failed_nonces) / n * 100.0,
					intensity / dt,
					cpu_limited ? ", limited by CPU" : "                "
				);
			}
			else
			{
				printf("%.0f h/s\t\r", intensity / dt);
			}
		}
		prev_time = cur_time;

		CL_CHECKED_CALL(clSetKernelArg, kernel_blake2b_initial_hash, 2, sizeof(uint32_t), &nonce);
		CL_CHECKED_CALL(clEnqueueNDRangeKernel, ctx.queue, kernel_blake2b_initial_hash, 1, nullptr, &global_work_size, &local_work_size, 0, nullptr, nullptr);
		CL_CHECKED_CALL(clEnqueueNDRangeKernel, ctx.queue, kernel_fillaes1rx4_scratchpad, 1, nullptr, &global_work_size4, &local_work_size, 0, nullptr, nullptr);
		CL_CHECKED_CALL(clEnqueueFillBuffer, ctx.queue, rounding_gpu, &zero, sizeof(zero), 0, intensity * sizeof(uint32_t), 0, nullptr, nullptr);

		for (size_t i = 0; i < RANDOMX_PROGRAM_COUNT; ++i)
		{
			CL_CHECKED_CALL(clEnqueueNDRangeKernel, ctx.queue, kernel_fillaes1rx4_entropy, 1, nullptr, &global_work_size4, &local_work_size, 0, nullptr, nullptr);
			CL_CHECKED_CALL(clEnqueueNDRangeKernel, ctx.queue, kernel_randomx_init, 1, nullptr, portable ? &global_work_size8 : &global_work_size, portable ? &local_work_size32 : &local_work_size, 0, nullptr, nullptr);
			if (portable)
			{
				uint32_t first = 1;
				uint32_t last = 0;
				CL_CHECKED_CALL(clSetKernelArg, kernel_randomx_run, 6, sizeof(uint32_t), &first);
				CL_CHECKED_CALL(clSetKernelArg, kernel_randomx_run, 7, sizeof(uint32_t), &last);
				for (int j = 0, n = 1 << bfactor; j < n; ++j)
				{
					if (j == n - 1)
					{
						last = 1;
						CL_CHECKED_CALL(clSetKernelArg, kernel_randomx_run, 7, sizeof(uint32_t), &last);
					}

					CL_CHECKED_CALL(clEnqueueNDRangeKernel, ctx.queue, kernel_randomx_run, 1, nullptr, (workers_per_hash == 16) ? &global_work_size16 : &global_work_size8, (workers_per_hash == 16) ? &local_work_size32 : &local_work_size16, 0, nullptr, nullptr);

					if (j == 0)
					{
						first = 0;
						CL_CHECKED_CALL(clSetKernelArg, kernel_randomx_run, 6, sizeof(uint32_t), &first);
					}
				}
			}
			else
			{
				//if (i == 0)
				//{
				//	CL_CHECKED_CALL(clFinish, ctx.queue);
				//	std::vector<char> buf((intensity / HASHES_PER_GROUP) * COMPILED_PROGRAM_SIZE);
				//	CL_CHECKED_CALL(clEnqueueReadBuffer, ctx.queue, compiled_programs_gpu, CL_TRUE, 0, buf.size(), buf.data(), 0, nullptr, nullptr);
				//	FILE* fp;
				//	fopen_s(&fp, "compiled_program.bin", "wb");
				//	fwrite(buf.data(), 1, buf.size(), fp);
				//	fclose(fp);
				//	return false;
				//}
				CL_CHECKED_CALL(clFinish, ctx.queue);
				CL_CHECKED_CALL(clEnqueueNDRangeKernel, ctx.queue, kernel_randomx_run, 1, nullptr, &global_work_size64, &local_work_size, 0, nullptr, nullptr);
			}

			if (i == RANDOMX_PROGRAM_COUNT - 1)
			{
				CL_CHECKED_CALL(clEnqueueNDRangeKernel, ctx.queue, kernel_hashaes1rx4, 1, nullptr, &global_work_size4, &local_work_size, 0, nullptr, nullptr);
				CL_CHECKED_CALL(clEnqueueNDRangeKernel, ctx.queue, kernel_blake2b_hash_registers_32, 1, nullptr, &global_work_size, &local_work_size, 0, nullptr, nullptr);
			}
			else
			{
				CL_CHECKED_CALL(clEnqueueNDRangeKernel, ctx.queue, kernel_blake2b_hash_registers_64, 1, nullptr, &global_work_size, &local_work_size, 0, nullptr, nullptr);
			}
		}

		CL_CHECKED_CALL(clFinish, ctx.queue);

		if (validate)
		{
			CL_CHECKED_CALL(clEnqueueReadBuffer, ctx.queue, hashes_gpu, CL_TRUE, 0, intensity * 32, hashes.data(), 0, nullptr, nullptr);

			cpu_limited = nonce_counter.load() < intensity;

			for (auto& thread : threads)
				thread.join();

			if (memcmp(hashes.data(), hashes_check.data(), intensity * 32) != 0)
			{
				for (uint32_t i = 0; i < intensity * 32; i += 32)
				{
					if (memcmp(hashes.data() + i, hashes_check.data() + i, 32))
					{
						std::cerr << "CPU validation error, failing nonce = " << (nonce + i / 32) << std::endl;
						++failed_nonces;
					}
				}
			}
		}
	}

	return true;
}
