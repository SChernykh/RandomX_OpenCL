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

#pragma once

#include <iostream>
#include <stdint.h>
#include <string>
#include <map>
#include <CL/cl.h>

#define CL_CHECK_RESULT(func) \
	if (err != CL_SUCCESS) \
	{ \
		std::cerr << #func##" failed: error " << err << std::endl; \
		return false; \
	}

#define CL_CHECKED_CALL(func, ...) \
	err = func(__VA_ARGS__); \
	CL_CHECK_RESULT(func);

struct OpenCLContext
{
	OpenCLContext()
		: context(0)
		, queue(0)
		, program(0)
	{}

	bool Init(uint32_t platform_id, uint32_t device_id, const std::initializer_list<std::string>& source_files, const std::initializer_list<std::string>& kernel_names);

	cl_context context;
	cl_command_queue queue;
	cl_program program;

	std::vector<char> device_name;
	cl_ulong device_global_mem_size;
	cl_ulong device_local_mem_size;
	cl_uint device_freq;
	cl_uint device_compute_units;
	cl_ulong device_max_alloc_size;
	std::vector<char> device_vendor;
	std::vector<char> device_version;
	std::vector<char> device_driver_version;
	std::vector<char> device_extensions;

	std::map<std::string, cl_kernel> kernels;
};

struct DevicePtr
{
	DevicePtr(const OpenCLContext& ctx, size_t size) : p(static_cast<cl_mem>(0)) { Init(ctx, size); }
	~DevicePtr() { if (p) clReleaseMemObject(p); }

	bool Init(const OpenCLContext& ctx, size_t size)
	{
		cl_int err;
		p = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, size, nullptr, &err);
		CL_CHECK_RESULT(clCreateBuffer);
		return true;
	}

	operator cl_mem() const { return p; }

private:
	cl_mem p;
};

static_assert(sizeof(DevicePtr) == sizeof(cl_mem), "Invalid DevicePtr struct, check your compiler options");

template<cl_uint> bool _clSetKernelArg(cl_kernel) { return true; }

template<cl_uint index, typename T, typename... Args>
bool _clSetKernelArg(cl_kernel kernel, T&& value, Args&& ... args)
{
	cl_int err;
	CL_CHECKED_CALL(clSetKernelArg, kernel, index, sizeof(T), &value);
	return _clSetKernelArg<index + 1>(kernel, std::forward<Args>(args)...);
}

template<typename... Args>
bool clSetKernelArgs(cl_kernel kernel, Args&& ... args)
{
	return _clSetKernelArg<0>(kernel, std::forward<Args>(args)...);
}
