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

#include <iostream>
#include <fstream>
#include <vector>
#include "opencl_helpers.h"

OpenCLContext::~OpenCLContext()
{
	for (auto& k : kernels)
		clReleaseKernel(k.second);

	clReleaseCommandQueue(queue);
	clReleaseContext(context);
}

bool OpenCLContext::Init(uint32_t platform_id, uint32_t device_id)
{
	cl_int err;

	cl_platform_id platforms[4];
	cl_uint num_platforms;
	CL_CHECKED_CALL(clGetPlatformIDs, 4, platforms, &num_platforms);

	if (platform_id >= num_platforms)
	{
		std::cerr << "Invalid platform ID (" << platform_id << "), " << num_platforms << " OpenCL platforms available" << std::endl;
		return false;
	}

	cl_device_id devices[32];
	cl_uint num_devices;
	CL_CHECKED_CALL(clGetDeviceIDs, platforms[platform_id], CL_DEVICE_TYPE_GPU, 32, devices, &num_devices);

	if (device_id >= num_devices)
	{
		std::cerr << "Invalid device ID (" << device_id << "), " << num_devices << " OpenCL GPU devices available" << std::endl;
		return false;
	}

	device = devices[device_id];

	size_t size;

	CL_CHECKED_CALL(clGetDeviceInfo, device, CL_DEVICE_NAME, 0, nullptr, &size);
	device_name.resize(size);
	CL_CHECKED_CALL(clGetDeviceInfo, device, CL_DEVICE_NAME, size, device_name.data(), nullptr);

	CL_CHECKED_CALL(clGetDeviceInfo, device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(device_global_mem_size), &device_global_mem_size, nullptr);
	CL_CHECKED_CALL(clGetDeviceInfo, device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(device_local_mem_size), &device_local_mem_size, nullptr);
	CL_CHECKED_CALL(clGetDeviceInfo, device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(device_freq), &device_freq, nullptr);
	CL_CHECKED_CALL(clGetDeviceInfo, device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(device_compute_units), &device_compute_units, nullptr);
	CL_CHECKED_CALL(clGetDeviceInfo, device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(device_max_alloc_size), &device_max_alloc_size, nullptr);

	CL_CHECKED_CALL(clGetDeviceInfo, device, CL_DEVICE_VENDOR, 0, nullptr, &size);
	device_vendor.resize(size);
	CL_CHECKED_CALL(clGetDeviceInfo, device, CL_DEVICE_VENDOR, size, device_vendor.data(), nullptr);

	CL_CHECKED_CALL(clGetDeviceInfo, device, CL_DEVICE_VERSION, 0, nullptr, &size);
	device_version.resize(size);
	CL_CHECKED_CALL(clGetDeviceInfo, device, CL_DEVICE_VERSION, size, device_version.data(), nullptr);

	CL_CHECKED_CALL(clGetDeviceInfo, device, CL_DRIVER_VERSION, 0, nullptr, &size);
	device_driver_version.resize(size);
	CL_CHECKED_CALL(clGetDeviceInfo, device, CL_DRIVER_VERSION, size, device_driver_version.data(), nullptr);

	CL_CHECKED_CALL(clGetDeviceInfo, device, CL_DEVICE_EXTENSIONS, 0, nullptr, &size);
	device_extensions.resize(size);
	CL_CHECKED_CALL(clGetDeviceInfo, device, CL_DEVICE_EXTENSIONS, size, device_extensions.data(), nullptr);

	std::cout << "Device name:    " << device_name.data() << std::endl;
	std::cout << "Device vendor:  " << device_vendor.data() << std::endl;
	std::cout << "Global memory:  " << (device_global_mem_size >> 20) << " MB" << std::endl;
	std::cout << "Local memory:   " << (device_local_mem_size >> 10) << " KB" << std::endl;
	std::cout << "Clock speed:    " << device_freq << " MHz" << std::endl;
	std::cout << "Compute units:  " << device_compute_units << std::endl;
	std::cout << "OpenCL version: " << device_version.data() << std::endl;
	std::cout << "Driver version: " << device_driver_version.data() << std::endl;
	std::cout << "Extensions:     " << device_extensions.data() << std::endl << std::endl;

	context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
	CL_CHECK_RESULT(clCreateContext);

	queue = clCreateCommandQueue(context, device, 0, &err);
	CL_CHECK_RESULT(clCreateCommandQueue);

	return true;
}

bool OpenCLContext::Compile(const char* binary_name, const std::initializer_list<std::string>& source_files, const std::initializer_list<std::string>& kernel_names, const std::string& options, CachingParameters caching)
{
	std::vector<std::string> source;
	source.reserve(source_files.size());
	for (const std::string& source_file : source_files)
	{
		std::ifstream f(source_file);
		if (!f.is_open())
		{
			std::cerr << "Couldn't open " << source_file << std::endl;
			return false;
		}
		source.emplace_back((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
	}

	std::vector<const char*> data;
	data.reserve(source_files.size());
	for (const std::string& s : source)
		data.emplace_back(s.data());

	const char** p = data.data();
	cl_int err;

	cl_program program = nullptr;
	bool created_with_binary = false;
	if (caching != ALWAYS_COMPILE)
	{
		std::ifstream f(binary_name, std::ios::binary);
		if (f.is_open())
		{
			std::vector<char> buf;
			buf.insert(buf.begin(), std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());

			const size_t data_length = buf.size();
			const unsigned char* binary_data = reinterpret_cast<const unsigned char*>(buf.data());
			program = clCreateProgramWithBinary(context, 1, &device, &data_length, &binary_data, nullptr, &err);
			CL_CHECK_RESULT(clCreateProgramWithBinary);

			created_with_binary = true;
		}
		else if (caching == ALWAYS_USE_BINARY)
		{
			std::cerr << "Couldn't open " << binary_name << std::endl;
			return false;
		}
	}

	if (!program)
	{
		if (caching == ALWAYS_USE_BINARY)
		{
			std::cerr << "Couldn't create program from binary " << binary_name << std::endl;
			return false;
		}
		program = clCreateProgramWithSource(context, static_cast<cl_uint>(source_files.size()), p, nullptr, &err);
		CL_CHECK_RESULT(clCreateProgramWithSource);
	}

	std::cout << "Compiling " << binary_name << "...";
	std::string s = "-Werror -I CL";
	if (!options.empty())
	{
		s += ' ';
		s += options;
	}
	err = clBuildProgram(program, 1, &device, s.c_str(), nullptr, nullptr);
	if (err != CL_SUCCESS)
	{
		std::cerr << "clBuildProgram failed: error " << err << std::endl;

		size_t size;
		CL_CHECKED_CALL(clGetProgramBuildInfo, program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &size);

		std::vector<char> build_log;
		build_log.resize(size);
		CL_CHECKED_CALL(clGetProgramBuildInfo, program, device, CL_PROGRAM_BUILD_LOG, size, build_log.data(), nullptr);

		std::cerr << build_log.data() << std::endl;

		return false;
	}
	std::cout << "done" << std::endl;

	if (!created_with_binary)
	{
		size_t bin_size;
		CL_CHECKED_CALL(clGetProgramInfo, program, CL_PROGRAM_BINARY_SIZES, sizeof(bin_size), &bin_size, nullptr);

		std::vector<char> binary_data(bin_size);
		char* tmp[1] = { binary_data.data() };
		CL_CHECKED_CALL(clGetProgramInfo, program, CL_PROGRAM_BINARIES, sizeof(tmp), tmp, NULL);

		std::ofstream f(binary_name, std::ios::binary);
		f.write(tmp[0], bin_size);
		f.close();
	}

	for (const std::string& name : kernel_names)
	{
		cl_kernel kernel = clCreateKernel(program, name.c_str(), &err);
		CL_CHECK_RESULT(clCreateKernel);

		kernels.emplace(name, kernel);
	}

	CL_CHECKED_CALL(clReleaseProgram, program);
	return true;
}
