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

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "tests.h"
#include "miner.h"

int main(int argc, char** argv)
{
	if (argc < 2)
	{
		printf("Usage: %s --mine [--validate] [--platform_id N] [--device_id N] [--intensity N] [--portable] [--workers N] [--bfactor N] [--dataset_host]\n\n", argv[0]);
		printf("platform_id  0 if you have only 1 OpenCL platform\n");
		printf("device_id    0 if you have only 1 GPU\n");
		printf("intensity    number of scratchpads to allocate, if it's not set then as many as possible will be allocated.\n\n");
		printf("portable     use generic OpenCL code that works on all GPUs.\n\n");
		printf("workers      number of parallel workers per hash to run in portable mode. Can be 2,4,8,16, default is 8.\n\n");
		printf("bfactor      splits main loop into multiple sub-steps. Use it to improve screen responsiveness. Can be 0-10, default is 5.\n\n");
		printf("dataset_host allocate dataset on host. This is required for 2 GB GPUs.\n\n");
		printf("Examples:\n%s --mine --validate --intensity 1984\n", argv[0]);
		return 0;
	}

	uint32_t platform_id = 0;
	uint32_t device_id = 0;
	size_t intensity = 0;
	uint32_t start_nonce = 0;
	uint32_t workers_per_hash = 8;
	uint32_t bfactor = 5;
	bool portable = false;
	bool dataset_host_allocated = false;
	bool validate = false;

	for (int i = 1; i < argc; ++i)
	{
		if ((strcmp(argv[i], "--platform_id") == 0) && (i + 1 < argc))
			platform_id = atoi(argv[i + 1]);
		else if ((strcmp(argv[i], "--device_id") == 0) && (i + 1 < argc))
			device_id = atoi(argv[i + 1]);
		else if ((strcmp(argv[i], "--intensity") == 0) && (i + 1 < argc))
			intensity = atoi(argv[i + 1]);
		else if ((strcmp(argv[i], "--nonce") == 0) && (i + 1 < argc))
			start_nonce = atoi(argv[i + 1]);
		else if ((strcmp(argv[i], "--workers") == 0) && (i + 1 < argc))
			workers_per_hash = atoi(argv[i + 1]);
		else if ((strcmp(argv[i], "--bfactor") == 0) && (i + 1 < argc))
			bfactor = atoi(argv[i + 1]);
		else if (strcmp(argv[i], "--portable") == 0)
			portable = true;
		else if (strcmp(argv[i], "--dataset_host") == 0)
			dataset_host_allocated = true;
		else if (strcmp(argv[i], "--validate") == 0)
			validate = true;
	}

	if (strcmp(argv[1], "--mine") == 0)
		return test_mining(platform_id, device_id, intensity, start_nonce, workers_per_hash, bfactor, portable, dataset_host_allocated, validate) ? 0 : 1;
	else if (strcmp(argv[1], "--test") == 0)
		return tests(platform_id, device_id, intensity) ? 0 : 1;

	return 0;
}
