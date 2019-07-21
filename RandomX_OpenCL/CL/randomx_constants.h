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

#define SCRATCHPAD_SIZE (1 << 21)
#define SCRATCHPAD_STRIDED 0

#define REGISTERS_SIZE 256
#define INITIAL_HASH_SIZE 64
#define RANDOMX_PROGRAM_SIZE 256
#define ENTROPY_SIZE (128 + 2048)
#define VM_STATE_SIZE 2048
#define INTERMEDIATE_PROGRAM_SIZE 4096
#define COMPILED_PROGRAM_SIZE 10048
#define LOCAL_GROUP_SIZE 64
#define WORKERS_PER_HASH 64
#define HASHES_PER_GROUP (LOCAL_GROUP_SIZE / WORKERS_PER_HASH)
#define NUM_VGPR_REGISTERS 128
