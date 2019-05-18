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

#include <string>

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
