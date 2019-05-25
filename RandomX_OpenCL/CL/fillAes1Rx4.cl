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

__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void fillAes1Rx4_name(__global void* state, __global void* out, uint batch_size)
{
	__local uint T[2048];

	const uint stride_size = batch_size * 4;
	const uint global_index = get_global_id(0);
	if (global_index >= stride_size)
		return;

	const uint idx = global_index / 4;
	const uint sub = global_index % 4;

	for (uint i = get_local_id(0), step = get_local_size(0); i < 2048; i += step)
		T[i] = AES_TABLE[i];

	barrier(CLK_LOCAL_MEM_FENCE);

	const uint k[4] = { AES_KEY_FILL[sub * 4], AES_KEY_FILL[sub * 4 + 1], AES_KEY_FILL[sub * 4 + 2], AES_KEY_FILL[sub * 4 + 3] };

	__global uint* s = ((__global uint*) state) + idx * (64 / sizeof(uint)) + sub * (16 / sizeof(uint));
	uint x[4] = { s[0], s[1], s[2], s[3] };

	const uint s1 = (sub & 1) ? 8 : 24;
	const uint s3 = (sub & 1) ? 24 : 8;

	__global uint4* p = strided ? (((__global uint4*) out) + idx * 4 + sub) : (((__global uint4*) out) + idx * (outputSize0 / sizeof(uint4)) + sub);

	const __local uint* const t0 = (sub & 1) ? T : (T + 1024);
	const __local uint* const t1 = (sub & 1) ? (T + 256) : (T + 1792);
	const __local uint* const t2 = (sub & 1) ? (T + 512) : (T + 1536);
	const __local uint* const t3 = (sub & 1) ? (T + 768) : (T + 1280);

	#pragma unroll(unroll_factor)
	for (uint i = 0; i < outputSize / sizeof(uint4); i += 4, p += strided ? stride_size : 4)
	{
		uint y[4];

		y[0] = t0[get_byte(x[0], 0)] ^ t1[get_byte(x[1], s1)] ^ t2[get_byte(x[2], 16)] ^ t3[get_byte(x[3], s3)] ^ k[0];
		y[1] = t0[get_byte(x[1], 0)] ^ t1[get_byte(x[2], s1)] ^ t2[get_byte(x[3], 16)] ^ t3[get_byte(x[0], s3)] ^ k[1];
		y[2] = t0[get_byte(x[2], 0)] ^ t1[get_byte(x[3], s1)] ^ t2[get_byte(x[0], 16)] ^ t3[get_byte(x[1], s3)] ^ k[2];
		y[3] = t0[get_byte(x[3], 0)] ^ t1[get_byte(x[0], s1)] ^ t2[get_byte(x[1], 16)] ^ t3[get_byte(x[2], s3)] ^ k[3];

		*p = *(uint4*)(y);

		x[0] = y[0];
		x[1] = y[1];
		x[2] = y[2];
		x[3] = y[3];
	}

	*(__global uint4*)(s) = *(uint4*)(x);
}
