/**********************************************************************
  Copyright(c) 2011-2013 Intel Corporation All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions
  are met:
	* Redistributions of source code must retain the above copyright
	  notice, this list of conditions and the following disclaimer.
	* Redistributions in binary form must reproduce the above copyright
	  notice, this list of conditions and the following disclaimer in
	  the documentation and/or other materials provided with the
	  distribution.
	* Neither the name of Intel Corporation nor the names of its
	  contributors may be used to endorse or promote products derived
	  from this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**********************************************************************/
#define __USE_GNU
#include <sched.h>
#include <pthread.h>
#include <getopt.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "raid.h"
#include "cauchy.h"
#include "galois.h"
#include "match.h"
#include "xorMatchCode.h"
#include "vandermonde.h"
#include "slp.h"

// #include "types.h"
#include <time.h>
#define TEST_SOURCES 15
#define TEST_LEN 16 * 1024
#define MMAX 255
#define KMAX 255
int codenum = 0;
int alldata = 0;
int decodedata = 0;
long long int encode_time_arrs[15500];
long long int decode_time_arrs[15500];
long long int thread_time_arrs[4];
int slp_encode_test(int k, int p, int w, int len, int packetsize, int e);
int slp_encode_test_split(int k, int p, int w, int len, int packetsize, int e, int n);
int usage(void)
{
	fprintf(stderr,
			"Usage: ec_simple_example [options]\n"
			"  -h        Help\n"
			"  -k <val>  Number of source fragments\n"
			"  -p <val>  Number of parity fragments\n"
			"  -w <val>  \n"
			"  -len <val>  Length of fragments\n"
			"  -packetsize \n");
	exit(0);
}

int en_n = 1000;
int de_n = 1000;
int *b;

int mds_prove(int k, int p, int w)
{
	int i;
	// unsigned int encode_matrix[6]={1,1,1,2,2,2};
	int *encode_matrix;
	encode_matrix = cauchy_good_general_coding_matrix(k, p, w);
	// encode_bitmatrix=matrix_to_bitmatrix(k, p, w, encode_matrix);
	if (check_coding_matrix(k, p, w, encode_matrix))
	{
		printf("No MDS\n");
		return 1;
	}
	// int count=0;
	// for(i=0;i<k*w*p*w;i++)
	// {
	// 	if(encode_bitmatrix[i]==1)
	// 	{
	// 		count++;
	// 	}
	// }
	printf("MDS\n");
	return 0;
}

#include <immintrin.h> // AVX2 头文件
#define SLICE 32
#define SLICE_EXP 5
#define ENCODE_WINDOW_SIZE 16
#define DECODE_WINDOW_SIZE 16
#define DECODE_WINDOW_SIZE4 32
#define EXTRA_DATA_SIZE 704
#define Mval 3
#define Mval4 4
#define Mval2 2

void two_tone_encode4(int k, int m, int blocksize, uint8_t **chunks)
{

	const int TOTAL_SLICE_NUM = blocksize / SLICE;
	const int window_size = ENCODE_WINDOW_SIZE;
	const int windows_num = TOTAL_SLICE_NUM / window_size;

	uint64_t slice_offset_1;
	uint64_t slice_offset_2;
	uint64_t slice_offset_3;
	uint64_t slice_offset_4;

	__m256i mxx1, mxx2, mxx3, mxx4, res1, res3, tmp;

	int window_id, slice_id, chunk_idx;

	for (window_id = 0; window_id < windows_num; window_id++)
	{
		const int slice_id_start = window_id * window_size;
		const int slice_id_end = slice_id_start + window_size;

		for (slice_id = slice_id_start; slice_id < slice_id_end; slice_id++)
		{
			slice_offset_1 = slice_id * SLICE;
			slice_offset_2 = slice_id * SLICE;
			slice_offset_3 = (slice_id + k - 2) * SLICE;	   // 7 - 1
			slice_offset_4 = (slice_id + (k - 1) * 2) * SLICE; // kval - 1

			// Load two data chunks into registers and XOR them
			mxx1 = _mm256_load_si256((__m256i *)&chunks[0][slice_offset_2]);
			mxx2 = _mm256_load_si256((__m256i *)&chunks[1][slice_offset_2]);
			res1 = _mm256_xor_si256(mxx1, mxx2);

			// Store the result into parity chunks
			_mm256_store_si256((__m256i *)&chunks[k + 1][slice_offset_2], res1);
			if (slice_id == slice_id_start)
			{
				res1 = mxx1;
				res3 = mxx2;
			}
			else
			{
				res1 = _mm256_xor_si256(mxx1, mxx4);
				res3 = _mm256_xor_si256(mxx2, mxx3);
			}
			if (window_id != 0)
			{
				if ((slice_id % window_size) < k - 1)
				{
					tmp = _mm256_load_si256((__m256i *)&chunks[k][slice_offset_1]);
					res1 = _mm256_xor_si256(res1, tmp);
				}
				if (slice_id == slice_id_start)
				{
					tmp = _mm256_load_si256((__m256i *)&chunks[k + 2][slice_offset_3]);
					res3 = _mm256_xor_si256(res3, tmp);
				}
			}
			_mm256_store_si256((__m256i *)&chunks[k][slice_offset_1], res1);
			_mm256_store_si256((__m256i *)&chunks[k + 2][slice_offset_3], res3);
			if (slice_id != slice_id_end - 1)
			{
				mxx3 = mxx1;
				mxx4 = mxx2;
			}
			else
			{
				// [!] Assert window_size >= k - 1, or overwriting occurs
				_mm256_store_si256((__m256i *)&chunks[k][slice_offset_1 + SLICE], mxx2);
				_mm256_store_si256((__m256i *)&chunks[k + 2][slice_offset_3 + SLICE], mxx1);
			}

			// chunk1 for parity4
			if (window_id != 0 || slice_id >= slice_id_start + 2)
			{
				tmp = _mm256_load_si256((__m256i *)&chunks[k + 3][slice_offset_4 - 2 * SLICE]);
				mxx2 = _mm256_xor_si256(mxx2, tmp);
			}
			// store chunk0 to pairty4
			_mm256_store_si256((__m256i *)&chunks[k + 3][slice_offset_4], mxx1);
			// store chunk1 to pairty4
			_mm256_store_si256((__m256i *)&chunks[k + 3][slice_offset_4 - 2 * SLICE], mxx2);
		}

		for (chunk_idx = 2; chunk_idx < k; chunk_idx += 2)
		{
			for (slice_id = slice_id_start; slice_id < slice_id_end; slice_id++)
			{
				slice_offset_2 = slice_id * SLICE;
				slice_offset_1 = (slice_id + chunk_idx) * SLICE;
				slice_offset_3 = (slice_id + (k - 2 - chunk_idx)) * SLICE;	   // 7 - 1
				slice_offset_4 = (slice_id + (k - 1 - chunk_idx) * 2) * SLICE; // kval - 1

				// Load two data chunks into registers and XOR them
				mxx1 = _mm256_load_si256((__m256i *)&chunks[chunk_idx][slice_offset_2]);
				mxx2 = _mm256_load_si256((__m256i *)&chunks[chunk_idx + 1][slice_offset_2]);
				res1 = _mm256_xor_si256(mxx1, mxx2);

				// Store the result into parity chunks
				tmp = _mm256_load_si256((__m256i *)&chunks[k + 1][slice_offset_2]);
				res1 = _mm256_xor_si256(res1, tmp);
				_mm256_store_si256((__m256i *)&chunks[k + 1][slice_offset_2], res1);

				if (slice_id == slice_id_start)
				{
					res1 = mxx1;
					res3 = mxx2;
				}
				else
				{
					res1 = _mm256_xor_si256(mxx1, mxx4);
					res3 = _mm256_xor_si256(mxx2, mxx3);
				}

				if (slice_id > 1)
				{
					tmp = _mm256_load_si256((__m256i *)&chunks[k + 2][slice_offset_3]);
					res3 = _mm256_xor_si256(res3, tmp);
				}
				_mm256_store_si256((__m256i *)&chunks[k + 2][slice_offset_3], res3);

				if (slice_id != slice_id_end - 1)
				{
					tmp = _mm256_load_si256((__m256i *)&chunks[k][slice_offset_1]);
					res1 = _mm256_xor_si256(res1, tmp);
					_mm256_store_si256((__m256i *)&chunks[k][slice_offset_1], res1);
					mxx3 = mxx1;
					mxx4 = mxx2;
				}
				else
				{
					_mm256_store_si256((__m256i *)&chunks[k][slice_offset_1], res1);
					// the slice of shift end
					_mm256_store_si256((__m256i *)&chunks[k][slice_offset_1 + SLICE], mxx2);
					tmp = _mm256_load_si256((__m256i *)&chunks[k + 2][slice_offset_3 + SLICE]);
					res3 = _mm256_xor_si256(mxx1, tmp);
					_mm256_store_si256((__m256i *)&chunks[k + 2][slice_offset_3 + SLICE], res3);
				}

				// chunk01 for parity4
				if (window_id != 0 || slice_id >= slice_id_start + 2)
				{
					tmp = _mm256_load_si256((__m256i *)&chunks[k + 3][slice_offset_4]);
					mxx1 = _mm256_xor_si256(mxx1, tmp);
					tmp = _mm256_load_si256((__m256i *)&chunks[k + 3][slice_offset_4 - 2 * SLICE]);
					mxx2 = _mm256_xor_si256(mxx2, tmp);
				}
				// store chunk01 to pairty4
				_mm256_store_si256((__m256i *)&chunks[k + 3][slice_offset_4], mxx1);
				_mm256_store_si256((__m256i *)&chunks[k + 3][slice_offset_4 - 2 * SLICE], mxx2);
			}
		}
	}
}

void two_tone_encode2(int k, int m, int blocksize, uint8_t **chunks)
{

	const int TOTAL_SLICE_NUM = blocksize / SLICE;
	const int window_size = ENCODE_WINDOW_SIZE;
	const int windows_num = TOTAL_SLICE_NUM / window_size;

	uint64_t slice_offset_1;
	uint64_t slice_offset_2;

	__m256i mxx1, mxx2, mxx4, res1, tmp;

	int window_id, slice_id, chunk_idx;

	for (window_id = 0; window_id < windows_num; window_id++)
	{
		const int slice_id_start = window_id * window_size;
		const int slice_id_end = slice_id_start + window_size;

		for (slice_id = slice_id_start; slice_id < slice_id_end; slice_id++)
		{
			slice_offset_1 = slice_id * SLICE;
			slice_offset_2 = slice_id * SLICE;

			// Load two data chunks into registers and XOR them
			mxx1 = _mm256_load_si256((__m256i *)&chunks[0][slice_offset_2]);
			mxx2 = _mm256_load_si256((__m256i *)&chunks[1][slice_offset_2]);
			res1 = _mm256_xor_si256(mxx1, mxx2);

			// Store the result into parity chunks
			_mm256_store_si256((__m256i *)&chunks[k + 1][slice_offset_2], res1);
			if (slice_id == slice_id_start)
			{
				res1 = mxx1;
			}
			else
			{
				res1 = _mm256_xor_si256(mxx1, mxx4);
			}
			if (window_id != 0)
			{
				if ((slice_id % window_size) < k - 1)
				{
					tmp = _mm256_load_si256((__m256i *)&chunks[k][slice_offset_1]);
					res1 = _mm256_xor_si256(res1, tmp);
				}
			}
			_mm256_store_si256((__m256i *)&chunks[k][slice_offset_1], res1);
			if (slice_id != slice_id_end - 1)
			{
				mxx4 = mxx2;
			}
			else
			{
				// [!] Assert window_size >= k - 1, or overwriting occurs
				_mm256_store_si256((__m256i *)&chunks[k][slice_offset_1 + SLICE], mxx2);
			}
		}

		for (chunk_idx = 2; chunk_idx < k; chunk_idx += 2)
		{
			for (slice_id = slice_id_start; slice_id < slice_id_end; slice_id++)
			{
				slice_offset_2 = slice_id * SLICE;
				slice_offset_1 = (slice_id + chunk_idx) * SLICE;

				// Load two data chunks into registers and XOR them
				mxx1 = _mm256_load_si256((__m256i *)&chunks[chunk_idx][slice_offset_2]);
				mxx2 = _mm256_load_si256((__m256i *)&chunks[chunk_idx + 1][slice_offset_2]);
				res1 = _mm256_xor_si256(mxx1, mxx2);

				// Store the result into parity chunks
				tmp = _mm256_load_si256((__m256i *)&chunks[k + 1][slice_offset_2]);
				res1 = _mm256_xor_si256(res1, tmp);
				_mm256_store_si256((__m256i *)&chunks[k + 1][slice_offset_2], res1);

				if (slice_id == slice_id_start)
				{
					res1 = mxx1;
				}
				else
				{
					res1 = _mm256_xor_si256(mxx1, mxx4);
				}

				if (slice_id != slice_id_end - 1)
				{
					tmp = _mm256_load_si256((__m256i *)&chunks[k][slice_offset_1]);
					res1 = _mm256_xor_si256(res1, tmp);
					_mm256_store_si256((__m256i *)&chunks[k][slice_offset_1], res1);
					mxx4 = mxx2;
				}
				else
				{
					_mm256_store_si256((__m256i *)&chunks[k][slice_offset_1], res1);
					// the slice of shift end
					_mm256_store_si256((__m256i *)&chunks[k][slice_offset_1 + SLICE], mxx2);
				}
			}
		}
	}
}

void two_tone_encode(int k, int m, int blocksize, uint8_t **chunks)
{

	const int TOTAL_SLICE_NUM = blocksize / SLICE;
	const int window_size = ENCODE_WINDOW_SIZE;
	const int windows_num = TOTAL_SLICE_NUM / window_size;

	uint64_t slice_offset_1;
	uint64_t slice_offset_2;
	uint64_t slice_offset_3;
	__m256i mxx1, mxx2, mxx3, mxx4, res1, res3, tmp;

	int window_id, slice_id, chunk_idx;

	for (window_id = 0; window_id < windows_num; window_id++)
	{
		const int slice_id_start = window_id * window_size;
		const int slice_id_end = slice_id_start + window_size;

		for (slice_id = slice_id_start; slice_id < slice_id_end; slice_id++)
		{
			slice_offset_1 = slice_id * SLICE;
			slice_offset_2 = slice_id * SLICE;
			slice_offset_3 = (slice_id + k - 2) * SLICE; // 7 - 1

			// Load two data chunks into registers and XOR them
			mxx1 = _mm256_load_si256((__m256i *)&chunks[0][slice_offset_2]);
			mxx2 = _mm256_load_si256((__m256i *)&chunks[1][slice_offset_2]);
			res1 = _mm256_xor_si256(mxx1, mxx2);

			// Store the result into parity chunks
			_mm256_store_si256((__m256i *)&chunks[k + 1][slice_offset_2], res1);
			if (slice_id == slice_id_start)
			{
				res1 = mxx1;
				res3 = mxx2;
			}
			else
			{
				res1 = _mm256_xor_si256(mxx1, mxx4);
				res3 = _mm256_xor_si256(mxx2, mxx3);
			}
			if (window_id != 0)
			{
				if ((slice_id % window_size) < k - 1)
				{
					tmp = _mm256_load_si256((__m256i *)&chunks[k][slice_offset_1]);
					res1 = _mm256_xor_si256(res1, tmp);
				}
				if (slice_id == slice_id_start)
				{
					tmp = _mm256_load_si256((__m256i *)&chunks[k + 2][slice_offset_3]);
					res3 = _mm256_xor_si256(res3, tmp);
				}
			}
			_mm256_store_si256((__m256i *)&chunks[k][slice_offset_1], res1);
			_mm256_store_si256((__m256i *)&chunks[k + 2][slice_offset_3], res3);
			if (slice_id != slice_id_end - 1)
			{
				mxx3 = mxx1;
				mxx4 = mxx2;
			}
			else
			{
				// [!] Assert window_size >= k - 1, or overwriting occurs
				_mm256_store_si256((__m256i *)&chunks[k][slice_offset_1 + SLICE], mxx2);
				_mm256_store_si256((__m256i *)&chunks[k + 2][slice_offset_3 + SLICE], mxx1);
			}
		}

		for (chunk_idx = 2; chunk_idx < k; chunk_idx += 2)
		{
			for (slice_id = slice_id_start; slice_id < slice_id_end; slice_id++)
			{
				slice_offset_2 = slice_id * SLICE;
				slice_offset_1 = (slice_id + chunk_idx) * SLICE;
				slice_offset_3 = (slice_id + (k - 2 - chunk_idx)) * SLICE; // 7 - 1

				// Load two data chunks into registers and XOR them
				mxx1 = _mm256_load_si256((__m256i *)&chunks[chunk_idx][slice_offset_2]);
				mxx2 = _mm256_load_si256((__m256i *)&chunks[chunk_idx + 1][slice_offset_2]);
				res1 = _mm256_xor_si256(mxx1, mxx2);

				// Store the result into parity chunks
				tmp = _mm256_load_si256((__m256i *)&chunks[k + 1][slice_offset_2]);
				res1 = _mm256_xor_si256(res1, tmp);
				_mm256_store_si256((__m256i *)&chunks[k + 1][slice_offset_2], res1);

				if (slice_id == slice_id_start)
				{
					res1 = mxx1;
					res3 = mxx2;
				}
				else
				{
					res1 = _mm256_xor_si256(mxx1, mxx4);
					res3 = _mm256_xor_si256(mxx2, mxx3);
				}

				if (slice_id > 1)
				{
					tmp = _mm256_load_si256((__m256i *)&chunks[k + 2][slice_offset_3]);
					res3 = _mm256_xor_si256(res3, tmp);
				}
				_mm256_store_si256((__m256i *)&chunks[k + 2][slice_offset_3], res3);

				if (slice_id != slice_id_end - 1)
				{
					tmp = _mm256_load_si256((__m256i *)&chunks[k][slice_offset_1]);
					res1 = _mm256_xor_si256(res1, tmp);
					_mm256_store_si256((__m256i *)&chunks[k][slice_offset_1], res1);
					mxx3 = mxx1;
					mxx4 = mxx2;
				}
				else
				{
					_mm256_store_si256((__m256i *)&chunks[k][slice_offset_1], res1);
					// the slice of shift end
					_mm256_store_si256((__m256i *)&chunks[k][slice_offset_1 + SLICE], mxx2);
					tmp = _mm256_load_si256((__m256i *)&chunks[k + 2][slice_offset_3 + SLICE]);
					res3 = _mm256_xor_si256(mxx1, tmp);
					_mm256_store_si256((__m256i *)&chunks[k + 2][slice_offset_3 + SLICE], res3);
				}
			}
		}
	}
}

void decode_one_data_chunk(int kval, uint8_t **data, uint8_t *coding, int blocksize, int erasure_idx)
{
	const int TOTAL_SLICE_NUM = blocksize / SLICE;
	const int UNROLL = 4;

	__m256i sum[UNROLL];
	__m256i tmp[UNROLL];

	uint8_t *tmp_ptr = data[erasure_idx];
	data[erasure_idx] = coding;
	coding = tmp_ptr;

	int slice_id = 0;
	int slice_offset[UNROLL];
	// Unrolled loop
	int u, data_idx;
	for (; slice_id + UNROLL - 1 < TOTAL_SLICE_NUM; slice_id += UNROLL)
	{
		for (u = 0; u < UNROLL; ++u)
			slice_offset[u] = (slice_id + u) << SLICE_EXP;

		// Load first data chunk for each unrolled iteration
		for (u = 0; u < UNROLL; ++u)
			sum[u] = _mm256_load_si256((__m256i *)&data[0][slice_offset[u]]);

		// XOR with remaining data chunks
		for (data_idx = 1; data_idx < kval; data_idx++)
		{
			for (u = 0; u < UNROLL; ++u)
			{
				tmp[u] = _mm256_load_si256((__m256i *)&data[data_idx][slice_offset[u]]);
			}
			for (u = 0; u < UNROLL; ++u)
			{
				sum[u] = _mm256_xor_si256(sum[u], tmp[u]);
			}
		}

		// Store result
		for (u = 0; u < UNROLL; ++u)
		{
			_mm256_stream_si256((__m256i *)&coding[slice_offset[u]], sum[u]);
		}
	}

	// Handle remaining slices
	// for (; slice_id < TOTAL_SLICE_NUM; ++slice_id)
	// {
	// 	int offset = slice_id << SLICE_EXP;
	// 	__m256i sum1 = _mm256_load_si256((__m256i *)&data[0][offset]);
	// 	for (data_idx = 1; data_idx < kval; data_idx++)
	// 	{
	// 		__m256i tmp1 = _mm256_load_si256((__m256i *)&data[data_idx][offset]);
	// 		sum1 = _mm256_xor_si256(sum1, tmp1);
	// 	}
	// 	_mm256_stream_si256((__m256i *)&coding[offset], sum1);
	// }

	return 0;
}

void two_tone_decode_data(int k, int m, int blocksize, uint8_t **chunks, int *eraseds)
{
	int eraseds_cnt = 0;
	int coding_data_map[Mval] = {0, 0, k - 1};

	int erasure_data[Mval] = {-1, -1, -1};
	int erasure_data_count = 0;

	int retension_data[k];
	int retension_data_count = 0;

	uint8_t *data[k];
	uint8_t *coding[Mval];

	int i, j;
	for (i = 0; i < k + m; i++)
	{
		if (i < k)
		{
			// 数据块
			data[i] = chunks[i];
			if (eraseds[eraseds_cnt] == i)
			{
				erasure_data[erasure_data_count] = i;
				erasure_data_count++;
				eraseds_cnt++;
			}
			else
			{
				retension_data[retension_data_count] = i;
				retension_data_count++;
			}
		}
		else
		{
			// 校验块
			int parity_idx = i - k;
			coding[parity_idx] = chunks[i];
		}
	}

	if (erasure_data_count == 1)
	{
		decode_one_data_chunk(k, data, coding[1], blocksize, erasure_data[0]);
		return;
	}

	retension_data[retension_data_count] = -1;

	const int TOTAL_SLICE_NUM = blocksize / SLICE;
	const int window_size = DECODE_WINDOW_SIZE;
	const int windows_num = TOTAL_SLICE_NUM / window_size;

	long slice_offset[3];
	__m256i mxx0, mxx1, mxx2, mxx3, tmp;
	__m256i res[3];

	const bool need_decoded[3] = {erasure_data[0] != -1, erasure_data[1] != -1, erasure_data[2] != -1};

	// uint8_t *zeros = (uint8_t *)aligned_alloc(SLICE, SLICE);
	// _mm256_store_si256((__m256i *)zeros, _mm256_setzero_si256());

	int window_id, slice_id, retension_data_idx;
	for (window_id = 0; window_id <= windows_num; window_id++)
	{

		// substitute
		if (window_id < windows_num)
		{
			const int slice_id_start = window_id * window_size;
			const int slice_id_end = slice_id_start + window_size;

			for (retension_data_idx = 0; retension_data_idx < retension_data_count; retension_data_idx++)
			{

				int retension_data_chunk_idx = retension_data[retension_data_idx];
				if (retension_data[retension_data_idx + 1] - retension_data_chunk_idx == 1)
				{
					for (slice_id = slice_id_start; slice_id < slice_id_end; slice_id++)
					{

						slice_offset[1] = slice_id * SLICE;

						slice_offset[0] = (slice_id + retension_data_chunk_idx) * SLICE;
						slice_offset[2] = (slice_id + k - retension_data_chunk_idx - 2) * SLICE; // for retension_data_chunk_idx + 1

						mxx0 = _mm256_load_si256((__m256i *)&data[retension_data_chunk_idx][slice_offset[1]]);
						mxx1 = _mm256_load_si256((__m256i *)&data[retension_data_chunk_idx + 1][slice_offset[1]]);

						// if (erasure_data[0] != -1)
						if (need_decoded[0])
						{
							if (slice_id == slice_id_start)
							{
								res[0] = mxx0;
							}
							else
							{
								res[0] = _mm256_xor_si256(mxx0, mxx3);
							}
							tmp = _mm256_load_si256((__m256i *)&coding[0][slice_offset[0]]);
							res[0] = _mm256_xor_si256(res[0], tmp);
							_mm256_store_si256((__m256i *)&coding[0][slice_offset[0]], res[0]);
							if (slice_id == slice_id_end - 1)
							{
								tmp = _mm256_load_si256((__m256i *)&coding[0][slice_offset[0] + SLICE]);
								res[0] = _mm256_xor_si256(mxx1, tmp);
								_mm256_store_si256((__m256i *)&coding[0][slice_offset[0] + SLICE], res[0]);
							}
						}
						// if (erasure_data[1] != -1)
						if (need_decoded[1])
						{

							res[1] = _mm256_xor_si256(mxx1, mxx0);
							tmp = _mm256_load_si256((__m256i *)&coding[1][slice_offset[1]]);
							res[1] = _mm256_xor_si256(res[1], tmp);
							_mm256_store_si256((__m256i *)&coding[1][slice_offset[1]], res[1]);
						}
						// if (erasure_data[2] != -1)
						if (need_decoded[2])
						{
							if (slice_id == slice_id_start)
							{
								res[2] = mxx1;
							}
							else
							{
								res[2] = _mm256_xor_si256(mxx1, mxx2);
							}
							tmp = _mm256_load_si256((__m256i *)&coding[2][slice_offset[2]]);
							res[2] = _mm256_xor_si256(res[2], tmp);
							_mm256_store_si256((__m256i *)&coding[2][slice_offset[2]], res[2]);
							if (slice_id == slice_id_end - 1)
							{
								tmp = _mm256_load_si256((__m256i *)&coding[2][slice_offset[2] + SLICE]);
								res[2] = _mm256_xor_si256(mxx0, tmp);
								_mm256_store_si256((__m256i *)&coding[2][slice_offset[2] + SLICE], res[2]);
							}
						}

						mxx2 = mxx0;
						mxx3 = mxx1;
					}

					retension_data_idx++;
				}
				else
				{
					for (slice_id = slice_id_start; slice_id < slice_id_end; slice_id++)
					{

						slice_offset[1] = slice_id * SLICE;

						mxx1 = _mm256_load_si256((__m256i *)&data[retension_data_chunk_idx][slice_offset[1]]);

						slice_offset[0] = (slice_id + retension_data_chunk_idx) * SLICE;
						slice_offset[2] = (slice_id + k - 1 - retension_data_chunk_idx) * SLICE;

						for (i = 0; i < 3; i++)
						{
							// if (erasure_data[i] != -1)
							if (need_decoded[i])
							{
								tmp = _mm256_load_si256((__m256i *)&coding[i][slice_offset[i]]);
								mxx2 = _mm256_xor_si256(mxx1, tmp);
								_mm256_store_si256((__m256i *)&coding[i][slice_offset[i]], mxx2);
							}
						}
					}
				}
			}
		}

		// elimination
		if (window_id > 0)
		{
			const int solve_slice_id_start = (window_id - 1) * window_size;
			const int solve_slice_id_end = solve_slice_id_start + window_size;

			for (slice_id = solve_slice_id_start; slice_id < solve_slice_id_end; slice_id++)
			{
				slice_offset[1] = slice_id * SLICE;
				if (need_decoded[0])
				{
					int erasure_data_idx = erasure_data[0];
					slice_offset[0] = (slice_id + erasure_data_idx) * SLICE;

					mxx1 = _mm256_load_si256((__m256i *)&coding[0][slice_offset[0]]);
					_mm256_stream_si256((__m256i *)&data[erasure_data_idx][slice_offset[1]], mxx1);

					if (need_decoded[1])
					{
						tmp = _mm256_load_si256((__m256i *)&coding[1][slice_offset[1]]);
						mxx2 = _mm256_xor_si256(mxx1, tmp);
						_mm256_store_si256((__m256i *)&coding[1][slice_offset[1]], mxx2);
					}

					if (need_decoded[2])
					{
						slice_offset[2] = (slice_id + k - 1 - erasure_data_idx) * SLICE;
						tmp = _mm256_load_si256((__m256i *)&coding[2][slice_offset[2]]);
						mxx2 = _mm256_xor_si256(mxx1, tmp);
						_mm256_store_si256((__m256i *)&coding[2][slice_offset[2]], mxx2);
					}
				}
				if (need_decoded[2])
				{
					int erasure_data_idx = erasure_data[2];
					slice_offset[2] = (slice_id + k - 1 - erasure_data_idx) * SLICE;

					mxx1 = _mm256_load_si256((__m256i *)&coding[2][slice_offset[2]]);
					_mm256_stream_si256((__m256i *)&data[erasure_data_idx][slice_offset[1]], mxx1);

					if (need_decoded[0])
					{
						slice_offset[0] = (slice_id + erasure_data_idx) * SLICE;
						tmp = _mm256_load_si256((__m256i *)&coding[0][slice_offset[0]]);
						mxx2 = _mm256_xor_si256(mxx1, tmp);
						_mm256_store_si256((__m256i *)&coding[0][slice_offset[0]], mxx2);
					}

					if (need_decoded[1])
					{
						tmp = _mm256_load_si256((__m256i *)&coding[1][slice_offset[1]]);
						mxx2 = _mm256_xor_si256(mxx1, tmp);
						_mm256_store_si256((__m256i *)&coding[1][slice_offset[1]], mxx2);
					}
				}
				if (need_decoded[1])
				{
					int erasure_data_idx = erasure_data[1];

					mxx1 = _mm256_load_si256((__m256i *)&coding[1][slice_offset[1]]);
					_mm256_stream_si256((__m256i *)&data[erasure_data_idx][slice_offset[1]], mxx1);

					if (need_decoded[0])
					{
						slice_offset[0] = (slice_id + erasure_data_idx) * SLICE;
						tmp = _mm256_load_si256((__m256i *)&coding[0][slice_offset[0]]);
						mxx2 = _mm256_xor_si256(mxx1, tmp);
						_mm256_store_si256((__m256i *)&coding[0][slice_offset[0]], mxx2);
					}

					if (need_decoded[2])
					{
						slice_offset[2] = (slice_id + k - 1 - erasure_data_idx) * SLICE;
						tmp = _mm256_load_si256((__m256i *)&coding[2][slice_offset[2]]);
						mxx2 = _mm256_xor_si256(mxx1, tmp);
						_mm256_store_si256((__m256i *)&coding[2][slice_offset[2]], mxx2);
					}
				}
			}
		}
	}
}

void two_tone_decode_data2(int k, int m, int blocksize, uint8_t **chunks, int *eraseds)
{
	int eraseds_cnt = 0;
	int coding_data_map[Mval2] = {0, 0};

	int erasure_data[Mval2] = {-1, -1};
	int erasure_data_count = 0;

	int retension_data[k];
	int retension_data_count = 0;

	uint8_t *data[k];
	uint8_t *coding[Mval2];

	int i, j;
	for (i = 0; i < k + m; i++)
	{
		if (i < k)
		{
			// 数据块
			data[i] = chunks[i];
			if (eraseds[eraseds_cnt] == i)
			{
				erasure_data[erasure_data_count] = i;
				erasure_data_count++;
				eraseds_cnt++;
			}
			else
			{
				retension_data[retension_data_count] = i;
				retension_data_count++;
			}
		}
		else
		{
			// 校验块
			int parity_idx = i - k;
			coding[parity_idx] = chunks[i];
		}
	}

	if (erasure_data_count == 1)
	{
		decode_one_data_chunk(k, data, coding[1], blocksize, erasure_data[0]);
		return;
	}

	retension_data[retension_data_count] = -1;

	const int TOTAL_SLICE_NUM = blocksize / SLICE;
	const int window_size = DECODE_WINDOW_SIZE;
	const int windows_num = TOTAL_SLICE_NUM / window_size;

	long slice_offset[2];
	__m256i mxx0, mxx1, mxx2, mxx3, tmp, res;

	const bool need_decoded[2] = {erasure_data[0] != -1, erasure_data[1] != -1};

	// uint8_t *zeros = (uint8_t *)aligned_alloc(SLICE, SLICE);
	// _mm256_store_si256((__m256i *)zeros, _mm256_setzero_si256());

	int window_id, slice_id, retension_data_idx;
	for (window_id = 0; window_id <= windows_num; window_id++)
	{

		// substitute
		if (window_id < windows_num)
		{
			const int slice_id_start = window_id * window_size;
			const int slice_id_end = slice_id_start + window_size;

			for (retension_data_idx = 0; retension_data_idx < retension_data_count; retension_data_idx++)
			{

				int retension_data_chunk_idx = retension_data[retension_data_idx];
				if (retension_data[retension_data_idx + 1] - retension_data_chunk_idx == 1)
				{
					for (slice_id = slice_id_start; slice_id < slice_id_end; slice_id++)
					{

						slice_offset[1] = slice_id * SLICE;
						slice_offset[0] = (slice_id + retension_data_chunk_idx) * SLICE;

						mxx0 = _mm256_load_si256((__m256i *)&data[retension_data_chunk_idx][slice_offset[1]]);
						mxx1 = _mm256_load_si256((__m256i *)&data[retension_data_chunk_idx + 1][slice_offset[1]]);

						// if (erasure_data[0] != -1)
						if (need_decoded[0])
						{
							if (slice_id == slice_id_start)
							{
								res = mxx0;
							}
							else
							{
								res = _mm256_xor_si256(mxx0, mxx3);
							}
							tmp = _mm256_load_si256((__m256i *)&coding[0][slice_offset[0]]);
							res = _mm256_xor_si256(res, tmp);
							_mm256_store_si256((__m256i *)&coding[0][slice_offset[0]], res);
							if (slice_id == slice_id_end - 1)
							{
								tmp = _mm256_load_si256((__m256i *)&coding[0][slice_offset[0] + SLICE]);
								res = _mm256_xor_si256(mxx1, tmp);
								_mm256_store_si256((__m256i *)&coding[0][slice_offset[0] + SLICE], res);
							}
						}
						// if (erasure_data[1] != -1)
						if (need_decoded[1])
						{

							res = _mm256_xor_si256(mxx1, mxx0);
							tmp = _mm256_load_si256((__m256i *)&coding[1][slice_offset[1]]);
							res = _mm256_xor_si256(res, tmp);
							_mm256_store_si256((__m256i *)&coding[1][slice_offset[1]], res);
						}
						mxx3 = mxx1;
					}

					retension_data_idx++;
				}
				else
				{
					for (slice_id = slice_id_start; slice_id < slice_id_end; slice_id++)
					{

						slice_offset[1] = slice_id * SLICE;

						mxx1 = _mm256_load_si256((__m256i *)&data[retension_data_chunk_idx][slice_offset[1]]);

						slice_offset[0] = (slice_id + retension_data_chunk_idx) * SLICE;

						for (i = 0; i < 2; i++)
						{
							// if (erasure_data[i] != -1)
							if (need_decoded[i])
							{
								tmp = _mm256_load_si256((__m256i *)&coding[i][slice_offset[i]]);
								mxx2 = _mm256_xor_si256(mxx1, tmp);
								_mm256_store_si256((__m256i *)&coding[i][slice_offset[i]], mxx2);
							}
						}
					}
				}
			}
		}

		// elimination
		if (window_id > 0)
		{
			const int solve_slice_id_start = (window_id - 1) * window_size;
			const int solve_slice_id_end = solve_slice_id_start + window_size;

			for (slice_id = solve_slice_id_start; slice_id < solve_slice_id_end; slice_id++)
			{
				slice_offset[1] = slice_id * SLICE;
				if (need_decoded[0])
				{
					int erasure_data_idx = erasure_data[0];
					slice_offset[0] = (slice_id + erasure_data_idx) * SLICE;

					mxx1 = _mm256_load_si256((__m256i *)&coding[0][slice_offset[0]]);
					_mm256_stream_si256((__m256i *)&data[erasure_data_idx][slice_offset[1]], mxx1);

					if (need_decoded[1])
					{
						tmp = _mm256_load_si256((__m256i *)&coding[1][slice_offset[1]]);
						mxx2 = _mm256_xor_si256(mxx1, tmp);
						_mm256_store_si256((__m256i *)&coding[1][slice_offset[1]], mxx2);
					}
				}
				if (need_decoded[1])
				{
					int erasure_data_idx = erasure_data[1];

					mxx1 = _mm256_load_si256((__m256i *)&coding[1][slice_offset[1]]);
					_mm256_stream_si256((__m256i *)&data[erasure_data_idx][slice_offset[1]], mxx1);

					if (need_decoded[0])
					{
						slice_offset[0] = (slice_id + erasure_data_idx) * SLICE;
						tmp = _mm256_load_si256((__m256i *)&coding[0][slice_offset[0]]);
						mxx2 = _mm256_xor_si256(mxx1, tmp);
						_mm256_store_si256((__m256i *)&coding[0][slice_offset[0]], mxx2);
					}
				}
			}
		}
	}
}

void two_tone_decode_data4(int k, int m, int blocksize, uint8_t **chunks, int *eraseds)
{
	int eraseds_cnt = 0;
	int coding_data_map[Mval4] = {0, 0, k - 1, k - 1};

	int erasure_data[Mval4] = {-1, -1, -1, -1};
	int erasure_data_count = 0;

	int retension_data[k];
	int retension_data_count = 0;

	uint8_t *data[k];
	uint8_t *coding[Mval4];

	int i, j;
	for (i = 0; i < k + m; i++)
	{
		if (i < k)
		{
			// 数据块
			data[i] = chunks[i];
			if (eraseds[eraseds_cnt] == i)
			{
				erasure_data[erasure_data_count] = i;
				erasure_data_count++;
				eraseds_cnt++;
			}
			else
			{
				retension_data[retension_data_count] = i;
				retension_data_count++;
			}
		}
		else
		{
			// 校验块
			int parity_idx = i - k;
			coding[parity_idx] = chunks[i];
		}
	}

	if (erasure_data_count == 1)
	{
		decode_one_data_chunk(k, data, coding[1], blocksize, erasure_data[0]);
		return;
	}

	retension_data[retension_data_count] = -1;
	int offset4 = erasure_data[3] - erasure_data[2];

	const int TOTAL_SLICE_NUM = blocksize / SLICE;
	const int window_size = DECODE_WINDOW_SIZE4;
	const int windows_num = TOTAL_SLICE_NUM / window_size;

	long slice_offset[4];
	__m256i mxx0, mxx1, mxx2, mxx3, tmp, res;
	// __m256i res[3];

	const bool need_decoded[4] = {erasure_data[0] != -1, erasure_data[1] != -1, erasure_data[2] != -1, erasure_data[3] != -1};

	// printf("erasure_data[3] = %d\n", erasure_data[3]);

	// uint8_t *zeros = (uint8_t *)aligned_alloc(SLICE, SLICE);
	// _mm256_store_si256((__m256i *)zeros, _mm256_setzero_si256());

	int window_id,
		slice_id, retension_data_idx;
	for (window_id = 0; window_id <= windows_num; window_id++)
	{

		// substitute
		if (window_id < windows_num)
		{
			const int slice_id_start = window_id * window_size;
			const int slice_id_end = slice_id_start + window_size;

			for (retension_data_idx = 0; retension_data_idx < retension_data_count; retension_data_idx++)
			{

				int retension_data_chunk_idx = retension_data[retension_data_idx];
				if (retension_data[retension_data_idx + 1] - retension_data_chunk_idx == 1)
				{
					for (slice_id = slice_id_start; slice_id < slice_id_end; slice_id++)
					{

						slice_offset[1] = slice_id * SLICE;

						slice_offset[0] = (slice_id + retension_data_chunk_idx) * SLICE;
						slice_offset[2] = (slice_id + k - retension_data_chunk_idx - 2) * SLICE;	   // for retension_data_chunk_idx + 1
						slice_offset[3] = (slice_id + (k - 1 - retension_data_chunk_idx) * 2) * SLICE; // for retension_data_chunk_idx

						mxx0 = _mm256_load_si256((__m256i *)&data[retension_data_chunk_idx][slice_offset[1]]);
						mxx1 = _mm256_load_si256((__m256i *)&data[retension_data_chunk_idx + 1][slice_offset[1]]);

						// if (erasure_data[0] != -1)
						if (need_decoded[0])
						{
							if (slice_id == slice_id_start)
							{
								res = mxx0;
							}
							else
							{
								res = _mm256_xor_si256(mxx0, mxx3);
							}
							tmp = _mm256_load_si256((__m256i *)&coding[0][slice_offset[0]]);
							res = _mm256_xor_si256(res, tmp);
							_mm256_store_si256((__m256i *)&coding[0][slice_offset[0]], res);
							if (slice_id == slice_id_end - 1)
							{
								tmp = _mm256_load_si256((__m256i *)&coding[0][slice_offset[0] + SLICE]);
								res = _mm256_xor_si256(mxx1, tmp);
								_mm256_store_si256((__m256i *)&coding[0][slice_offset[0] + SLICE], res);
							}
						}
						// if (erasure_data[1] != -1)
						if (need_decoded[1])
						{

							res = _mm256_xor_si256(mxx1, mxx0);
							tmp = _mm256_load_si256((__m256i *)&coding[1][slice_offset[1]]);
							res = _mm256_xor_si256(res, tmp);
							_mm256_store_si256((__m256i *)&coding[1][slice_offset[1]], res);
						}
						// if (erasure_data[2] != -1)
						if (need_decoded[2])
						{
							if (slice_id == slice_id_start)
							{
								res = mxx1;
							}
							else
							{
								res = _mm256_xor_si256(mxx1, mxx2);
							}
							tmp = _mm256_load_si256((__m256i *)&coding[2][slice_offset[2]]);
							res = _mm256_xor_si256(res, tmp);
							_mm256_store_si256((__m256i *)&coding[2][slice_offset[2]], res);
							if (slice_id == slice_id_end - 1)
							{
								tmp = _mm256_load_si256((__m256i *)&coding[2][slice_offset[2] + SLICE]);
								res = _mm256_xor_si256(mxx0, tmp);
								_mm256_store_si256((__m256i *)&coding[2][slice_offset[2] + SLICE], res);
							}
						}
						if (need_decoded[3])
						{
							tmp = _mm256_load_si256((__m256i *)&coding[3][slice_offset[3]]);
							res = _mm256_xor_si256(mxx0, tmp);
							_mm256_store_si256((__m256i *)&coding[3][slice_offset[3]], res);

							tmp = _mm256_load_si256((__m256i *)&coding[3][slice_offset[3] - 2 * SLICE]);
							res = _mm256_xor_si256(mxx1, tmp);
							_mm256_store_si256((__m256i *)&coding[3][slice_offset[3] - 2 * SLICE], res);
						}

						mxx2 = mxx0;
						mxx3 = mxx1;
					}

					retension_data_idx++;
				}
				else
				{
					for (slice_id = slice_id_start; slice_id < slice_id_end; slice_id++)
					{

						slice_offset[1] = slice_id * SLICE;

						mxx1 = _mm256_load_si256((__m256i *)&data[retension_data_chunk_idx][slice_offset[1]]);

						slice_offset[0] = (slice_id + retension_data_chunk_idx) * SLICE;
						slice_offset[2] = (slice_id + k - 1 - retension_data_chunk_idx) * SLICE;
						slice_offset[3] = (slice_id + (k - 1 - retension_data_chunk_idx) * 2) * SLICE;

						for (i = 0; i < 4; i++)
						{
							// if (erasure_data[i] != -1)
							if (need_decoded[i])
							{
								tmp = _mm256_load_si256((__m256i *)&coding[i][slice_offset[i]]);
								mxx2 = _mm256_xor_si256(mxx1, tmp);
								_mm256_store_si256((__m256i *)&coding[i][slice_offset[i]], mxx2);
							}
						}
					}
				}
			}
		}

		// elimination
		if (window_id > 0)
		{
			const int solve_slice_id_start = (window_id - 1) * window_size;
			const int solve_slice_id_end = solve_slice_id_start + window_size;

			// parity4 for the first window
			if (need_decoded[3] && window_id == 1)
			{
				int erasure_data_idx = erasure_data[3];
				for (slice_id = solve_slice_id_start; slice_id < solve_slice_id_start + offset4; slice_id++)
				{
					slice_offset[0] = (slice_id + erasure_data_idx) * SLICE;
					slice_offset[1] = slice_id * SLICE;
					slice_offset[2] = (slice_id + k - 1 - erasure_data_idx) * SLICE;
					slice_offset[3] = (slice_id + (k - 1 - erasure_data_idx) * 2) * SLICE;

					mxx1 = _mm256_load_si256((__m256i *)&coding[3][slice_offset[3]]);
					_mm256_stream_si256((__m256i *)&data[erasure_data_idx][slice_offset[1]], mxx1);

					for (j = 0; j < 3; j++)
					{
						if (need_decoded[j])
						{
							tmp = _mm256_load_si256((__m256i *)&coding[j][slice_offset[j]]);
							mxx2 = _mm256_xor_si256(mxx1, tmp);
							_mm256_store_si256((__m256i *)&coding[j][slice_offset[j]], mxx2);
						}
					}
				}
			}

			for (slice_id = solve_slice_id_start; slice_id < solve_slice_id_end; slice_id++)
			{
				slice_offset[1] = slice_id * SLICE;
				if (need_decoded[0])
				{
					int erasure_data_idx = erasure_data[0];
					slice_offset[0] = (slice_id + erasure_data_idx) * SLICE;

					mxx1 = _mm256_load_si256((__m256i *)&coding[0][slice_offset[0]]);
					_mm256_stream_si256((__m256i *)&data[erasure_data_idx][slice_offset[1]], mxx1);

					if (need_decoded[1])
					{
						tmp = _mm256_load_si256((__m256i *)&coding[1][slice_offset[1]]);
						mxx2 = _mm256_xor_si256(mxx1, tmp);
						_mm256_store_si256((__m256i *)&coding[1][slice_offset[1]], mxx2);
					}

					if (need_decoded[2])
					{
						slice_offset[2] = (slice_id + k - 1 - erasure_data_idx) * SLICE;
						tmp = _mm256_load_si256((__m256i *)&coding[2][slice_offset[2]]);
						mxx2 = _mm256_xor_si256(mxx1, tmp);
						_mm256_store_si256((__m256i *)&coding[2][slice_offset[2]], mxx2);
					}

					if (need_decoded[3])
					{
						slice_offset[3] = (slice_id + (k - 1 - erasure_data_idx) * 2) * SLICE;
						tmp = _mm256_load_si256((__m256i *)&coding[3][slice_offset[3]]);
						mxx2 = _mm256_xor_si256(mxx1, tmp);
						_mm256_store_si256((__m256i *)&coding[3][slice_offset[3]], mxx2);
					}
				}

				if (need_decoded[3])
				{
					slice_id += offset4;
					slice_offset[1] = slice_id * SLICE;

					int erasure_data_idx = erasure_data[3];
					slice_offset[3] = (slice_id + (k - 1 - erasure_data_idx) * 2) * SLICE;

					mxx1 = _mm256_load_si256((__m256i *)&coding[3][slice_offset[3]]);
					_mm256_stream_si256((__m256i *)&data[erasure_data_idx][slice_offset[1]], mxx1);

					if (need_decoded[0])
					{
						slice_offset[0] = (slice_id + erasure_data_idx) * SLICE;
						tmp = _mm256_load_si256((__m256i *)&coding[0][slice_offset[0]]);
						mxx2 = _mm256_xor_si256(mxx1, tmp);
						_mm256_store_si256((__m256i *)&coding[0][slice_offset[0]], mxx2);
					}

					if (need_decoded[1])
					{
						tmp = _mm256_load_si256((__m256i *)&coding[1][slice_offset[1]]);
						mxx2 = _mm256_xor_si256(mxx1, tmp);
						_mm256_store_si256((__m256i *)&coding[1][slice_offset[1]], mxx2);
					}

					if (need_decoded[2])
					{
						slice_offset[2] = (slice_id + k - 1 - erasure_data_idx) * SLICE;
						tmp = _mm256_load_si256((__m256i *)&coding[2][slice_offset[2]]);
						mxx2 = _mm256_xor_si256(mxx1, tmp);
						_mm256_store_si256((__m256i *)&coding[2][slice_offset[2]], mxx2);
					}

					slice_id -= offset4;
					slice_offset[1] = slice_id * SLICE;
				}

				if (need_decoded[2])
				{
					int erasure_data_idx = erasure_data[2];
					slice_offset[2] = (slice_id + k - 1 - erasure_data_idx) * SLICE;

					mxx1 = _mm256_load_si256((__m256i *)&coding[2][slice_offset[2]]);
					_mm256_stream_si256((__m256i *)&data[erasure_data_idx][slice_offset[1]], mxx1);

					if (need_decoded[0])
					{
						slice_offset[0] = (slice_id + erasure_data_idx) * SLICE;
						tmp = _mm256_load_si256((__m256i *)&coding[0][slice_offset[0]]);
						mxx2 = _mm256_xor_si256(mxx1, tmp);
						_mm256_store_si256((__m256i *)&coding[0][slice_offset[0]], mxx2);
					}

					if (need_decoded[1])
					{
						tmp = _mm256_load_si256((__m256i *)&coding[1][slice_offset[1]]);
						mxx2 = _mm256_xor_si256(mxx1, tmp);
						_mm256_store_si256((__m256i *)&coding[1][slice_offset[1]], mxx2);
					}

					if (need_decoded[3])
					{
						slice_offset[3] = (slice_id + (k - 1 - erasure_data_idx) * 2) * SLICE;
						tmp = _mm256_load_si256((__m256i *)&coding[3][slice_offset[3]]);
						mxx2 = _mm256_xor_si256(mxx1, tmp);
						_mm256_store_si256((__m256i *)&coding[3][slice_offset[3]], mxx2);
					}
				}

				if (need_decoded[1])
				{
					int erasure_data_idx = erasure_data[1];

					mxx1 = _mm256_load_si256((__m256i *)&coding[1][slice_offset[1]]);
					_mm256_stream_si256((__m256i *)&data[erasure_data_idx][slice_offset[1]], mxx1);

					if (need_decoded[0])
					{
						slice_offset[0] = (slice_id + erasure_data_idx) * SLICE;
						tmp = _mm256_load_si256((__m256i *)&coding[0][slice_offset[0]]);
						mxx2 = _mm256_xor_si256(mxx1, tmp);
						_mm256_store_si256((__m256i *)&coding[0][slice_offset[0]], mxx2);
					}

					if (need_decoded[2])
					{
						slice_offset[2] = (slice_id + k - 1 - erasure_data_idx) * SLICE;
						tmp = _mm256_load_si256((__m256i *)&coding[2][slice_offset[2]]);
						mxx2 = _mm256_xor_si256(mxx1, tmp);
						_mm256_store_si256((__m256i *)&coding[2][slice_offset[2]], mxx2);
					}

					if (need_decoded[3])
					{
						slice_offset[3] = (slice_id + (k - 1 - erasure_data_idx) * 2) * SLICE;
						tmp = _mm256_load_si256((__m256i *)&coding[3][slice_offset[3]]);
						mxx2 = _mm256_xor_si256(mxx1, tmp);
						_mm256_store_si256((__m256i *)&coding[3][slice_offset[3]], mxx2);
					}
				}
			}
		}
	}
}

void two_tone_decode_data_compare(int k, int m, int blocksize, uint8_t **chunks, int *eraseds)
{
	int eraseds_cnt = 0;
	int coding_data_map[Mval] = {0, 0, k - 1};
	int need_decoded[Mval] = {-1, -1, -1};

	int erasure_data[Mval];
	int erasure_data_count = 0;

	int retension_data[k];
	int retension_data_count = 0;

	int retension_coding_count = 0;

	uint8_t *data[k];
	uint8_t *coding[Mval];
	// int last_erasure_coding_idx = -1;

	uint8_t *maperasure_ptr[Mval] = {NULL, NULL, NULL};
	uint8_t *parity_ptr[Mval] = {NULL, NULL, NULL};

	int i, j;
	for (i = 0; i < k + m; i++)
	{
		if (i < k)
		{
			// 数据块
			data[i] = chunks[i];
			if (eraseds[eraseds_cnt] == i)
			{
				erasure_data[erasure_data_count] = i;
				erasure_data_count++;
				eraseds_cnt++;
			}
			else
			{
				retension_data[retension_data_count] = i;
				retension_data_count++;
			}
		}
		else
		{
			// 校验块
			int parity_idx = i - k;
			coding[parity_idx] = chunks[i];
			if (eraseds[eraseds_cnt] != i)
			{
				if (retension_coding_count < erasure_data_count)
				{
					need_decoded[parity_idx] = (coding_data_map[parity_idx] = erasure_data[retension_coding_count]);
					maperasure_ptr[parity_idx] = data[coding_data_map[parity_idx]];
					parity_ptr[parity_idx] = coding[parity_idx];
				}
				retension_coding_count++;
			}
			else
			{
				// maperasure_ptr[parity_idx] = coding[parity_idx];
				// last_erasure_coding_idx = parity_idx;
				eraseds_cnt++;
			}
		}
	}

	retension_data[retension_data_count] = -1;
	// if (erasure_data_count == 1 && retension_coding_count == Mval)
	// {
	// 	return decode_one_data_chunk(data, coding[1], blocksize, erasure_data[0]);
	// }

	const int TOTAL_SLICE_NUM = blocksize / SLICE;
	const int window_size = DECODE_WINDOW_SIZE;
	const int windows_num = TOTAL_SLICE_NUM / window_size;

	long slice_offset[3];
	__m256i mxx0, mxx1, mxx2, mxx3, tmp;
	__m256i res[3];

	// uint8_t *zeros = (uint8_t *)aligned_alloc(SLICE, SLICE);
	// _mm256_store_si256((__m256i *)zeros, _mm256_setzero_si256());

	int window_id, slice_id, retension_data_idx;
	for (window_id = -1; window_id <= windows_num; window_id++)
	{
		// init maperasure by parity
		if (window_id < windows_num)
		{
			const int slice_id_start = (window_id + 1) * window_size;
			const int slice_id_end = slice_id_start + window_size;

			slice_offset[0] = (slice_id_start - coding_data_map[0]) * SLICE;
			slice_offset[1] = slice_id_start * SLICE;
			slice_offset[2] = (slice_id_start - (k - 1) + coding_data_map[2]) * SLICE;
			for (i = 0; i < 3; i++)
			{
				if (maperasure_ptr[i] != NULL)
				{
					for (slice_id = slice_id_start; slice_id < slice_id_end; slice_id++)
					{
						if (slice_offset[i] >= 0)
						{
							tmp = _mm256_load_si256((__m256i *)&parity_ptr[i][slice_id * SLICE]);
							_mm256_store_si256((__m256i *)&maperasure_ptr[i][slice_offset[i]], tmp);
						}
						slice_offset[i] += SLICE;
					}
				}
			}
		}

		// substitute
		if (0 <= window_id && window_id < windows_num)
		{
			const int slice_id_start = window_id * window_size;
			const int slice_id_end = slice_id_start + window_size;

			for (retension_data_idx = 0; retension_data_idx < retension_data_count; retension_data_idx++)
			{

				int retension_data_chunk_idx = retension_data[retension_data_idx];
				if (retension_data[retension_data_idx + 1] - retension_data_chunk_idx == 1)
				{
					// mxx2 = _mm256_load_si256((__m256i *)data[retension_data_chunk_idx][(slice_id_start - 1) * SLICE]);
					// mxx3 = _mm256_load_si256((__m256i *)data[retension_data_chunk_idx + 1][(slice_id_start - 1) * SLICE]);
					for (slice_id = slice_id_start; slice_id < slice_id_end; slice_id++)
					{

						slice_offset[1] = slice_id * SLICE;
						slice_offset[0] = (slice_id + retension_data_chunk_idx - coding_data_map[0]) * SLICE;
						slice_offset[2] = (slice_id + coding_data_map[2] - retension_data_chunk_idx - 1) * SLICE; // for retension_data_chunk_idx + 1

						mxx0 = _mm256_load_si256((__m256i *)&data[retension_data_chunk_idx][slice_offset[1]]);
						mxx1 = _mm256_load_si256((__m256i *)&data[retension_data_chunk_idx + 1][slice_offset[1]]);

						if (maperasure_ptr[0] != NULL && slice_offset[0] >= 0)
						{
							if (slice_id == slice_id_start)
							{
								res[0] = mxx0;
							}
							else
							{
								res[0] = _mm256_xor_si256(mxx0, mxx3);
							}
							tmp = _mm256_load_si256((__m256i *)&maperasure_ptr[0][slice_offset[0]]);
							res[0] = _mm256_xor_si256(res[0], tmp);
							_mm256_store_si256((__m256i *)&maperasure_ptr[0][slice_offset[0]], res[0]);
							if (slice_id == slice_id_end - 1)
							{
								tmp = _mm256_load_si256((__m256i *)&maperasure_ptr[0][slice_offset[0] + SLICE]);
								res[0] = _mm256_xor_si256(mxx1, tmp);
								_mm256_store_si256((__m256i *)&maperasure_ptr[0][slice_offset[0] + SLICE], res[0]);
							}
						}
						if (maperasure_ptr[1] != NULL && slice_offset[1] >= 0)
						{

							res[1] = _mm256_xor_si256(mxx1, mxx0);
							tmp = _mm256_load_si256((__m256i *)&maperasure_ptr[1][slice_offset[1]]);
							res[1] = _mm256_xor_si256(res[1], tmp);
							_mm256_store_si256((__m256i *)&maperasure_ptr[1][slice_offset[1]], res[1]);
						}
						if (maperasure_ptr[2] != NULL && slice_offset[2] >= 0)
						{
							if (slice_id == slice_id_start)
							{
								res[2] = mxx1;
							}
							else
							{
								res[2] = _mm256_xor_si256(mxx1, mxx2);
							}
							tmp = _mm256_load_si256((__m256i *)&maperasure_ptr[2][slice_offset[2]]);
							res[2] = _mm256_xor_si256(res[2], tmp);
							_mm256_store_si256((__m256i *)&maperasure_ptr[2][slice_offset[2]], res[2]);
							if (slice_id == slice_id_end - 1)
							{
								tmp = _mm256_load_si256((__m256i *)&maperasure_ptr[2][slice_offset[2] + SLICE]);
								res[2] = _mm256_xor_si256(mxx0, tmp);
								_mm256_store_si256((__m256i *)&maperasure_ptr[2][slice_offset[2] + SLICE], res[2]);
							}
						}

						mxx2 = mxx0;
						mxx3 = mxx1;
					}

					retension_data_idx++;
				}
				else
				{
					for (slice_id = slice_id_start; slice_id < slice_id_end; slice_id++)
					{

						slice_offset[1] = slice_id * SLICE;

						mxx1 = _mm256_load_si256((__m256i *)&data[retension_data_chunk_idx][slice_offset[1]]);

						slice_offset[0] = (slice_id + retension_data_chunk_idx - coding_data_map[0]) * SLICE;
						slice_offset[2] = (slice_id + coding_data_map[2] - retension_data_chunk_idx) * SLICE;

						for (i = 0; i < 3; i++)
						{
							if (maperasure_ptr[i] != NULL && slice_offset[i] >= 0)
							{
								tmp = _mm256_load_si256((__m256i *)&maperasure_ptr[i][slice_offset[i]]);
								mxx2 = _mm256_xor_si256(mxx1, tmp);
								_mm256_store_si256((__m256i *)&maperasure_ptr[i][slice_offset[i]], mxx2);
							}
						}
					}
				}
			}
		}

		// elimination
		if (window_id > 0)
		{
			const int solve_slice_id_start = (window_id - 1) * window_size;
			const int solve_slice_id_end = solve_slice_id_start + window_size;

			for (slice_id = solve_slice_id_start; slice_id < solve_slice_id_end; slice_id++)
			{
				slice_offset[1] = slice_id * SLICE;
				int two_tone_idx[5] = {0, 2, 1, 0, 2};
				for (i = 0; i < 3; i++)
				{

					int maperasure_idx = two_tone_idx[i];
					if (maperasure_ptr[maperasure_idx] != NULL)
					{

						mxx1 = _mm256_load_si256((__m256i *)&data[coding_data_map[maperasure_idx]][slice_offset[1]]);

						slice_offset[0] = (slice_id + coding_data_map[maperasure_idx] - coding_data_map[0]) * SLICE;
						slice_offset[2] = (slice_id + coding_data_map[2] - coding_data_map[maperasure_idx]) * SLICE;

						for (j = 1; j < 3; j++)
						{

							int other_idx = two_tone_idx[i + j];
							if (maperasure_ptr[other_idx] != NULL && slice_offset[other_idx] >= 0)
							{

								tmp = _mm256_load_si256((__m256i *)&maperasure_ptr[other_idx][slice_offset[other_idx]]);
								mxx2 = _mm256_xor_si256(mxx1, tmp);
								_mm256_store_si256((__m256i *)&maperasure_ptr[other_idx][slice_offset[other_idx]], mxx2);
							}
						}
					}
				}
			}
		}
	}
}

void two_tone_test(int k, int p, int len, int *eraseds)
{
	struct timespec time1 = {0, 0};
	struct timespec time2 = {0, 0};
	long encode_time = 0;

	int m = p;
	int n = m + k;
	int blocksize = len / k;

	uint8_t *chunks[MMAX];
	uint8_t *copy_ptrs[MMAX];

	alldata = len;
	decodedata = len;

	int i, j, bufsize = blocksize + EXTRA_DATA_SIZE;
	for (i = 0; i < n; i++)
	{
		void *buf;

		if (posix_memalign(&buf, 32, bufsize))
		{
			printf("alloc error: Fail");
			return 1;
		}
		chunks[i] = buf;
	}

	for (i = 0; i < n; i++)
	{
		for (j = 0; j < bufsize; j++)
		{
			// chunks[i][j] = rand();
			chunks[i][j] = 'X';
		}
	}

	for (i = 0; i < en_n; i++)
	{
		clock_gettime(CLOCK_REALTIME, &time1);
		two_tone_encode(k, m, blocksize, chunks);
		// two_tone_encode2(k, m, blocksize, chunks);
		// two_tone_encode4(k, m, blocksize, chunks);
		// encode_base_deforestation(k, p, w, schedule, frag_ptrs, &frag_ptrs[k], alignlen, packetsize);
		clock_gettime(CLOCK_REALTIME, &time2);
		encode_time = (time2.tv_sec - time1.tv_sec) * 1000000000 + (time2.tv_nsec - time1.tv_nsec);
		encode_time_arrs[i] = encode_time;
	}

	for (i = 0; i < n; i++)
	{
		void *buf;
		if (posix_memalign(&buf, 32, bufsize))
		{
			printf("alloc error: Fail");
			return 1;
		}
		copy_ptrs[i] = buf;
	}

	for (i = 0; i < n; i++)
	{
		for (j = 0; j < bufsize; j++)
		{
			copy_ptrs[i][j] = chunks[i][j];
		}
	}

	// print copy_ptrs[0] to log1.txt with hexadecimal
	FILE *fp1 = fopen("log1.txt", "w");
	if (fp1 == NULL)
	{
		printf("Error opening file!\n");
		return;
	}
	for (i = 0; i < blocksize; i++)
	{
		fprintf(fp1, "%02x ", copy_ptrs[0][i]);
		if ((i + 1) % 16 == 0)
			fprintf(fp1, "\n");
	}
	fclose(fp1);

	// print copy_ptrs[k + m - 1] to log3.txt with hexadecimal
	FILE *fp3 = fopen("log3.txt", "w");
	if (fp3 == NULL)
	{
		printf("Error opening file!\n");
		return;
	}
	for (i = 0; i < bufsize; i++)
	{
		fprintf(fp3, "%02x ", copy_ptrs[k + m - 1][i]);
		if ((i + 1) % 16 == 0)
			fprintf(fp3, "\n");
	}
	fclose(fp3);

	int erro_num = 0;
	for (i = 0; i < n; i++)
	{
		if (eraseds[erro_num] == i)
		{
			erro_num++;
			for (j = 0; j < bufsize; j++)
			{
				chunks[i][j] = rand();
			}
		}
	}

	// print eraseds
	printf("erro_num:%d\n", erro_num);
	for (i = 0; i < erro_num; i++)
		printf("e%d ", eraseds[i]);
	printf("\n");

	for (i = 0; i < de_n; i++)
	{
		clock_gettime(CLOCK_REALTIME, &time1);
		// two_tone_decode(k, m, blocksize, chunks, eraseds);
		two_tone_decode_data(k, m, blocksize, chunks, eraseds);
		// two_tone_decode_data2(k, m, blocksize, chunks, eraseds);
		// two_tone_decode_data4(k, m, blocksize, chunks, eraseds);
		// two_tone_decode_data_compare(k, m, blocksize, chunks, eraseds);
		clock_gettime(CLOCK_REALTIME, &time2);
		encode_time = (time2.tv_sec - time1.tv_sec) * 1000000000 + (time2.tv_nsec - time1.tv_nsec);
		decode_time_arrs[i] = encode_time;

		// in place decode
		two_tone_encode(k, m, blocksize, chunks);
		// two_tone_encode2(k, m, blocksize, chunks);
		// two_tone_encode4(k, m, blocksize, chunks);
	}

	// print chunks[0] to log2.txt with hexadecimal
	FILE *fp2 = fopen("log2.txt", "w");
	if (fp2 == NULL)
	{
		printf("Error opening file!\n");
		return;
	}
	for (i = 0; i < blocksize; i++)
	{
		fprintf(fp2, "%02x ", chunks[0][i]);
		if ((i + 1) % 16 == 0)
			fprintf(fp2, "\n");
	}
	fclose(fp2);

	// check ptrs and copy_ptrs
	int flag = 1;
	for (i = 0; i < n; i++)
	{

		if (i == k || i == k + m - 1)
		{
			bufsize = blocksize + EXTRA_DATA_SIZE;
		}
		else
		{
			bufsize = blocksize;
		}
		for (j = 0; j < bufsize; j++)
		{
			if (chunks[i][j] != copy_ptrs[i][j])
			{
				flag = 0;
				printf("error %d %d\n", i, j);
				break;
			}
		}
		if (flag == 0)
			break;
	}
	if (flag)
	{
		printf("decode success\n");
	}
	else
	{
		printf("decode fail\n");
	}
}

int main(int argc, char *argv[])
{
	int i, j;
	int c;
	int k = 8, p = 3, w = 8;
	// int packetsize = 1024;
	int packetsize = 512;
	int e = -1;
	int d = p;
	// int n = 1;
	int n = 1000;
	int l = 16;

	int code = 0; // 0:Cerasure, 1:TSX

	int m = k + p;
	int *eraseds = (int *)malloc(sizeof(int) * m);
	int erro_num = 0;

	while ((c = getopt(argc, argv, "k:p:l:s:e:w:d:c:n:")) != -1)
	{
		switch (c)
		{
		case 'k':
			k = atoi(optarg);
			break;
		case 'p':
			p = atoi(optarg);
			break;
		case 'l':
			l = atoi(optarg);
			break;
		case 's':
			packetsize = atoi(optarg);
			break;
		case 'e':
			e = atoi(optarg);
			eraseds[erro_num] = e;
			erro_num++;
			break;
		case 'w':
			w = atoi(optarg);
			break;
		case 'd':
			d = atoi(optarg);
			break;
		case 'c':
			code = atoi(optarg);
			break;
		case 'n':
			n = atoi(optarg);
			break;
			break;
		}
	}

	eraseds[erro_num] = -1;
	// e=2;
	// int flag=0;
	// gen_best_matrix_all(k, p,w);
	// vandermonde_sub_coding_matrix_all(k, p, w,0,&flag);
	// vandermonde_all(w);
	// count_one_R_all(w);
	// int len = 10 * 1024 * 1024;
	int len = l * 1024 * 1024;
	alldata = len;
	decodedata = len;

	en_n = (de_n = n);

	// mds_prove(k,p,w);

	// RS encoding
	// encode_deforest_test(k, p, w, len, packetsize, e);
	if (code == 0)
	{
		encode_deforest_test(k, p, w, len, packetsize, eraseds);
	}
	else if (code == 1)
	{
		two_tone_test(k, p, len, eraseds);
	}

	// the number of ones of matrices, the runtime of searching matrices
	// matrix_test(k,p,w,len,packetsize,e);

	// wide stripe
	// slp_encode_test_split(k,p,w,len,packetsize,e,n);

	// LRC encoding
	// encode_lrc_test(k,p,w,len,packetsize,e,l);

	// multiple thread
	// test_threads_new_time(k,p,w,len,packetsize,e,n);

	long long int encode_time_all = 0;
	long long int decode_time_all = 0;
	for (i = 0; i < en_n; i++)
	{
		encode_time_all += encode_time_arrs[i];
	}

	for (i = 0; i < de_n; i++)
	{
		decode_time_all += decode_time_arrs[i];
	}
	encode_time_all = encode_time_all / en_n;
	decode_time_all = decode_time_all / de_n;
	double encode_v = ((((double)alldata)) / 1024 / 1024 / 1024) / ((double)encode_time_all / 1000 / 1000 / 1000);
	double decode_v = ((((double)decodedata)) / 1024 / 1024 / 1024) / ((double)decode_time_all / 1000 / 1000 / 1000);
	// printf("%ld ",encode_time_all);
	printf("%.2lf\n", encode_v);
	printf("%.2lf\n", decode_v);

	return 0;
}

// int encode_deforest_test(int k, int p, int w, int len, int packetsize, int e)
int encode_deforest_test(int k, int p, int w, int len, int packetsize, int *eraseds)
{
	struct timespec time1 = {0, 0};
	struct timespec time2 = {0, 0};
	long encode_time = 0;

	int i, j, m, a;
	int alignlen;
	// Fragment buffer pointers
	u8 *frag_ptrs[MMAX];
	u8 *copy_ptrs[MMAX];
	u8 *tmp[MMAX];
	unsigned int *encode_matrix;

	int *encode_bitmatrix;
	int *slp;
	int **schedule;
	m = k + p;

	// int datasize = 10 * 1024 * 1024;
	int datasize = len;

	alignlen = datasize / k;
	// align len with w*packetsize
	if (packetsize != 0)
	{
		if (alignlen % (w * packetsize) != 0)
		{
			while (alignlen % (w * packetsize) != 0)
				alignlen++;
		}
	}
	else
	{
		if (alignlen % w != 0)
		{
			while (alignlen % w != 0)
				alignlen++;
		}
	}
	alldata = alignlen * (k);
	decodedata = alignlen * (k);
	// Allocate the src & parity buffers
	for (i = 0; i < m; i++)
	{
		void *buf;
		if (posix_memalign(&buf, 32, alignlen))
		{
			printf("alloc error: Fail");
			return 1;
		}
		frag_ptrs[i] = buf;
	}

	for (i = 0; i < m; i++)
		for (j = 0; j < alignlen; j++)
			frag_ptrs[i][j] = rand();

	encode_matrix = gen_best_matrix_from_any_element_lowtime(k, p, w);

	// for(i=0;i<k*(p);i++)
	// {
	// 	if(i%k==0)
	// 		printf("\n");
	// 	printf("%d ",encode_matrix[i]);
	// }

	encode_bitmatrix = matrix_to_bitmatrix(k, p, w, encode_matrix);
	// printf("%d ",count_one_numbers(k,p,w,encode_bitmatrix));

	// int count=0;
	// for(i=0;i<k*w*p*w;i++)
	// {
	// 	if(encode_bitmatrix[i]==1)
	// 	{
	// 		count++;
	// 	}
	// }

	schedule = bitmatrix_to_smart_schedule(k, p, w, encode_bitmatrix);
	// schedule=bitmatrix_to_schedule(k,p,w,encode_bitmatrix);

	// int index=0;
	// int xor_num = 0;
	// while(schedule[index][0]!=-1)
	// {
	// 	xor_num += (schedule[index][0]-1);
	//     i=0;
	//     index++;
	//     // printf("\n");
	// }
	// printf("%d ",xor_num);

	for (i = 0; i < en_n; i++)
	{
		clock_gettime(CLOCK_REALTIME, &time1);
		slp_encode_best_finally(k, p, w, schedule, frag_ptrs, &frag_ptrs[k], alignlen, packetsize);
		// encode_base_deforestation(k, p, w, schedule, frag_ptrs, &frag_ptrs[k], alignlen, packetsize);
		clock_gettime(CLOCK_REALTIME, &time2);
		encode_time = (time2.tv_sec - time1.tv_sec) * 1000000000 + (time2.tv_nsec - time1.tv_nsec);
		encode_time_arrs[i] = encode_time;
	}

	for (i = 0; i < m; i++)
	{
		void *buf;
		if (posix_memalign(&buf, 32, alignlen))
		{
			printf("alloc error: Fail");
			return 1;
		}
		copy_ptrs[i] = buf;
	}

	for (i = 0; i < m; i++)
	{
		for (j = 0; j < alignlen; j++)
		{
			copy_ptrs[i][j] = frag_ptrs[i][j];
		}
	}

	u8 **ptrs;
	int *decode_bitmatrix;
	int **de_operations;
	int *survives;
	int *recover;
	int *de_recover;

	survives = (int *)malloc(sizeof(int) * (m));
	recover = (int *)malloc(sizeof(int) * (m));
	de_recover = (int *)malloc(sizeof(int) * (m));

	// int *eraseds;
	// eraseds = (int *)malloc(sizeof(int) * (m));
	// int erro_num = e;
	// for (i = 0; i < erro_num; i++)
	// {
	// 	eraseds[i] = i;
	// }
	// eraseds[erro_num] = -1;
	// for (i = 0; i < m - erro_num; i++)
	// {
	// 	survives[i] = i + erro_num;
	// 	de_recover[i] = i + erro_num;
	// }

	int erro_num = 0;
	int survive_num = 0;
	for (i = 0; i < m; i++)
	{
		if (eraseds[erro_num] == i)
		{
			erro_num++;
			for (j = 0; j < alignlen; j++)
			{
				frag_ptrs[i][j] = rand();
			}
		}
		else
		{
			survives[survive_num] = i;
			de_recover[survive_num] = i;
			survive_num++;
		}
	}

	// print eraseds and survives
	printf("erro_num:%d survive_num:%d\n", erro_num, survive_num);
	for (i = 0; i < erro_num; i++)
		printf("e%d ", eraseds[i]);
	printf("\n");
	for (i = 0; i < survive_num; i++)
		printf("s%d ", survives[i]);
	printf("\n");

	decode_bitmatrix = generate_decoding_bitmatrix(k, p, w, encode_bitmatrix, eraseds, de_recover);
	// printf("decode bitmatrix:%d ",count_one_numbers(k,p,w,decode_bitmatrix));

	ptrs = set_up_ptrs_for_opertaions(k, erro_num, eraseds, frag_ptrs, &frag_ptrs[k], de_recover);

	// schedule=bitmatrix_to_schedule(k,erro_num,w,decode_bitmatrix);
	schedule = bitmatrix_to_smart_schedule(k, erro_num, w, decode_bitmatrix);
	// printf("!!\n");

	// printf("!!\n");
	for (i = 0; i < de_n; i++)
	{
		clock_gettime(CLOCK_REALTIME, &time1);
		slp_encode_best_finally(k, erro_num, w, schedule, ptrs, &ptrs[k], alignlen, packetsize);
		clock_gettime(CLOCK_REALTIME, &time2);
		encode_time = (time2.tv_sec - time1.tv_sec) * 1000000000 + (time2.tv_nsec - time1.tv_nsec);
		decode_time_arrs[i] = encode_time;
	}

	// check ptrs and copy_ptrs
	int flag = 1;
	for (i = 0; i < m; i++)
	{
		for (j = 0; j < alignlen; j++)
		{
			if (frag_ptrs[i][j] != copy_ptrs[i][j])
			{
				flag = 0;
				printf("error %d %d\n", i, j);
				break;
			}
		}
		if (flag == 0)
			break;
	}
	if (flag)
	{
		printf("decode success\n");
	}
	else
	{
		printf("decode fail\n");
	}
}

int encode_lrc_test(int k, int p, int w, int len, int packetsize, int e, int l)
{
	struct timespec time1 = {0, 0};
	struct timespec time2 = {0, 0};
	long encode_time = 0;

	int i, j, m, a;
	int alignlen;
	// Fragment buffer pointers
	u8 *frag_ptrs[MMAX];
	u8 *copy_ptrs[MMAX];
	u8 *tmp[MMAX];
	unsigned int *encode_matrix;
	int *encode_bitmatrix;
	int *slp;
	int **schedule;
	m = k + p + l;

	int datasize = 10 * 1024 * 1024;

	alignlen = datasize / k;
	// align len with w*packetsize
	if (packetsize != 0)
	{
		if (alignlen % (w * packetsize) != 0)
		{
			while (alignlen % (w * packetsize) != 0)
				alignlen++;
		}
	}
	else
	{
		if (alignlen % w != 0)
		{
			while (alignlen % w != 0)
				alignlen++;
		}
	}
	alldata = alignlen * (k);
	decodedata = alignlen * (k);
	// Allocate the src & parity buffers
	for (i = 0; i < m; i++)
	{
		void *buf;
		if (posix_memalign(&buf, 32, alignlen))
		{
			printf("alloc error: Fail");
			return 1;
		}
		frag_ptrs[i] = buf;
	}

	for (i = 0; i < m; i++)
		for (j = 0; j < alignlen; j++)
			frag_ptrs[i][j] = rand();

	encode_matrix = gen_best_matrix_from_any_element_lowtime_lrc(k, p, l, w);
	// for(i=0;i<k*(p+l);i++)
	// {
	// 	if(i%k==0)
	// 		printf("\n");
	// 	printf("%d ",encode_matrix[i]);
	// }

	encode_bitmatrix = matrix_to_bitmatrix(k, p + l, w, encode_matrix);
	// printf("%d ",count_one_numbers(k,p,w,encode_bitmatrix));

	// int count=0;
	// for(i=0;i<k*w*p*w;i++)
	// {
	// 	if(encode_bitmatrix[i]==1)
	// 	{
	// 		count++;
	// 	}
	// }

	schedule = bitmatrix_to_smart_schedule(k, p + l, w, encode_bitmatrix);
	// schedule=bitmatrix_to_schedule(k,p,w,encode_bitmatrix);

	// int index=0;
	// int xor_num = 0;

	// while(schedule[index][0]!=-1)
	// {
	// 	xor_num += (schedule[index][0]-1);
	//     i=0;
	//     index++;
	//     // printf("\n");
	// }
	// printf("%d",xor_num);

	for (i = 0; i < en_n; i++)
	{
		clock_gettime(CLOCK_REALTIME, &time1);
		slp_encode_best_finally(k, p + l, w, schedule, frag_ptrs, &frag_ptrs[k], alignlen, packetsize);
		// encode_base_deforestation(k, p, w, schedule, frag_ptrs, &frag_ptrs[k], alignlen, packetsize);
		clock_gettime(CLOCK_REALTIME, &time2);
		encode_time = (time2.tv_sec - time1.tv_sec) * 1000000000 + (time2.tv_nsec - time1.tv_nsec);
		encode_time_arrs[i] = encode_time;
	}

	return 1;
}

int matrix_test(int k, int p, int w, int len, int packetsize, int e)
{
	struct timespec time1 = {0, 0};
	struct timespec time2 = {0, 0};
	long long encode_time = 0;

	int i, j, m, a;
	int alignlen;
	// Fragment buffer pointers
	u8 *frag_ptrs[MMAX];
	u8 *copy_ptrs[MMAX];
	u8 *tmp[MMAX];
	unsigned int *encode_matrix;
	int *encode_bitmatrix;
	int *slp;
	int **schedule;
	m = k + p;

	int datasize = 10 * 1024 * 1024;
	// int datasize=10321920;
	alignlen = datasize / k;
	// align len with w*packetsize
	if (packetsize != 0)
	{
		if (alignlen % (w * packetsize) != 0)
		{
			while (alignlen % (w * packetsize) != 0)
				alignlen++;
		}
	}
	else
	{
		if (alignlen % w != 0)
		{
			while (alignlen % w != 0)
				alignlen++;
		}
	}
	alldata = alignlen * (k);
	decodedata = alignlen * (k);
	// Allocate the src & parity buffers
	for (i = 0; i < m; i++)
	{
		void *buf;
		if (posix_memalign(&buf, 32, alignlen))
		{
			printf("alloc error: Fail");
			return 1;
		}
		frag_ptrs[i] = buf;
		// frag_ptrs_copy[i] = frag_ptrs[i];
	}

	for (i = 0; i < m; i++)
		for (j = 0; j < alignlen; j++)
			frag_ptrs[i][j] = rand();

	clock_gettime(CLOCK_REALTIME, &time1);
	// encode_matrix=cauchy_good_general_coding_matrix(k,p,w);
	encode_matrix = gen_best_matrix_all(k, p, w);
	// encode_matrix=gen_best_matrix_from_any_element(k,p,w);
	// encode_matrix=gen_best_matrix_from_any_element_lowtime(k,p,w);
	clock_gettime(CLOCK_REALTIME, &time2);
	encode_bitmatrix = matrix_to_bitmatrix(k, p, w, encode_matrix);
	encode_time = (time2.tv_sec - time1.tv_sec) * 1000000000 + (time2.tv_nsec - time1.tv_nsec);
	printf("%ld ", encode_time);

	// printf("%d ",count_one_numbers(k,p,w,encode_bitmatrix));

	int **smart_operations = smart_bitmatrix_to_schedule(k, p, w, encode_bitmatrix);
	// int count_of_smart_operations=count_xor_num_of_smart_schedule(smart_operations);
	// printf("%d ",count_of_smart_operations);

	int least = the_least_one_num_of_matrix_in_gf(k, p, w);
	// printf("%d",least);

	// int split=1;
	// find_best_split(k,p,w,split,encode_bitmatrix);

	// int k_sub=k/split;
	// int* sub_encode_matrix=malloc(sizeof(int)*p*k_sub);
	// int s;
	// int total=0;
	// for(s=0;s<split;s++)
	// {
	// 	for(i=0;i<p;i++)
	// 	{
	// 		for(j=0;j<k_sub;j++)
	// 		{
	// 			sub_encode_matrix[i*k_sub+j]=encode_matrix[i*k+j+s*k_sub];
	// 		}
	// 	}
	// 	encode_bitmatrix=matrix_to_bitmatrix(k_sub, p, w, sub_encode_matrix);
	// 	smart_operations=smart_bitmatrix_to_schedule(k_sub,p,w,encode_bitmatrix);
	// 	count_of_smart_operations=count_xor_num_of_smart_schedule(smart_operations);
	// 	total+=count_of_smart_operations;
	// }
	// total+=(split-1)*w*p;
	// printf("%d\n",total);

	return 1;
}

int V_serach_test(int k, int p, int w)
{
	struct timespec time1 = {0, 0};
	struct timespec time2 = {0, 0};
	long encode_time = 0;

	int i, j, m, a;
	int alignlen;
	// Fragment buffer pointers
	u8 *frag_ptrs[MMAX];
	u8 *copy_ptrs[MMAX];
	u8 *tmp[MMAX];
	unsigned int *encode_matrix;
	int *encode_bitmatrix;
	int *slp;
	int **schedule;
	int ***operations;
	m = k + p;
	clock_gettime(CLOCK_REALTIME, &time1);
	encode_matrix = gen_best_matrix_all(k, p, w);
	clock_gettime(CLOCK_REALTIME, &time2);
	encode_time = (time2.tv_sec - time1.tv_sec) * 1000000000 + (time2.tv_nsec - time1.tv_nsec);
	printf("%lf\n", encode_time / 1000000000.0);

	encode_bitmatrix = matrix_to_bitmatrix(k, p, w, encode_matrix);
	int count = 0;
	for (i = 0; i < k * w * p * w; i++)
	{
		if (encode_bitmatrix[i] == 1)
		{
			count++;
		}
	}
	int count_old = 0;
	encode_matrix = cauchy_good_general_coding_matrix(k, p, w);
	encode_bitmatrix = matrix_to_bitmatrix(k, p, w, encode_matrix);
	for (i = 0; i < k * w * p * w; i++)
	{
		if (encode_bitmatrix[i] == 1)
		{
			count_old++;
		}
	}
	printf("%d %d %lf\n", count, count_old, 1 - count / (double)count_old);
	return 1;
}

int slp_encode_test_split(int k, int p, int w, int len, int packetsize, int e, int n)
{
	struct timespec time1 = {0, 0};
	struct timespec time2 = {0, 0};
	long encode_time = 0;

	int i, j, m, a;
	int alignlen;
	// Fragment buffer pointers
	u8 *frag_ptrs[MMAX];
	u8 *tmp_ptrs;
	u8 **tmp_parity[p * w];
	u8 *copy_ptrs[MMAX];
	u8 *tmp[MMAX];
	unsigned int *encode_matrix;
	int *encode_bitmatrix;
	int *slp;
	int **schedule;
	int ***operations;
	m = k + p;

	int datasize = 10 * 1024 * 1024;
	// int datasize=10321920;
	alignlen = datasize / k;
	// align len with w*packetsize
	if (packetsize != 0)
	{
		if (alignlen % (w * packetsize) != 0)
		{
			while (alignlen % (w * packetsize) != 0)
				alignlen++;
		}
	}
	else
	{
		if (alignlen % w != 0)
		{
			while (alignlen % w != 0)
				alignlen++;
		}
	}
	alldata = alignlen * (k);
	decodedata = alignlen * (k);
	// Allocate the src & parity buffers
	for (i = 0; i < m; i++)
	{
		void *buf;
		if (posix_memalign(&buf, 32, alignlen))
		{
			printf("alloc error: Fail");
			return 1;
		}
		frag_ptrs[i] = buf;
		// frag_ptrs_copy[i] = frag_ptrs[i];
	}

	for (i = 0; i < 1; i++)
	{
		void *buf;
		if (posix_memalign(&buf, 32, w * packetsize * p * n))
		{
			printf("alloc error: Fail");
			return 1;
		}
		tmp_ptrs = buf;
	}

	for (i = 0; i < k; i++)
		for (j = 0; j < alignlen; j++)
			frag_ptrs[i][j] = rand();

	for (i = k; i < m; i++)
		for (j = 0; j < alignlen; j++)
			frag_ptrs[i][j] = rand();

	for (i = 0; i < w * packetsize * p * n; i++)
		tmp_ptrs[i] = rand();

	for (j = 0; j < p * w; j++)
	{
		tmp_parity[j] = malloc(sizeof(long) * n);
	}
	int tmp_ptr_index = 0;
	for (i = 0; i < p * w; i++)
	{
		for (j = 0; j < n; j++)
		{
			tmp_parity[i][j] = tmp_ptrs + (i * n + j) * packetsize;
			// tmp_parity[i][j]=malloc(sizeof(u8)*packetsize);
			// for(a=0;a<packetsize;a++)
			// 	tmp_parity[i][j][a]=0;
		}
	}

	// encode_matrix=gen_best_matrix_all(k,p,w);
	encode_matrix = cauchy_good_general_coding_matrix(k, p, w);

	encode_bitmatrix = matrix_to_bitmatrix(k, p, w, encode_matrix);
	// printf("%d ",count_one_numbers(k,p,w,encode_bitmatrix));

	// operations=bitmatrix_to_schedule_split(k,p,w,n,encode_bitmatrix);
	operations = bitmatrix_to_smart_schedule_split(k, p, w, n, encode_bitmatrix);

	// printf("\n");
	// int index=0;
	// int total=0;
	// for(i=0;i<n;i++)
	// {
	// 	for(index=0;index<p*w;index++)
	// 	{
	// 		total+=operations[i][index][0]-1;
	// 		j=1;
	// 		// printf("%d  %d,%d  ",operations[i][index][0],operations[i][index][j],operations[i][index][j+1]);
	// 		j=3;
	// 		while(j <= 2 * operations[i][index][0]+2)
	// 		{
	// 			// printf("%d,%d  ",operations[i][index][j],operations[i][index][j+1]);
	// 			j=j+2;
	// 		}
	// 		// printf("\n");
	// 	}
	// 	// printf("\n");
	// }
	// total+=(n-1)*w*p;
	// printf("%d\n",total);

	// printf("%d\n",count);
	//  for(i=0;i<n;i++)
	//  {
	//  	int index=0;
	//  	while(operations[i][index][0]>0)
	//  	{
	//  		j=0;
	//  		while(j <= 2 * operations[i][index][0]+2)
	//  		{
	//  			printf("%d ",operations[i][index][j]);
	//  			j++;
	//  		}
	//  		index++;
	//  		printf("\n");
	//  	}
	//  }

	for (i = 0; i < en_n; i++)
	{
		clock_gettime(CLOCK_REALTIME, &time1);

		slp_encode_best_spilt_with_smart_schedule(k, p, w, operations, frag_ptrs, &frag_ptrs[k], tmp_parity, n, alignlen, packetsize);

		// slp_encode_best_spilt(k, p, w, operations, frag_ptrs, &frag_ptrs[k],n,alignlen, packetsize);

		clock_gettime(CLOCK_REALTIME, &time2);
		encode_time = (time2.tv_sec - time1.tv_sec) * 1000000000 + (time2.tv_nsec - time1.tv_nsec);
		encode_time_arrs[i] = encode_time;
	}

	// for (i = 0; i < m; i++) {
	// 	void *buf;
	//     if (posix_memalign(&buf, 32, alignlen)) {
	//         printf("alloc error: Fail");
	//         return 1;
	//     }
	//     copy_ptrs[i] = buf;
	// }

	// for(i=0;i<k+p;i++)
	// {
	// 	for(j=0;j<alignlen;j++)
	// 	{
	// 		copy_ptrs[i][j] = frag_ptrs[i][j];
	// 	}
	// }
	// //printf("%d\n",e);
	// u8 **ptrs;
	// int *decode_bitmatrix;
	// int **de_operations;
	// int *eraseds;
	// int *survives;
	// int *recover;
	// int *de_recover;

	// eraseds = (int *)malloc(sizeof(int)*(m));
	// survives = (int *)malloc(sizeof(int)*(m));
	// recover = (int *)malloc(sizeof(int)*(m));
	// de_recover = (int *)malloc(sizeof(int)*(m));
	// int erro_num=e;

	// for(i=0;i < erro_num;i++)
	// {
	// 	eraseds[i]=i;
	// }
	// eraseds[erro_num] = -1;
	// for(i=0;i<m-erro_num;i++)
	// {
	// 	survives[i]=i+erro_num;
	// 	de_recover[i] = i+ erro_num;
	// }
	// decode_bitmatrix = generate_decoding_bitmatrix(k, erro_num, w, encode_bitmatrix, eraseds,de_recover);
	// ptrs = set_up_ptrs_for_opertaions(k, erro_num, eraseds, frag_ptrs, &frag_ptrs[k],de_recover);
	// // operations=bitmatrix_to_schedule_split(k,erro_num,w,n,decode_bitmatrix);
	// operations=bitmatrix_to_smart_schedule_split(k,erro_num,w,n,decode_bitmatrix);

	// int value;
	// for(i=0;i<value/w-k-p+1;i++)
	// {
	// 	void *buf;
	// 	posix_memalign(&buf, 32, packetsize * w);
	//     tmp[i] = buf;
	// }
	// for(i=0;i<de_n;i++)
	// {
	// 	clock_gettime(CLOCK_REALTIME, &time1);

	// 	slp_encode_best_spilt_with_smart_schedule(k, erro_num, w, operations, ptrs, &ptrs[k],tmp_parity,n,alignlen, packetsize);

	// 	// slp_encode_best_spilt(k, erro_num, w, operations, ptrs, &ptrs[k],n,alignlen, packetsize);

	// 	clock_gettime(CLOCK_REALTIME, &time2);
	// 	encode_time=(time2.tv_sec-time1.tv_sec)*1000000000+(time2.tv_nsec-time1.tv_nsec);
	// 	decode_time_arrs[i]=encode_time;
	// }

	// printf("!!!\n");
	// printf("%d\n",alignlen);
	//  for(i=0;i<m;i++)
	//  {
	//  	for(j=0;j<alignlen;j++)
	//  	{
	//  		if(copy_ptrs[i][j] != frag_ptrs[i][j])
	//  		{
	//  			printf("%d %d error!!!!!\n",i,j);
	//  			//printf("%d %d error!!!!!\n",copy_ptrs[i][j],frag_ptrs[i][j]);
	//  			break;
	//  		}
	//  	}
	//  	//printf("%d %d\n",copy_ptrs[i][0],frag_ptrs[i][0]);
	//  }
	return 1;
}

int slp_encode_test_split_matirx(int k, int p, int w, int len, int packetsize, int e)
{
	int n = 10;
	struct timespec time1 = {0, 0};
	struct timespec time2 = {0, 0};
	long encode_time = 0;

	int i, j, m, a;
	int alignlen;
	// Fragment buffer pointers
	u8 *frag_ptrs[MMAX];
	u8 *copy_ptrs[MMAX];
	u8 *tmp[MMAX];
	unsigned int *encode_matrix;
	int *encode_bitmatrix;
	int *slp;
	int **schedule;
	int ***operations;
	m = k + p;

	int datasize = 10 * 1024 * 1024;
	// int datasize=10321920;
	alignlen = datasize / k;
	// align len with w*packetsize
	if (packetsize != 0)
	{
		if (alignlen % (w * packetsize) != 0)
		{
			while (alignlen % (w * packetsize) != 0)
				alignlen++;
		}
	}
	else
	{
		if (alignlen % w != 0)
		{
			while (alignlen % w != 0)
				alignlen++;
		}
	}
	alldata = alignlen * (k);
	decodedata = alignlen * (k);
	// Allocate the src & parity buffers
	for (i = 0; i < m; i++)
	{
		void *buf;
		if (posix_memalign(&buf, 32, alignlen))
		{
			printf("alloc error: Fail");
			return 1;
		}
		frag_ptrs[i] = buf;
		// frag_ptrs_copy[i] = frag_ptrs[i];
	}

	for (i = 0; i < m; i++)
		for (j = 0; j < alignlen; j++)
			frag_ptrs[i][j] = rand();

	encode_matrix = gen_best_matrix_all(k, p, w);
	// encode_matrix=cauchy_good_general_coding_matrix(k,p,w);
	//  for(i=0;i<k*p;i++)
	//  {
	//  	printf("%d ",encode_matrix[i]);
	//  }
	encode_bitmatrix = matrix_to_bitmatrix(k, p, w, encode_matrix);
	operations = bitmatrix_to_schedule_split(k, p, w, n, encode_bitmatrix);
	for (i = 0; i < n; i++)
	{
		int index = 0;
		while (operations[i][index][0] > 0)
		{
			j = 0;
			while (j <= 2 * operations[i][index][0] + 2)
			{
				printf("%d ", operations[i][index][j]);
				j++;
			}
			index++;
			printf("\n");
		}
	}

	return 1;
}

void *encode_threads(void *pt)
{
	int i, j;
	ptInfo *ptinfo = (ptInfo *)pt;
	int k = ptinfo->k;
	int p = ptinfo->p;
	int w = ptinfo->w;
	int **schedule = ptinfo->schedule;
	u8 **src = ptinfo->src;
	u8 **dest = ptinfo->dest;
	int len = ptinfo->len;
	int packetsize = ptinfo->packetsize;
	int value = ptinfo->value;
	int begin = ptinfo->begin;
	int n = ptinfo->n;
	u8 *copy_ptr[k + p];

	// 构造指针加法
	long *add_offset;
	long *buf;
	if (posix_memalign(&buf, 32, 16 * sizeof(long)))
	{
		return 1;
	}
	add_offset = buf;
	long add_value = n * w * packetsize;
	for (i = 0; i < 16; i++)
	{
		add_offset[i] = add_value;
	}

	// 构造头指针
	for (i = 0; i < k; i++)
	{
		copy_ptr[i] = src[i];
	}
	for (i = 0; i < p; i++)
	{
		copy_ptr[i + k] = dest[i];
	}

	void **indexs;
	void **data_buffs[p * w];
	void **parity_buffs[p * w];
	for (i = 0; i < p * w; i++)
	{
		posix_memalign(&buf, 32, k * w * sizeof(long));
		data_buffs[i] = buf;
		parity_buffs[i] = malloc(sizeof(void *));
	}
	posix_memalign(&buf, 32, p * w * k * w * sizeof(long));
	indexs = buf;
	int ptr_index = 0;
	// int xor_bytes=0;
	int index = 0;
	while (schedule[index][0] >= 0)
	{
		// printf("%d\n",index);
		data_buffs[index] = &indexs[ptr_index];
		for (j = 0; j < schedule[index][0]; j++)
		{
			indexs[ptr_index++] = copy_ptr[schedule[index][2 * j + 3]] + schedule[index][2 * j + 4] * packetsize + begin * w * packetsize;
		}
		parity_buffs[index] = &indexs[ptr_index];
		indexs[ptr_index++] = copy_ptr[schedule[index][1]] + schedule[index][2] * packetsize + begin * w * packetsize;
		index++;
	}
	int loop = mceil(ptr_index, 8) * 64;
	for (i = 0; i < len; i += add_value)
	{
		for (j = 0; j < index; j++)
		{
			// printf("%d\n",schedule[j][1]);
			xor_gen(schedule[j][0], packetsize, data_buffs[j], parity_buffs[j]);
		}
		// printf("!!\n");
		buff_finally(loop, add_offset, indexs);
	}
	// printf("!!!\n");
}

int test_threads(int k, int p, int w, int len, int packetsize, int e, int n)
{
	struct timespec time1 = {0, 0};
	struct timespec time2 = {0, 0};
	long encode_time = 0;

	int i, j, m, a;
	int alignlen;
	// Fragment buffer pointers
	u8 *frag_ptrs[MMAX];
	u8 *copy_ptrs[MMAX];
	u8 *tmp[MMAX];
	unsigned int *encode_matrix;
	int *encode_bitmatrix;
	int *slp;
	int **schedule;
	m = k + p;

	int datasize = 10 * 1024 * 1024;
	// int datasize=10321920;
	alignlen = datasize / k;
	// align len with w*packetsize
	if (packetsize != 0)
	{
		if (alignlen % (w * packetsize * n) != 0)
		{
			while (alignlen % (w * packetsize * n) != 0)
				alignlen++;
		}
	}
	else
	{
		if (alignlen % w != 0)
		{
			while (alignlen % w != 0)
				alignlen++;
		}
	}
	alldata = alignlen * (k);
	decodedata = alignlen * (k);
	// Allocate the src & parity buffers
	for (i = 0; i < m; i++)
	{
		void *buf;
		if (posix_memalign(&buf, 32, alignlen))
		{
			printf("alloc error: Fail");
			return 1;
		}
		frag_ptrs[i] = buf;
		// frag_ptrs_copy[i] = frag_ptrs[i];
	}
	for (i = 0; i < m; i++)
		for (j = 0; j < alignlen; j++)
			frag_ptrs[i][j] = rand();

	// encode_matrix=gen_best_matrix_all(k,p,w);
	encode_matrix = gen_best_matrix_from_any_element_lowtime(k, p, w);
	encode_bitmatrix = matrix_to_bitmatrix(k, p, w, encode_matrix);

	slp = rustslp(encode_bitmatrix, p * w, k * w, packetsize, 0, 2);
	schedule = rustslp_to_schedule(k, p, w, slp);
	int value = max_var(slp);
	for (i = 0; i < value / w - k - p + 1; i++)
	{
		void *buf;
		posix_memalign(&buf, 32, packetsize * w);
		tmp[i] = buf;
	}

	void **data_indexs;
	void **parity_indexs;
	long *buf;
	posix_memalign(&buf, 32, p * w * 2 * k * w * sizeof(long));
	data_indexs = buf;
	posix_memalign(&buf, 32, 2 * 2 * k * w * sizeof(long));
	parity_indexs = buf;
	void **data_buffs;
	void **parity_buffs;
	data_buffs = malloc(sizeof(void *) * (2 * k * w));
	parity_buffs = malloc(sizeof(void *));
	pthread_t pt[n];
	ptInfo ptInfos[n];

	for (i = 0; i < n; i++)
	{
		ptInfos[i].k = k;
		ptInfos[i].p = p;
		ptInfos[i].w = w;
		ptInfos[i].schedule = schedule;
		ptInfos[i].src = frag_ptrs;
		ptInfos[i].dest = &frag_ptrs[k];
		ptInfos[i].len = alignlen;
		ptInfos[i].packetsize = packetsize;
		ptInfos[i].value = value;
		ptInfos[i].begin = i;
		ptInfos[i].n = n;
	}
	// printf("!!\n");
	for (i = 0; i < en_n; i++)
	{
		clock_gettime(CLOCK_REALTIME, &time1);
		for (j = 0; j < n; j++)
		{
			pthread_create(&pt[j], NULL, encode_threads, &ptInfos[j]);
		}
		for (j = 0; j < n; j++)
		{
			pthread_join(pt[j], NULL);
		}
		clock_gettime(CLOCK_REALTIME, &time2);
		encode_time = (time2.tv_sec - time1.tv_sec) * 1000000000 + (time2.tv_nsec - time1.tv_nsec);
		encode_time_arrs[i] = encode_time;
	}

	return 1;
}

void *encode_threads_new(void *pt)
{
	int i, j;
	ptInfo *ptinfo = (ptInfo *)pt;
	int k = ptinfo->k;
	int p = ptinfo->p;
	int w = ptinfo->w;
	int **schedule = ptinfo->schedule;
	u8 **src = ptinfo->src;
	u8 **dest = ptinfo->dest;
	int len = ptinfo->len;
	int packetsize = ptinfo->packetsize;
	int value = ptinfo->value;
	int begin = ptinfo->begin;
	int n = ptinfo->n;
	int offset = begin * len;
	u8 *copy_ptr[k + p];

	long *add_offset;
	long *buf;
	if (posix_memalign(&buf, 32, 16 * sizeof(long)))
	{
		return 1;
	}
	add_offset = buf;
	long add_value = w * packetsize;
	for (i = 0; i < 16; i++)
	{
		add_offset[i] = add_value;
	}

	for (i = 0; i < k; i++)
	{
		copy_ptr[i] = src[i] + offset;
	}
	for (i = 0; i < p; i++)
	{
		copy_ptr[i + k] = dest[i] + offset;
	}

	void **indexs;
	void **data_buffs[p * w];
	void **parity_buffs[p * w];
	for (i = 0; i < p * w; i++)
	{
		posix_memalign(&buf, 32, k * w * sizeof(long));
		data_buffs[i] = buf;
		parity_buffs[i] = malloc(sizeof(void *));
	}
	posix_memalign(&buf, 32, p * w * k * w * sizeof(long));
	indexs = buf;
	int ptr_index = 0;
	// int xor_bytes=0;
	int index = 0;
	while (schedule[index][0] >= 0)
	{
		data_buffs[index] = &indexs[ptr_index];
		for (j = 0; j < schedule[index][0]; j++)
		{
			indexs[ptr_index++] = copy_ptr[schedule[index][2 * j + 3]] + schedule[index][2 * j + 4] * packetsize + begin * w * packetsize;
		}
		parity_buffs[index] = &indexs[ptr_index];
		indexs[ptr_index++] = copy_ptr[schedule[index][1]] + schedule[index][2] * packetsize + begin * w * packetsize;
		index++;
	}
	int loop = mceil(ptr_index, 8) * 64;
	for (i = 0; i < len; i += add_value)
	{
		for (j = 0; j < index; j++)
		{
			xor_gen(schedule[j][0], packetsize, data_buffs[j], parity_buffs[j]);
		}
		buff_finally(loop, add_offset, indexs);
	}
}

int test_threads_new(int k, int p, int w, int len, int packetsize, int e, int n)
{
	struct timespec time1 = {0, 0};
	struct timespec time2 = {0, 0};

	long long encode_time;
	// long long *encode_time;
	// encode_time=malloc(sizeof(long long)*n);

	int i, j, m, a;
	int alignlen;
	// Fragment buffer pointers
	u8 *frag_ptrs[MMAX];
	u8 *copy_ptrs[MMAX];
	u8 *tmp[MMAX];
	unsigned int *encode_matrix;
	int *encode_bitmatrix;
	int *slp;
	int **schedule;
	m = k + p;

	int datasize = 10 * 1024 * 1024;
	// int datasize=10321920;
	alignlen = datasize / k;
	// align len with w*packetsize
	if (packetsize != 0)
	{
		if (alignlen % (w * packetsize * n) != 0)
		{
			while (alignlen % (w * packetsize * n) != 0)
				alignlen++;
		}
	}
	else
	{
		if (alignlen % w != 0)
		{
			while (alignlen % w != 0)
				alignlen++;
		}
	}
	alldata = alignlen * (k);
	decodedata = alignlen * (k);
	// Allocate the src & parity buffers
	for (i = 0; i < m; i++)
	{
		void *buf;
		if (posix_memalign(&buf, 32, alignlen + n * w * packetsize))
		{
			printf("alloc error: Fail");
			return 1;
		}
		frag_ptrs[i] = buf;
		// frag_ptrs_copy[i] = frag_ptrs[i];
	}
	for (i = 0; i < m; i++)
		for (j = 0; j < alignlen; j++)
			frag_ptrs[i][j] = rand();

	// encode_matrix=gen_best_matrix_all(k,p,w);
	encode_matrix = gen_best_matrix_from_any_element_lowtime(k, p, w);
	encode_bitmatrix = matrix_to_bitmatrix(k, p, w, encode_matrix);

	slp = rustslp(encode_bitmatrix, p * w, k * w, packetsize, 0, 2);
	schedule = rustslp_to_schedule(k, p, w, slp);
	int value = max_var(slp);
	for (i = 0; i < value / w - k - p + 1; i++)
	{
		void *buf;
		posix_memalign(&buf, 32, packetsize * w);
		tmp[i] = buf;
	}
	// schedule=bitmatrix_to_smart_schedule(k,p,w,encode_bitmatrix);
	void **data_indexs;
	void **parity_indexs;
	long *buf;
	posix_memalign(&buf, 32, p * w * 2 * k * w * sizeof(long));
	data_indexs = buf;
	posix_memalign(&buf, 32, 2 * 2 * k * w * sizeof(long));
	parity_indexs = buf;
	void **data_buffs;
	void **parity_buffs;
	data_buffs = malloc(sizeof(void *) * (2 * k * w));
	parity_buffs = malloc(sizeof(void *));
	pthread_t pt[n];
	ptInfo ptInfos[n];

	for (i = 0; i < n; i++)
	{
		ptInfos[i].k = k;
		ptInfos[i].p = p;
		ptInfos[i].w = w;
		ptInfos[i].schedule = schedule;
		ptInfos[i].src = frag_ptrs;
		ptInfos[i].dest = &frag_ptrs[k];
		ptInfos[i].len = alignlen / n;
		ptInfos[i].packetsize = packetsize;
		ptInfos[i].value = value;
		ptInfos[i].begin = i;
		ptInfos[i].n = n;
	}
	// printf("!!\n");
	long long encode_time_average = 0;
	for (i = 0; i < 1000; i++)
	{
		clock_gettime(CLOCK_REALTIME, &time1);
		for (j = 0; j < n; j++)
		{
			pthread_create(&pt[j], NULL, encode_threads_new, &ptInfos[j]);
		}
		for (j = 0; j < n; j++)
		{
			pthread_join(pt[j], NULL);
		}
		clock_gettime(CLOCK_REALTIME, &time2);
		encode_time = (time2.tv_sec - time1.tv_sec) * 1000000000 + (time2.tv_nsec - time1.tv_nsec);
		encode_time_arrs[i] = encode_time;
	}

	return 1;
}

void *encode_threads_new_time(void *pt)
{
	struct timespec time1 = {0, 0};
	struct timespec time2 = {0, 0};
	clock_gettime(CLOCK_REALTIME, &time1);
	int i, j;
	ptInfo *ptinfo = (ptInfo *)pt;
	int k = ptinfo->k;
	int p = ptinfo->p;
	int w = ptinfo->w;
	int **schedule = ptinfo->schedule;
	u8 **src = ptinfo->src;
	u8 **dest = ptinfo->dest;
	int len = ptinfo->len;
	int packetsize = ptinfo->packetsize;
	int value = ptinfo->value;
	int begin = ptinfo->begin;
	int n = ptinfo->n;
	int thread_num = ptinfo->thread_num;
	int offset = begin * len;
	u8 *copy_ptr[k + p];

	long *add_offset;
	long *buf;
	if (posix_memalign(&buf, 32, 16 * sizeof(long)))
	{
		return 1;
	}
	add_offset = buf;
	long add_value = w * packetsize;
	for (i = 0; i < 16; i++)
	{
		add_offset[i] = add_value;
	}

	for (i = 0; i < k; i++)
	{
		copy_ptr[i] = src[i] + offset;
	}
	for (i = 0; i < p; i++)
	{
		copy_ptr[i + k] = dest[i] + offset;
	}

	void **indexs;
	void **data_buffs[p * w];
	void **parity_buffs[p * w];
	for (i = 0; i < p * w; i++)
	{
		posix_memalign(&buf, 32, k * w * sizeof(long));
		data_buffs[i] = buf;
		parity_buffs[i] = malloc(sizeof(void *));
	}
	posix_memalign(&buf, 32, p * w * k * w * sizeof(long));
	indexs = buf;
	int ptr_index = 0;
	// int xor_bytes=0;
	int index = 0;

	while (schedule[index][0] >= 0)
	{
		data_buffs[index] = &indexs[ptr_index];
		for (j = 0; j < schedule[index][0]; j++)
		{
			indexs[ptr_index++] = copy_ptr[schedule[index][2 * j + 3]] + schedule[index][2 * j + 4] * packetsize + begin * w * packetsize;
		}
		parity_buffs[index] = &indexs[ptr_index];
		indexs[ptr_index++] = copy_ptr[schedule[index][1]] + schedule[index][2] * packetsize + begin * w * packetsize;
		index++;
	}
	int loop = mceil(ptr_index, 8) * 64;
	for (i = 0; i < len; i += add_value)
	{
		for (j = 0; j < index; j++)
		{
			xor_gen(schedule[j][0], packetsize, data_buffs[j], parity_buffs[j]);
		}
		buff_finally(loop, add_offset, indexs);
	}
	clock_gettime(CLOCK_REALTIME, &time2);
	thread_time_arrs[thread_num] = (time2.tv_sec - time1.tv_sec) * 1000000000 + (time2.tv_nsec - time1.tv_nsec);
}

int test_threads_new_time(int k, int p, int w, int len, int packetsize, int e, int n)
{
	struct timespec time1 = {0, 0};
	struct timespec time2 = {0, 0};

	long long encode_time;
	// long long *encode_time;
	// encode_time=malloc(sizeof(long long)*n);
	long long each_thread_encode_time[4][1000];
	long long each_thread_total_time[4];
	double each_thread_encode_v[4];

	int i, j, m, a;
	int alignlen;
	// Fragment buffer pointers
	u8 *frag_ptrs[MMAX];
	u8 *copy_ptrs[MMAX];
	u8 *tmp[MMAX];
	unsigned int *encode_matrix;
	int *encode_bitmatrix;
	int *slp;
	int **schedule;
	m = k + p;

	int datasize = 10 * 1024 * 1024;
	// int datasize=10321920;
	alignlen = datasize / k;
	// align len with w*packetsize
	if (packetsize != 0)
	{
		if (alignlen % (w * packetsize * n) != 0)
		{
			while (alignlen % (w * packetsize * n) != 0)
				alignlen++;
		}
	}
	else
	{
		if (alignlen % w != 0)
		{
			while (alignlen % w != 0)
				alignlen++;
		}
	}
	alldata = alignlen * (k);
	decodedata = alignlen * (k);
	// Allocate the src & parity buffers
	for (i = 0; i < m; i++)
	{
		void *buf;
		if (posix_memalign(&buf, 32, alignlen + n * w * packetsize))
		{
			printf("alloc error: Fail");
			return 1;
		}
		frag_ptrs[i] = buf;
		// frag_ptrs_copy[i] = frag_ptrs[i];
	}
	for (i = 0; i < m; i++)
		for (j = 0; j < alignlen; j++)
			frag_ptrs[i][j] = rand();

	// encode_matrix=gen_best_matrix_all(k,p,w);
	encode_matrix = gen_best_matrix_from_any_element_lowtime(k, p, w);
	encode_bitmatrix = matrix_to_bitmatrix(k, p, w, encode_matrix);

	slp = rustslp(encode_bitmatrix, p * w, k * w, packetsize, 0, 2);
	schedule = rustslp_to_schedule(k, p, w, slp);
	int value = max_var(slp);
	for (i = 0; i < value / w - k - p + 1; i++)
	{
		void *buf;
		posix_memalign(&buf, 32, packetsize * w);
		tmp[i] = buf;
	}
	// schedule=bitmatrix_to_smart_schedule(k,p,w,encode_bitmatrix);
	void **data_indexs;
	void **parity_indexs;
	long *buf;
	posix_memalign(&buf, 32, p * w * 2 * k * w * sizeof(long));
	data_indexs = buf;
	posix_memalign(&buf, 32, 2 * 2 * k * w * sizeof(long));
	parity_indexs = buf;
	void **data_buffs;
	void **parity_buffs;
	data_buffs = malloc(sizeof(void *) * (2 * k * w));
	parity_buffs = malloc(sizeof(void *));
	pthread_t pt[n];
	ptInfo ptInfos[n];

	for (i = 0; i < n; i++)
	{
		ptInfos[i].k = k;
		ptInfos[i].p = p;
		ptInfos[i].w = w;
		ptInfos[i].schedule = schedule;
		ptInfos[i].src = frag_ptrs;
		ptInfos[i].dest = &frag_ptrs[k];
		ptInfos[i].len = alignlen / n;
		ptInfos[i].packetsize = packetsize;
		ptInfos[i].value = value;
		ptInfos[i].begin = i;
		ptInfos[i].n = n;
	}
	// printf("!!\n");
	for (i = 0; i < 1000; i++)
	{
		for (j = 0; j < n; j++)
		{
			ptInfos[j].thread_num = j;
			pthread_create(&pt[j], NULL, encode_threads_new_time, &ptInfos[j]);
		}
		for (j = 0; j < n; j++)
		{
			pthread_join(pt[j], NULL);
		}
		for (j = 0; j < n; j++)
		{
			each_thread_encode_time[j][i] = thread_time_arrs[j];
			// printf("%d\n",each_thread_encode_time[j][i]);
		}
	}

	for (j = 0; j < n; j++)
	{
		each_thread_total_time[j] = 0;
		for (i = 0; i < 1000; i++)
		{
			each_thread_total_time[j] += each_thread_encode_time[j][i];
		}
		each_thread_total_time[j] /= 1000;
	}

	double total_encode_v = 0.0;
	for (j = 0; j < n; j++)
	{
		each_thread_encode_v[j] = ((((double)alldata / n)) / 1024 / 1024 / 1024) / ((double)each_thread_total_time[j] / 1000 / 1000 / 1000);
		total_encode_v += each_thread_encode_v[j];
	}

	printf("%.2lf ", total_encode_v);

	return 1;
}