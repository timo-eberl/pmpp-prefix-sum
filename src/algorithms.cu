#include "scan.h"
#include <omp.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <numeric>

// block size maximum of my GPU is 1024 threads
#define BLOCK_SIZE 1024

size_t workspace_none(int n) { return 0; }

void scan_sequential(const int* input, int n, int* output, void* workspace) {
	if (n == 0) return;
	output[0] = input[0];
	for (int i = 1; i < n; i++) {
		output[i] = output[i - 1] + input[i];
	}
}

void scan_std(const int* input, int n, int* output, void* workspace) {
	// C++17 standard inclusive scan
	std::inclusive_scan(input, input + n, output);
}

size_t workspace_omp(int n) { 
	return (omp_get_max_threads() + 1) * sizeof(int); 
}

void scan_omp(const int* input, int n, int* output, void* workspace) {
	int n_threads = omp_get_max_threads();
	int* offsets = (int*)workspace; // Use provided workspace
	offsets[0] = 0;

	#pragma omp parallel
	{
		int id = omp_get_thread_num();
		int chunk = (n + n_threads - 1) / n_threads;
		int start = id * chunk;
		int end = (start + chunk < n) ? start + chunk : n;

		// 1. Local Prefix Sum
		if (start < n) {
			output[start] = input[start];
			for (int i = start + 1; i < end; i++) {
				output[i] = output[i - 1] + input[i];
			}
			offsets[id + 1] = output[end - 1];
		}
		else {
			offsets[id + 1] = 0;
		}
		#pragma omp barrier

		// 2. Calculate Offsets (Single thread)
		#pragma omp single
		{
			for (int i = 1; i <= n_threads; i++) offsets[i] += offsets[i - 1];
		} // Implicit barrier

		// 3. Apply Offsets
		if (start < n && id > 0) {
			int off = offsets[id];
			for (int i = start; i < end; i++) output[i] += off;
		}
	}
}

void scan_thrust(const int* d_input, int n, int* d_out, void* workspace) {
	thrust::device_ptr<const int> t_in(d_input);
	thrust::device_ptr<int> t_out(d_out);
	thrust::inclusive_scan(t_in, t_in + n, t_out);
}

__global__ void kogge_stone_double_buffered_kernel(const int* d_in, int* d_out, int n) {
	__shared__ int buf_A[1024]; __shared__ int buf_B[1024]; // allocate 2 buffers
	// convert to pointers so we can do pointer arithmetic
	int* temp_in = buf_A;
	int* temp_out = buf_B;

	int i = threadIdx.x;

	// Load input into the first buffer
	if (i < n) temp_in[i] = d_in[i];
	else temp_in[i] = 0;

	for (int stride = 1; stride < blockDim.x; stride *= 2) {
		__syncthreads(); // write-after-read protection: ensure all threads finished writing

		if (i >= stride) {
			temp_out[i] = temp_in[i] + temp_in[i - stride];
		} else {
			// In double buffering, we must copy the value forward even if 
			// no addition happens, so it exists in the buffer for the next step
			temp_out[i] = temp_in[i];
		}

		// Swap pointers: Output of this step becomes input of the next
		int* t = temp_in; temp_in = temp_out; temp_out = t;
	}

	// Write result to global memory, temp_in always holds the most recent result
	if (i < n) d_out[i] = temp_in[i];
}

__global__ void kogge_stone_kernel(const int* d_in, int* d_out, int n) {
	__shared__ int temp[BLOCK_SIZE];
	int i = threadIdx.x;

	// Load input into shared memory
	if (i < n) temp[i] = d_in[i];
	else temp[i] = 0;

	// We use a register 'v' to handle the read-after-write dependency
	for (int stride = 1; stride < blockDim.x; stride *= 2) {
		__syncthreads(); // write-after-read protection: ensure all threads finished writing
		int v = 0;
		if (i >= stride) {
			v = temp[i] + temp[i - stride];
		}

		__syncthreads(); // read-after-write protection: ensure all threads finished reading
		if (i >= stride) {
			temp[i] = v;
		}
	}

	// Write result to global memory
	if (i < n) d_out[i] = temp[i];
}

void scan_kogge_stone(const int* d_input, int n, int* d_out, void* workspace) {
	assert(n <= BLOCK_SIZE);
	kogge_stone_kernel<<<1, BLOCK_SIZE>>>(d_input, d_out, n);
}

__global__ void brent_kung_kernel(const int* d_in, int* d_out, int n) {
	// Each thread handles 2 elements
	const int SECTION_SIZE = 2 * BLOCK_SIZE;
	__shared__ int temp[SECTION_SIZE];
	
	// index for the first of the two elements
	int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;

	// Load input into shared memory
	if (i < n) temp[tid] = d_in[i];
	else temp[tid] = 0;
	if (i + blockDim.x < n) temp[tid + blockDim.x] = d_in[i + blockDim.x];
	else temp[tid + blockDim.x] = 0;

	// Phase 1: Reduction Tree (Up-Sweep)
	for (int stride = 1; stride <= blockDim.x; stride *= 2) {
		__syncthreads();
		// Map thread ID to the element indices being summed
		int index = (tid + 1) * 2 * stride - 1;
		if (index < SECTION_SIZE) {
			temp[index] += temp[index - stride];
		}
	}

	// Phase 2: Reverse Tree (Down-Sweep)
	for (int stride = SECTION_SIZE / 4; stride > 0; stride /= 2) {
		__syncthreads();
		int index = (tid + 1) * 2 * stride - 1;
		if (index + stride < SECTION_SIZE) {
			temp[index + stride] += temp[index];
		}
	}

	// Write result to global memory
	__syncthreads();
	if (i < n) d_out[i] = temp[tid];
	if (i + blockDim.x < n) d_out[i + blockDim.x] = temp[tid + blockDim.x];
}

void scan_brent_kung(const int* d_input, int n, int* d_out, void* workspace) {
	assert(n <= BLOCK_SIZE*2);
	brent_kung_kernel<<<1, BLOCK_SIZE>>>(d_input, d_out, n);
}
