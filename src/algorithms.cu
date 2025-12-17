#include "scan.h"
#include <omp.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <numeric>

// block size maximum of my GPU is 1024 threads
#define BLOCK_SIZE 1024
// shared memory maximum: 49152 bytes -> coarse factor up to 12
#define COARSE_FACTOR 12

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

__global__ void kogge_stone_double_buffered_kernel(const int* d_in, int* d_out, int n, int* d_aux) {
	__shared__ int buf_A[BLOCK_SIZE]; __shared__ int buf_B[BLOCK_SIZE]; // allocate 2 buffers
	// convert to pointers so we can do pointer arithmetic
	int* temp_in = buf_A;
	int* temp_out = buf_B;

	int tid = threadIdx.x;
	int global_idx = blockIdx.x * blockDim.x + tid;

	// Load input into the first buffer
	if (global_idx < n) temp_in[tid] = d_in[global_idx];
	else temp_in[tid] = 0;

	for (int stride = 1; stride < blockDim.x; stride *= 2) {
		__syncthreads(); // write-after-read protection: ensure all threads finished writing

		if (tid >= stride) {
			temp_out[tid] = temp_in[tid] + temp_in[tid - stride];
		} else {
			// In double buffering, we must copy the value forward even if 
			// no addition happens, so it exists in the buffer for the next step
			temp_out[tid] = temp_in[tid];
		}

		// Swap pointers: Output of this step becomes input of the next
		int* t = temp_in; temp_in = temp_out; temp_out = t;
	}

	// Write result to global memory, temp_in always holds the most recent result
	if (global_idx < n) d_out[global_idx] = temp_in[tid];

	// Write block sum to auxiliary array (last thread has the total)
	if (d_aux != nullptr && tid == BLOCK_SIZE - 1) {
		d_aux[blockIdx.x] = temp_in[BLOCK_SIZE - 1];
	}
}

__global__ void kogge_stone_kernel(const int* d_in, int* d_out, int n, int* d_aux) {
	__shared__ int temp[BLOCK_SIZE];
	int tid = threadIdx.x;
	int i = blockIdx.x * blockDim.x + tid;

	// Load input into shared memory
	if (i < n) temp[tid] = d_in[i];
	else temp[tid] = 0;

	// We use a register v to handle the read-after-write dependency
	for (int stride = 1; stride < blockDim.x; stride *= 2) {
		__syncthreads(); // write-after-read protection: ensure all threads finished writing
		int v = 0;
		if (tid >= stride) {
			v = temp[tid] + temp[tid - stride];
		}

		__syncthreads(); // read-after-write protection: ensure all threads finished reading
		if (tid >= stride) {
			temp[tid] = v;
		}
	}

	// Write result to global memory
	if (i < n) d_out[i] = temp[tid];

	// Write block sum to auxiliary array (last thread has the total)
	if (d_aux != nullptr && tid == BLOCK_SIZE - 1) {
		d_aux[blockIdx.x] = temp[BLOCK_SIZE - 1];
	}
}

void scan_kogge_stone(const int* d_input, int n, int* d_out, void* workspace) {
	assert(n <= BLOCK_SIZE);
	kogge_stone_double_buffered_kernel<<<1, BLOCK_SIZE>>>(d_input, d_out, n, nullptr);
}

__global__ void brent_kung_kernel(const int* d_in, int* d_out, int n, int* d_aux) {
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

	// Write block sum to auxiliary array (last thread has the total)
	if (d_aux != nullptr && tid == 0) {
		d_aux[blockIdx.x] = temp[SECTION_SIZE - 1];
	}
}

void scan_brent_kung(const int* d_input, int n, int* d_out, void* workspace) {
	assert(n <= BLOCK_SIZE*2);
	brent_kung_kernel<<<1, BLOCK_SIZE>>>(d_input, d_out, n, nullptr);
}

__device__ void perform_coarse_scan(
	int* temp, const int* d_in, int* d_out, int* d_aux, int n, int block_offset
) {
	int tid = threadIdx.x;

	// Threads read from global memory in a coalesced manner (with optional offset)
	for (int i = 0; i < COARSE_FACTOR; ++i) {
		int local_idx = tid + i * BLOCK_SIZE;
		int global_idx = block_offset + local_idx;
		if (global_idx < n) temp[local_idx] = d_in[global_idx];
		else temp[local_idx] = 0;
	}
	__syncthreads();

	// Phase 1: Sequential Scan per Thread
	int start_idx = tid * COARSE_FACTOR;
	// The first element stays the same
	for (int i = 1; i < COARSE_FACTOR; ++i) {
		temp[start_idx + i] += temp[start_idx + i - 1];
	}
	__syncthreads();

	// Phase 2: Kogge-Stone Parallel Scan on the end of each chunk
	for (int stride = 1; stride < BLOCK_SIZE; stride *= 2) {
		int v = 0;
		if (tid >= stride) {
			v = temp[(tid - stride + 1) * COARSE_FACTOR - 1];
		}
		__syncthreads();
		if (tid >= stride) {
			temp[(tid + 1) * COARSE_FACTOR - 1] += v;
		}
		__syncthreads();
	}

	// Phase 3: Distribute Block Sums to Local Chunks
	if (tid > 0) {
		int prev_sum = temp[tid * COARSE_FACTOR - 1];
		for (int i = 0; i < COARSE_FACTOR - 1; ++i) {
			temp[start_idx + i] += prev_sum;
		}
	}
	__syncthreads();

	// Write back to global memory in coalesced pattern
	for (int i = 0; i < COARSE_FACTOR; ++i) {
		int local_idx = tid + i * BLOCK_SIZE;
		int global_idx = block_offset + local_idx;
		if (global_idx < n) d_out[global_idx] = temp[local_idx];
	}

	// Write Total Block Sum to Auxiliary Array (last thread) if d_aux is provided
	if (d_aux != nullptr && tid == BLOCK_SIZE - 1) {
		d_aux[blockIdx.x] = temp[BLOCK_SIZE * COARSE_FACTOR - 1];
	}
}

__global__ void coarsened_scan_kernel(const int* d_in, int* d_out, int n) {
	__shared__ int temp[BLOCK_SIZE * COARSE_FACTOR];
	perform_coarse_scan(temp, d_in, d_out, nullptr, n, 0);
}

void scan_coarsened(const int* d_input, int n, int* d_out, void* workspace) {
	// Ensure input fits in one coarsened block
	assert(n <= BLOCK_SIZE * COARSE_FACTOR);
	coarsened_scan_kernel<<<1, BLOCK_SIZE>>>(d_input, d_out, n);
}

// Segmented Scan Kernel 1: Block-wise Coarsened Scan
__global__ void segmented_scan_block_kernel(const int* d_in, int* d_out, int* d_aux, int n) {
	const int SECTION_SIZE = BLOCK_SIZE * COARSE_FACTOR;
	__shared__ int temp[SECTION_SIZE];
	int block_offset = blockIdx.x * SECTION_SIZE;
	perform_coarse_scan(temp, d_in, d_out, d_aux, n, block_offset);
}

// Segmented Scan Kernel 3: Add Offsets Kernel
__global__ void add_block_sums_kernel(int* d_out, const int* d_aux, int n, int elems_per_block) {
	if (blockIdx.x == 0) return;

	int offset = d_aux[blockIdx.x - 1];
	int block_offset = blockIdx.x * elems_per_block;
	int tid = threadIdx.x;

	for (int i = 0; i < elems_per_block / BLOCK_SIZE; ++i) {
		int global_idx = block_offset + tid + i * BLOCK_SIZE;
		if (global_idx < n) {
			d_out[global_idx] += offset;
		}
	}
}

size_t workspace_segm_coarse(int n) {
	const int SECTION_SIZE = BLOCK_SIZE * COARSE_FACTOR;
	int num_blocks = (n + SECTION_SIZE - 1) / SECTION_SIZE;
	return num_blocks * sizeof(int);
}

void scan_segm_coarse(const int* d_input, int n, int* d_out, void* workspace) {
	const int SECTION_SIZE = BLOCK_SIZE * COARSE_FACTOR;
	int num_blocks = (n + SECTION_SIZE - 1) / SECTION_SIZE;

	assert(num_blocks <= SECTION_SIZE);
	assert(n <= SECTION_SIZE*SECTION_SIZE); // essentially the same check

	int* d_aux = (int*)workspace;

	// Step 1: Perform Local Scans and generate Block Sums
	segmented_scan_block_kernel<<<num_blocks, BLOCK_SIZE>>>(d_input, d_out, d_aux, n);

	if (num_blocks < 2) return; // exit early

	// Step 2: Perform Scan on Block Sums
	coarsened_scan_kernel<<<1, BLOCK_SIZE>>>(d_aux, d_aux, num_blocks);

	// Step 3: Add scanned block sums to the result
	add_block_sums_kernel<<<num_blocks, BLOCK_SIZE>>>(d_out, d_aux, n, SECTION_SIZE);
}

size_t workspace_segm_brent_kung(int n) {
	const int SECTION_SIZE = BLOCK_SIZE * 2;
	int num_blocks = (n + SECTION_SIZE - 1) / SECTION_SIZE;
	return num_blocks * sizeof(int);
}

void scan_segm_brent_kung(const int* d_input, int n, int* d_out, void* workspace) {
	// Brent-Kung processes 2 * BLOCK_SIZE elements per block
	const int SECTION_SIZE = BLOCK_SIZE * 2;
	int num_blocks = (n + SECTION_SIZE - 1) / SECTION_SIZE;

	assert(num_blocks <= SECTION_SIZE);
	assert(n <= SECTION_SIZE*SECTION_SIZE); // essentially the same check
	
	int* d_aux = (int*)workspace;

	// Step 1: Perform Local Scans and generate Block Sums
	brent_kung_kernel<<<num_blocks, BLOCK_SIZE>>>(d_input, d_out, n, d_aux);

	if (num_blocks < 2) return; // exit early

	// Step 2: Perform Scan on Block Sums
	brent_kung_kernel<<<1, BLOCK_SIZE>>>(d_aux, d_aux, num_blocks, nullptr);

	// Step 3: Add scanned block sums to the result
	add_block_sums_kernel<<<num_blocks, BLOCK_SIZE>>>(d_out, d_aux, n, SECTION_SIZE);
}

size_t workspace_segm_kogge_stone(int n) {
	const int SECTION_SIZE = BLOCK_SIZE;
	int num_blocks = (n + SECTION_SIZE - 1) / SECTION_SIZE;
	return num_blocks * sizeof(int);
}

void scan_segm_kogge_stone(const int* d_input, int n, int* d_out, void* workspace) {
	// Kogge-Stone processes 1 * BLOCK_SIZE element per block
	const int SECTION_SIZE = BLOCK_SIZE;
	int num_blocks = (n + SECTION_SIZE - 1) / SECTION_SIZE;

	assert(num_blocks <= SECTION_SIZE);
	assert(n <= SECTION_SIZE*SECTION_SIZE);

	int* d_aux = (int*)workspace;

	// Step 1: Perform Local Scans and generate Block Sums
	kogge_stone_double_buffered_kernel<<<num_blocks, BLOCK_SIZE>>>(d_input, d_out, n, d_aux);

	if (num_blocks < 2) return; // exit early

	// Step 2: Perform Scan on Block Sums
	kogge_stone_double_buffered_kernel<<<1, BLOCK_SIZE>>>(d_aux, d_aux, num_blocks, nullptr);

	// Step 3: Add scanned block sums to the result
	add_block_sums_kernel<<<num_blocks, BLOCK_SIZE>>>(d_out, d_aux, n, SECTION_SIZE);
}
