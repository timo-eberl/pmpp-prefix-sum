#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <string.h>
#include <stdbool.h>
#include <omp.h>
#include <limits.h>
#include <cuda_runtime.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>

// --- Helpers ---

void print_array_sample(const int* arr, int n, int limit) {
	int width = limit / 2;
	for (int i = 0; i < n; i++) {
		// If array is too big and we reached the middle, jump to the end
		if (n > limit && i == width) { printf(" ..."); i = n - width - 1; continue; }
		printf("%12d", arr[i]);
	} printf("\n");
}

double get_time() {
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// --- Algorithms ---

void prefix_sum_sequential(const int* input, int n, int* output) {
	if (n == 0) return;
	output[0] = input[0];
	for (int i = 1; i < n; i++) {
		output[i] = output[i - 1] + input[i];
	}
}

void prefix_sum_omp(const int* input, int n, int* output) {
	int n_threads = omp_get_max_threads();
	int* offsets = (int*)malloc((n_threads + 1) * sizeof(int));
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
	free(offsets);
}

void prefix_sum_thrust(const int* d_input, int n, int* d_output) {
	thrust::device_ptr<const int> t_in(d_input);
	thrust::device_ptr<int> t_out(d_output);
	thrust::inclusive_scan(t_in, t_in + n, t_out);
}

// --- Testing Framework ---

typedef void (*ps_func)(const int*, int, int*);

void run_test(
	const char* name, ps_func func, const int* input, int n, const int* ref, bool is_gpu
) {
	size_t bytes = n * sizeof(int);
	int* output = (int*)malloc(bytes);
	double duration = 0.0;

	if (is_gpu) {
		int *d_in, *d_out;
		if (cudaMalloc(&d_in, bytes) != cudaSuccess || cudaMalloc(&d_out, bytes) != cudaSuccess) {
			fprintf(stderr, "GPU Malloc Failed\n"); free(output); return;
		}
		cudaMemcpy(d_in, input, bytes, cudaMemcpyHostToDevice);
		cudaMemset(d_out, 0, bytes);
		cudaDeviceSynchronize();

		double start = get_time();
		func(d_in, n, d_out);
		cudaDeviceSynchronize();
		duration = get_time() - start;

		cudaMemcpy(output, d_out, bytes, cudaMemcpyDeviceToHost);
		cudaFree(d_in); cudaFree(d_out);
	} 
	else {
		memset(output, 0, bytes);

		double start = get_time();
		func(input, n, output);
		duration = get_time() - start;
	}

	// Verification
	bool valid = true;
	for (int i = 0; i < n; i++) {
		if (output[i] != ref[i]) valid = false;
	}

	// Print info
	printf("%-20s Time: %.5fs | Valid: %s\n", name, duration, valid ? "YES" : "NO");
	printf("\tInput:     "); print_array_sample(input, n, 16);
	printf("\tOutput:    "); print_array_sample(output, n, 16);
	printf("\tReference: "); print_array_sample(ref, n, 16);

	free(output);
}

int main() {
	const int n = 100000000; // above 500000000 may fail at GPU memory allocation
	const int max_input_value = 3;

	assert((unsigned long long)n * max_input_value < INT_MAX); // ensure it can't overflow
	size_t bytes = n * sizeof(int);

	printf("Initializing %d elements (%.2f MiB)...\n", n, bytes / (1024.0 * 1024.0));

	int* input = (int*)malloc(bytes);
	int* ref = (int*)malloc(bytes);

	// Initialize input data
	for (int i = 0; i < n; i++) { input[i] = rand() % (max_input_value+1); }
	// Prepare reference data (use sequential sum as source of truth)
	prefix_sum_sequential(input, n, ref);

	bool is_gpu = false;
	run_test("CPU Sequential", prefix_sum_sequential, input, n, ref, is_gpu);
	run_test("CPU Multi-threaded", prefix_sum_omp, input, n, ref, is_gpu);
	is_gpu = true;
	run_test("GPU Thrust library", prefix_sum_thrust, input, n, ref, is_gpu);

	free(input); free(ref);

	return 0;
}
