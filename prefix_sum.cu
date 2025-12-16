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

size_t workspace_none(int n) { return 0; }

void prefix_sum_sequential(const int* input, int n, int* output, void* workspace) {
	if (n == 0) return;
	output[0] = input[0];
	for (int i = 1; i < n; i++) {
		output[i] = output[i - 1] + input[i];
	}
}

size_t workspace_omp(int n) { return (omp_get_max_threads() + 1) * sizeof(int); }

void prefix_sum_omp(const int* input, int n, int* output, void* workspace) {
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

void prefix_sum_thrust(const int* d_input, int n, int* d_out, void* workspace) {
	thrust::device_ptr<const int> t_in(d_input);
	thrust::device_ptr<int> t_out(d_out);
	thrust::inclusive_scan(t_in, t_in + n, t_out);
}

// --- Testing Framework ---

typedef void (*ps_func)(const int*, int, int*, void*);
typedef size_t (*workspace_func)(int);

typedef struct {
	const char* name;
	ps_func func;
	workspace_func calc_workspace;
	bool is_gpu;
	int max_n; // max supported n by algorithm; -1 for no limit
} ScanAlgorithm;

void run_test(ScanAlgorithm algo, const int* input, int n, const int* ref) {
	if (algo.max_n != -1 && n > algo.max_n) return; // Skip if N is too large

	size_t bytes = n * sizeof(int);
	size_t workspace_bytes = algo.calc_workspace(n);

	int* output = (int*)malloc(bytes);
	void* workspace = NULL;
	double duration = 0.0;

	if (algo.is_gpu) {
		int *d_in, *d_out;
		if (cudaMalloc(&d_in, bytes) != cudaSuccess || cudaMalloc(&d_out, bytes) != cudaSuccess) {
			fprintf(stderr, "GPU Malloc Failed\n"); free(output); return;
		}
		if (workspace_bytes > 0 && cudaMalloc(&workspace, workspace_bytes) != cudaSuccess) {
			fprintf(stderr, "GPU Workspace Malloc Failed\n");
			free(output); cudaFree(d_in); cudaFree(d_out); return;
		}

		cudaMemcpy(d_in, input, bytes, cudaMemcpyHostToDevice);
		cudaMemset(d_out, 0, bytes); cudaMemset(workspace, 0, workspace_bytes);
		cudaDeviceSynchronize();

		// run once to "warm up" (GPU clocks, Driver Init, ...)
		algo.func(d_in, n, d_out, workspace);
		cudaMemset(d_out, 0, bytes); cudaMemset(workspace, 0, workspace_bytes);
		cudaDeviceSynchronize();

		double start = get_time();
		algo.func(d_in, n, d_out, workspace);
		cudaDeviceSynchronize();
		duration = get_time() - start;

		cudaMemcpy(output, d_out, bytes, cudaMemcpyDeviceToHost);
		cudaFree(d_in); cudaFree(d_out);
		if (workspace) cudaFree(workspace);
	} 
	else {
		if (workspace_bytes > 0) workspace = malloc(workspace_bytes);
		
		memset(output, 0, bytes); memset(workspace, 0, workspace_bytes);

		// run once to "warm up" (OS paging, clock speed, ...)
		algo.func(input, n, output, workspace);
		memset(output, 0, bytes); memset(workspace, 0, workspace_bytes);

		double start = get_time();
		algo.func(input, n, output, workspace);
		duration = get_time() - start;

		if (workspace) free(workspace);
	}

	// Verification
	bool valid = true;
	for (int i = 0; i < n; i++) {
		if (output[i] != ref[i]) { valid = false; break; }
	}

	// Print info
	printf("    %-20s Time: %.5fs | Valid: %s\n", algo.name, duration, valid ? "YES" : "NO");
	if (!valid) {
		printf("        Input:     "); print_array_sample(input, n, 16);
		printf("        Output:    "); print_array_sample(output, n, 16);
		printf("        Reference: "); print_array_sample(ref, n, 16);
	}

	free(output);
}

void run_test_suite(const int n, const int max_input_value) {
	assert((unsigned long long)n * max_input_value < INT_MAX); // ensure it can't overflow
	size_t bytes = n * sizeof(int);

	printf("Running with %d elements (%.2f MiB)...\n", n, bytes / (1024.0 * 1024.0));
	int* input = (int*)malloc(bytes);
	int* ref = (int*)malloc(bytes);

	for (int i = 0; i < n; i++) { input[i] = rand() % (max_input_value+1); }
	prefix_sum_sequential(input, n, ref, NULL);

	ScanAlgorithm algorithms[] = {
		{ "CPU Sequential",      prefix_sum_sequential, workspace_none, false, -1 },
		{ "CPU Multi-threaded",  prefix_sum_omp,        workspace_omp,  false, -1 },
		{ "GPU Thrust library",  prefix_sum_thrust,     workspace_none, true,  -1 },
		// Add PMPP algorithms here (e.g., Kogge-Stone with max_n = 1024)
	};

	int num_algos = sizeof(algorithms) / sizeof(ScanAlgorithm);
	for (int i = 0; i < num_algos; i++) {
		run_test(algorithms[i], input, n, ref);
	}

	free(input); free(ref);
}

int main() {
	// Small Case (Single Block algorithms will also run here)
	run_test_suite(1024, 100);

	run_test_suite(10000000, 50);
	run_test_suite(500000000, 2);

	return 0;
}
