#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <climits>
#include <chrono>
#include <cuda_runtime.h>
#include "scan.h"

void print_array_sample(const int* arr, int n, int limit) {
	int width = limit / 2;
	for (int i = 0; i < n; i++) {
		if (n > limit && i == width) { printf(" ..."); i = n - width - 1; continue; }
		printf("%12d", arr[i]);
	} printf("\n");
}

double get_time() {
	using namespace std::chrono;
	auto now = high_resolution_clock::now();
	return duration_cast<duration<double>>(now.time_since_epoch()).count();
}

void run_test(ScanAlgorithm algo, const int* input, int n, const int* ref) {
	if (algo.max_n != -1 && n > algo.max_n) return;

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
		cudaMemset(d_out, 0, bytes); 
		if (workspace) cudaMemset(workspace, 0, workspace_bytes);
		cudaDeviceSynchronize();

		// run once to "warm up" (GPU clocks, Driver Init, ...)
		algo.func(d_in, n, d_out, workspace);
		cudaMemset(d_out, 0, bytes); 
		if (workspace) cudaMemset(workspace, 0, workspace_bytes);
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
		
		memset(output, 0, bytes); 
		if (workspace) memset(workspace, 0, workspace_bytes);

		// run once to "warm up" (OS paging, clock speed, ...)
		algo.func(input, n, output, workspace);
		memset(output, 0, bytes); 
		if (workspace) memset(workspace, 0, workspace_bytes);

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

	printf("    %-21s Time: %.6fs | Valid: %s\n", algo.name, duration, valid ? "YES" : "NO");
	if (!valid) {
		printf("        Input:     "); print_array_sample(input, n, 16);
		printf("        Output:    "); print_array_sample(output, n, 16);
		printf("        Reference: "); print_array_sample(ref, n, 16);
	}

	free(output);
}

void run_test_suite(const int n, const int max_input_value) {
	assert((unsigned long long)n * max_input_value < INT_MAX); // ensure result can't overflow
	size_t bytes = n * sizeof(int);

	printf("Running with %d elements (%.2f MiB)...\n", n, bytes / (1024.0 * 1024.0));
	int* input = (int*)malloc(bytes);
	int* ref = (int*)malloc(bytes);

	for (int i = 0; i < n; i++) { input[i] = rand() % (max_input_value+1); }
	scan_sequential(input, n, ref, NULL); // Get reference data from sequential algorithm

	ScanAlgorithm algorithms[] = {
		{"CPU Sequential",       scan_sequential,      workspace_none,            false,-1       },
		{"CPU Multi-threaded",   scan_omp,             workspace_omp,             false,-1       },
		{"CPU C++ Std library",  scan_std,             workspace_none,            false,-1       },
		{"GPU Kogge-Stone",      scan_kogge_stone,     workspace_none,            true, 1024     },
		{"GPU Brent-Kung",       scan_brent_kung,      workspace_none,            true, 2048     },
		{"GPU Coarsened",        scan_coarsened,       workspace_none,            true, 12288    },
		{"GPU Segm. Kogge-Stone",scan_segm_kogge_stone,workspace_segm_kogge_stone,true, 1048576  },
		{"GPU Segm. Brent-Kung", scan_segm_brent_kung, workspace_segm_brent_kung, true, 4194304  },
		{"GPU Segm. Coarse",     scan_segm_coarse,     workspace_segm_coarse,     true, 150994944},
		{"GPU Segm. Single-pass",scan_segm_single,     workspace_segm_single,     true, -1       },
		{"GPU Thrust library",   scan_thrust,          workspace_none,            true, -1       },
	};

	int num_algos = sizeof(algorithms) / sizeof(ScanAlgorithm);
	for (int i = 0; i < num_algos; i++) {
		run_test(algorithms[i], input, n, ref);
	}

	free(input); free(ref);
}

int main() {
	// some algorithms will only run up to specific sizes
	run_test_suite(1024, 100);
	// run_test_suite(2048, 100);
	run_test_suite(12288, 100);
	// run_test_suite(2000000, 2); // omp is sometimes faster
	run_test_suite(1048576, 2);
	run_test_suite(4194304, 2);
	run_test_suite(150994944, 2);
	run_test_suite(500000000, 2); // can't go much higher, because GPU malloc fails

	return 0;
}
