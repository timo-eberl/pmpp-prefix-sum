#define _POSIX_C_SOURCE 199309L // we use POSIX time
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <string.h>
#include <stdbool.h>
#include <omp.h>
#include <limits.h>

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

// --- Testing Framework ---

typedef void (*ps_func)(const int*, int, int*);

void run_test(const char* name, ps_func func, const int* input, int n, int* output, const int* ref) {
	// Reset output to 0 to ensure no data leaks from previous runs
	memset(output, 0, n * sizeof(int));

	// Measure Time
	double start = get_time();
	func(input, n, output);
	double duration = get_time() - start;

	// Verify (if ref provided)
	bool valid = true;
	if (ref) {
		for (int i = 0; i < n; i++) {
			if (output[i] != ref[i]) { valid = false; break; }
		}
	}

	// Print info
	printf("%-20s Time: %.5fs", name, duration);
	if(ref) printf(" | Valid: %s", valid ? "YES" : "NO");
	printf("\n");
	printf("\tInput:     "); print_array_sample(input, n, 16);
	printf("\tOutput:    "); print_array_sample(output, n, 16);
	if(ref) { printf("\tReference: "); print_array_sample((int*)ref, n, 16); }
}

int main() {
	const int n = 1000000000; // don't set above 1000000000, otherwise it will crash
	const int max_input_value = 2;
	assert((unsigned long long)n * max_input_value < INT_MAX); // ensure it can't overflow
	size_t bytes = n * sizeof(int);

	printf("Initializing %d elements (%.2f MiB)...\n", n, bytes / (1024.0 * 1024.0));
	int* data = (int*)malloc(bytes);
	int* result = (int*)malloc(bytes);
	int* ref = (int*)malloc(bytes);
	for (int i = 0; i < n; i++) { data[i] = rand() % (max_input_value+1); }

	// Sequential (Source of Truth)
	run_test("CPU Sequential", prefix_sum_sequential, data, n, ref, NULL);

	// OpenMP
	run_test("CPU Multi-threaded", prefix_sum_omp, data, n, result, ref);

	free(data); free(result); free(ref);

	return 0;
}
