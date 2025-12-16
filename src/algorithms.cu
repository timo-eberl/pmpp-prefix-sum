#include "scan.h"
#include <omp.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>

size_t workspace_none(int n) { return 0; }

void scan_sequential(const int* input, int n, int* output, void* workspace) {
	if (n == 0) return;
	output[0] = input[0];
	for (int i = 1; i < n; i++) {
		output[i] = output[i - 1] + input[i];
	}
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
