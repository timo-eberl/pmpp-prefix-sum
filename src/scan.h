#ifndef SCAN_H
#define SCAN_H

#include <stddef.h>
#include <stdbool.h>

// Function pointer types
typedef void (*ps_func)(const int*, int, int*, void*);
typedef size_t (*workspace_func)(int);

// Algorithm metadata
typedef struct {
	const char* name;
	ps_func func;
	workspace_func calc_workspace;
	bool is_gpu;
	int max_n; // -1 for no limit
} ScanAlgorithm;

// Algorithm Declarations
// CPU
size_t workspace_none(int n);
size_t workspace_omp(int n);
void prefix_sum_sequential(const int* input, int n, int* output, void* workspace);
void prefix_sum_omp(const int* input, int n, int* output, void* workspace);
// GPU
void prefix_sum_thrust(const int* d_input, int n, int* d_out, void* workspace);

// Helpers
void print_array_sample(const int* arr, int n, int limit);
double get_time();

#endif
