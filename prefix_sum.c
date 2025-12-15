#define _POSIX_C_SOURCE 199309L // we use POSIX time
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>

void prefix_sum_sequential(int input[], int n, int output[]) {
	if (n == 0) return;
	output[0] = input[0];
	for (int i = 1; i < n; i++) {
		output[i] = output[i - 1] + input[i];
	}
}

void print_array_head(int arr[], int n, int limit) {
	int m = (limit < n) ? limit : n;
	for (int i = 0; i < m; i++) printf("%d\t", arr[i]);
	printf(n > limit ? "... (truncated)\n" : "\n");
}

// unix only
double get_time() {
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
	int n = 200000000;
	size_t bytes = n * sizeof(int);
	printf("Initializing %d elements (%.2f MB)...\n", n, bytes / (1024.0 * 1024.0));
	int* data = (int*)malloc(bytes);
	for (int i = 0; i < n; i++) data[i] = i; // Initialize data

	{
		int* result = (int*)malloc(n * sizeof(int));
		double start = get_time();
		prefix_sum_sequential(data, n, result);
		double end = get_time();

		printf("CPU Sequential:\n");
		printf("\tTime: %.4f s\n", end - start);
		printf("\tOriginal Array:   "); print_array_head(data, n, 20);
		printf("\tPrefix Sum Array: "); print_array_head(result, n, 20);
		free(result);
	}

	free(data);

	return 0;
}
