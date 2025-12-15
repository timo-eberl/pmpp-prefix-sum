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

void print_array_sample(int arr[], int n, int limit) {
	int width = limit / 2;
	for (int i = 0; i < n; i++) {
		// If array is too big and we reached the middle, jump to the end
		if (n > limit && i == width) { printf(" .......... "); i = n - width - 1; continue; }
		printf("%12d", arr[i]);
	} printf("\n");
}

// unix only
double get_time() {
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
	int n = 500000000;
	size_t bytes = n * sizeof(int);
	printf("Initializing %d elements (%.2f MB)...\n", n, bytes / (1024.0 * 1024.0));
	int* input_data = (int*)malloc(bytes);
	for (int i = 0; i < n; i++) input_data[i] = 1; // Initialize data

	int* reference_result = (int*)malloc(n * sizeof(int));

	{
		double start = get_time();
		prefix_sum_sequential(input_data, n, reference_result);
		double end = get_time();

		printf("CPU Sequential:\n");
		printf("\tTime: %.4f s\n", end - start);
		printf("\tOriginal Array:   "); print_array_sample(input_data, n, 20);
		printf("\tPrefix Sum Array: "); print_array_sample(reference_result, n, 20);
	}

	free(input_data);
	free(reference_result);

	return 0;
}
