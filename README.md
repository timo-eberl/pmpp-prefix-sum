# Parallel Prefix Sum

## Compile

```
clang -O3 -march=native -fopenmp -o build/prefix_sum prefix_sum.c
```

or

```
nvcc -ccbin clang -O3 -Xcompiler "-march=native -fopenmp" -o build/prefix_sum prefix_sum.cu
```
