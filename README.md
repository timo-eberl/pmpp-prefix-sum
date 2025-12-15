# Parallel Prefix Sum

## Compile

```
nvcc -arch=native -ccbin clang -O3 -Xcompiler "-O3 -march=native -fopenmp" -o build/prefix_sum prefix_sum.cu
```
