# Parallel Prefix Sum

## Compile

```
clang -O3 -march=native -o build/prefix_sum prefix_sum.c
```

or

```
nvcc -ccbin clang -O3 -Xcompiler "-march=native" -o build/prefix_sum prefix_sum.cu
```
