# Parallel Prefix Sum (Scan)

## Compile

```
nvcc -arch=native -ccbin clang -O3 -Xcompiler "-O3 -march=native -fopenmp" src/algorithms.cu src/main.cu -o build/scan
```
