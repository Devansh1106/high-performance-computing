# High-Performance Computing

Educational implementations of parallel and distributed computing with MPI, OpenMP, and CUDA.

## Contents

- **Root directory**: MPI basics, collective operations, and serial/parallel algorithms (dot product, matrix operations, numerical methods)
- **openmp/**: Shared memory parallelism (parallel loops, reductions, iterative solvers)
- **mpi/**: Distributed memory examples
- **assignment/**: Advanced implementations (sorting, 1D/2D heat diffusion)
- **hpc_project/**: GPU acceleration with CUDA (multi-GPU matrix-vector multiplication)

## Compilation

**MPI:**
```bash
mpicc -o program program.c
mpirun -np <num_processes> ./program
```

**OpenMP:**
```bash
gcc -fopenmp -o program program.c
./program
```

**CUDA:**
```bash
nvcc -o program program.cu
./program
```

## Requirements

- OpenMPI/MPICH (for MPI examples)
- GCC with OpenMP support
- NVIDIA CUDA Toolkit (for GPU code)

See `notes.md` for additional technical notes and `hpc_project/project_data.md` for performance benchmarks.
