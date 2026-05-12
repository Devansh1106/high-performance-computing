- Compiler flags for mpicc:
    - `showme`
        - `showme:compile`; `showme:link`
- use `man mpicc`: for all other options.
- strong scaling: fix problem size and change no. of processors (Amdahl's law)
- weak scaling: fix problem size for each processor i.e. scale the problem as processors increase
- Gustafson's law

## OpenMP
- Scheduling: changing the data distribution.
    - `#pragma omp parallel for schedule(schedule-type, chunk-size)`
    - Options for schedule-type: 
