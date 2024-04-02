# CHAMPS Hackathon 2024

![airfoil](https://github.com/wvannoordt/helmholtz2024/blob/main/assets/streak.png)

## Summary
This repository contains two smaller solvers that are representative of workloads that may 
be encountered in the CFD industry. One of them is a smaller, simpler case that simulates
an incompressible, free, periodic vortex that breaks down into turbulence. The other is more
complex, and features an airfoil at cruise conditions.

The cases are located in the `cases` directory, each contains a makefile that will compile the mini-solver.
Installation instructions are in the dedicated section below.

## Basic Requirements
- NVIDIA GPU with >= 8GB of global memory
- G++ version 11.2 or newer
- CUDA version 12.3 or newer
- OpenMPI version 4.1.2 or newer, headers and compiler wrappers
- GNU Make

## Installation
The test cases rely on the presence of `mpicxx` and `nvcc` on the system path, as well as two libraries:

- [spade](https://github.com/wvannoordt/spade): this library is the primary workhorse for the new version of the CHAMPS code. It is header only and requires no compilation at time of writing.
- [scidf](https://github.com/wvannoordt/scidf): this library is a library for reading the `.sdf` files used for parametric inputs for solvers. It is also header-only.

Both solvers will rely on the presence of two environment variables available at compile time:

- `SPADE` should point to the `spade` installation path, e.g. `export SPADE=/path/to/spade`, and
- `SCIDF` should do the same for `scidf`, e.g. `export SPADE=/path/to/spade`

Once this is the case, both solvers can be built in their own directories with `make`. Currently, both solvers produce a single executable, `vdf.x`, that
can be run either as simply `./vdf.x` or as `./vdf.x file.sdf`, where `file.sdf` is the file containing the runtime parameters.