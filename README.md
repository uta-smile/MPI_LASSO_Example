# MPI_LASSO_Example

This project is an simple example to demonstrate the usage of MPI for numerical computing.
It implements a distributd alternating direction method of multipliers (ADMM) algorithm for solving lasso problem. 

## Pre-requirement

This repository provide implementations in three languages:
- C. The C implementation requires [GNU Scientific Library (GSL)](http://www.gnu.org/software/gsl/) and Open MPI (or other MPI implementation).
- Python. The python version requires Numpy/SciPy and [MPI4Py](http://mpi4py.scipy.org/).
- Lua. The Lua version is implemented based on [Torch7](http://torch.ch/) with [MPIT](https://github.com/sixin-zh/mpiT).

## How to run it

### Run on local machine

First of all, one should load the require modules on TACC before running the code.
Using the follow script to load modules

```bash
. ./loadmod.sh
```

**Note:** the dot before "./loadmod.sh" is critical and cannot be eliminated.

Then, one can choose the following implementation to run.

C version
> make

> make run

Python version
> make runpy

Lua version 
> make runlua

### Run on TACC Stampede

C version
> make clean; make

> sbatch my_c_lasso_job

Python version
> sbatch my_py_lasso_job

Lua version 
> sbatch my_lua_lasso_job

**Note:** Here assumes Torch7 is installed in $HOME/torch/install.

## Preliminary result

Here is some very **unofficial** result on running time comparison of the three implementations on a Macbook Pro labtop computer and the TACC stampede node. 
The speed depends on the implementation of the MPI and numerial library.

| Language      | C/GSL         | Lua/Torch  | Python/Numpy  |
| ------------- |:-------------:|:----------:|:-------------:|
| Macbook       | 0.0264 sec    | 0.0553 sec | 0.669 sec     |
| Stampede      | 0.009936 s    | 0.009416 s | 0.064316      |


## Acknowledgement

The c implementation is from [here](https://web.stanford.edu/~boyd/papers/admm/mpi/).
