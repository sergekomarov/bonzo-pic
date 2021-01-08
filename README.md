# bonzo-pic

Particle-in-cell code for plasma modelling: a version of the TRISTAN code by O. Buneman (1993) implemented in Cython and C.

## Prerequisites

- python 3
- cython 3
- optional (parallelization): MPI (tested with OpenMPI and Intel MPI) + mpi4py
- optional (output file format): HDF5 + h5py, necessary with MPI

## Compilation

1) See available compilation options:
```
python setup.py --help 
```
2) Compile:
```
python setup.py build_ext --inplace --problem=problem_name [options]
```
``problem_name`` corresponds to the name of the chosen problem generator found in the ``src/problem`` directory

## Running

1) Create a user directory in the root folder:
```
mkdir user_dir
```
2) Place a configuration file (example provided in the root folder) in the user folder:
```
cp config.cfg user_dir
```
3) Run: 
```
python main.py ./user_dir
```
4) Output is written to ``user_dir/out``, restart files saved in ``user_dir/rst``.

