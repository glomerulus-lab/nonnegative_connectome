# nonnegative_connectome

This repository requires lowrank_connectome by kuerschner to generate initial connectome approximations.

https://gitlab.mpi-magdeburg.mpg.de/kuerschner/lowrank_connectome.git

The lowrank_connectome repository should be located beside this repository as shown below:
```
...
├── lowrank_connectome/
└── nonnegative_connectome/
```

## run_matlab.sh
This file provides a wrapper to run matlab code in lowrank_connectome/matlab/.
example invocation:
```
./run_matlab.sh test_allvis_completion
```

## load_mat.py
Python module to load connectome solutions from .mat files. Used by plot_svectors.py.


## plot_svectors.py
Visualize dominant singular vectors of solution. Based on plot_svectors.m in lowrank_connectome/matlab.

Parameters:
- testname: either 'top_view' or 'flatmap'. Used to locate voxel coordinate mapping.
- solution_name: name of .mat solution file to plot.
- n: number of factors to plot.

Flags:
- greedy: Look for solution name in the lowrank_connectome/data/ directory rather than nonnegative_connectome/data/.
- raw: Plot the raw solution, rather than scaled QR decompositions.

example invocation:
```
python3 plot_svectors.py flatmap flatmap_solution 4 --greedy
```
