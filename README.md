# nonnegative_connectome

This repository requires lowrank_connectome by kuerschner to generate initial connectome approximations.

https://gitlab.mpi-magdeburg.mpg.de/kuerschner/lowrank_connectome.git

The lowrank_connectome repository should be located beside this repository as shown below:
```
...
├── lowrank_connectome/
└── nonnegative_connectome/
```

### run_matlab.sh
This file provides a wrapper to run matlab code in lowrank_connectome/matlab/.
example invocation:
```
./run_matlab.sh test_allvis_completion
```

### load_mat.py
Python module to load connectome solutions from .mat files.