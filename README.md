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

https://github.com/harrispopgen/mushi


## nonnegative_converter.py 
Computes nonnegative connectome solution using greedy solution.

Arguments:
- testname: 'test', 'top_view', or 'flatmap' 
- solution_name: filename of unconstrained solution
- data_directory: path to folder where output data is stored
- images_directory: path to folder where output images are stored (test heatmaps)
- output_suffix: String added to the output solution's filename
- init_max_outer_iter: Number of alternating iterations in initialization refinement
- init_max_inner_iter: Number of updates on each factor before alternating in initialization refinement
- init_max_line_iter: Maximum number of line-search iterations in initialization refinement
- max_outer_iter: Number of alternating iterations
- max_inner_iter: Number of updates on each factor before alternating
- max_line_iter: Maximum number of line-search iterations

Flags:
- from_lc: look for unconstrained solution in ../lowrank_connectome/data rather than /data
- tol: PGD stopping criteria tolerance
- init_tol: PGD stopping criteria tolerance for initialization refinement
- alt_tol: tolerance for alt_acc_prox_grad
- lamb: value of lambda
- -load_lamb: load lambda value from unconstrained solution in ../lowrank_connectome/matlab/solution

Example invocation:
```
python3 nonnegative_converter.py top_view top_view_solution data/ images/ nonneg_top_view 400 14 50 400 14 50 -init_tol 1e-6 -tol 1e-7 --load_lamb -from_lc
```
<br />

# Useful Modules 

## nonnegative_connectome.py
Module containing code for computing the nonnegative connectome. Used by nonnegative_converter.py

## nonnegative_intialization.py
Module containing code for computing initial nonnegative factors. Used by nonnegative_converter.py

```
...
├── nonnegative_converter.py
├── nonnegative
│   ├── nonnegative_connectome.py
│   └── nonnegative_initialization.py
...
```

<br />

------

## alt_acc_prox_grad.py
Implementation of heirarchical alternating PGD.
Note: contains acc_prox_grad_method function from: 
https://github.com/harrispopgen/mushi/blob/master/mushi/optimization.py
https://github.com/harrispopgen/mushi/blob/master/LICENSE

```
...
├── optimization
│   └──  alt_acc_prox_grad.py
...
```
<br />

------

## plot_svectors.py
Visualize dominant singular vectors of solution. Based on plot_svectors.m in lowrank_connectome/matlab. used by nonnegative_converter.py

Arguments:
- testname: either 'top_view' or 'flatmap'. Used to locate voxel coordinate mapping
- solution_name: name of .mat solution file to plot
- n: number of factors to plot

Flags:
- greedy: Look for solution name in the lowrank_connectome/data/ directory rather than nonnegative_connectome/data/.
- raw: Plot the raw solution, rather than scaled QR decompositions.

example invocation:
```
python3 plot_svectors.py flatmap flatmap_solution 4 -greedy
```

## plot_test_heatmap.py
Module to plot test solutions on a heatmap. Used by nonnegative_converter.py

Arguments:
- solution_name: Name of solution to plot
- output_file: Name of file to save heatmap


## plot_l_curve.py
Module to plot regularization vs cost l-curve of both nonnegative and greedy solutions. Corner of L-Curve represents optimal lambda value.

Arguments:
- testname: Can be 'test', 'top_view' or 'flatmap'. Used to determine where figures are saved.
- solution_name: Name of solutions to plot (Ex. 'test_*.mat')
- path_to_solution: Path to solution to plot
- title: Title of l-curve (note: title cannot contain spaces)

Flags:
- nneg: Plot the nonnegative solution using proper keywords in data retrieval 

Example invocation:
```
python3 plot_l_curve.py test lambda_test*.mat path/to/solutions/ Nonneg_Test_L_Curve -nneg
```

<br />

> **__NOTE:__** For plot_l_curve and plot_error, add the following directories to save images: 
> ```
> ...
> ├── plot
> ├── data
> │   ├── lambda_tests
> │   ├── lambda_tv
> │   └── lambda_fm
> ...
> ``` 


<br />



## plot_error.py 
Module to plot the error between true solution and estimated solution for both nonnegative and greedy test solutions, given various lambdas.

Arguments:
- solution_name: Name of solutions to plot (Ex. 'test_*.mat')
- path_to_solution: Path to solution to plot

Flags:
- nneg: Plot the nonnegative solution using proper keywords in data retrieval 

Example invocation:
```
python3 plot_error.py lambda_test*.mat path/to/solutions/ -nneg
```


```
...
├── plot
│   ├── plots
│   ├── plot_svectors.py
│   ├── plot_test_heatmap.py
│   ├── plot_l_curve.py
│   └── plot_error.py
...
```

<br />

------

## load_mat.py
Module to load connectome solutions from .mat files. Used by nonnegative_converter.py, plot_svectors.py, plot_test_heatmap.py

```
...
├── util
    ├── load_mat.py
    ├── math_util.py
    └── read_hyperparameters.py
 ```



