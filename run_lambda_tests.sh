#!/bin/bash

# create different nonneg solutions (where i = lambda and initialized connectivity data is neg_1e-2_lambda)
for i in 1e-2 1e-1 0 1e1 1e2 1e3 1e4 1e5 1e6 1e7 1e8 1e9 1e10
    do  
        mkdir lambda_tests/lambda_images/"$i"_images
        python main.py test neg_1e-2_lambda lambda_tests/lambda_data/ lambda_tests/lambda_images/"$i"_images/ "$i" 400 14 50 400 14 50 -init_tol 1e-6 -tol 1e-7 -lamb $i
    done