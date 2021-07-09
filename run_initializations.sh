for i in 1e-5 1e-4 1e-3 1e-2 1e-1 1e0 1e1 1e2 1e3
    do  
        mkdir lambda_tests/l_initializations/images/"$i"_images
        python main.py test lambda_"$i"_solution lambda_tests/l_initializations/data/ lambda_tests/l_initializations/images/"$i"_images/ "$i" 400 14 50 400 14 50 -init_tol 1e-6 -tol 1e-7 -lamb 1e4
    done