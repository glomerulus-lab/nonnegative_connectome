import os
import glob

for filepath in glob.glob('/home/stillwj3/Documents/research/lowrank_connectome/matlab/solution/lambda_test_*.mat'):
    base = os.path.basename(filepath)
    filename = os.path.splitext(base)
    path = os.path.join("lambda_tests/lambda_images/", filename[0])
    os.mkdir(path)

    #python main.py test filename[0] lambda_tests/lambda_data/ lambda_tests/lambda_images/"$i"_images/ "$i" 400 14 50 400 14 50 -init_tol 1e-6 -tol 1e-7 -lamb $i --load_lambda

