import os
import glob

filename = []
for filepath in glob.glob('/home/stillwj3/Documents/research/lowrank_connectome/matlab/solution/lambda_test_*.mat'):
    base = os.path.basename(filepath)
    filename = os.path.splitext(base)

    path = os.path.join("lambda_tests/lambda_images/", filename[0])
    os.mkdir(path)

    command = "python nonnegative_converter.py test " + filename[0] + " lambda_tests/lambda_data/ lambda_tests/lambda_images/"+filename[0]+"/" +" "+ filename[0].replace("test_","") + " 1000 20 50 1000 20 50 -init_tol 1e-6 -tol 1e-7 --load_lamb -from_lc"
    os.system(command)

    print("\n")
    print("\n")
    print("\n")
    print("\n")

