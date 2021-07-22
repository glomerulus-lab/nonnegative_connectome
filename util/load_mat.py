import numpy as np
import scipy.io


# Loads solution factors from .mat file, returns lamb
def load_lamb(name, folder, greedy=False,):
    try:
        filename = folder+name
        print(filename)
        data = scipy.io.loadmat(filename)


        return data["lamb"][0][0]

    except:
        # Solution file not found
        if(greedy):
            print("Solution from '" + filename + "' could not be found, make sure a solution exists by running test_allvis_completion.m. Exitting.")
        else:
            print("Solution from '" + filename + "' could not be found, Exitting.")
        exit(1)

# Loads solution factors from .mat file, returns U, V
# Where the full (low rank) solution X = U @ V
# U has shape (nx * r)
# V has shape (r * ny)
def load_solution(name, folder, greedy=False):
    
    try:
        filename = folder+ name
        print(filename)
        data = scipy.io.loadmat(filename)


        return data["W"][0][0], np.transpose(data["W"][1][0])

    except:
        # Solution file not found
        if(greedy):
            print("Solution from '" + filename + "' could not be found, make sure a solution exists by running test_allvis_completion.m. Exitting.")
        else:
            print("Solution from '" + filename + "' could not be found, Exitting.")
        exit(1)
    


# Loads solution returns W_true with shape (200, 200)
def load_test_truth():
    try:
        data = scipy.io.loadmat("../lowrank_connectome/data/test_matrices.mat")
        return data["W_true"]
    except:
        print("W_true from '../lowrank_connectome/data/test_matrices.mat' could not be found, Exitting.")
        exit(1)


# Loads the voxel coordinates and look up table (lut)
def load_voxel_coords(testname):

    try:
        data = scipy.io.loadmat("../lowrank_connectome/data/"+testname+"_matrices.mat")

        voxel_coords_source = data["voxel_coords_source"]
        voxel_coords_target = data["voxel_coords_target"]
        view_lut = data["view_lut"]

        return  voxel_coords_source, voxel_coords_target, view_lut
    except:
        print("'Voxel coordinates from '../lowrank_connectome/data/"+testname+"_matrices.mat' could not be found, Exitting.")
        exit(1)

def load_all_matricies(testname):
    try:
        data = scipy.io.loadmat("../lowrank_connectome/data/"+testname+"_matrices.mat")
        data["Omega"] = data["Omega"].astype(np.int8)
        data["Lx"] = data["Lx"].astype(np.int8)
        data["Ly"] = data["Ly"].astype(np.int8)
        return  data
    except:
        print("'Matricies data from '../lowrank_connectome/data/"+testname+"_matrices.mat' could not be found, Exitting.")
        exit(1)
