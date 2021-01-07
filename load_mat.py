import numpy as np
import scipy.io


# Loads solution factors from .mat file, returns U, V
# Where the full (low rank) solution X = U @ V
# U has shape (nx * r)
# V has shape (r * ny)
def load_solution(name, greedy=False):
    
    try:
        filename = "data/"+name

        if(greedy):
            # Load solution from lowrank_connectome repository
            filename = "../lowrank_connectome/" + filename + "_solution"

        filename = filename + ".mat"
        data = scipy.io.loadmat(filename)

        return data["W"][0][0], np.transpose(data["W"][1][0])

    except:
        # Solution file not found
        print("Solution from '" + filename + "' could not found, Exitting.")
        exit(1)
    


# Loads solution returns W_true with shape (200, 200)
def load_test_truth():
    try:
        data = scipy.io.loadmat("../lowrank_connectome/data/test_matricies.mat")
        return data["W_true"]
    except:
        print("W_true from '../lowrank_connectome/data/test_matricies.mat' could not be found, Exitting.")
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