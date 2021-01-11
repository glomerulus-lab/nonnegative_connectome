import argparse
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap
import load_mat
import palettable # https://jiffyclub.github.io/palettable/

parser = argparse.ArgumentParser(description="Plot dominant factors of connectome solution")
parser.add_argument('testname',  type=str, nargs=1,
                   help='Name of test to plot. "flatmap" or "top_view"')
parser.add_argument('solution_name',  type=str, nargs=1,
                   help='Name of .mat solution file, including or excluding file extension.')
parser.add_argument('n',  type=str, nargs=1,
                   help='number of factors to plot')
parser.add_argument("--greedy", action="store_true", help="Search ../lowrank_connectome/data for solution.")

parser.add_argument("--raw", action="store_true", help="Plot raw solution, rather than scaled QR decompositions.")


def plot_svectors(U, V, testname, output_name, n, raw=False):
    voxel_coords_source, voxel_coords_target, view_lut = load_mat.load_voxel_coords(testname)
    
    if(raw):
        Q1 = U
        Q2 = V.T
    else:
        Q1, R1 = np.linalg.qr(U)
        Q2, R2 = np.linalg.qr(V.T)
        u, S, v = np.linalg.svd(R1@R2.T)
        Q1 = Q1 @ u
        Q2 = Q2 @ v

    for i in range(int(n)):
        if(raw):
            target_img = map_to_grid(Q1[:,i], voxel_coords_target, view_lut)
        else:
            target_img = map_to_grid(Q1[:,i] * S[i], voxel_coords_target, view_lut)
        source_img = map_to_grid(Q2[:,i] , voxel_coords_source, view_lut)

        plot_factor(target_img, source_img, "plots/qr/"+str(output_name)+"_factor_"+str(i+1))



# Create 2D image using known size of view, and coordinate mapping for vectorized solution
def map_to_grid(image_vec, voxel_coords, view_lut):
    #initialize the image to nans
    new_image = np.empty(view_lut.shape)
    new_image[:] = np.nan
    for i in range(image_vec.shape[0]):
        new_image[voxel_coords[i,0], voxel_coords[i,1]] = image_vec[i]
    return new_image

# Plots the image in the provided subplot
def create_plot_im(ax, img):
    #Find colormap range
    imgMin = np.nanmin(img)
    imgMax = np.nanmax(img)
    colormapLimit = max(np.abs(imgMin), np.abs(imgMax))

    #Include blue in colormap if any values are negative
    if(imgMin < 0):
        # plt.set_cmap("RdBu_r")
        im = ax.imshow(img, cmap=ListedColormap(palettable.colorbrewer.diverging.RdBu_11_r.mpl_colors))
        im.set_clim(-colormapLimit,colormapLimit)
    else:
        # plt.set_cmap("Reds")
        im = ax.imshow(img, cmap=ListedColormap(palettable.colorbrewer.sequential.Reds_9.mpl_colors))
        im.set_clim(0,colormapLimit)


    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="20%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    return im

#Visualize the source and target for the given factor.
def plot_factor(target_img, source_img, filename):
    print(filename)

    fig, (ax1, ax2) = plt.subplots(1,2) 
    
    im1 = create_plot_im(ax1, target_img)
    im2 = create_plot_im(ax2, source_img)
    
    plt.savefig(filename)
    plt.clf()
    plt.close()



if __name__ == '__main__':
    args = parser.parse_args()

    U, V = load_mat.load_solution(args.solution_name[0], args.greedy)
    plot_svectors(U, V, args.testname[0], args.solution_name[0].split("/")[-1], args.n[0], args.raw)
    