import os
import argparse
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap
from matplotlib import gridspec
import sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
import load_mat
import palettable # https://jiffyclub.github.io/palettable/





parser = argparse.ArgumentParser(description='Plot dominant factors of connectome solution')
# Arguments
parser.add_argument('testname',         type=str, nargs=1, help='Name of test to plot. "flatmap" or "top_view"')
parser.add_argument('solution_name',    type=str, nargs=1, help='Name of .mat solution file, including or excluding file extension.')
parser.add_argument('n',                type=str, nargs=1, help='number of factors to plot')
# Flags
parser.add_argument('-greedy', action='store_true', help='Search ../lowrank_connectome/data for solution.')
parser.add_argument('-nneg',    action='store_true', help='Plot reordered & scaled solution, rather than scaled QR decomposition.')


def plot_svectors(U, V, testname, output_name, n, nneg=False):
    voxel_coords_source, voxel_coords_target, view_lut = load_mat.load_voxel_coords(testname)

    if(nneg):
        outputpath = 'plots/nneg/'

        Q1 = U.copy()
        Q2 = (V.T).copy()

        r = Q1.shape[1]
        factor_norms_sq = np.zeros(r)

        for rank in range(r):
            # Q1_r = Q1[:, rank][:, np.newaxis]
            # Q2_r = Q2[:, rank][:, np.newaxis].T
            
            Q1_r = Q1[:, rank]
            Q2_r = Q2[:, rank]
            
            Q2_norm = np.linalg.norm(Q2_r)

            #factor_norms_sq[rank] = np.trace(Q2_r @ Q2_r.T @ Q1_r.T @ Q1_r)
            factor_norms_sq[rank] = np.linalg.norm(Q1_r) *Q2_norm 

            Q1[:, rank] *= Q2_norm
            Q2[:, rank] /= Q2_norm

        indexlist = np.argsort(-1*factor_norms_sq)
        
        Q1 = Q1[:, indexlist]
        Q2 = Q2[:, indexlist]

    else:
        outputpath = 'plots/qr/'

        Q1, R1 = np.linalg.qr(U, mode='reduced')
        Q2, R2 = np.linalg.qr(V.T)
        u, S, vh = np.linalg.svd(R1 @ (R2.T) )
        Q1 = Q1 @ u * S
        Q2 = Q2 @ vh.T
    
    
    outputpath = 'plots/plot_test/' #TODO remove this line

    if not os.path.exists('plots'):
        os.makedirs('plots')   
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)   


    for i in range(int(n)):
        # Correct the sign such that the maximum element is positive
        argmax = np.argmax(np.abs(Q1[:,i]))
        sign = 1/np.sign(Q1[:,i][argmax])
        
        target_img = map_to_grid(Q1[:,i] * sign, voxel_coords_target, view_lut)
        source_img = map_to_grid(Q2[:,i] * sign , voxel_coords_source, view_lut)

        filename = outputpath+str(output_name)+'_factor_'+str(i+1)
        plot_factor(target_img, source_img, testname, filename, np.linalg.norm(Q1[:,i]), i+1)



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
    colormapLimit = max(np.abs(imgMin), np.abs(imgMax)) * 0.9

    #Include blue in colormap if any values are negative
    if(imgMin < 0):
        # plt.set_cmap('RdBu_r')
        im = ax.imshow(img, cmap=ListedColormap(palettable.colorbrewer.diverging.RdBu_11_r.mpl_colors))
        im.set_clim(-colormapLimit,colormapLimit)
    else:
        #im = ax.imshow(img, cmap=ListedColormap(palettable.colorbrewer.sequential.Reds_9.mpl_colors))
        im = ax.imshow(img, cmap="Reds")

        im.set_clim(0,colormapLimit)


    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size=0.25, pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_ticks([])
    return im

#Visualize the source and target for the given factor.
def plot_factor(target_img, source_img, testname, filename, magnitude, factor):
    print(filename)
    figsize = (2,1)
    #fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8,4)) 
    fig = plt.figure()

    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    
    fig = plt.figure(figsize=(8, 6)) 
    
    ratios = [1.73, 1]
    title_height = 0.85
    title_testname = "Top-View"
    if(testname == 'flatmap'):
        ratios = [1.8, 1]
        title_height = 0.66
        title_testname = "Flatmap"
    
    gs = gridspec.GridSpec(1, 2, width_ratios=ratios) 

    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])


    im1 = create_plot_im(ax1, target_img)
    im2 = create_plot_im(ax2, remove_left_of_image(source_img))
    
    plt.figtext(0.52,title_height,title_testname + " Factor " + str(factor) + ", Norm: " + "{:.2f}".format(magnitude), ha='center', va='center', fontsize="20")
    plt.savefig(filename.replace(".","_"), dpi=400, bbox_inches='tight')
    plt.clf()
    plt.close()

def remove_left_of_image(img):
    shape = img.shape
    return img[:,int(shape[1]/2)-10:]


if __name__ == '__main__':
    args = parser.parse_args()
    U, V = load_mat.load_solution(args.solution_name[0], args.greedy)
    plot_svectors(U, V, args.testname[0], args.solution_name[0].split('/')[-1], args.n[0], args.nneg)
    