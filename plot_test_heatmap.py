import argparse
import matplotlib.pyplot as plt
import seaborn as sns # for heatmap generation
import os
import load_mat

parser = argparse.ArgumentParser(description="Creates a heatmap of solution.")
parser.add_argument('solution_name',  type=str, nargs=1,
                   help='Name of solution to plot')
parser.add_argument('output_file',  type=str, nargs=1,
                   help='name of file to save to')


def create_heatmap(U, V, output_file):

    if not os.path.exists('plots'):
        os.makedirs('plots')   
    if not os.path.exists('plots/test'):
        os.makedirs('plots/test')   
    # Ask user to confirm plots for large solutions
    if(U.shape[0] > 1000 or V.shape[1] > 1000):
        cont = input("Warning: Xolution is large, ("+str(U.shape[0])+", "+str(V.shape[1])+"). Continue? (y/n)")
        if (cont.lower() != "y"):
            print("Exitting plot_solution.py")
            exit()

    #Compute solution        
    X = U @ V
    rank = U.shape[1]

    #Plot solution
    ax = sns.heatmap( X, cmap="Reds", cbar=True, xticklabels=[], yticklabels=[])
    ax.tick_params(left=False, bottom=False) ## other options are right and top

    plt.title("$\mathregular{W_{"+str(rank)+"}}$")
    #Save plot as file
    plt.savefig(output_file, bbox_inches='tight')
    plt.clf()


def create_heatmap_test_truth(output_file):
    W = load_mat.load_test_truth()
    #Plot solution
    ax = sns.heatmap( W, cmap="Reds", cbar=True, xticklabels=[], yticklabels=[])
    plt.title("$\mathregular{W_{truth}}$")
    
    #Save plot as file
    plt.savefig(output_file, bbox_inches='tight')
    plt.clf()
    
    
#Wrapper for create_heatmap that accepts a solution file to load
def _create_heatmap_from_solution(solution_name, output_file, greedy=False):
    U, V = load_mat.load_solution(solution_name, greedy)
    create_heatmap(U, V, output_file)


if __name__ == '__main__':
    args = parser.parse_args()
    # create_heatmap_from_solution(args.solution_name[0], args.output_file[0])
    create_heatmap_test_truth(args.output_file[0])