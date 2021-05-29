import matplotlib.pyplot as plt
import scipy.io
import numpy as np



tests_dir = "/home/mcculls5/Documents/glomerulus/nonnegative_connectome/data/"
tests = [
    "nonnegative_top_view_top_view_20_e4e5.mat",
    "nonnegative_top_view_top_view_50_e4e5.mat",
    "nonnegative_top_view_top_view_100_e4e5.mat",
    "nonnegative_top_view_top_view_250_full.mat",
    "nonnegative_top_view_top_view_500_e4e5.mat",
]

test_names = [
    "Rank 20",
    "Rank 50",
    "Rank 100",
    "Rank 250",
    "Rank 500",
]

greedy = []
data = []
ranks = []
for test in tests:
    test_data = scipy.io.loadmat(tests_dir + test)
   
    greedy.append(test_data["cost_greedy"][0][0])
    data.append(test_data["cost_final"][0][0])
    ranks.append(test_data["W"][0][0].shape[1])

greedy_ranks = []
for rank in ranks:
    greedy_ranks.append(int((rank+1)/2))


plt.figure()
plt.clf()

for i in range(len(ranks)):
    plt.plot([ranks[i],greedy_ranks[i]], [data[i], greedy[i]], color="grey", linestyle="--")


plt.plot(ranks, data, label="Nonnegative", color="navy", linestyle="-", marker=".", markersize="10")
plt.plot(greedy_ranks, greedy, label="Unconstrained", color="tomato", linestyle="-", marker=".", markersize="10")
plt.plot([], [], color="grey", linestyle="--", label="Related Solutions")

# plt.yscale("log")
plt.xlabel("Solution Rank")
plt.xticks(ticks=range(0,1100,100))
plt.ylabel("Final Cost")
plt.title("Final Cost by Rank of Solution")
plt.legend(loc="best")
plt.savefig("final_cost_by_rank", bbox_inches="tight")
plt.close()