import matplotlib.pyplot as plt
import scipy.io
import numpy as np

tests_dir = "/home/mcculls5/Documents/glomerulus/nonnegative_connectome/data/"
tests2 = [
    "nonnegative_test_test_init_quality_2_0.mat",
    "nonnegative_test_test_init_quality_2_10.mat",
    "nonnegative_test_test_init_quality_2_20.mat",
    "nonnegative_test_test_init_quality_2_30.mat",
    "nonnegative_test_test_init_quality_2_40.mat",
    "nonnegative_test_test_init_quality_2_50.mat",
    "nonnegative_test_test_init_quality_2_60.mat",
    "nonnegative_test_test_init_quality_2_70.mat",
    "nonnegative_test_test_init_quality_2_80.mat",
    "nonnegative_test_test_init_quality_2_90.mat",
    "nonnegative_test_test_init_quality_2_100.mat",
]
tests = [
    "nonnegative_test_test_init_quality_0.mat",
    "nonnegative_test_test_init_quality_10.mat",
    "nonnegative_test_test_init_quality_20.mat",
    "nonnegative_test_test_init_quality_30.mat",
    "nonnegative_test_test_init_quality_40.mat",
    "nonnegative_test_test_init_quality_50.mat",
    "nonnegative_test_test_init_quality_60.mat",
    "nonnegative_test_test_init_quality_70.mat",
    "nonnegative_test_test_init_quality_80.mat",
    "nonnegative_test_test_init_quality_90.mat",
    "nonnegative_test_test_init_quality_100.mat",
]

tests3 = [
    "nonnegative_top_view_top_view_init_quality_0.mat",
    "nonnegative_top_view_top_view_init_quality_10.mat",
    "nonnegative_top_view_top_view_init_quality_20.mat",
    "nonnegative_top_view_top_view_init_quality_30.mat",
    "nonnegative_top_view_top_view_init_quality_40.mat",
    "nonnegative_top_view_top_view_init_quality_50.mat",
    "nonnegative_top_view_top_view_init_quality_60.mat",
    "nonnegative_top_view_top_view_init_quality_70.mat",
    "nonnegative_top_view_top_view_init_quality_80.mat",
    "nonnegative_top_view_top_view_init_quality_90.mat",
    "nonnegative_top_view_top_view_init_quality_100.mat",
]

testSuffixes= ["nonnegative_test_test_init_quality_", "nonnegative_test_test_init_quality_2_", "nonnegative_test_test_init_quality_3_", "nonnegative_test_test_init_quality_4_", "nonnegative_test_test_init_quality_5_", "nonnegative_test_test_init_quality_6_"]
steps = np.arange(0,160,10)

refining = []
final = []

for step in steps:
    avg_ref = 0
    avg_fin = 0
    for suffix in testSuffixes:
        test_data = scipy.io.loadmat(tests_dir + suffix +str(step))
        avg_ref += test_data["time_refining"][0][0]
        avg_fin += test_data["time_final_solution"][0][0]

    refining.append(avg_ref/len(testSuffixes))
    final.append(avg_fin/len(testSuffixes))





plt.figure()
plt.clf()
plt.bar(steps, final, width=8, label="Final Regression")
plt.bar(steps, refining, width=8, label="Initialization Refinement", bottom=final)

plt.xlabel("Refinement Iterations")
plt.ylabel("Time")
plt.title("Total Time by Refinement Iterations")
plt.legend(loc="best")
plt.savefig("init_quality")
plt.close()