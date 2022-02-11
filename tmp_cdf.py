# %%
import numpy as np
import matplotlib.pyplot as plt
from maxsum_rl import *
from maxsum_rl_validation import N_TEST

sumrates_mp = np.load("sumrates_mp.npy")
sumrates_rl = np.load("sumrates_rl.npy")
sumrates_hungarian = np.load("sumrates_hungarian.npy")
sumrates_sorted = {"mp": np.zeros(N_TEST),
                   "rl": np.zeros(N_TEST),
                   "hungarian": np.zeros(N_TEST)}
sumrates_sorted["hungarian"] = np.sort(sumrates_hungarian)
sumrates_sorted["mp"] = np.sort(sumrates_mp)
sumrates_sorted["rl"] = np.sort(sumrates_rl)
cdf_bins = np.arange(N_TEST)/float(N_TEST-1)
plt.semilogx(sumrates_sorted["hungarian"], cdf_bins, label="Hungarian", linewidth=3, color='yellow', alpha=1)
plt.semilogx(sumrates_sorted["mp"], cdf_bins, ':', color='black', label="Ising")
plt.semilogx(sumrates_sorted["rl"], cdf_bins, '--', color='r', label="RL")
plt.legend()
plt.xlim(3, 30)
plt.title("Test set CDF")
plt.xlabel("sum-rate [bps]")
plt.savefig("cdf.pdf")
plt.savefig("cdf.png")
plt.close("all")
