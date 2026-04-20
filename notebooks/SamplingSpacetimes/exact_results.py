import numpy as np
from CST_helpers import *
import time
n = 7
start = time.time()

_, unique_CS = get_unique_matrices(n)
mid = time.time()
print(f"Time taken to find unique causal sets: {mid - start:.2f} seconds")

heights_list = []
num_relations_list = []
for unique in unique_CS:
    unique = np.frombuffer(unique, dtype=np.int32).reshape(n,n)
    height_ = height(unique)
    
    num_relation = num_relations(unique)
    heights_list.append(height_)
    num_relations_list.append(num_relation)

print("Time to compute heights and number of relations: {:.2f} seconds".format(time.time() - mid))

#hs, bins_h = np.histogram(heights_list, bins=np.arange(0.5, n+1.5), density=True)



bins_h,hs = np.unique(heights_list, return_counts=True)#
hs = hs / sum(hs)  # Normalize to get probabilities
print("bins_h:", bins_h)
print("hs:", hs)
plt.scatter(bins_h, hs)
plt.xlabel("Height")
plt.ylabel("Frequency")
plt.title(f"Height Distribution of Unique Causal Sets (C={n})")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


#nr, bins_n = np.histogram(num_relations_list, bins=np.arange(-0.5, (n*(n-1)//2)+1.5), density=True)


bins_n, nr = np.unique(num_relations_list, return_counts=True)
nr = nr / sum(nr)  # Normalize to get probabilities
print("bins_n:", bins_n)
print("nr:", nr)

plt.scatter(bins_n, nr)
plt.xlabel("Number of Relations")
plt.ylabel("Frequency")
plt.title(f"Number of Relations Distribution of Unique Causal Sets (C={n})")
plt.grid(True, alpha=0.3)
plt.tight_layout()  
plt.show()




# Save hs, bins and nr, bins to a file for later use
import pickle
save_path =  os.path.dirname(__file__)
with open(os.path.join(save_path, f"save_files/exact_results_{n}.pkl"), "wb") as f:
    pickle.dump({
        "height_histogram": (bins_h,hs),
        "num_relations_histogram": (bins_n, nr)
    }, f)
