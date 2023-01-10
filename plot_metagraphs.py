import numpy as np
import matplotlib.pyplot as plt 

# plt.rcParams["figure.dpi"] = 150
import matplotlib.style as style
style.use('default')

def get_opponents(index, alg_type):
    if alg_type == "fp":
        return range(index)
    
    if alg_type == "afp":
        eligible = list(range(0, index, 2))
        if index % 2 == 0:
            eligible.append(index - 1)
        return eligible

    if alg_type == "afp-fp1":
        eligible = []
        for i in range(index):
            if i % 2 == 1 or i == 0 or i == index-1:
                eligible.append(i)
        return eligible

    if alg_type == "afp-fp2":
        eligible = []
        for i in range(index):
            if i % 2 == 0 or i == 1 or i == index-1:
                eligible.append(i)
        return eligible

    if alg_type == "pfp-1":
        if index % 2 == 0:
            return get_opponents(index, "afp")
        else:
            return get_opponents(index, "fp")
        
    if alg_type == "pfp-2":
        if index % 2 == 0:
            return get_opponents(index, "afp")
        else:
            return [index-1]

    assert False
 
def get_fp_indices(population_size, alg_type):
    if alg_type == "fp":
        return range(population_size)
    
    if alg_type == "afp":
        return range(0, population_size, 2)

    if alg_type == "afp-fp1":
        indices = set(range(1, population_size, 2))
        indices.add(0)
        return indices

    if alg_type == "afp-fp2":
        indices = set(range(0, population_size, 2))
        indices.add(1)
        return indices    

    assert False        

def get_metagraph(population_size, alg_name):
    graph = np.zeros((population_size,population_size))
    for i in range(1,population_size):
        eligible_indices = get_opponents(i, alg_name)
        graph[i,eligible_indices] = 1/len(eligible_indices)
    return graph, get_fp_indices(population_size, alg_name)

if __name__ == "__main__":
    population_size = 11
    metagraphs = {
        alg : get_metagraph(population_size, alg)[0]
        for alg in ["fp", "afp", "afp-fp2"]
    }
    
    for i in range(1,population_size):
        for alg, metagraph in metagraphs.items():
            eligible_indices = get_opponents(i, alg)
            metagraph[i,eligible_indices] = 1/len(eligible_indices)
    
        
    fig, axes = plt.subplots(ncols=3, figsize=(9,7))
    cmap = "Blues"
    axes[0].imshow(metagraphs["fp"], cmap=cmap)
    axes[1].imshow(metagraphs["afp"], cmap=cmap)
    axes[2].imshow(metagraphs["afp-fp2"], cmap=cmap)
    
    first_idx = 1
    for ax_idx, ax in enumerate(axes):
        ax.set_xticks(np.arange(0,population_size))
        ax.set_xticklabels([str(i) for i in range(first_idx,population_size+first_idx)], fontsize=8)
        ax.set_yticks(np.arange(0,population_size))
        ax.set_yticklabels([str(i) for i in range(first_idx,population_size+first_idx)], fontsize=8)
    
    title_size = 14
    axis_label_size = 11
    axes[0].set_title("FP", fontsize=title_size)
    axes[1].set_title("AFP", fontsize=title_size)
    axes[2].set_title("AFP (2 initial FP steps)", fontsize=title_size)
    axes[0].set_ylabel("Learner", fontsize=axis_label_size)
    axes[0].set_xlabel("Opponent probability", fontsize=axis_label_size)
    
    save=True
    if save:
        plt.savefig("plots//generated//metagraphs", bbox_inches='tight')
        