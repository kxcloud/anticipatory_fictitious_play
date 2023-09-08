# AFP
This repo corresponds to the paper [Anticipatory Fictitious Play](https://www.ijcai.org/proceedings/2023/0009) (Cloud, Wang, and Kerr, 2023). It contains matrix game simulations for FP, AFP, and other variants, and scripts for generating the plots in the paper. Deep learning environments, algorithms, and experiment parameters are property of Riot Games and are *not* included here. **Note: this code is not cleaned up, organized, or optimized, and is merely presented for the sake of replicability.**


The key scripts are...

* `games.py` - stores many different matrix games
* `InteractivePlayer.py` - implements FP, AFP within a more general framework of FP-like algorithms
* `lp_solver.py` - solve matrix games with linear programming
* `plot_metagraphs` - used to plot the interaction graphs for FP, AFP, and AFP with initial steps of FP
* `plots_for_paper_*` - generate plots for the paper 

The other scripts were used for temporary investigations, like `interaction_graph_search.py` and `optimized_graphs.py` which were used to search over interaction graphs to find algorithms with better small-sample performance, and `plot_vector_fields.py` for showing FP and AFP as vector fields on a simplex.
