import pickle
from matplotlib import pyplot as plt
import numpy as np

# Load payoffs
with open("data\\spinning_top_payoffs.pkl", "rb") as fh:
  payoffs = pickle.load(fh)


# Iterate over games

if __name__ == "__main__":
    for game_name in payoffs:
    
      print(f"Game name: {game_name}")
      print(f"Number of strategies: {payoffs[game_name].shape[0]}")
      print()
    
      # Sort strategies by mean winrate for nice presentation
      order = np.argsort(-payoffs[game_name].mean(1))
    
      # Plot the payoff
      plt.figure()
      plt.title(game_name)
      plt.imshow(payoffs[game_name][order, :][:, order])
      plt.axis('off')
      plt.show()
      plt.close()