import numpy as np
import matplotlib.pyplot as plt

"""
    This script implements specifics 
"""

num_steps = 200
num_actions = 10
delta = np.zeros((num_steps, num_actions))

def get_argmaxes(array):
    max_val = np.max(array)
    return np.argwhere(array == max_val).flatten(), max_val


idx = num_actions // 2
indexes = np.zeros(num_steps-1)
for step in range(num_steps-1):
    argmaxes, max_val = get_argmaxes(delta[step])
    idx = np.random.choice(argmaxes)
    # idx_plus_one = (idx + 1) % num_actions
    # if delta[step, idx_plus_one] == max_val:
    #     idx = idx_plus_one
    # elif delta[step, idx] < max_val:
    #     idx = np.random.choice(argmaxes)
        
    # if np.max(delta[step]) > delta[step, idx]:
    #     idx = np.argmax(delta[step]) 
    indexes[step] = idx
    delta[step+1] = delta[step]
    delta[step+1, (idx + 1) % num_actions] += 1
    delta[step+1, (idx - 1) % num_actions] -= 1
    
for d, i in zip(delta, indexes):
    print(d, int(i))
    
max_delta = np.max(delta, axis=1)

fig, ax = plt.subplots()
ax.plot(range(num_steps), max_delta, label="$\max\, \Delta$")
ax.plot(range(num_steps), np.sqrt(np.array(range(num_steps))/3), label="$\sqrt{t}/2$")
ax.set_xlabel("$t$")
ax.set_title("FP delta sequence")
ax.legend()

fig, ax = plt.subplots()
for a in range(num_actions):
    ax.plot(range(num_steps), delta[:,a], label="$\Delta[t,"+str(a)+"]$")
ax.legend()