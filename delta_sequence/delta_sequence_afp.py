import numpy as np
import matplotlib.pyplot as plt


def get_argmaxes(array):
    max_val = np.max(array)
    return np.argwhere(array == max_val).flatten(), max_val


def random_argmax(array):
    argmaxes, max_val = get_argmaxes(array)
    return np.random.choice(argmaxes), max_val


def next_delta(delta, idx):
    next_delta = delta.copy()
    next_delta[(idx + 1) % len(delta)] += 1
    next_delta[(idx - 1) % len(delta)] -= 1
    return next_delta


def argmax_function(func, args):
    max_val = -np.inf
    arg_max = None
    for arg in args:
        val = func(arg)
        if val > max_val:
            max_val = val
            arg_max = arg
    return arg_max, max_val


def inconvenient_argmax(prev_array, array, step, depth):
    """
    Searches ahead 'depth' number of steps to find the argmax to select now
    in order to maximize the array in the future (according to AFP evolution).
    
    Returns the current argmax and future max value of delta.
    """
    argmaxes, _ = get_argmaxes(array)
    
    if depth <= 0 and step % 2 == 1:
        # If this play will be kept, look ahead to see what will maximize
        def objective(idx):
            return max(next_delta(prev_array, idx))
        return argmax_function(objective, argmaxes)
    
    array_to_add_to = prev_array if step % 2 == 1 else array
    def objective(idx):
        amax, aval = inconvenient_argmax(array, next_delta(array_to_add_to, idx), step+1, depth-1)
        return aval
    
    return argmax_function(objective, argmaxes)

num_steps = 1000
num_actions = 7
delta = np.zeros((num_steps, num_actions))

indexes = np.zeros(num_steps-1)
for step in range(num_steps-1):
    if step == 0 or True:
        idx, _ = random_argmax(delta[step])
    else:
        idx, _ = inconvenient_argmax(delta[step-1], delta[step], step, 3)
    indexes[step] = idx
    
    is_odd_step = (step % 2) == 1
    delta[step+1] = delta[step - is_odd_step]
    delta[step+1, (idx + 1) % num_actions] += 1
    delta[step+1, (idx - 1) % num_actions] -= 1
    
for timestep, (d,i) in enumerate(zip(delta, indexes)):
    print(d, int(i), "<--" if timestep % 2 == 0 else "   ")
    
max_delta = np.max(delta, axis=1)
print(max_delta)
print("All timesteps ", max(max_delta))
print("Even timesteps", max(max_delta[::2]))

idx_of_max_even = 2*np.argmax(max_delta[::2])
for idx in range(idx_of_max_even-6, idx_of_max_even+1):
    print(delta[idx], idx, "<--" if idx % 2 == 0 else "  ", "*" if idx == idx_of_max_even else " ")
    

fig, ax = plt.subplots()
ax.plot(range(num_steps), max_delta, label="$\max\, \Delta$")
ax.set_xlabel("$t$")
ax.set_title("FP delta sequence")
ax.legend()
