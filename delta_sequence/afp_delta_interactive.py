import numpy as np

"""
This standalone script lets you run AFP interactively in the form of a "delta
sequence." In a delta sequence, you pick an index, then -1 gets added to the
entry below and +1 gets added to the entry above. The max entry of the delta 
sequence divided by the timestep is the exploitability of AFP on RPS at that
step. Because this is AFP, every odd-indexed play will be forgotten.

Note that you can toggle PLAY_EVEN_STEPS_ONLY on or off. If *off*, you will play
both the "ephemeral" (odd-indexed) FP steps and the "permanent" (even-indexed)
AFP steps. If *on*, you will be shown the indexes where AFP could play 
directly, and every move will be "permanent."

"""

def next_delta(delta, idx):
    next_delta = delta.copy()
    next_delta[(idx + 1) % len(delta)] += 1
    next_delta[(idx - 1) % len(delta)] -= 1
    return next_delta


def get_argmaxes(array):
    max_val = np.max(array)
    argmaxes = []
    for idx in range(len(array)):
        if array[idx] == max_val:
            argmaxes.append(idx)
    return argmaxes, max_val


def get_2step_argmaxes(array):
    argmaxes, _ = get_argmaxes(array)
    argmaxes2 = set()
    for idx in argmaxes:
        array2 = next_delta(array, idx)
        argmaxes2.update(get_argmaxes(array2)[0])
    return argmaxes2
    

def argprint(indexes, num_actions, end=None):
    argmax_str = "idx:  "
    for idx in range(num_actions):
        if idx in indexes:
            argmax_str += f"{idx:3.0f} "
        else:
            argmax_str += "    "
    argmax_str += "  "
    print(argmax_str, end=end)


def eprint(array, append=""):
    array_as_str ="Δ =  ["
    for idx, a in enumerate(array):
        array_as_str += f"{a:3.0f} "
    array_as_str += " ]"+append
    print(array_as_str, end=None)


def get_argmax_input(delta, step_is_ephemeral, argmaxes):
    eprint(delta, "~~~" if step_is_ephemeral else "<--")
    argprint(argmaxes, len(delta), end="")
    print(end="", flush=True)  # For notebook compatibility.
    
    argmax = None
    while argmax not in argmaxes:
        argmax_str = input("Select index ('q' to quit): ")
        if argmax_str == "q":
            return None
        else:
            argmax = int(argmax_str)
    return argmax
        

def check_for_isomorphism(delta, list_of_deltas):
    """ Check if delta is equivalent (under permutation) to list element """
    num_actions = len(delta)
    twice_delta = 2*list(delta)
    for other_delta in list_of_deltas:
        # TODO: make more efficient
        other_delta = list(other_delta)
        for start_idx in range(num_actions):
            if twice_delta[start_idx:start_idx+num_actions] == other_delta:
                return other_delta
    return None

if __name__ == "__main__": 
    num_actions = 7
    
    # Setting this to True automatically looks ahead through "FP" steps to show 
    # you directly where you can play for AFP. Setting it to False allows you 
    # to manually select the in-between moves that will be "forgotten."
    PLAY_EVEN_STEPS_ONLY = False
    
    seed_argmaxes = []
    delta = np.zeros(num_actions, dtype=int)
    delta_hist = [delta]
    argmax_hist = []
    step = 0
    skip = 1 if PLAY_EVEN_STEPS_ONLY else 2
    
    print("Welcome to the AFP delta game!")
    print("Try to make the max entry as large as possible.")
    print("If the max is bounded, then AFP is O(1/t)!")
    print()
    while True:
        if PLAY_EVEN_STEPS_ONLY:
            argmaxes = get_2step_argmaxes(delta)
        else:
            argmaxes, _ = get_argmaxes(delta)
        
        if step < len(seed_argmaxes):
            argmax = seed_argmaxes[step]
            assert argmax in argmaxes
        else:
            argmax = get_argmax_input(delta, step % skip, argmaxes)
            print()
            if argmax is None:
                break
            
        argmax_hist.append(argmax)    
        delta_to_add_to = delta if step % skip == 0 else delta_hist[-2]
        delta = next_delta(delta_to_add_to, argmax)
        
        equivalent_delta = check_for_isomorphism(delta, delta_hist[(step-1) % skip::skip])
        if equivalent_delta is not None:
            print("Equivalent state visited previously: ")
            eprint(equivalent_delta)            
        
        delta_hist.append(delta)
        step += 1
    
    delta_hist = np.array(delta_hist)
    max_even_idx = np.max(delta_hist[::skip])
    print(f"You played {argmax_hist}.")
    print(f"Your score was {max_even_idx:0.0f}.")