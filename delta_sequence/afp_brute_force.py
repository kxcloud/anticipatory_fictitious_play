import itertools

import numpy as np

def get_argmaxes(array):
    max_val = np.max(array)
    return np.argwhere(array == max_val).flatten(), max_val

def next_delta_nowrap(delta, idx):
    next_delta = delta.copy()
    if idx + 1 < num_actions:
        next_delta[(idx + 1) % len(delta)] += 1
    if idx - 1 >= 0:
        next_delta[(idx - 1) % len(delta)] -= 1
    return next_delta

def eprint(array, argmax=None):
    max_val = max(array)    
    print("[", end="")
    for idx, a in enumerate(array):
        end = "!" if idx == argmax else ("." if max_val == a else " ")
        print(f"{a:3.0f}", end=end)
    print(" ]")

# num_actions = 5
# upper_bound = 3
# count = 0

# print(f"num values: {upper_bound**num_actions}")
# for delta in itertools.product(range(upper_bound), repeat=num_actions):
#     delta = np.array(delta)
#     found_one = False
#     for idx in get_argmaxes(delta)[0]:
#         if not found_one:
#             temp_delta = next_delta_nowrap(delta, idx)
#             for next_idx in get_argmaxes(temp_delta)[0]:
#                 result = next_delta_nowrap(delta, next_idx)
#                 if upper_bound in result:
#                     eprint(delta, idx)
#                     eprint(temp_delta, next_idx)
#                     eprint(result)
#                     print()
#                     count += 1
#                     found_one = True
#                     break
#         else:
#             break
# print(count)


def next_delta(delta, idx):
    next_delta = delta.copy()
    next_delta[(idx + 1) % len(delta)] += 1
    next_delta[(idx - 1) % len(delta)] -= 1
    return next_delta

num_actions = 5
upper_bound = 3
count = 0

print(f"num values: {upper_bound**num_actions}")
for delta in itertools.product(range(-3*upper_bound-1, upper_bound), repeat=num_actions):
    delta = np.array(delta)
    if np.abs(np.sum(delta) - 0) < 0.01:
        found_one = False
        for idx in get_argmaxes(delta)[0]:
            if not found_one:
                temp_delta = next_delta(delta, idx)
                for next_idx in get_argmaxes(temp_delta)[0]:
                    result = next_delta(delta, next_idx)
                    if upper_bound in result:
                        eprint(delta, idx)
                        eprint(temp_delta, next_idx)
                        eprint(result)
                        print()
                        count += 1
                        found_one = True
                        break
            else:
                break