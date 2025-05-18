import numpy as np
import random
from collections import defaultdict

def get_min_beneficial_expert_on_gpu(i):
    residents = gpu_assignments[i]
    min_e = min(residents, key=lambda e: B[e][i])
    return min_e, B[min_e][i]

# ==== CONFIG ====
# M = 3   # number of GPUs in node...
# K = 3   # max number of experts per GPU
# N = M * K   # Total number of experts

def relocate_algo(total_num_gpus, num_experts_per_gpu):
    total_num_experts = total_num_gpus * num_experts_per_gpu
    cost = 
    benefit_matrix = 

    gpu_assignments = defaultdict(list)
    expert_assignment = [-1] * N



# TODO(MASAAD): Double check this assumption about NVLink:
# (On NVLink - Fixed cost to move from GPU A to B)
# Not necessarily across nodes...
C = 5

# random.seed(42)
# np.random.seed(42)

# # ==== BENEFIT MATRIX ====
# B = np.random.randint(10, 100, size=(N, M))  # B[n][i] = BENEFIT if Expert n goes to GPU g

# ==== INITIAL ASSIGNMENT ====


# Random initial assignment (respecting K cap)
available_slots = list(range(M)) * K
random.shuffle(available_slots)

for n in range(N):
    i = available_slots.pop()
    gpu_assignments[i].append(n)
    expert_assignment[n] = i
initial_assignment = expert_assignment[:]

# ==== UTILITY ====


# ==== OPTIMIZATION LOOP ====
MAX_ITERS = 50
for _ in range(MAX_ITERS):
    best_gain = 0
    best_move = None
    for n in range(N):
        i_from = expert_assignment[n]
        for i_to in range(M):
            if i_to == i_from:
                continue
            if len(gpu_assignments[i_to]) < K:
                gain = B[n][i_to] - B[n][i_from] - C
                if gain > best_gain:
                    best_gain = gain
                    best_move = (n, None, i_from, i_to)
            else:
                p_out, r_out = get_min_beneficial_expert_on_gpu(i_to)
                if B[n][i_to] > r_out:
                    gain = (B[n][i_to] - r_out) - C
                    if gain > best_gain:
                        best_gain = gain
                        best_move = (n, p_out, i_from, i_to)
    if not best_move:
        break

    n, p_out, i_from, i_to = best_move
    gpu_assignments[i_from].remove(n)
    gpu_assignments[i_to].append(n)
    expert_assignment[n] = i_to

    if p_out is not None:
        gpu_assignments[i_to].remove(p_out)
        gpu_assignments[i_from].append(p_out)
        expert_assignment[p_out] = i_from

# ==== OUTPUT ====
print("\nFinal Assignments (expert → gpu):")
for n in range(N):
    print(f"Expert {n}: GPU {expert_assignment[n]} | Benefit: {B[n][expert_assignment[n]]}")

net_revenue = sum(B[n][expert_assignment[n]] for n in range(N))
move_costs = sum(C for n in range(N) if expert_assignment[n] != initial_assignment[n])
total_net = net_revenue - move_costs

print("\nInitial Assignments:", initial_assignment)
print("Final Assignments:  ", expert_assignment)
print("Net Revenue:        ", net_revenue)
print("Movement Cost:      ", move_costs)
print("Total Net Value:    ", total_net)
