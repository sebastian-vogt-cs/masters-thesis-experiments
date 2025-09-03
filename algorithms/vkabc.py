import math
import numpy as np
from multiprocessing import Pool, cpu_count

processes = cpu_count()
print(f"using {processes} processes")

def _g(x, y):
    return np.exp(-(np.dot(y - x, y - x)) / 5000)

def _sample(n, arms):
    data = []
    for arm in arms:
        s = arm.sample(n)
        data.append(s)
    return data

# Author: Claude code
def _calculate_single_arm_variance(arm_data):
    """Calculate empirical variance for a single arm's data."""
    n = len(arm_data)
    v_hat = 0.0
    for t in range(n):
        v_hat += _g(arm_data[t], arm_data[t])
        nested_sum = 0.0
        for s in range(n):
            nested_sum += _g(arm_data[t], arm_data[s])
        nested_sum /= n
        v_hat -= nested_sum 
    v_hat /= n - 1
    return v_hat

def _calculate_empirical_variances(data):
    with Pool(processes=processes) as pool:
        vars = pool.map(_calculate_single_arm_variance, data)
    return vars

# Author: Claude code
def _calculate_single_distance(args):
    """Calculate distance between a pair of arms."""
    i, j, arm_i, arm_j = args
    n = len(arm_i)
    assert len(arm_j) == n
    d_squared = 0.0
    for k in range(n):
        for l in range(n):
            d_squared += _g(arm_i[k], arm_i[l]) - 2 * _g(arm_i[k], arm_j[l]) + _g(arm_j[k], arm_j[l])
    d_squared /= n * n
    return i, j, math.sqrt(d_squared)

# Author: Claude code
def _calculate_distances(data):
    distances = np.zeros((len(data), len(data)))
    
    # Prepare arguments for parallel processing
    args_list = []
    for i in range(len(data)):
        for j in range(i):
            args_list.append((i, j, data[i], data[j]))
    
    # Calculate distances in parallel
    with Pool(processes=processes) as pool:
        results = pool.map(_calculate_single_distance, args_list)
    
    # Fill the distance matrix
    for i, j, distance in results:
        distances[i][j] = distance
        distances[j][i] = distance
    
    return distances

# Author: Claude code
def _calculate_variances_and_distances(data):
    """Calculate both variances and distances in parallel using a single process pool."""
    n_arms = len(data)
    
    # Prepare all tasks: variances and distance pairs
    all_tasks = []
    
    # Add variance tasks
    for i, arm_data in enumerate(data):
        all_tasks.append(('variance', i, arm_data))
    
    # Add distance tasks
    for i in range(n_arms):
        for j in range(i):
            all_tasks.append(('distance', (i, j), (data[i], data[j])))
    
    # Process all tasks in parallel
    with Pool(processes=processes) as pool:
        results = pool.map(_process_mixed_task, all_tasks)
    
    # Separate results
    variances = [None] * n_arms
    distances = np.zeros((n_arms, n_arms))
    
    for result in results:
        if result[0] == 'variance':
            _, arm_idx, variance = result
            variances[arm_idx] = variance
        else:  # distance
            _, (i, j), distance = result
            distances[i][j] = distance
            distances[j][i] = distance
    
    return variances, distances

# Author: Claude code
def _process_mixed_task(task):
    """Process either a variance or distance calculation task."""
    task_type, index, data = task
    
    if task_type == 'variance':
        arm_data = data
        variance = _calculate_single_arm_variance(arm_data)
        return ('variance', index, variance)
    else:  # distance
        i, j = index
        arm_i, arm_j = data
        n = len(arm_i)
        assert len(arm_j) == n
        d_squared = 0.0
        for k in range(n):
            for l in range(n):
                d_squared += _g(arm_i[k], arm_i[l]) - 2 * _g(arm_i[k], arm_j[l]) + _g(arm_j[k], arm_j[l])
        d_squared /= n * n
        distance = math.sqrt(d_squared)
        return ('distance', (i, j), distance)

def _get_connected_components(adjacency):
    # https://www.geeksforgeeks.org/dsa/connected-components-in-an-undirected-graph/
    n = len(adjacency)

    def dfs(node, adj_list, visited, component):
        visited[node] = True
        component.append(node)
        for neighbor in adj_list[node]:
            if not visited[neighbor]:
                dfs(neighbor, adj_list, visited, component)

    adj_list = [[] for _ in range(n)]
    for i in range(n):
        m = len(adjacency[i])
        assert n == m
        for j in range(i):
            if adjacency[i][j]:
                adj_list[i].append(j)
                adj_list[j].append(i)

    visited = [False for _ in range(n)]
    res = []

    for i in range(n):
        if not visited[i]:
            component = []
            dfs(i, adj_list, visited, component)
            res.append(component)

    return res


def _calculate_tau(arms, delta, vars, dists):
    N = len(arms)
    log_term = math.log((32 * (N*N - N))/delta)
    distances_different_clusters = []
    for (i, arm_i) in enumerate(arms):
        for (j, arm_j) in enumerate(arms):
            if not arm_i.cluster == arm_j.cluster:
                d = dists[i][j]
                distances_different_clusters.append(d)
    Delta_min = min(distances_different_clusters)
    V_max = max(vars)
    frac_1 = (128 * V_max) / (Delta_min * Delta_min)
    frac_2 = (112 + 16) / (3 * Delta_min)
    max_term = max([frac_1,frac_2])
    ceil_term = math.ceil(math.log2(max_term))
    return 8 * N * ((2 * math.log(ceil_term)) + log_term) * max_term

def _VKABC_CLUSTER(k, delta, arms):
    N = len(arms)
    # First, we need to calculate the sample size
    log_term = 2 * math.log(k) + math.log((32 * (N*N - N))/delta)
    nk = math.ceil(2**k * log_term)
    delta_k = delta / (4 * (k * k))
    data = _sample(nk, arms)
    varis, dists = _calculate_variances_and_distances(data)
    tau = _calculate_tau(arms, delta, varis, dists)
    incidence = np.zeros((N, N))

    # print(f"sampling {nk} values")

    psi_tilde = 1

    bound_log = math.log((8 * (N * N - N)) / delta_k) / nk
    bound_constant_part = (32/3) * math.sqrt(psi_tilde) * bound_log

    print(f"sampled {nk} times per arm")
    for i in range(N):
        for j in range(i):
            bound = bound_constant_part + ((math.sqrt(varis[i]) + math.sqrt(varis[j])) * (math.sqrt(2 * bound_log)))

            d = dists[i][j]
            print(f"comparing arm {i} and {j}: {d} <= {bound}")
            if dists[i][j] <= bound:
                incidence[i][j] = 1
                incidence[j][i] = 1

    return _get_connected_components(incidence), N * nk, tau

def _KABC_CLUSTER(k, delta, arms):
    N = len(arms)
    # First, we need to calculate the sample size
    log_term = 2 * math.log(k) + math.log((8 * (N*N - N))/delta)
    nk = math.ceil(2**k * log_term)
    delta_k = delta / (4 * (k * k))
    data = _sample(nk, arms)
    dists = _calculate_distances(data)
    incidence = np.zeros((N, N))

    g_bar = 1

    bound = (2 * math.sqrt(g_bar/nk)) + (2 * math.sqrt((2 * g_bar * math.log((2 * (N*N - N))/delta_k))/nk))

    print(f"sampled {nk} times per arm")
    for i in range(N):
        for j in range(i):
            d = dists[i][j]
            print(f"comparing arm {i} and {j}: {d} <= {bound}")
            if d <= bound:
                incidence[i][j] = 1
                incidence[j][i] = 1

    return _get_connected_components(incidence), N * nk, -1


def _adaptive(delta, K, arms, CLUSTER):
    k = 2
    sampling_complexity = 0
    while True:
        # print(f"iteration {k}")
        clusters, samples_drawn, tau = CLUSTER(k, delta, arms)
        sampling_complexity += samples_drawn
        if len(clusters) >= K:
            return clusters, sampling_complexity, tau
        k += 1

def VKABC(delta, K, arms):
    return _adaptive(delta, K, arms, _VKABC_CLUSTER)

def KABC(delta, K, arms):
    return _adaptive(delta, K, arms, _KABC_CLUSTER)