import numpy as np
rng = np.random.default_rng()
import pickle
from model.arm import Arm, MultimodalArm
from algorithms.vkabc import VKABC, KABC
from drawing.bandit_drawer import draw

D = 50

mean_0 = rng.uniform(low=-100, high=100, size=2)
diff = rng.uniform(low=-100, high=100, size=2)
diff = diff / np.linalg.norm(diff)
diff = diff * D
mean_1 = mean_0 + diff
A_0 = rng.uniform(low=-5, high=5, size=(2, 2))
covariance_0 = np.dot(A_0, A_0.transpose()) # https://stackoverflow.com/questions/619335/
A_1 = rng.uniform(low=-5, high=5, size=(2, 2))
covariance_1 = np.dot(A_1, A_1.transpose()) # https://stackoverflow.com/questions/619335/

# Author: Claude code
def get_multimodal_experiment(mix):
    """
    Returns a model with multimodal arms where some arms are mixtures of two Gaussians.
    
    Args:
        mix: Mixing ratio for the multimodal arms
        D: Distance between the two Gaussian components
    
    Returns:
        arms: List of arms (2 unimodal, 2 multimodal)
        N: Number of arms
        D: Distance parameter
        K: Number of clusters
    """
    arms = []

    arms.append(Arm(mean_0, covariance_0, 0))
    arms.append(Arm(mean_0, covariance_0, 0))
    arms.append(MultimodalArm(mean_1, covariance_1, mean_0, covariance_0, mix, 1))
    arms.append(MultimodalArm(mean_1, covariance_1, mean_0, covariance_0, mix, 1))
    return arms, len(arms), D, 2

# Author: Claude code
def execute():
    arms, N, D, K = get_multimodal_experiment(0.5)
    with open('data/multimodal_experiment_data.p', 'wb') as fp:
        pickle.dump(arms, fp, protocol=pickle.HIGHEST_PROTOCOL)


    sampling_complexities_VKABC = {}
    sampling_complexities_KABC = {}
    taus = {}

    for t in range(10):
        frac = float(t) * 0.1
        arms, N, D, K = get_multimodal_experiment(frac)
        print("----------------------------")
        print(f"Trying it with ratio on the other size: {frac}")
        
        result, sampling_complexity, tau = VKABC(0.5, K, arms)
        result2, sampling_complexity2, _ = KABC(0.5, K, arms)
        print(f"{result}, {sampling_complexity}, {tau}")
        print(f"{result2}, {sampling_complexity2}")
        sampling_complexities_VKABC[frac] = sampling_complexity
        sampling_complexities_KABC[frac] = sampling_complexity2
        taus[frac] = tau

        # Extract and print true clustering
        clusters = [[] for _ in range(K)]
        for i, arm in enumerate(arms):
            cluster = arm.get_cluster()
            clusters[cluster].append(i)
        print(f"{clusters}")

    with open('data/multimodal_experiment_VKABC.p', 'wb') as fp:
        pickle.dump(sampling_complexities_VKABC, fp, protocol=pickle.HIGHEST_PROTOCOL)
    with open('data/multimodal_experiment_KABC.p', 'wb') as fp:
        pickle.dump(sampling_complexities_KABC, fp, protocol=pickle.HIGHEST_PROTOCOL)
    with open('data/multimodal_experiment_taus.p', 'wb') as fp:
        pickle.dump(taus, fp, protocol=pickle.HIGHEST_PROTOCOL)

    return sampling_complexities_VKABC


if __name__ == '__main__':
    execute()
