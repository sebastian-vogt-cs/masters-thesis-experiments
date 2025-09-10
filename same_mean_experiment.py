import numpy as np
rng = np.random.default_rng()
import pickle
from model.arm import Arm
from algorithms.vkabc import VKABC, KABC

def get_same_mean_experiment(V):
    """Generates a model with two distributions/clusters and two arms per cluster. The two clusters have the same mean 
    but one has unit covariance, the other has a covariance with V on the diagonal.

    Args:
        V (int): The number on the diagonal of the covariance matrix of the second distribution.

    Returns:
        (list[Arm], int): The model as a list of arms and the number of clusters K.
    """

    # The model is a list of arms
    arms = []

    # The mean of both distributions is the origin
    mean = np.array([0, 0])

    # The first distribution has unit covariance
    covariance_0 = np.array([[1, 0], [0, 1]])

    # The second covariance has V on the diagonal
    covariance_1 = V * covariance_0

    # Add all arms to the list
    arms.append(Arm(mean, covariance_0, 0))
    arms.append(Arm(mean, covariance_0, 0))
    arms.append(Arm(mean, covariance_1, 1))
    arms.append(Arm(mean, covariance_1, 1))

    return arms, 2

def execute():
    """Executes the same mean experiment for different variances of the second distribution.
    """

    sampling_complexities_VKABC = {}
    sampling_complexities_KABC = {}
    taus = {}

    for V in [200, 400, 800, 1600, 3200, 6400]:
        arms, K = get_same_mean_experiment(V)
        print("----------------------------")
        print(f"Trying it with covariance of: {V} * id")

        result, sampling_complexity, tau = VKABC(0.5, K, arms)
        result2, sampling_complexity2, _ = KABC(0.5, K, arms)
        print(f"{result}, {sampling_complexity}, {tau}")
        print(f"{result2}, {sampling_complexity2}")
        sampling_complexities_VKABC[V] = sampling_complexity
        sampling_complexities_KABC[V] = sampling_complexity2
        taus[V] = tau

        # Extract and print true clustering
        clusters = [[] for _ in range(K)]
        for i, arm in enumerate(arms):
            cluster = arm.get_cluster()
            clusters[cluster].append(i)
        print(f"{clusters}")

    with open('data/same_mean_experiment_VKABC.p', 'wb') as fp:
        pickle.dump(sampling_complexities_VKABC, fp, protocol=pickle.HIGHEST_PROTOCOL)
    with open('data/same_mean_experiment_KABC.p', 'wb') as fp:
        pickle.dump(sampling_complexities_KABC, fp, protocol=pickle.HIGHEST_PROTOCOL)
    with open('data/same_mean_experiment_taus.p', 'wb') as fp:
        pickle.dump(taus, fp, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    execute()
