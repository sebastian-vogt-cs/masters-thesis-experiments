import numpy as np
rng = np.random.default_rng()

class Arm:
    def __init__(self, mean, covariance, cluster):
        self.mean = mean
        self.covariance = covariance
        self.cluster = cluster

    def sample(self, size):
        return rng.multivariate_normal(self.mean, self.covariance, size=size, check_valid='raise')

    def get_cluster(self):
        return self.cluster

class MultimodalArm(Arm):
    def __init__(self, mean, covariance, mean2, covariance2, mix2, cluster):
        super().__init__(mean, covariance, cluster)
        self.mean2 = mean2
        self.mix2 = mix2
        self.covariance2 = covariance2

    def sample(self, size):
        samples = []
        for _ in range(size):
            random = rng.uniform()
            if (random < self.mix2):
                samples.append(rng.multivariate_normal(self.mean2, self.covariance2, check_valid='raise'))
            else:
                samples.append(rng.multivariate_normal(self.mean, self.covariance, check_valid='raise'))
        return np.array(samples)