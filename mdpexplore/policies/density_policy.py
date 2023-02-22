import autograd.numpy as np

from mdpexplore.policies.policy_base import SummarizedPolicy
from mdpexplore.policies.simple_policy import SimplePolicy
from mdpexplore.env.discrete_env import DiscreteEnv


class DensityPolicy(SummarizedPolicy):
    def __init__(self, env: DiscreteEnv, density: np.ndarray, density_sa: np.ndarray) -> None:
        super().__init__(env)
        self.density = density
        self.density_sa = density_sa

        policy = np.zeros(shape = self.density_sa.shape)
        temp = np.tile(self.density.reshape(-1,1), (1,self.density_sa.shape[1]))
        mask = temp > 0
        policy[mask] = self.density_sa[mask] / temp[mask]
        self.policy = SimplePolicy(env, policy)

    def next_action(self, state):
        return self.policy.next_action(state)