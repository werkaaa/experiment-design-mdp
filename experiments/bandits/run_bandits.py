import numpy as np
import torch

from stpy.continuous_processes.gauss_procc import GaussianProcess
from stpy.kernels import KernelFunction

from mdpexplore.solvers.min import MinSolver
from mdpexplore.mdpexplore import MdpExplore
from mdpexplore.env.bandits import Bandits
from mdpexplore.utils.reward_functionals import DesignSimpleBandits, DesignBandits
from mdpexplore.policies.density_policy import DensityPolicy

if __name__ == "__main__":
    theta_star = np.array([1, 1])
    action_space = np.array([
        [1, 0],
        [0, 1],
        [np.sqrt(0.5), np.sqrt(0.5)],
    ])
    sigma = 1
    delta = 1
    lambd = 1 / 50

    env = Bandits(
        action_space=action_space,
        theta_star=theta_star,
        sigma=sigma)

    kernel = KernelFunction(kernel_name='linear')
    estimator = GaussianProcess(kernel=kernel)

    design = DesignSimpleBandits(
        action_space_size=action_space.shape[0]
    )


    # def callback(trajectory, objective):
    #     action = trajectory[-1]
    #     z = action_space[action]
    #     eps = np.random.normal(0, sigma)
    #     fun_value = z @ theta_star + eps
    #     estimator.add_data_point(torch.tensor(np.expand_dims(z, 0)),
    #                              torch.tensor([[fun_value]]))  # TODO: Q: Do the dims look okay?
    #     estimator.fit()
    #     objective.mu = estimator.ucb(torch.tensor(action_space)).numpy().squeeze()
    #
    #
    # design = DesignBandits(
    #     action_space_size=action_space.shape[0],
    #     lambd=lambd
    # )


    def callback(trajectory, objective):
        action = trajectory[-1]
        z = action_space[action]
        eps = np.random.normal(0, sigma)
        fun_value = z @ theta_star + eps
        estimator.add_data_point(torch.tensor(np.expand_dims(z, 0)),
                                 torch.tensor([[fun_value]]))  # TODO: Q: Do the dims look okay?
        estimator.fit()
        objective.ucbs = estimator.ucb(torch.tensor(action_space)).numpy().squeeze()
        objective.lcbs = estimator.lcb(torch.tensor(action_space)).numpy().squeeze()


    me = MdpExplore(
        env,
        objective=design,
        solver=MinSolver,
        method='frank-wolfe',
        verbosity=True,
        callback=callback)

    val, opt_val = me.run(
        num_components=1,
        episodes=20,
        SummarizedPolicyType=DensityPolicy,
    )
