import matplotlib.pyplot as plt
import numpy as np
import os
import wandb

from utils.argparser import Parser
from utils.reward_functionals import design_B, design_D
from env.env_builder import EnvBuilder
from maxent import MaxEnt
from policies.average_policy import AveragePolicy
from policies.density_policy import DensityPolicy
from policies.mixture_policy import MixturePolicy

def main():
    config = Parser().parse()
    #wandb.init(project="max-ent", entity="mojusko")
    #wandb.config.update(config)

    if config.num_runs < 2:
        print('Number of runs must be at least 2')
        exit(1)

    env = EnvBuilder(config).build()

    me = MaxEnt(
        env,
        objective=design_D,
        solver=config.solver,
        step=config.step,
        method=config.method,
        verbosity=config.verbosity
    )

    for policy in [MixturePolicy, DensityPolicy, AveragePolicy]:

        means, stds, opt = me.run(
            num_components=config.num_components,
            num_unrolls=config.num_unrolls,
            num_runs=config.num_runs,
            SummarizedPolicyType=policy,
            accuracy=config.accuracy,
            plot=False,
        )

        if policy == MixturePolicy:
            mixed_means = np.array(means)
            mixed_std = np.array(stds)

        elif policy == DensityPolicy:
            density_means = np.array(means)
            density_std = np.array(stds)

        elif policy == AveragePolicy:
            average_means = np.array(means)
            average_std = np.array(stds)
            
    
    me = MaxEnt(
        env,
        objective=design_B,
        solver=config.solver,
        step=config.step,
        method=config.method,
        verbosity=config.verbosity
    )

    bayesian_means, bayesian_stds, _ = me.run(
        num_components=config.num_components,
        num_unrolls=config.num_unrolls,
        num_runs=config.num_runs,
        SummarizedPolicyType=DensityPolicy,
        accuracy=config.accuracy,
        plot=False,
    )

    me = MaxEnt(
        env,
        objective=design_B,
        solver=config.solver,
        step=config.step,
        method=config.method,
        verbosity=config.verbosity
    )
    alt_bayesian_means, alt_bayesian_stds, _ = me.run(
        num_components=1,
        num_unrolls=config.num_unrolls,
        num_runs=config.num_runs,
        SummarizedPolicyType=DensityPolicy,
        accuracy=config.accuracy,
        plot=False,
    )

    me = MaxEnt(
        env,
        objective=design_D,
        solver=config.solver,
        step=config.step,
        method=config.method,
        verbosity=config.verbosity
    )
    random_means, random_stds, _ = me.run(
        num_components=0,
        num_unrolls=config.num_unrolls,
        num_runs=config.num_runs,
        SummarizedPolicyType=MixturePolicy,
        accuracy=config.accuracy,
        plot=False,
    )

    np.save('average_means.npy', average_means)
    np.save('average_std.npy', average_std)
    np.save('mixed_means.npy', mixed_means)
    np.save('mixed_std.npy', mixed_std)
    np.save('density_means.npy', density_means)
    np.save('density_std.npy', density_std)
    np.save('bayesian_means.npy', bayesian_means)
    np.save('bayesian_std.npy', bayesian_stds)
    np.save('random_means.npy', random_means)
    np.save('random_std.npy', random_stds)
    np.save('opt.npy', opt)

    plt.figure(figsize=(10, 10))
    x_axis = 1 + np.arange(len(mixed_means))
    plt.plot(x_axis, average_means, label='Average', color='red')
    plt.fill_between(x_axis, average_means - average_std, average_means + average_std, alpha=0.2, color='red')

    plt.plot(x_axis, mixed_means, label='Mixed', color='blue')
    plt.fill_between(x_axis, mixed_means - mixed_std, mixed_means + mixed_std, alpha=0.2, color='blue')

    plt.plot(x_axis, density_means, label='Density', color='green')
    plt.fill_between(x_axis, density_means - density_std, density_means + density_std, alpha=0.2, color='green')

    plt.plot(x_axis, bayesian_means, label='Bayesian', color='yellow')
    plt.fill_between(x_axis, bayesian_means - bayesian_stds, bayesian_means + bayesian_stds, alpha=0.2, color='yellow')

    plt.plot(x_axis, alt_bayesian_means, label='1-Bayesian', color='orange')
    plt.fill_between(x_axis, alt_bayesian_means - alt_bayesian_stds, alt_bayesian_means + alt_bayesian_stds, alpha=0.2, color='orange')

    plt.plot(x_axis, random_means, label='Random', color='black')
    plt.fill_between(x_axis, random_means - random_stds, random_means + random_stds, alpha=0.2, color='black')

    plt.axhline(y=opt, color='magenta', label='Expert')
    plt.legend(loc='lower right')

    plt.grid()
    plt.xlabel('Number of trajectories rolled out')
    plt.ylabel('Objective function value')

    plt.savefig(os.path.join('fig', 'objectives.png'))
    wandb.log({'objectives': wandb.Image(plt)})


if __name__ == "__main__":
    main()