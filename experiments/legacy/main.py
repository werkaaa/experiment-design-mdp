from mdpexplore.maxent import MaxEnt
from mdpexplore.env.env_builder import EnvBuilder
from mdpexplore.utils.argparser import Parser

import wandb

def main():
    config = Parser().parse()
    wandb.init(project="max-ent", entity="tjanik")
    wandb.config.update(config)

    env = EnvBuilder(config).build()

    me = MaxEnt(
        env,
        objective=config.objective_fn,
        solver=config.solver,
        step=config.step,
        method=config.method,
        verbosity=config.verbosity
    )
    me.run(
        num_components=config.num_components,
        num_unrolls=config.num_unrolls,
        num_runs=config.num_runs,
        SummarizedPolicyType=config.policy,
        accuracy=config.accuracy,
        plot=config.plot,
    )


if __name__ == "__main__":
    main()
