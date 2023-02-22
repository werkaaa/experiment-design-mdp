import autograd.numpy as np
import autograd.numpy.linalg as la
from typing import List, Union
from abc import ABC, abstractmethod
from mdpexplore.env.discrete_env import Environment

class RewardFunctional(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def eval(self,
             emissions: np.ndarray,
            distribution: np.ndarray,
             episodes: int):
        pass


    def get_type(self):
        return self.type


class DesignA(RewardFunctional):

    def __init__(self,
        env: Environment,
        lambd: float = 1e-3):
        super().__init__()
        self.lambd = lambd
        self.type = "static"
        self.env = env

    def eval(self,
             emissions: np.ndarray,
            distribution: np.ndarray,
             episodes: int = 0
             ) -> float:
        z = emissions.T @ np.diag(distribution) @ emissions
        return -np.trace(la.inv(z + self.lambd*np.eye(z.shape[0])))

class DesignD(RewardFunctional):
    def __init__(self,
                 env: Environment,
                 lambd: float = 1e-3,
                 scale_reg = True,
                 sigma: float = 1. ):
        super().__init__()
        self.env = env
        self.dim = self.env.get_dim()
        self.lambd = lambd
        self.scale_reg = scale_reg
        self.type = "static"
        self.lambd = lambd * np.eye(self.dim)

        if isinstance(lambd, float):
            self.Sigma = sigma * np.ones(self.env.get_states_num())
            self.Sigma_true = self.Sigma
        else:
            self.Sigma = sigma
            self.Sigma_true = self.Sigma

    def eval(self,
        emissions: np.ndarray,
        distribution: np.ndarray,
        episodes: int
        ) -> float:
        z = emissions.T @ np.diag(distribution/(self.Sigma**2)) @ emissions
        return np.linalg.slogdet(z + self.lambd/episodes)[1]

    def eval_full(self,
        emissions: np.ndarray,
        distribution: np.ndarray,
        episodes: int,
        ) -> float:
        z = emissions.T @ np.diag(distribution/(self.Sigma_true**2)) @ emissions
        return np.linalg.slogdet(z + self.lambd/episodes)[1]

class DesignC(RewardFunctional):

    def __init__(self,
                 env:Environment,
                 lambd: float = 1e-3,
                 sigma: float = 1.,
                 C : Union[np.array,List,None] = None):
        super().__init__()

        self.env = env
        self.dim = self.env.get_dim()
        self.lambd = lambd * np.eye(self.dim)
        self.sigma = sigma

        self.C = C
        self.type = "static"

    def eval(self,
        emissions: np.ndarray,
        distribution: np.ndarray,
        episodes: int = 0) -> float:

        z = np.multiply(emissions.T, distribution/(self.sigma**2)) @ emissions
        if isinstance(self.C, list):
            return np.max([np.trace(la.inv(C @ la.inv(z + self.lambd / episodes) @ C.T)) for C in self.C])
        else:
            return np.trace(la.inv(self.C @ la.inv(z + self.lambd/episodes)@ self.C.T))

    def eval_full(self,
        emissions: np.ndarray,
        distribution: np.ndarray,
        episodes: int = 0) -> float:
        return self.eval(emissions,distribution,episodes)

class DesignBayesD(RewardFunctional):

    def __init__(self,
                 env:Environment,
                 lambd: float =1e-3,
                 scale_reg: bool = True,
                 uniform_alpha: bool = False,
                 sigma: float = 1.0):

        super().__init__()

        self.dim = env.get_dim()
        self.scale_reg = scale_reg
        self.uniform_alpha = uniform_alpha
        self.env = env
        self.lambd = lambd * np.eye(self.dim)

        if isinstance(lambd, float):
            self.Sigma = sigma * np.ones(self.env.get_states_num())
            self.Sigma_true = self.Sigma
        else:
            self.Sigma = sigma
            self.Sigma_true = self.Sigma

        self.type = "adaptive"

    def eval_basic(self,
        emissions: np.ndarray,
        distribution: np.ndarray,
        unrolls: List[np.ndarray],
        episodes: int,
        ) -> float:
        """

        """
        alpha = len(unrolls)/episodes
        aggregated_unrolls = sum(unrolls)/len(unrolls) if len(unrolls) > 0 else np.zeros(emissions.shape[0])

        new_z = np.multiply(emissions.T, distribution/(self.Sigma**2)) @ emissions
        agg_z =  np.multiply(emissions.T, aggregated_unrolls/(self.Sigma**2)) @ emissions

        #new_z = emissions.T @ np.diag(distribution/(self.Sigma**2)) @ emissions
        #agg_z = emissions.T @ np.diag(aggregated_unrolls/(self.Sigma**2)) @ emissions


        if self.uniform_alpha:
            z = 1./episodes * new_z + \
                alpha * agg_z
        else:
            z = (1 - alpha) * new_z + \
                alpha * agg_z
        return z

    def eval(self,
        emissions: np.ndarray,
        distribution: np.ndarray,
        unrolls: List[np.ndarray],
        episodes: int,
        ) -> float:
        alpha = len(unrolls)/episodes

        z = self.eval_basic(emissions,distribution,unrolls,episodes)

        if not self.scale_reg:
            return np.linalg.slogdet(z + (1-alpha) * self.lambd)[1]
        else:
            return np.linalg.slogdet(z +  self.lambd/episodes)[1]

    def eval_full(self,
        emissions: np.ndarray,
        distribution: np.ndarray,
        episodes: int,
        ) -> float:

        z = emissions.T @ np.diag(distribution/(self.Sigma_true**2)) @ emissions

        if not self.scale_reg:
            return np.linalg.slogdet(z + self.lambd)[1]
        else:
            return np.linalg.slogdet(z + self.lambd/episodes)[1]

class DesignBayesC(DesignBayesD):
    def __init__(self, env: Environment, lambd: float = 1e-3, scale_reg: bool = False, sigma: float = 1.0, C=None):
        super().__init__(env,lambd = lambd, scale_reg=scale_reg, sigma=sigma)
        self.C = C

    def eval(self,
        emissions: np.ndarray,
        distribution: np.ndarray,
        unrolls: List[np.ndarray],
        episodes: int,
        ) -> float:
        z = self.eval_basic(emissions,distribution,unrolls,episodes)
        if isinstance(self.C,list):
            return np.max([np.trace(la.inv(C @ la.inv(z + (1. / episodes) * self.lambd) @ C.T)) for C in self.C])
        else:
            return np.trace(la.inv(self.C@la.inv(z + (1./episodes) * self.lambd)@self.C.T))

    def eval_full(self,
        emissions: np.ndarray,
        distribution: np.ndarray,
        episodes: int,
        ) -> float:
        z = np.multiply(emissions.T, distribution/(self.Sigma_true**2)) @ emissions
        if isinstance(self.C,list):
            return np.max([np.trace(la.inv(C @ la.inv(z + (1. / episodes) * self.lambd) @ C.T)) for C in self.C])
        else:
            return np.trace(la.inv(self.C@la.inv(z + (1./episodes) * self.lambd)@self.C.T))





class DesignBayesA(RewardFunctional):
    def eval(self,
        emissions: np.ndarray,
        distribution: np.ndarray,
        unrolls: List[np.ndarray],
        episodes: int,
        ) -> float:
        alpha = len(unrolls)/episodes
        z = self.eval_basic(emissions,distribution,unrolls,episodes)
        if not self.scale_reg:
            return -np.trace(la.inv(z + (1-alpha) * self.lambd))
        else:
            return -np.trace(la.inv(z + (1./episodes) * self.lambd))


