import numpy as np
import scipy

from bandit_machines import BernoulliMAB, GaussianMAB

def randmax(a):
    """ return a random maximum """
    a = np.array(a)  
    max_indices = np.flatnonzero(a == a.max())
    return np.random.choice(max_indices)

def evaluate_run(env, pi, T=1000):
    '''
    Run a bandit agent on a bandit instance (environment) for T steps.
    '''
    r_log = []
    a_log = []
    pi.clear()
    for t in range(T):
        a = pi.act()
        r = env.rwd(a)
        pi.update(a,r)
        r_log.append(r)
        a_log.append(a)
    return a_log, r_log

def evaluate_runs(env, pi, T=1000, N=10):
    '''
        Parameters
        ----------

        env : 
            bandit machine
        pi : 
            bandit algorithm
        N : int
            number of experiments
        T : int
            number of trails per experiment
    '''
    R_log = np.zeros((N,T))
    A_log = np.zeros((N,T),dtype=int)
    for n in range(N):
        np.random.seed()
        A_log[n], R_log[n] = evaluate_run(env, pi, T)

    return A_log, R_log

class UCB():

    """UCB1 with parameter alpha"""

    def __init__(self,n_arms,alpha=1/2):
        self.n_arms = n_arms
        self.alpha = alpha
        self.clear()

    def clear(self):
        self.nbDraws = np.zeros(self.n_arms)
        self.cumRewards = np.zeros(self.n_arms)
        self.t = 0

    def act(self):
        ''' balance exploration and exploitation by encourages the agent to pick arms with high average rewards, but ensures that arms with fewer pulls are favored '''
        self.t += 1
        if (min(self.nbDraws)==0):
            return randmax(-self.nbDraws)
        else:
            return randmax(self.cumRewards/self.nbDraws + np.sqrt(self.alpha*np.log(self.t)/self.nbDraws))
    
    def update(self,a,r):
        self.cumRewards[a] = self.cumRewards[a] + r
        self.nbDraws[a] = self.nbDraws[a] + 1

    def name(self):
        return "UCB(%3.2f)" % self.alpha

