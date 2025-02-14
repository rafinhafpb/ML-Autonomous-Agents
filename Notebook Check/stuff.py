import numpy as np

def gen_traj(env, T=5):
    ''' Generate a path with associated observations.

        Paramaters
        ----------

        T : int
            how long is the path

        Returns
        -------

        o : (T,d)-shape array
            sequence of observations
        s : T-length array of states
            sequence of tiles
    '''
    o = []
    s = []

    s.append(env.step()[0])
    o.append(env.step()[1])

    for i in range(T-1):
        temp_s, temp_o = env.step(s[i])
        s.append(temp_s)
        o.append(temp_o)

    return np.array(o), np.array(s)

