import numpy as np
import matplotlib.pyplot as plt

class Agent: 

    def __init__(self, env): 
        '''
            env : Environment 
                of the type provided to you
        '''
        self.env = env

    def s_tree(self, T):
        '''
            Provides all possible paths for T iterations.

            Parameters
            ----------

            T : int
                number of iterations

            Returns
            -------

            all_paths : list of lists
                all possible paths of states for the given number of iterations, with size T
        '''

        all_paths = []

        for t in range(T):
            if t == 0:
                first_s = [i[-1] for i in np.argwhere(self.env.P_1)]
                all_paths.extend([[s] for s in first_s])

            elif t == 1:
                current_s = []
                for s in first_s:
                    for j in range(len(np.argwhere(self.env.P_S[s]))):
                        new = [s]
                        new.append(np.argwhere(self.env.P_S[s])[j, -1])
                        current_s.append(new)
                last_s = current_s
                all_paths.extend(last_s)

            else:
                current_s = []
                for s in last_s:
                    for j in range(len(np.argwhere(self.env.P_S[s[-1]]))):
                        new = s.copy()
                        new.append(np.argwhere(self.env.P_S[s[-1]])[j, -1])
                        current_s.append(new)
                last_s = current_s
                all_paths.extend(last_s)

        all_paths = [s for s in all_paths if len(s) == T]

        return all_paths

    def P_traj(self, ooo, M=-1):
        '''
        Provides full conditional distribution P(SSS | ooo) where SSS and ooo are sequences of length T.
        $$
            P( Y_1,\ldots,Y_T | o_1,\ldots,o_T )
        $$

        Parameters
        ----------

        ooo : array_like(t, d)
            t observations (of d dimensions each)

        M : int
            -1 indicates to use a brute force solution (exact recovery of the distribution) 
            M > 0 indicates to use M Monte Carlo simulations (this parameter is used in Week 2)


        Returns
        -------

        p : dict(str:float)
            such that p[sss] = P(sss | ooo)
            and if sss not in p, it implies P(sss | ooo) = 0

            important: let sss be a string representation of the state sequence, separated by spaces, e.g., 
            the string representation of np.array([1,2,3,4],dtype=int) should be '1 2 3 4'. 
        '''        
        prob = {}
        env = self.env
        
        if M == -1:
            possible_paths = []

            s_rustle = np.array([env.P_O[s, 0, 1] for s in range(env.n_states)]) != 0
            s_rustle = [i for i, s in enumerate(s_rustle) if s]
            p_rustle = len(s_rustle) / env.n_states

            s_crinkle = np.array([env.P_O[s, 1, 1] for s in range(env.n_states)]) != 0
            s_crinkle = [i for i, s in enumerate(s_crinkle) if s]
            p_crinkle = len(s_crinkle) / env.n_states

            s_both = [s for s in s_rustle if s in s_crinkle]
            p_both = len(s_both) / env.n_states

            s_none = [s for s in range(env.n_states) if s not in s_rustle and s not in s_crinkle]
            p_none = len(s_none) / env.n_states

            for i, o in enumerate(ooo):
                all_paths = self.s_tree(i+1)

                if i == 0:
                    for s in all_paths:
                        if len(s) == 1:
                            possible_paths.append(s)
                            prob[str(np.array(s))[1:-1]] = round(float(env.P_1[s]), 5)
                else:
                    if o[0] == 1 and o[1] == 1:
                        for s in all_paths:
                            if s[:-1] in possible_paths and s[-1] in s_both:
                                possible_paths.append(s)
                                prob[str(np.array(s))[1:-1]] = round(prob[str(np.array(s[:-1]))[1:-1]] * p_both, 5)

                    elif o[0] == 1 and o[1] == 0:
                        for s in all_paths:
                            if s[:-1] in possible_paths and s[-1] in s_rustle:
                                possible_paths.append(s)
                                if s[-1] in s_both:
                                    prob[str(np.array(s))[1:-1]] = round(prob[str(np.array(s[:-1]))[1:-1]] * p_rustle * (1-p_crinkle), 5)
                                else:
                                    prob[str(np.array(s))[1:-1]] = round(prob[str(np.array(s[:-1]))[1:-1]] * p_rustle, 5)

                    elif o[0] == 0 and o[1] == 1:
                        for s in all_paths:
                            if s[:-1] in possible_paths and s[-1] in s_crinkle:
                                possible_paths.append(s)
                                if s[-1] in s_both:
                                    prob[str(np.array(s))[1:-1]] = round(prob[str(np.array(s[:-1]))[1:-1]] * p_crinkle * (1-p_rustle), 5)
                                else:
                                    prob[str(np.array(s))[1:-1]] = round(prob[str(np.array(s[:-1]))[1:-1]] * p_crinkle, 5)
                    else:
                        for s in all_paths:
                            if s[:-1] in possible_paths:
                                possible_paths.append(s)
                                if s[-1] in s_both:
                                    prob[str(np.array(s))[1:-1]] = round(prob[str(np.array(s[:-1]))[1:-1]] * p_both, 5)
                                elif s[-1] in s_rustle:
                                    prob[str(np.array(s))[1:-1]] = round(prob[str(np.array(s[:-1]))[1:-1]] * p_rustle, 5)
                                elif s[-1] in s_crinkle:
                                    prob[str(np.array(s))[1:-1]] = round(prob[str(np.array(s[:-1]))[1:-1]] * p_crinkle, 5)
                                else:
                                    prob[str(np.array(s))[1:-1]] = round(prob[str(np.array(s[:-1]))[1:-1]] * p_none, 5)

            # Filter the paths and probabilities added in the last loop
            last_possible_paths = [s for s in possible_paths if len(s) == len(possible_paths[-1])]
            p = {k: v for k, v in prob.items() if len(k.split()) == len(last_possible_paths[0])}
            
            # Fill the missing paths with probability 0
            for path in all_paths:
                if str(np.array(path))[1:-1] not in p.keys():
                    p[str(np.array(path))[1:-1]] = 0
                            
        return p

        
    def P_S(self, ooo, t=-1): 
        '''
        Provide P(s_t | ooo) given observations o from 1,...,T.  

        $$
            P(S_t | o_1,...,o_T ).
        $$
        
        The probability (distribution) of the t-th state, given the observed evidence 'o'.

        Parameters
        ----------

        ooo : array_like(t, d)
            t observations (of d dimensions each)

        t : int
            the state being queried, e.g., 3, or -1 for final state (corresponding to o[-1])

        Returns
        -------

        P : array_like(float,ndim=1) 
            such that P[s] = P(S_t = s | o_1,...,o_t)
        '''

        P = np.zeros(self.env.n_states)

        if t != -1:
            ooo = ooo[:t+1]

        p = self.P_traj(ooo)
        prob_paths = [path for path in p.keys() if p[path] != 0]
        prob_tiles = [int(s.split()[-1]) for s in prob_paths]
        
        for tile in prob_tiles:
            P[tile] += 1 / len(prob_tiles)

        return P

    def Q(self, o): 
        '''
            Provide Q(o,a) for all a i.e., the value for any given a under observation o. 

            Parameters
            ----------

            o : array_like(int,ndim=2)
                t observations (of 2 bits each)

            Returns
            -------

            Q : array_like(float,ndim=n_actions)
                such that Q[a] is the value (expected reward) of action a.

        '''
        Q = np.zeros(self.env.n_states)
        # TODO 
        return Q

    def act(self, obs): 
        '''
        Decide on the best action to take, under the provided observation. 

        Parameters
        ----------

        obs : array_like(int,ndim=2)
            t observations (of 2 bits each)

        Returns
        -------

        a : int
            the chosen action a
        '''

        a = -1
        # TODO 
        return a