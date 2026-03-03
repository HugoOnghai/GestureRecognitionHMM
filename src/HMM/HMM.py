import numpy as np

class HMM():
    """
    A class to hold the parameters of each of my HMMs, one for each gesture type:
    pi: (N,) # initial sate distribution
    A: (N,N) the transition matrix (transition probabilities from hidden state to hidden state between subsequent time steps)
    B: (N,M) the observation/emission matrix (observation probabilities from given hidden state to observed observation)
    """

    def __init__(self, N, M, pi=None, A=None, B=None, label=None, init="uniform"):
        self.N = int(N) # the number of hidden states within the full gesture sequence (number of partial gesture movements)
        self.M = int(M) # the number of observation types (the k-means cluster ids)
        self.label = label # the gesture type label

        if init == "uniform":
            if pi is None: # then initialize to uniform
                pi = np.full(self.N, 1.0 / self.N)
            if A is None: 
                A = np.full((self.N, self.N), 1.0 / self.N)
            if B is None:
                B = np.full((self.N, self.M), 1.0 / self.M)
        elif init == "random":
            rng = np.random.default_rng(0)
            if pi is None:
                pi = rng.random(self.N)
                pi /= pi.sum()
            if A is None:
                A = rng.random((self.N, self.N))
                A /= A.sum(axis=1, keepdims=True)
            if B is None:
                B = rng.random((self.N, self.M))
                B /= B.sum(axis=1, keepdims=True)
        else:
            raise ValueError("init param must be uniform or random")

        self.pi = pi
        self.A = A # A[i, j] = a_ij = P(X_t = j | X_t-1 = i), time-indep stochastic transition matrix
        self.B = B # B[i, j] = b_j(y_i) = P(Y_t = y_i | X_t = j), j in all possible states, i in all possible observations

    def construct_forward(self, seq: np.ndarray):
        T = len(seq)

        alpha_hat = np.zeros((T, self.N)) # hat denotes after scaling, as per Rabiner's notation
        alpha = np.zeros((T, self.N)) # before scaling
        c = np.zeros(T) # the scaling coefficient at each time step

        # t = 0
        # [1 by N] = [N] * [N by 1]
        alpha[0,:] = self.pi[:]*self.B[:,seq[0]] # calculate alpha_i(0) for every i in 0 to N-1.
        c[0] = 1.0 / (alpha[0,:].sum()) # calculate scaling coefficient c_0.
        alpha_hat[0, :] = c[0]*alpha[0,:] # like a geometric series, we keep applying scaling coefficients

        # t > 0
        for t in range(1, T):
            # [1 by N] = [1 by N]*[N by N] * [N by 1] *
            pred = alpha_hat[t-1,:] @ self.A[:, :] # the probability of getting all the way to state i at t-1, then transition from i to j. 
            alpha[t,:] = pred * self.B[:,seq[t]]
            c[t] = 1.0 / (alpha[t, :].sum())
            alpha_hat[t,:] = c[t]*alpha[t,:]

        return alpha_hat, c
    
    def construct_backward(self, seq: np.ndarray, c): # pass through the scale factors from the forward
        # in the backward procedure, we constract beta_i(t) which tells us the probability of the ending partial sequence being observed
        # given that we start at state i at time t.
        T = len(seq)

        beta = np.zeros((T, self.N))
        beta_hat = np.zeros((T, self.N))

        # t = T-1, the last time
        beta[T-1, :] = np.ones((self.N)) # beta(T)=1 for any i in N
        beta_hat[T-1, :] = c[T-1] * beta[T-1, :]

        # t < T-1
        for t in range(T-2, -1, -1):
            beta[t] = self.A[:,:] @ (self.B[:,seq[t+1]] * beta_hat[t+1,:])
            beta_hat[t] = c[t] * beta[t, :]

        return beta_hat, c

    def calculate_gamma(self, alpha_hat, beta_hat):
        # gamma is the probability of being in state i at time t given the observed sequence Y and the parameters theta (source: wikipedia)
        numer = alpha_hat * beta_hat # (T,N) element wise operation gives us alpha_i(t) * beta_i(t)
        denom = numer.sum(axis=1, keepdims=True) # (T,1) sum across each row (over states)
        gamma = numer / denom
        return gamma # (T,N)

    def calculate_xi(self, seq: np.ndarray, alpha_hat, beta_hat):
        # xi is the probability of being in state i and j at times t and t+1, given the observed sequence Y (source: wikipedia)
        T = len(seq)
        xi = np.zeros((T-1, self.N, self.N))

        for t in range(T-1):
            # prob of being in j from the backwards
            bj = beta_hat[t+1,:]*self.B[:,seq[t+1]]

            # prob of being in i from the forwards
            ai = alpha_hat[t,:]*self.A[:,:]

            numer = ai*bj

            # normalize the numerator to get the probability of being in state i and j at t and t+1 given seq and model
            denom = numer.sum()

            xi[t,:,:] = numer / denom

        return xi
    
    def update_params(self, seq, gamma, xi): # updating params based on a single input
        # pi
        self.pi = gamma[0].copy()

        # A
        self.A = xi.sum(axis=0, keepdims=True) / gamma[:-1, :].sum(axis=0, keepdims=True)
        
        # B
        Y = np.eye(self.M)[seq]
        B_numer = gamma.T @ Y
        B_denom = gamma.sum(axis=0, keepdims=True).T
        self.B = B_numer / (B_denom + 1e-6)

    def fit_once(self, seqs): # the whole shebang, source: wikipedia
        # instead of using update_params, I need to sum the numerators and denominators up first, across what was found from each sequences
        pi = np.zeros(self.N)
        A_numer = np.zeros((self.N, self.N))
        A_denom = np.zeros((self.N))
        B_numer = np.zeros((self.N, self.M))
        B_denom = np.zeros((self.N))

        total_loglikelihood = 0.0

        for seq in seqs:
            # forward-backward calculations
            alpha_hat, c = self.construct_forward(seq)
            beta_hat, _ = self.construct_backward(seq, c)

            # E-step
            gamma = self.calculate_gamma(alpha_hat, beta_hat)
            xi = self.calculate_xi(seq, alpha_hat, beta_hat)

            # log P(seq|model)
            total_loglikelihood += -np.sum(np.log(c + 1e-12))

            # accumulate updates
            pi += gamma[0] # (1,N)

            A_numer += xi.sum(axis=0) # shape N-by-N
            A_denom += gamma[:-1].sum(axis=0)

            Y = np.eye(self.M)[seq]
            B_numer += gamma.T @ Y
            B_denom += gamma.sum(axis=0)

        # normalize (also M-step)
        self.pi = pi / len(seqs)
        self.A = A_numer / A_denom[:, None] # fixes broadcasting issues
        self.B = B_numer / B_denom[:, None] # fixes broadcasting issues

        return total_loglikelihood
    
    def score(self, seqs, eps=1e-12):
        total = 0.0
        for seq in seqs:
            _, c = self.construct_forward(seq)
            total += -np.sum(np.log(c + eps))
        return total