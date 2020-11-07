import numpy as np


## 计算两个矩阵样本之间的距离, 即成对距离(pairwise distances)
# P(pij)分子和分母中的一部分
def PairwiseDistance(X):
    '''
    X: sample matrix, every row is a sample
    D: pairwise distances matrix, D[a][b]=distance of a and b
    '''
    # (a-b)^2 = a^2 + b^2 -2ab
    (n, m) = X.shape
    sum_X = np.sum(np.square(X), axis=1)
    PD = (sum_X - 2.*np.matmul(X, X.T) ).T + sum_X
    return PD

## 由当前sigma和距离，计算当下的熵和手动构造的概率
def getHP(D, beta):
    '''
    D: pairwise distances of X
    beta: coefficient which are equivalent to sigma

    Pi: items in probability we will constructed
    H: entropy of Pi
    P: probability we constructed
    '''
    Pi = np.exp(-D * beta)
    sum_Pi = np.sum(Pi)
    # H(P) = -Sigma[j] [(Pij * log(Pij))]
    # H = np.log(sum_Pi) + beta * np.sum(D * Pi) / sum_Pi # ？
    P = Pi / sum_Pi
    H = -np.sum(P * np.log(P))
    return H, P

## convert sample matrix X to conditonal probabilities/P value
def X2P(X, tol=1e-5, perplexity=30.0):
    '''
    X: sample matrix
    perplexity: parameters, between 5-50 is better
    tol: error tolerance

    sigma: a group of parameters of Gaussian Distribution/P, shape(n,1)
    beta = 1/sigma^2, transformation of sigma as a easier parameters
    '''
    (n, m) = X.shape
    D = PairwiseDistance(X) # pairwise distances

    P = np.zeros((n, n)) # probability we constructed
    beta = np.ones((n, 1)) # equivalent to sigma 
    objective_H = np.log(perplexity) # Perp(P) = e^H(P), objective entropy

    # use binary search to find
    for i in range(n):
        
        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))
        
        # boundaries
        betamin = -np.inf
        betamax = np.inf
        # np.r_是按行连接两个矩阵
        # np.r_[0:3] = [0,1,2]
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] # 从D中获取一行中的部分值，空一个因为对角为0
        (H, tmp_P) = getHP(Di, beta[i])

        diff_H = H - objective_H # gap between current value and objective value
        cnt = 0
        # binary search
        while np.abs(diff_H) > tol and cnt < 50:
            # increase or decrease
            if diff_H > 0:
                betamin = beta[i]
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i]
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # update values
            (H, tmp_P) = getHP(Di, beta[i])
            diff_H = H - objective_H
            cnt += 1
        
        # set value in the relative position of P as the value in tmp_P
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = tmp_P

    # print sigma
    print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
    return P


## reduce original dimension to objective dimension
def tsne(X, dims=2, perplexity=30.0, epoch=1000):
    '''
    X: sample matrix
    dims: objective dimension

    P: original distribution
    Q: objective distribution
    '''
    ## 0.initial paramters
    # we can run PCA to pre-reduction dimension
    # pca(X)
    (n, m) = X.shape
    d = dims
    initial_momentum = 0.5
    final_momentum = 0.8
    # gradient descending withparamters
    eta = 500 # learning rate
    min_gain = 0.01
    gains = np.ones((n, d)) # 增值
    beta = 0.3 # weight parameters in my GD with Momentum
    v = np.ones((n, d)) # speed parameters in my GD with Momentum

    ## 1.compute P
    P = X2P(X, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = P * 4. # optimization: early exaggeration
    P = np.maximum(P, 1e-12) # optimization: prevent anomaly

    ## 3.initial Y
    Y = np.random.randn(n, d)
    dY = np.zeros((n, d)) # gradient of Y
    Y_pre = np.zeros((n, d)) # previous Y

    for e in range(epoch):

        ## 4.compute Q, t-distribution
        sum_Y = np.sum(np.square(Y), axis=1)
        # pairwise affinites(not distances)
        PA = 1. / (1. + (sum_Y - 2. * np.matmul(Y, Y.T) ).T + sum_Y)
        PA[range(n), range(n)] = 0. # diagonal value is 0
        Q = PA / np.sum(PA)
        Q = np.maximum(Q, 1e-12) # optimization: prevent anomaly

        ## 5.compute gradient
        PQ = P - Q
        # np.tile：沿着坐标复制
        # np.tile(a,(2,3)) 第一个维度复制2倍，第二个维度复制3倍
        for i in range(n):
            # dc/dYi = Sigma[j] [(1+(yi-yj)^2)^-1 * (yi-yj) * (pij-qij) ]
            dY[i, :] = np.sum(np.tile(PQ[:, i] * PA[:, i], (d, 1)).T * (Y[i,:] - Y), axis=0)

        ## 6.update Y
        # original
        # if epoch < 20:
        #     momentum = initial_momentum
        # else:
        #     momentum = final_momentum
        # gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
        #         (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        # gains[gains < min_gain] = min_gain
        # iY = momentum * iY - eta * (gains * dY)
        # Y = Y + iY # eta*dC/dY
        # Y = Y - np.tile(np.mean(Y, axis=0), (n,1))

        # normal GD - too slow
        # Y = Y - 0.05 * dY

        # GD with momentum in paper
        # Y(t) = Y(t-1) + eta*dC/dY + alpha*(Y(t-1)-Y(t-2))
        # Y = Y - eta * dY - alpha * (Y - Y_pre)
        # Y_pre = Y

        # my GD with momentum
        v = beta * v + (1 - beta) * dY
        Y = Y - eta * v

        # print current loss
        if (e + 1) % 10 == 0:
            # use KL divergence as loss function
            C = np.sum(P * np.log(P / Q))
            print("Iteration %d: loss is %.6f" % (e + 1, C))
    
        # optimization: Stop lying about P-values
        if e == 100:
            P = P / 4.

    return Y