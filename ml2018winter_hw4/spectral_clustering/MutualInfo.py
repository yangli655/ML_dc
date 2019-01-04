import numpy as np


def MutualInfo(L1, L2):
    '''
    mutual information
        INPUT:
            L1: labels of L1, shape of (N,) vector
            L2: labels of L2, shape of (N,) vector

        OUTPUT:
            MIhat: normalized mutual information between L1 and L2, float
    version 1.0 --December/2018
    Modified from MutualInfo.m (written by Deng Cai)
    '''
    import copy
    L1 = L1.copy()
    L2 = L2.copy()
    if L1.shape[0] != L2.shape[0] or len(L1.shape) > 1 or len(L2.shape) > 1:
        raise Exception('L1 shape must equal L2 shape')
        return
    Label = np.unique(L1)
    nClass = Label.shape[0]
    Label2 = np.unique(L2)
    nClass2 = Label2.shape[0]
    if nClass2 < nClass:
        # smooth
        L1 = np.vstack((L1, Label))
        L2 = np.vstack((L2, Label))
    elif nClass2 > nClass:
        # smooth
        L1 = np.vstack((L1, Label2))
        L2 = np.vstack((L2, Label2))
    G = np.zeros((nClass, nClass))
    for i in range(nClass):
        for j in range(nClass):
            G[i, j] = np.sum((np.logical_and(L1 == Label[i], L2 == Label[j])).astype(np.int64))
    sumG = np.sum(G)

    P1 = np.sum(G, axis=1)
    P1 = P1 / sumG
    P2 = np.sum(G, axis=0)
    P2 = P2 / sumG
    if (P1 == 0).any() or (P2 == 0).any():
        # smooth
        raise Exception('Smooth Failed')
        return
    else:
        H1 = np.sum(np.multiply(-P1, np.log2(P1)))
        H2 = np.sum(np.multiply(-P2, np.log2(P2)))
        P12 = G / sumG
        PPP = np.divide(np.divide(P12, np.tile(P2, (nClass, 1))),
                        np.tile(P1, (nClass, 1)))
        PPP[np.abs(PPP) < 1e-12] = 1
        MI = np.sum(np.multiply(P12, np.log2(PPP)))
        MIhat = MI / max(H1, H2)
    return MIhat
