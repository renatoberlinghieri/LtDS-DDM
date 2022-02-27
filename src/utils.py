#This file contains the functions that are used to estimate the parameters
#of DDM with the wavelet estimator. To see these in action, refer to the
#demo notebooks in the "Notebooks" folder on the repo.

#imports
import numpy as np
from numpy.typing import ArrayLike
from typing import Dict, Union
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10, 6]

def get_l(P : ArrayLike):
    """
    get_l computes the log-odds for each pair of alternatives.

    Args:
        P -> observed choice probabilities (np.ndarray of size NxN)
    Returns:
        l -> log-odds matrix (np.ndarray of size NxN)
    """
    l = np.log(P/(1-P))
    return l

def get_comp(a : int, obs : ArrayLike):
    """
    get_comp finds which alternatives are actually compared to "a"
    in the experiment.

    Args:
        a -> choice option of interest (int)
        obs -> observables of interest, can either be P or DT
              (np.ndarray of size NxN)
    Returns:
        choice_set -> list of alternatives compared to "a"
                     (np.ndarray of size NxN)
    """
    choice_set = np.arange(np.shape(obs)[0])[np.abs(obs[a,:]) != np.Inf]
    return choice_set

def l_tilde_pair(a : int, b : int, l : ArrayLike):
    """
    l_tilde_pair computes the log-odds "tilde" as in equation [2] of
    the reference paper, for a pair of alternatives.

    Args:
        a -> first choice alternative of interest (int)
        b -> second choice alternative of interest (int)
        l -> log-odds matrix (np.ndarray of size NxN)
    Returns:
        bool -> boolean about whether a and b are connected
        l_tilde -> modified log-odds for a and b
    """
    C_a = get_comp(a, l)
    C_b = get_comp(b, l)
    C = set(C_a).intersection(set(C_b))
    l̃ = 0
    N = len(C)
    if N == 0:
        return (False, -np.Inf) #a and b are not connected
    for c in C:
        l̃ += l[a,c] + l[c,b]
    return (True, l̃/N)

def l_tilde(l : ArrayLike):
    """
    l_tilde computes the log-odds "tilde" matrix as in equation [2] of
    the reference paper.

    Args:
        l -> log-odds matrix (np.ndarray of size NxN)
    Returns:
        L̃ -> modified log-odds matrix (np.ndarray of size NxN)
        conn -> adjacency matrix for the alternative (np.ndarray of size NxN)
    """
    n = np.shape(l)[0]
    L̃ = -np.Inf * np.ones([n,n])
    for a in range(n):
        assert np.abs(l_tilde_pair(a,a,l)[1]) < 1e-7
        L̃[a,a] = 0.0
        for b in range(a+1, n):
            conn, l̃ = l_tilde_pair(a,b,l)
            if not conn:
                L̃[a,b] = L̃[b,a] = -np.Inf
            else:
                L̃[a,b] = l̃
                L̃[b,a] = -l̃
    conn = (L̃ > -np.Inf)
    return L̃, conn

def DT_tilde_pair(a : int, b : int, l : ArrayLike, DT : ArrayLike):
    """
    DT_tilde_pair works as l_tilde_pair, just on the DT observable.

    Args:
        a -> first choice alternative of interest (int)
        b -> second choice alternative of interest (int)
        l -> log-odds matrix (np.ndarray of size NxN)
        DT -> observed decision times (np.ndarray of size NxN)
    Returns:
        ... -> ...
        SPIEGARE BENE QUI
    """
    A_l = (l > -np.Inf)
    A_dt = (DT < np.Inf)
    assert ((A_l & A_dt) == A_dt).all()
    n = np.shape(A_l)[0]
    if not A_l[a,b]:
        return np.Inf
    if a == b:
        return 0.0
    R = np.abs(l[a,b]) * (np.exp(np.abs(l[a,b])) + 1)/(np.exp(np.abs(l[a,b])) - 1)
    res = 0.0
    for c in range(n):
        for d in range(c+1,n):
            if A_dt[c,d]:
                k = np.abs(l[c,d])
                r = k * (np.exp(k) + 1)/(np.exp(k) - 1)
                res += DT[c,d] * r
    return res / (R * (np.sum(A_dt) - n) / 2)

def DT_tilde(l : ArrayLike, DT : ArrayLike):
    """
    DT_tilde works as l_tilde, just on the DT observable.

    Args:
        l -> log-odds matrix (np.ndarray of size NxN)
        DT -> observed decision times (np.ndarray of size NxN)
    Returns:
        ... -> ...
        SPIEGARE BENE QUI
    """
    n = np.shape(DT)[0]
    DT_t = np.Inf * np.ones([n,n])
    for a in range(n):
        for b in range(a, n):
            DT_t[a,b] = DT_tilde_pair(a,b,l,DT)
            DT_t[b,a] = DT_t[a,b]
    return DT_t

def wavelet(l : ArrayLike, DT : ArrayLike, new : bool = True):
    """
    wavelet computes the wavelet estimator as defined in equation [3]
    of the reference paper.

    Args:
        l -> log-odds matrix (np.ndarray of size NxN)
        DT -> observed decision times (np.ndarray of size NxN)
    Returns:
        λ -> estimated lambda for the ddm (float)
        udiff -> estimated utility difference for the ddm
                (np.ndarray of size NxN)
    """
    assert np.shape(l)[0] == np.shape(l)[1] == np.shape(DT)[0] == np.shape(DT)[1]
    n = np.shape(l)[0]
    conn = (l > -np.Inf)
    assert (conn == (DT < np.Inf)).all()
    if new:
        k = np.abs(l[1,2])
        λ = DT[1,2] * k * (np.exp(k) + 1) / (np.exp(k) - 1)
    else:
        λ = 0
        edges = (np.sum(conn) - n) / 2
        for a in range(n):
            for b in range(a+1,n):
                if conn[a,b]:
                    k = np.abs(l[a,b])
                    λ += DT[a,b] * k * (np.exp(k) + 1) / (np.exp(k) -1)
        λ /= edges
    return np.sqrt(λ), l/np.sqrt(λ)

def DT_DDM(λ : float, udiff : ArrayLike):
    """
    DT_DDM computes the theoretical decision times predicted by the DDM.

    Args:
        λ -> lambda parameter in the DDM (float)
        udiff -> utility difference for alternatives (np.ndarray of size NxN)
    Returns:
        DT -> theoretically defined decision times (np.ndarray of size NxN)
    """
    return λ * np.tanh(λ * udiff / 2) / udiff

def P_DDM(λ : float, udiff : ArrayLike):
    """
    P_DDM computes the theoretical choice probabilities predicted by the DDM.

    Args:
        λ -> lambda parameter in the DDM (float)
        udiff -> utility difference for alternatives (np.ndarray of size NxN)
    Returns:
        P -> theoretically defined choice probabilities (np.ndarray of size NxN)
    """
    return 1 / (1 + np.exp(-λ * udiff))

def plot_obs(obs : ArrayLike, λ : float, udiff : ArrayLike):
    """
    plot_obs scatter plots the observables of interest.

    Args:
        obs -> observables of interest, can either be P or DT
              (np.ndarray of size NxN)
        λ -> lambda parameter in the DDM (float)
        udiff -> utility difference for alternatives (np.ndarray of size NxN)
    """
    n = np.shape(obs)[0]
    v = λ * udiff
    idx = (obs != 0.0)
    y = obs[idx]
    x = v[idx]
    plt.scatter(x,y)
    return

def plot_obs_v(obs : ArrayLike, λ : float, udiff : ArrayLike):
    """
    plot_obs_v scatter plots the observables of interest, given value difference v.

    Args:
        obs -> observables of interest, can either be P or DT
              (np.ndarray of size NxN)
        λ -> lambda parameter in the DDM (float)
        udiff -> utility difference for alternatives (np.ndarray of size NxN)
    """
    n = np.shape(obs)[0]
    idx = (obs != 0.0)
    y = obs[idx]
    x = v[idx]
    plt.scatter(x,y)
    return

def plot_P(P : ArrayLike, λ : float, udiff : ArrayLike):
    """
    plot_P plots the estimated choice probabilites against the
    observed ones.

    Args:
        P -> observed choice probabilities (np.ndarray of size NxN)
        λ -> lambda parameter in the DDM (float)
        udiff -> utility difference for alternatives (np.ndarray of size NxN)
    """
    n = np.shape(P)[0]
    v = udiff
    idx = (P != 0.0) & (P != 0.5)
    x = v[idx]
    y = P[idx]

    u = udiff[idx]
    grid = np.min(u) + (np.max(u) - np.min(u)) * np.arange(1000) / 1000
    P_th = P_DDM(λ, grid)

    #actual plot
    plt.scatter(x, y, label="observed choice freq.")
    plt.plot(grid, P_th, label = "estimated DDM", c = 'red')
    plt.title("Wavelet estimator and empirical data (P)")
    plt.legend()
    plt.xlabel("Δu (estimated)")
    plt.ylabel("P(a,b)")
    plt.show()
    return

def plot_DT(DT : ArrayLike, λ : float, udiff : ArrayLike):
    """
    plot_DT plots the estimated decision times against the
    observed ones.

    Args:
        DT -> observed decision times (np.ndarray of size NxN)
        λ -> lambda parameter in the DDM (float)
        udiff -> utility difference for alternatives (np.ndarray of size NxN)
    """
    n = np.shape(DT)[0]
    v = udiff
    idx = (DT < np.Inf) & (DT != 0.0)
    x = v[idx]
    y = DT[idx]

    u = udiff[idx]
    grid = np.min(u) + (np.max(u) - np.min(u)) * np.arange(1000) / 1000
    DT_th = DT_DDM(λ, grid)

    #actual plot
    plt.scatter(x, y, label="observed choice freq.")
    plt.plot(grid, DT_th, label = "estimated DDM", c = 'red')
    plt.title("Wavelet estimator and empirical data (DT)")
    plt.legend()
    plt.xlabel("Δu (estimated)")
    plt.ylabel("DT(a,b)")
    plt.show()
    return
