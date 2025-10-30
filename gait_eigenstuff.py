# this script does a bunch of weird linear algebra on coordination matrices of gait trials

import numpy as np
import pickle
from numpy.linalg import eigh
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr
from scipy.linalg import subspace_angles

# Load trial data
with open('../../out/clustering/trials_all.pkl', 'rb') as f:
    trials_all = pickle.load(f)
with open('../../out/clustering/trials_metadata.pkl', 'rb') as f:
    trials_metadata = pickle.load(f)

def corr_matrix_for_trial(trial, n_joints):
    """
    Compute Pearson coordination matrix for a single trial.
    
    Parameters:
        trial (np.ndarray): array of shape [T, 3*n_joints]
        n_joints (int): number of joints
    Returns:
        C (np.ndarray): coordination matrix of shape [n_joints, n_joints]
    """
    T = trial.shape[0]
    joint_coords = trial.reshape(T, n_joints, 3).transpose(1, 0, 2).reshape(n_joints, -1) # reshape to [n_njoints, T*3] for pearsonr
    C = np.corrcoef(joint_coords) # compute Pearson correlation matrix
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)  # clean NaNs and infs
    C = np.clip(C,-1.0, 1.0)
    C = 0.5 * (C + C.T)  # ensure symmetry
    np.fill_diagonal(C, 1.0)  # set diagonal to 1
    return C

def build_coord_matrices(all_trials, joint_names=['Trunk1_COG','Trunk2_COG','NeckBase','TrunkPelvis','MidBack','Rshoulder_PROX','Lshoulder_PROX','RHipJnt','RKneeJnt','RAnkJnt','LHipJnt','LKneeJnt','LAnkJnt'], return_adj=True):
    """
    Build per-trial coordination matrices (pearson correlation) and optionally adjacency matrices.
    
    Parameters:
        all_trials (list): list of trial arrays, each [T, 3*n_joints]
        joint_names (list): list of joint names
        return_adj (bool): if True, also return adjacency W = |C| with zero diagonal
    Returns:
        C_all: [n_trials, n_joints, n_joints] signed correlation matrices
        W_all (optional): [n_trials, n_joints, n_joints] adjacency (|r|) matrices, diag=0
    """
    n_joints = len(joint_names)
    C_list = []

    for trial in tqdm(all_trials):
        C = corr_matrix_for_trial(trial, n_joints)
        C_list.append(C)
    
    C_all = np.stack(C_list, axis=0) # [n_trials, n_joints, n_joints]

    if not return_adj:
        return C_all
    
    W_all = np.abs(C_all) # build adjacency as |r| 
    # zero diagonal for Laplacian/spectral clustering/etc
    for i in range(W_all.shape[0]):
        np.fill_diagonal(W_all[i], 0.0)
    return C_all, W_all

# ––––––– Helpers ––––––––

def _safe_probs(lmbda, eps=1e-12):
    """Safely converts eigenvalues to a probability distribution."""
    s = np.clip(lmbda, eps, None)
    probs = s / np.sum(s)
    return probs

def participation_ratio(lmbda, eps=1e-12):
    """Computes the participation ratio of a set of eigenvalues."""
    s1 = np.sum(lmbda)
    s2 = np.sum(lmbda**2)
    return (s1 ** 2) / max(s2, eps)

def spectral_entropy(lmbda, eps=1e-12):
    """Computes the spectral entropy of a set of eigenvalues."""
    probs = _safe_probs(lmbda, eps)
    return -np.sum(probs * np.log(probs + eps)) # using nats for now (switch to np.log2 for bits for explainability maybe?)

def k_for_variance(lmbda, var_thresh=0.8):
    """Computes the number of eigenmodes needed to explain a given variance threshold."""
    p = _safe_probs(lmbda)
    cumulative = np.cumsum(p)
    k = np.searchsorted(cumulative, var_thresh) + 1  # +1 for 1-based count
    return k

def topk_eig(C_or_L, k, largest=True):
    """Computes the top-k eigenvalues and eigenvectors of a covariance or Laplacian matrix."""
    vals, vecs = eigh(C_or_L) # eigh returns in ascending order
    if largest:
        vals = vals[::-1]
        vecs = vecs[:, ::-1]
    return vals[:k], vecs[:, :k], vals, vecs

def normalized_laplacian(W, eps=1e-12):
    """Computes the normalized graph Laplacian from an adjacency matrix W."""
    d = np.sum(W, axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.clip(d, eps, None)))
    L = np.eye(W.shape[0]) - D_inv_sqrt @ W @ D_inv_sqrt # L = I - D^(-1/2) W D^(-1/2)
    return L

def principal_angle_deg(U, U_ref):
    """Computes the principal angle (in degrees) between two k-dimensional subspaces spanned by U and U_ref."""
    angles_rad = subspace_angles(U, U_ref) # this is in radians
    max_angle_rad = np.max(angles_rad)
    return np.degrees(max_angle_rad) if angles_rad.size > 0 else np.nan

def grassmann_mean_subspace(U_list, k):
    """Computes the mean subspace on the Grassmann manifold from a list of orthonormal bases U_list."""
    if len(U_list) == 0:
        return None
    J = U_list[0].shape[0]
    P_mean = np.zeros((J, J))
    for U in U_list:
        P_mean += U @ U.T
    P_mean /= len(U_list)
    _, V_k, _, _ = topk_eig(P_mean, k, largest=True)
    return V_k

def align_eigenvector_signs(U, U_ref):
    """Aligns the signs of eigenvectors in U to match those in U_ref."""
    U_aligned = U.copy()
    m = min(U.shape[1], U_ref.shape[1])
    for i in range(m):
        if np.dot(U[:, i], U_ref[:, i]) < 0:
            U_aligned[:, i] *= -1
    return U_aligned

# –––––– Spectral features: CORRELATION matrices ––––––
def spectral_features_C(C_all, k_subspace=3):
    """
    Compute spectral features from correlation matrices C_all.
    
    Parameters:
        C_all (np.ndarray): [n_trials, n_joints, n_joints] correlation matrices
        k_subspace (int): number of top eigenvectors to consider for subspace angles
    Returns:
        feats (dict): dictionary of the following np.ndarrays
            - 'spec_entropy': [n_trials,] spectral entropy of eigenvalues
            - 'part_ratio': [n_trials,] participation ratio of eigenvalues
            - 'k_var80': [n_trials,] number of eigenmodes to explain 80% variance
            - 'eigvals': [n_trials, n_joints] eigenvalues (descending)
            - 'U_topk': [n_trials, n_joints, k_subspace] top-k eigenvectors for each trial
    """
    n_trials, n_joints, _ = C_all.shape
    spec_entropy = np.zeros(n_trials)
    part_ratio = np.zeros(n_trials)
    k_var80 = np.zeros(n_trials, dtype=int)
    eigvals_all = np.zeros((n_trials, n_joints))
    U_list = []

    for i in tqdm(range(n_trials)):
        C = C_all[i]
        # largest eigenvalues and vectors
        _, U_k, vals_desc, vecs_desc = topk_eig(C, k=k_subspace, largest=True)
        eigvals_all[i] = vals_desc
        spec_entropy[i] = spectral_entropy(vals_desc)
        part_ratio[i] = participation_ratio(vals_desc)
        k_var80[i] = k_for_variance(vals_desc, var_thresh=0.8)
        U_list.append(U_k)
    
    return {
        'spec_entropy': spec_entropy,
        'part_ratio': part_ratio,
        'k_var80': k_var80,
        'eigvals': eigvals_all,
        'U_topk': np.stack(U_list, axis=0)
    }

# –––––– Spectral features: ADJACENCY matrices (via Laplacian) ––––––
def spectral_features_W(W_all, k_subspace=3, eigengap_K_max=6):
    """
    Compute spectral features from adjacency matrices W_all via normalized Laplacian.
    
    Parameters:
        W_all (np.ndarray): [n_trials, n_joints, n_joints] adjacency matrices (nonnegative, diag=0)
        k_subspace (int): number of smallest Laplacian eigenvectors to consider (smoothest modes)
        eigengap_K_max (int): maximum K to consider for first K_max eigengap calculation (L is ascending)
    Returns:
        feats (dict): dictionary of the following np.ndarrays
            - 'lambda2': [n_trials,] second smallest eigenvalue (algebraic connectivity)
            - 'eigengaps': [n_trials, eigengap_K_max-1] eigengaps for K=1 to K_max-1
            - 'L_eigvals': [n_trials, n_joints] Laplacian eigenvalues (ascending)
            - 'U_topk': [n_trials, n_joints, k_subspace] bottom-k eigenvectors for each trial
    """
    n_trials, n_joints, _ = W_all.shape
    lambda2 = np.full(n_trials, np.nan)
    eigengaps = np.zeros((n_trials, max(eigengap_K_max - 1, 1)))
    L_eigvals = np.zeros((n_trials, n_joints))
    U_list = []

    for i in tqdm(range(n_trials)):
        W = W_all[i]
        L = normalized_laplacian(W)
        # for Laplacian, smallest eigenvalues and vectors
        vals, vecs = eigh(L)
        L_eigvals[i] = vals
        lambda2[i] = vals[1] if n_joints > 1 else np.nan  # second smallest eigenvalue

        K_max = min(eigengap_K_max, n_joints - 1)
        if K_max > 1:
            eigengaps[i, :K_max - 1] = np.diff(vals[:K_max])
        
        U_k = vecs[:, :k_subspace]
        U_list.append(U_k)
    
    return {
        'lambda2': lambda2,
        'eigengaps': eigengaps,
        'L_eigvals': L_eigvals,
        'U_topk': np.stack(U_list, axis=0)
    }

# –––––– Group-reference subspaces and angles ––––––
def subspace_angles_to_ref(U_list, group_index, k_subspace=3):
    """
    Compute principal angles (degrees) between each trial's subspace to a group reference subspace.

    Parameters:
        U_list (list): list of np.ndarrays [n_joints, k_subspace] eigenvector bases for each trial
        group_index (iterable): indices of trials belonging to the reference group
        k_subspace (int): number of top eigenvectors to consider for subspace angles

    Returns:
        angles_deg (np.ndarray): [n_trials,] principal angles in degrees between each trial and the group reference subspaces
        U_ref (np.ndarray): [n_joints, k_subspace] mean subspace basis of the reference group (columns are orthonormal)
    """
    U_group = [U_list[i] for i in range(len(U_list)) if i in group_index]
    U_ref = grassmann_mean_subspace(U_group, k_subspace)
    angles = np.zeros(len(U_list))

    for i, U in enumerate(U_list):
        angles[i] = principal_angle_deg(U, U_ref) if U_ref is not None else np.nan
    return angles, U_ref

def align_all_to_ref(U_list, U_ref):
    """
    Flips signs of eigenvectors to align each trial's eigenspace basis to U_ref (column-wise).
    Parameters:
        U_list (list): list of np.ndarrays [n_joints, k_subspace] eigenvector bases for each trial
        U_ref (np.ndarray): [n_joints, k_subspace] reference eigenvector basis (columns are orthonormal)
    Returns:
        U_aligned_list (list): list of sign-aligned eigenvector bases
    """
    if U_ref is None:
        return U_list
    aligned = []
    for U in U_list:
        U_aligned = align_eigenvector_signs(U, U_ref)
        aligned.append(U_aligned)
    return aligned