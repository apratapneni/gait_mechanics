import numpy as np
from numpy.linalg import eigh
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix
from scipy.optimize import linear_sum_assignment
from itertools import combinations
from tqdm import tqdm

# –––––––––– Laplacian & eigengap ––––––––––
def normalized_laplacian(W: np.ndarray, eps = 1e-12) -> np.ndarray:
    """Computes the normalized graph Laplacian from an adjacency matrix W."""
    d = np.sum(W, axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.clip(d, eps, None)))
    L = np.eye(W.shape[0]) - D_inv_sqrt @ W @ D_inv_sqrt # L = I - D^(-1/2) W D^(-1/2)
    return L

def choose_k_by_eigengap(W: np.ndarray, k_min: int = 2, k_max: int = 6) -> int:
    """Picks K for clustering via largest eigengap on the normalized Laplacian's smallest eigenvalues."""
    L = normalized_laplacian(W)
    vals, _ = eigh(L) # ascending
    Kmax = min(k_max, W.shape[0]-1)
    gaps = np.diff(vals[k_min-1:Kmax+1])
    k = np.argmax(gaps) + k_min # shift index to start from k_min
    return int(k)

def spectral_embedding(W: np.ndarray, K: int) -> np.ndarray:
    """Returns the K smallest eigenvectors of the normalized Laplacian of W, row-normalized."""
    L = normalized_laplacian(W)
    _, vecs = eigh(L) # ascending
    U = vecs[:, :K]
    # row normalize U to unit length
    U_norm = np.linalg.norm(U, axis=1, keepdims=True)
    U_normalized = U / np.clip(U_norm, 1e-12, None)
    return U_normalized

def spectral_partition(W: np.ndarray, K: int, n_init: int = 50, random_state: int = 42) -> np.ndarray:
    """Spectral clustering via k-means on Laplacian embeddings (stable + fast)"""
    U_normalized = spectral_embedding(W, K)
    kmeans = KMeans(n_clusters=K, n_init=n_init, random_state=random_state)
    labels = kmeans.fit_predict(U_normalized)
    return labels

# –––––––––– Trial clustering w/ K selection ––––––––––
def cluster_trial(W: np.ndarray, k_strategy: str = 'eigengap', k_fixed: int = 3, k_range: tuple = (2, 6), silhouette_guard: bool = True) -> tuple:
    """
    Returns labels and chosen K for clustering one trial's adjacency matrix W.
    
    Parameters:
        W: Adjacency matrix (n x n)
        k_strategy: 'eigengap' | 'fixed' | 'silhouette'
        k_fixed: if k_strategy=='fixed', use this K
        k_range: if k_strategy=='silhouette', range of K to try (min, max)
        silhouette_guard: if True, checks silhouette score to avoid empty clusters
    Returns:
        labels: Cluster labels for each node
        K: Chosen number of clusters
    """
    if k_strategy == 'fixed':
        K = k_fixed
    elif k_strategy == 'eigengap':
        K = choose_k_by_eigengap(W, k_min=k_range[0], k_max=k_range[1])
    elif k_strategy == 'silhouette':
        # try K in range and pick max silhouette score (on spectral embedding + k-means labels)
        best_score, best_K, best_labels = -np.inf, None, None
        for K in range(k_range[0], k_range[1]+1):
            labels = spectral_partition(W, K)
            U_normalized = spectral_embedding(W, K)
            s = silhouette_score(U_normalized, labels) if K > 1 else -np.inf
            if s > best_score:
                best_score, best_K, best_labels = s, K, labels
        return best_labels, best_K
    else:
        raise ValueError(f"Unknown k_strategy: {k_strategy}")
    
    labels = spectral_partition(W, K)
    if silhouette_guard and K > 1:
        U_normalized = spectral_embedding(W, K)
        _ = silhouette_score(U_normalized, labels)  # just to check cluster sizes
    return labels, K

# –––––––––– Co-association matrix and consensus partition ––––––––––
def coassociation_matrix(label_list: list) -> np.ndarray:
    """Computes the co-association matrix from a list of cluster labels.
    
    Parameters:
        label_list (list): List of length n, each element is an array of n_joints cluster labels.
    Returns:
        A (np.ndarray [n_joints, n_joints]): A_ij = frequency at which joints i and j are clustered together.
    """
    n_joints = len(label_list[0])
    A = np.zeros((n_joints, n_joints), dtype=float)
    for labels in label_list:
        for i, j in combinations(range(n_joints), 2):
            A[i, j] += float(labels[i] == labels[j])
            A[j, i] = A[i, j]
    A /= len(label_list)
    np.fill_diagonal(A, 1.0)
    return A

def consensus_partition(A: np.ndarray, k_strategy: str='eigengap', k_fixed: int=3, k_range: tuple=(2,6)) -> tuple:
    """
    Clusters the co-association matrix to obtain a consensus partition.
    
    Parameters:
        A (np.ndarray [n_joints, n_joints]): Co-association matrix
        k_strategy (str): 'eigengap' | 'fixed' | 'silhouette'
        k_fixed (int): if k_strategy=='fixed', use this K
        k_range (tuple): if k_strategy=='silhouette', range of K to try (min, max)
    Returns:
        np.ndarray [n_joints]: Consensus cluster labels
        int: Chosen number of clusters
    """
    # all A_ij must be in [0, 1] and A must be symmetric
    A = np.clip(0.5 * (A + A.T), 0.0, 1.0)
    if k_strategy == 'fixed':
        K = k_fixed
    elif k_strategy == 'eigengap':
        K = choose_k_by_eigengap(A, k_min=k_range[0], k_max=k_range[1])
    elif k_strategy == 'silhouette':
        # try K in range and pick max silhouette score (on spectral embedding + k-means labels)
        best_score, best_K, best_labels = -np.inf, None, None
        for K in range(k_range[0], k_range[1]+1):
            labels = spectral_partition(A, K)
            U_normalized = spectral_embedding(A, K)
            s = silhouette_score(U_normalized, labels) if K > 1 else -np.inf
            if s > best_score:
                best_score, best_K, best_labels = s, K, labels
        return best_labels, best_K
    else:
        raise ValueError(f"Unknown k_strategy: {k_strategy}")
    labels = spectral_partition(A, K)
    return labels, K

def stability_to_connsensus(label_list: list, consensus_labels: np.ndarray) -> np.ndarray:
    """
    Computes the fraction of trials in which each joint agrees with the consensus cluster.
    Parameters:
        label_list (list): List of length n_trials, each element is an array of n_joints cluster labels.
        consensus_labels (np.ndarray [n_joints]): Consensus cluster labels.
    Returns:
        np.ndarray [n_joints]: Stability score for each joint [0, 1].
    """
    n_joints = len(consensus_labels)
    stability = np.zeros(n_joints, dtype=float)
    for joint in range(n_joints):
        agreements = np.sum(labels[joint] == consensus_labels[joint] for labels in label_list)
        stability[joint] = agreements / len(label_list)
    return stability

# –––––––––– Align & compare group consensus partitions ––––––––––
def align_partitions(y_true, y_pred):
    """
    Relabel y_pred to best match y_true using the Hungarian algorithm on the contingency matrix.
    
    Parameters:
        y_true (np.ndarray): Ground truth cluster labels.
        y_pred (np.ndarray): Predicted cluster labels to be aligned.
    Returns:
        y_pred_aligned (np.ndarray): Aligned predicted cluster labels.
        ari (float): Adjusted Rand Index between y_true and y_pred_aligned.
        nmi (float): Normalized Mutual Information between y_true and y_pred_aligned.
    """
    cm = contingency_matrix(y_true, y_pred)
    # maximize trace by minimizing negative
    row_ind, col_ind = linear_sum_assignment(-cm)
    label_mapping = {old: new for old, new in zip(col_ind, row_ind)}
    y_pred_aligned = np.array([label_mapping[label] for label in y_pred])
    ari = adjusted_rand_score(y_true, y_pred_aligned)
    nmi = normalized_mutual_info_score(y_true, y_pred_aligned)
    return y_pred_aligned, ari, nmi

# –––––– Patient vs Control Pipeline –––––––
def spectral_consensus_pipeline(W_all: np.ndarray, group_mask: np.ndarray, k_strategy='eigengap', k_fixed=3, k_range=(2,6)) -> dict:
    """
    Full pipeline to compute and align consensus partitions for patient vs control groups.
    
    Parameters:
        W_all (np.ndarray [n_trials, n_joints, n_joints]): Adjacency matrices for all trials.
        group_mask (np.ndarray [n_trials]): Boolean mask indicating patient (True) vs control (False).
        k_strategy (str): 'eigengap' | 'fixed' | 'silhouette'
        k_fixed (int): if k_strategy=='fixed', use this K
        k_range (tuple): if k_strategy=='silhouette', range of K to try (min, max)
    Returns:
        dict: {
            'trial_indices': Indices of trials in the group,
            'trial_labels': List of cluster labels for each trial,
            'trial_Ks': Array of chosen K for each trial,
            'coassoc': Co-association matrix,
            'consensus_labels': Consensus cluster labels,
            'consensus_K': Chosen K for consensus,
            'stability': Stability scores for each joint
        }
    """
    idx = np.where(group_mask)[0]
    labels_list, Ks = [], []
    for i in tqdm(idx):
        labels, K = cluster_trial(W_all[i], k_strategy=k_strategy, k_fixed=k_fixed, k_range=k_range)
        labels_list.append(labels)
        Ks.append(K)
    
    A = coassociation_matrix(labels_list)
    labels_consensus, K_consensus = consensus_partition(A, k_strategy=k_strategy, k_fixed=k_fixed, k_range=k_range)
    stability = stability_to_connsensus(labels_list, labels_consensus)

    return {
        'trial_indices': idx,
        'trial_labels': labels_list, # list of arrays of size n_joints
        'trial_Ks': np.array(Ks), # array of size n_trials_in_group
        'coassoc': A, # n_joints x n_joints
        'consensus_labels': labels_consensus, # array of size n_joints
        'consensus_K': K_consensus, # int
        'stability': stability # array of size n_joints
    }

def compare_group_consensus(consensus_1, consensus_2):
    """Aligns 2's labels to 1's labels and computes ARI/NMI and a contingency summary."""
    y_1 = consensus_1['consensus_labels']
    y_2 = consensus_2['consensus_labels']
    y_2_aligned, ari, nmi = align_partitions(y_1, y_2)
    cm = contingency_matrix(y_1, y_2_aligned)
    return {
        'ARI': ari,
        'NMI': nmi,
        'contingency_matrix': cm,
        'y_2_aligned': y_2_aligned
    }