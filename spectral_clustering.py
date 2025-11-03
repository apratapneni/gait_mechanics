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

# –––––––––– Granular distribution-level group comparisons ––––––––––

def coassociation_entropy(A: np.ndarray, normalize: bool = True) -> float:
    """
    Computes the Shannon entropy of the co-association matrix to measure partition uncertainty.
    Higher entropy = more variability in how joints cluster together across trials.
    
    Parameters:
        A (np.ndarray [n_joints, n_joints]): Co-association matrix (values in [0, 1])
        normalize (bool): If True, normalize by max possible entropy
    Returns:
        float: Entropy of co-association distribution
    """
    # Get upper triangle (excluding diagonal) to avoid double counting
    idx = np.triu_indices_from(A, k=1)
    p = A[idx]
    
    # Treat each entry as a Bernoulli distribution and sum entropies
    p = np.clip(p, 1e-12, 1 - 1e-12)  # avoid log(0)
    H = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    total_entropy = np.sum(H)
    
    if normalize:
        # Max entropy occurs when all p_ij = 0.5
        n_pairs = len(p)
        max_entropy = n_pairs  # each pair contributes 1 bit at p=0.5
        return total_entropy / max_entropy
    return total_entropy

def partition_diversity_index(label_list: list) -> float:
    """
    Computes the average pairwise ARI between all trial partitions (Simpson-like diversity).
    Lower values = more diverse partitions across trials.
    
    Parameters:
        label_list (list): List of cluster label arrays from individual trials
    Returns:
        float: Average pairwise ARI (higher = more consistent clustering)
    """
    n_trials = len(label_list)
    if n_trials < 2:
        return 1.0
    
    ari_sum = 0.0
    count = 0
    for i, j in combinations(range(n_trials), 2):
        ari_sum += adjusted_rand_score(label_list[i], label_list[j])
        count += 1
    
    return ari_sum / count

def stability_distribution_test(stability_1: np.ndarray, stability_2: np.ndarray, test: str = 'ks') -> dict:
    """
    Statistical test comparing stability score distributions between two groups.
    
    Parameters:
        stability_1 (np.ndarray): Stability scores from group 1
        stability_2 (np.ndarray): Stability scores from group 2
        test (str): 'ks' (Kolmogorov-Smirnov) or 'mw' (Mann-Whitney U)
    Returns:
        dict: {'statistic': float, 'pvalue': float, 'effect_size': float}
    """
    from scipy.stats import ks_2samp, mannwhitneyu
    
    if test == 'ks':
        stat, pval = ks_2samp(stability_1, stability_2)
        # Effect size: max difference in CDFs
        effect = stat
    elif test == 'mw':
        stat, pval = mannwhitneyu(stability_1, stability_2, alternative='two-sided')
        # Effect size: rank-biserial correlation
        n1, n2 = len(stability_1), len(stability_2)
        effect = 1 - (2 * stat) / (n1 * n2)
    else:
        raise ValueError(f"Unknown test: {test}")
    
    return {
        'statistic': stat,
        'pvalue': pval,
        'effect_size': effect,
        'test': test
    }

def joint_stability_comparison(consensus_1: dict, consensus_2: dict, joint_names: list = None) -> dict:
    """
    Per-joint stability comparison between two groups with effect sizes.
    
    Parameters:
        consensus_1, consensus_2 (dict): Output from spectral_consensus_pipeline
        joint_names (list): Optional names for joints
    Returns:
        dict: {
            'joint_stability_diff': Array of stability differences (group1 - group2),
            'joint_stability_effect': Cohen's d effect size per joint,
            'mean_absolute_diff': Mean absolute stability difference,
            'joints_large_diff': Indices of joints with |diff| > threshold
        }
    """
    s1 = consensus_1['stability']
    s2 = consensus_2['stability']
    n_joints = len(s1)
    
    if joint_names is None:
        joint_names = [f"Joint_{i}" for i in range(n_joints)]
    
    # Point estimates
    diff = s1 - s2
    
    # Compute per-joint variance from trial-level stability
    def joint_stability_variance(trial_labels, consensus_labels):
        """Estimate variance in stability for each joint across trials"""
        n_trials = len(trial_labels)
        n_joints = len(consensus_labels)
        stability_matrix = np.zeros((n_trials, n_joints))
        
        for t, labels in enumerate(trial_labels):
            for j in range(n_joints):
                stability_matrix[t, j] = float(labels[j] == consensus_labels[j])
        
        return np.var(stability_matrix, axis=0, ddof=1)
    
    var1 = joint_stability_variance(consensus_1['trial_labels'], consensus_1['consensus_labels'])
    var2 = joint_stability_variance(consensus_2['trial_labels'], consensus_2['consensus_labels'])
    
    # Cohen's d per joint
    pooled_std = np.sqrt((var1 + var2) / 2)
    cohens_d = diff / np.clip(pooled_std, 1e-12, None)
    
    # Identify joints with large differences (|d| > 0.5 is medium effect)
    threshold = 0.5
    large_diff_idx = np.where(np.abs(cohens_d) > threshold)[0]
    
    return {
        'joint_names': joint_names,
        'stability_group1': s1,
        'stability_group2': s2,
        'stability_diff': diff,
        'cohens_d': cohens_d,
        'mean_absolute_diff': np.mean(np.abs(diff)),
        'joints_large_effect': large_diff_idx,
        'large_effect_names': [joint_names[i] for i in large_diff_idx]
    }

def coassociation_distance(A1: np.ndarray, A2: np.ndarray, metric: str = 'frobenius') -> float:
    """
    Distance between two co-association matrices.
    
    Parameters:
        A1, A2 (np.ndarray): Co-association matrices
        metric (str): 'frobenius' | 'hellinger' | 'js' (Jensen-Shannon divergence)
    Returns:
        float: Distance measure
    """
    if metric == 'frobenius':
        return np.linalg.norm(A1 - A2, ord='fro')
    
    elif metric == 'hellinger':
        # Hellinger distance: sqrt(sum((sqrt(p) - sqrt(q))^2))
        # Treat matrices as probability distributions over edges
        idx = np.triu_indices_from(A1, k=1)
        p = A1[idx]
        q = A2[idx]
        return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2)
    
    elif metric == 'js':
        # Jensen-Shannon divergence (symmetrized KL divergence)
        from scipy.spatial.distance import jensenshannon
        idx = np.triu_indices_from(A1, k=1)
        p = A1[idx]
        q = A2[idx]
        # JSD requires probability distributions; normalize to sum to 1
        p = p / np.sum(p)
        q = q / np.sum(q)
        return jensenshannon(p, q) ** 2  # squared JS distance
    
    else:
        raise ValueError(f"Unknown metric: {metric}")

def partition_variability_ratio(consensus_1: dict, consensus_2: dict) -> dict:
    """
    Compares partition variability between groups using multiple measures.
    
    Parameters:
        consensus_1, consensus_2 (dict): Output from spectral_consensus_pipeline
    Returns:
        dict: {
            'entropy_ratio': Ratio of co-association entropies,
            'diversity_ratio': Ratio of partition diversity indices,
            'K_variance_ratio': Ratio of variances in chosen K values,
            'interpretation': String describing which group is more variable
        }
    """
    H1 = coassociation_entropy(consensus_1['coassoc'])
    H2 = coassociation_entropy(consensus_2['coassoc'])
    
    D1 = partition_diversity_index(consensus_1['trial_labels'])
    D2 = partition_diversity_index(consensus_2['trial_labels'])
    
    K_var1 = np.var(consensus_1['trial_Ks'], ddof=1) if len(consensus_1['trial_Ks']) > 1 else 0
    K_var2 = np.var(consensus_2['trial_Ks'], ddof=1) if len(consensus_2['trial_Ks']) > 1 else 0
    
    entropy_ratio = H1 / np.clip(H2, 1e-12, None)
    diversity_ratio = (1 - D1) / np.clip(1 - D2, 1e-12, None)  # Higher = more variable
    K_var_ratio = K_var1 / np.clip(K_var2, 1e-12, None)
    
    # Interpretation
    if entropy_ratio > 1.2 or diversity_ratio > 1.2:
        interp = "Group 1 shows higher partition variability"
    elif entropy_ratio < 0.8 or diversity_ratio < 0.8:
        interp = "Group 2 shows higher partition variability"
    else:
        interp = "Groups show similar partition variability"
    
    return {
        'entropy_group1': H1,
        'entropy_group2': H2,
        'entropy_ratio': entropy_ratio,
        'diversity_group1': D1,
        'diversity_group2': D2,
        'diversity_ratio': diversity_ratio,
        'K_variance_group1': K_var1,
        'K_variance_group2': K_var2,
        'K_variance_ratio': K_var_ratio,
        'interpretation': interp
    }

def permutation_test_coassociation(consensus_1: dict, consensus_2: dict, 
                                   n_permutations: int = 1000, 
                                   metric: str = 'frobenius',
                                   random_state: int = 42) -> dict:
    """
    Permutation test for co-association matrix difference between groups.
    Tests null hypothesis that groups have same underlying clustering distribution.
    
    Parameters:
        consensus_1, consensus_2 (dict): Output from spectral_consensus_pipeline
        n_permutations (int): Number of permutations
        metric (str): Distance metric for co-association matrices
        random_state (int): Random seed
    Returns:
        dict: {
            'observed_distance': Distance between observed co-association matrices,
            'permuted_distances': Array of distances under null,
            'pvalue': Two-tailed p-value,
            'effect_size': Standardized effect size (z-score)
        }
    """
    rng = np.random.RandomState(random_state)
    
    # Observed distance
    A1 = consensus_1['coassoc']
    A2 = consensus_2['coassoc']
    observed_dist = coassociation_distance(A1, A2, metric=metric)
    
    # Pool all trial labels
    all_labels = consensus_1['trial_labels'] + consensus_2['trial_labels']
    n1 = len(consensus_1['trial_labels'])
    n_total = len(all_labels)
    
    # Permutation distribution
    permuted_dists = np.zeros(n_permutations)
    for i in tqdm(range(n_permutations), desc="Permutation test"):
        # Shuffle and split
        perm_idx = rng.permutation(n_total)
        perm_labels_1 = [all_labels[j] for j in perm_idx[:n1]]
        perm_labels_2 = [all_labels[j] for j in perm_idx[n1:]]
        
        # Compute co-association matrices under null
        A1_perm = coassociation_matrix(perm_labels_1)
        A2_perm = coassociation_matrix(perm_labels_2)
        
        permuted_dists[i] = coassociation_distance(A1_perm, A2_perm, metric=metric)
    
    # P-value (two-tailed)
    pvalue = np.mean(permuted_dists >= observed_dist) * 2
    pvalue = np.clip(pvalue, 0, 1)
    
    # Effect size (standardized distance)
    effect_size = (observed_dist - np.mean(permuted_dists)) / np.clip(np.std(permuted_dists), 1e-12, None)
    
    return {
        'observed_distance': observed_dist,
        'permuted_distances': permuted_dists,
        'pvalue': pvalue,
        'effect_size': effect_size,
        'metric': metric
    }

def bootstrap_consensus_stability(W_all: np.ndarray, group_mask: np.ndarray,
                                 n_bootstrap: int = 100,
                                 k_strategy: str = 'eigengap',
                                 k_fixed: int = 3,
                                 k_range: tuple = (2, 6),
                                 random_state: int = 42) -> dict:
    """
    Bootstrap resampling to estimate uncertainty in consensus partition and stability.
    
    Parameters:
        W_all (np.ndarray): All adjacency matrices
        group_mask (np.ndarray): Boolean mask for group
        n_bootstrap (int): Number of bootstrap samples
        k_strategy, k_fixed, k_range: Clustering parameters
        random_state (int): Random seed
    Returns:
        dict: {
            'consensus_labels_boot': List of consensus partitions from bootstrap samples,
            'stability_boot': Array [n_bootstrap, n_joints] of stability scores,
            'stability_ci': 95% confidence intervals for stability [n_joints, 2],
            'consensus_variability': ARI variability across bootstrap consensus partitions
        }
    """
    rng = np.random.RandomState(random_state)
    idx = np.where(group_mask)[0]
    n_trials = len(idx)
    n_joints = W_all.shape[1]
    
    consensus_labels_boot = []
    stability_boot = np.zeros((n_bootstrap, n_joints))
    
    for b in tqdm(range(n_bootstrap), desc="Bootstrap"):
        # Resample trials with replacement
        boot_idx = rng.choice(idx, size=n_trials, replace=True)
        labels_list = []
        
        for i in boot_idx:
            labels, _ = cluster_trial(W_all[i], k_strategy=k_strategy, 
                                     k_fixed=k_fixed, k_range=k_range)
            labels_list.append(labels)
        
        # Compute consensus for this bootstrap sample
        A_boot = coassociation_matrix(labels_list)
        labels_consensus, _ = consensus_partition(A_boot, k_strategy=k_strategy,
                                                 k_fixed=k_fixed, k_range=k_range)
        stability = stability_to_connsensus(labels_list, labels_consensus)
        
        consensus_labels_boot.append(labels_consensus)
        stability_boot[b, :] = stability
    
    # Stability confidence intervals (95%)
    stability_ci = np.percentile(stability_boot, [2.5, 97.5], axis=0).T
    
    # Consensus variability: average pairwise ARI across bootstrap consensus partitions
    consensus_variability = partition_diversity_index(consensus_labels_boot)
    
    return {
        'consensus_labels_boot': consensus_labels_boot,
        'stability_boot': stability_boot,
        'stability_mean': np.mean(stability_boot, axis=0),
        'stability_ci': stability_ci,
        'consensus_variability': consensus_variability
    }

def comprehensive_group_comparison(consensus_1: dict, consensus_2: dict,
                                  joint_names: list = None,
                                  permutation_test: bool = True,
                                  n_permutations: int = 1000) -> dict:
    """
    All-in-one comprehensive comparison between two group consensus partitions.
    
    Parameters:
        consensus_1, consensus_2 (dict): Output from spectral_consensus_pipeline
        joint_names (list): Optional joint names
        permutation_test (bool): Whether to run permutation test
        n_permutations (int): Number of permutations if test is run
    Returns:
        dict: Comprehensive comparison results including consensus alignment,
              stability comparison, variability ratios, and statistical tests
    """
    results = {}
    
    # 1. Basic consensus alignment
    results['consensus_alignment'] = compare_group_consensus(consensus_1, consensus_2)
    
    # 2. Joint-level stability comparison
    results['joint_stability'] = joint_stability_comparison(consensus_1, consensus_2, joint_names)
    
    # 3. Distribution-level stability test
    results['stability_distribution'] = stability_distribution_test(
        consensus_1['stability'], consensus_2['stability'], test='ks'
    )
    
    # 4. Partition variability comparison
    results['variability'] = partition_variability_ratio(consensus_1, consensus_2)
    
    # 5. Co-association matrix distance
    results['coassoc_distance'] = {
        'frobenius': coassociation_distance(consensus_1['coassoc'], consensus_2['coassoc'], 'frobenius'),
        'hellinger': coassociation_distance(consensus_1['coassoc'], consensus_2['coassoc'], 'hellinger'),
        'jensen_shannon': coassociation_distance(consensus_1['coassoc'], consensus_2['coassoc'], 'js')
    }
    
    # 6. Permutation test (optional, can be slow)
    if permutation_test:
        results['permutation_test'] = permutation_test_coassociation(
            consensus_1, consensus_2, n_permutations=n_permutations
        )
    
    return results