import numpy as np
import networkx as nx
import pandas as pd
from scipy.stats import pearsonr, ttest_ind
from scipy.signal import correlate, hilbert
from sklearn.cross_decomposition import CCA
import pickle
from tqdm import tqdm

def pearson(X, Y):
    """
    Static similarity between two joint trajectories.
    """
    return pearsonr(X.flatten(), Y.flatten())[0]

def pearson_magnitude(X, Y):
    """
    Static similarity between two joint trajectories, using magnitude.
    Computes the Pearson correlation of the magnitudes of the joint movements.
    """
    return pearsonr(np.linalg.norm(X, axis=1), np.linalg.norm(Y, axis=1))[0]

def cross_correlation_weight(X, Y, max_lag=50):
    """
    Temporal coupling between two joint trajectories.
    Computes the maximum normalized cross-correlation between the two trajectories
    across all three axes, within a specified lag range.
    """
    weights = []
    for axis in range(3):
        x = X[:, axis] - X[:, axis].mean()
        y = Y[:, axis] - Y[:, axis].mean()
        corr = correlate(x, y, mode='full')
        lags = np.arange(-len(x)+1, len(x))
        center = len(corr) // 2
        window = slice(center - max_lag, center + max_lag + 1)
        norm_corr = corr[window] / (np.std(x) * np.std(y) * len(x))
        weights.append(np.max(np.abs(norm_corr)))
    return np.mean(weights)

def phase_sync_weight(X, Y):
    """
    Phase synchronization (rhythmic locking) between two joint trajectories.
    Computes the phase locking value (PLV) across all three axes.
    """
    sync_vals = []
    for axis in range(3):
        x_phase = np.angle(hilbert(X[:, axis]))
        y_phase = np.angle(hilbert(Y[:, axis]))
        phase_diff = x_phase - y_phase
        plv = np.abs(np.mean(np.exp(1j * phase_diff)))
        sync_vals.append(plv)
    return np.mean(sync_vals)

def cca_weight(X, Y, n_components=1):
    """
    Multi-joint dependency using Canonical Correlation Analysis (CCA).
    Computes the mean correlation of the canonical variables.
    CCA finds linear combinations of the two sets of variables that are maximally correlated.
    """
    cca = CCA(n_components=n_components)
    X_std = (X - X.mean(axis=0)) / X.std(axis=0)
    Y_std = (Y - Y.mean(axis=0)) / Y.std(axis=0)
    
    try:
        X_c, Y_c = cca.fit_transform(X_std, Y_std)
        corr = np.corrcoef(X_c.T, Y_c.T).diagonal(offset=n_components)
        return np.mean(corr)
    except Exception as e:
        return 0.0  # Fallback if CCA fails due to numerical issues

def build_stride_graph(stride, joint_names, weight_func=pearson, threshold=0.6, ceiling=1.0):
    """
    Build a graph from one stride's time series of joint movements.
    Edges are formed by Pearson correlation between joint trajectories.

    Parameters:
        stride: [300, 39] array
        joint_names: list of joint names (13 joints assumed)
        weight_func: function to compute edge weights (default is Pearson correlation)
        threshold: correlation cutoff to include an edge
        ceiling: maximum correlation value to consider

    Returns:
        NetworkX graph object
    """
    n_joints = len(joint_names)
    G = nx.Graph()

    # Add nodes
    for joint in joint_names:
        G.add_node(joint)

    # Compute correlations between joints (using all 3 axes)
    for i in range(n_joints):
        for j in range(i + 1, n_joints):
            joint_i_xyz = stride[:, i*3:(i+1)*3]
            joint_j_xyz = stride[:, j*3:(j+1)*3]

            r = weight_func(joint_i_xyz, joint_j_xyz)

            if abs(r) >= threshold and abs(r) < ceiling:
                G.add_edge(joint_names[i], joint_names[j], weight=r)

    return G

def extract_graph_features(G):
    """
    Compute network features from a graph.

    Returns:
        Dictionary of graph features.
    """
    features = {}
    if len(G.nodes) == 0:
        return {k: np.nan for k in ['n_edges', 'mean_degree', 'global_efficiency', 'avg_clustering',
                                   'density', 'transitivity', 'avg_shortest_path_length', 'diameter',
                                   'assortativity', 'n_connected_components', 'largest_component_size',
                                   'degree_centrality_std', 'betweenness_centrality_mean', 'closeness_centrality_mean',
                                   'eigenvector_centrality_mean', 'small_worldness']}
    
    features['n_edges'] = G.number_of_edges()
    degrees = [d for n, d in G.degree()]
    features['mean_degree'] = np.mean(degrees)
    features['global_efficiency'] = nx.global_efficiency(G)
    features['avg_clustering'] = nx.average_clustering(G)
    
    # Network density - proportion of possible edges that exist
    features['density'] = nx.density(G)
    
    # Transitivity - overall clustering coefficient
    features['transitivity'] = nx.transitivity(G)
    
    # Path-based metrics
    if nx.is_connected(G):
        features['avg_shortest_path_length'] = nx.average_shortest_path_length(G)
        features['diameter'] = nx.diameter(G)
    else:
        # For disconnected graphs, compute on largest component
        largest_cc = max(nx.connected_components(G), key=len) if len(G.nodes) > 0 else set()
        if len(largest_cc) > 1:
            subgraph = G.subgraph(largest_cc)
            features['avg_shortest_path_length'] = nx.average_shortest_path_length(subgraph)
            features['diameter'] = nx.diameter(subgraph)
        else:
            features['avg_shortest_path_length'] = np.nan
            features['diameter'] = np.nan
    
    # Assortativity - tendency of nodes to connect to similar nodes
    try:
        features['assortativity'] = nx.degree_assortativity_coefficient(G)
    except:
        features['assortativity'] = np.nan
    
    # Connectivity metrics
    features['n_connected_components'] = nx.number_connected_components(G)
    if len(G.nodes) > 0:
        features['largest_component_size'] = len(max(nx.connected_components(G), key=len))
    else:
        features['largest_component_size'] = 0
    
    # Centrality measures
    if len(G.nodes) > 0:
        degree_centrality = nx.degree_centrality(G)
        features['degree_centrality_std'] = np.std(list(degree_centrality.values()))
        
        if nx.is_connected(G) and len(G.nodes) > 1:
            betweenness_centrality = nx.betweenness_centrality(G)
            closeness_centrality = nx.closeness_centrality(G)
            features['betweenness_centrality_mean'] = np.mean(list(betweenness_centrality.values()))
            features['closeness_centrality_mean'] = np.mean(list(closeness_centrality.values()))
            
            try:
                eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
                features['eigenvector_centrality_mean'] = np.mean(list(eigenvector_centrality.values()))
            except:
                features['eigenvector_centrality_mean'] = np.nan
        else:
            features['betweenness_centrality_mean'] = np.nan
            features['closeness_centrality_mean'] = np.nan
            features['eigenvector_centrality_mean'] = np.nan
    else:
        features['degree_centrality_std'] = np.nan
        features['betweenness_centrality_mean'] = np.nan
        features['closeness_centrality_mean'] = np.nan
        features['eigenvector_centrality_mean'] = np.nan
    
    # Small-worldness (Watts-Strogatz small-world coefficient)
    if nx.is_connected(G) and len(G.nodes) > 2:
        try:
            # Generate random graph with same degree sequence
            degree_sequence = [d for n, d in G.degree()]
            random_graph = nx.configuration_model(degree_sequence)
            random_graph = nx.Graph(random_graph)  # Remove parallel edges
            random_graph.remove_edges_from(nx.selfloop_edges(random_graph))  # Remove self-loops
            
            if nx.is_connected(random_graph):
                C = features['avg_clustering']
                C_rand = nx.average_clustering(random_graph)
                L = features['avg_shortest_path_length']
                L_rand = nx.average_shortest_path_length(random_graph)
                
                if C_rand > 0 and L_rand > 0:
                    features['small_worldness'] = (C / C_rand) / (L / L_rand)
                else:
                    features['small_worldness'] = np.nan
            else:
                features['small_worldness'] = np.nan
        except:
            features['small_worldness'] = np.nan
    else:
        features['small_worldness'] = np.nan
    
    return features

def compute_stride_graph_metrics(strides_all, metadata, joint_names, weight_func, threshold=0.6, ceiling=1.0):
    """
    Compute graph metrics for every stride.

    Returns:
        pd.DataFrame of graph metrics with metadata
    """
    graph_features = []
    
    for stride in tqdm(strides_all):
        G = build_stride_graph(stride, joint_names, weight_func, threshold, ceiling)
        feats = extract_graph_features(G)
        graph_features.append(feats)
    
    metrics_df = pd.DataFrame(graph_features)
    return pd.concat([metadata.reset_index(drop=True), metrics_df], axis=1)

joint_names =['Trunk1_COG', 'Trunk2_COG', 'NeckBase', 'TrunkPelvis', 'MidBack', 'Rshoulder_PROX',
              'Lshoulder_PROX', 'RHipJnt', 'RKneeJnt', 'RAnkJnt', 'LHipJnt', 'LKneeJnt', 'LAnkJnt']

with open('/Users/aniketpratapneni/Library/CloudStorage/Box-Box/MMC_Aniket/out/clustering/strides_all.pkl', 'rb') as f:
    strides_all = pickle.load(f)
with open('/Users/aniketpratapneni/Library/CloudStorage/Box-Box/MMC_Aniket/out/clustering/metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

for func in [pearson]:#, cross_correlation_weight, phase_sync_weight]:
    print(f"Graph metrics using {func.__name__}:")
    for bounds in [(0.6,0.8), (0.8, 0.9), (0.9, 1.0)]:
        print(f"Threshold: {bounds[0]}-{bounds[1]}")
        graph_metrics_df = compute_stride_graph_metrics(
            strides_all=strides_all, 
            metadata=metadata, 
            joint_names=joint_names, 
            weight_func=func, 
            threshold=bounds[0], 
            ceiling=bounds[1]
        )
        stats = ['mean_degree', 'global_efficiency', 'avg_clustering',
                 'density', 'transitivity', 'avg_shortest_path_length', 'diameter',
                 'assortativity', 'n_connected_components', 'largest_component_size',
                 'degree_centrality_std', 'betweenness_centrality_mean', 'closeness_centrality_mean',
                 'eigenvector_centrality_mean', 'small_worldness']

        for stat in stats:
            print(f"\n{stat.replace('_', ' ').title()}:")
            print(graph_metrics_df.groupby('group')[stat].mean())
            t_stat, p_val = ttest_ind(graph_metrics_df[graph_metrics_df['group'] == 'control'][stat],
                                      graph_metrics_df[graph_metrics_df['group'] == 'patient'][stat],
                                      equal_var=False)
            pooled_std = np.sqrt((
                (graph_metrics_df[graph_metrics_df['group'] == 'control'][stat].std() ** 2) +
                (graph_metrics_df[graph_metrics_df['group'] == 'patient'][stat].std() ** 2)) / 2
            )
            cohens_d = np.abs(graph_metrics_df[graph_metrics_df['group'] == 'control'][stat].mean() -
                              graph_metrics_df[graph_metrics_df['group'] == 'patient'][stat].mean()) / pooled_std
            print(f"p-value={p_val}, Cohen's d={cohens_d}")
        
        print("\n" + "="*30 + "\n")
        #graph_metrics_df.to_csv(f'/Users/aniketpratapneni/Library/CloudStorage/Box-Box/MMC_Aniket/out/clustering/graph_metrics/graph_metrics_{func.__name__}_thresh_{threshold}.csv', index=False)