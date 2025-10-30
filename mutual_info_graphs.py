import numpy as np
import networkx as nx
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.cross_decomposition import CCA
from tqdm import tqdm
import pickle
from scipy.stats import ttest_ind

with open('../../out/clustering/trials_all.pkl', 'rb') as f:
    trials_all = pickle.load(f)
with open('../../out/clustering/trials_metadata.pkl', 'rb') as f:
    trials_metadata = pickle.load(f)

def population_avg_corr(all_strides, joint_names):
    """
    Compute the average correlation per joint pair across the entire population.

    Parameters:
        all_strides: list of stride arrays, each [T, 3*njoints]
        joint_names: list of joint names

    Returns:
        avg_corr: dict {(joint_i, joint_j): avg_corr}
    """
    n_joints = len(joint_names)
    corr_sums = {}
    counts = {}

    for stride in tqdm(all_strides):
        for i in range(n_joints):
            for j in range(i + 1, n_joints):
                joint_i_xyz = stride[:, i*3:(i+1)*3]
                joint_j_xyz = stride[:, j*3:(j+1)*3]


                # r = max_xyz_correlation(joint_i_xyz, joint_j_xyz) # bad
                # r = rv_coefficient(joint_i_xyz, joint_j_xyz) # bad
                # r = plv_coefficient(joint_i_xyz, joint_j_xyz) # bad
                # r = cca(joint_i_xyz, joint_j_xyz)
                r = mutual_info_regression(joint_i_xyz.flatten().reshape(-1, 1), joint_j_xyz.flatten())[0]

                # r = pearsonr(joint_i_xyz.flatten(), joint_j_xyz.flatten())[0]
                key = (joint_names[i], joint_names[j])

                if not np.isnan(r):
                    corr_sums[key] = corr_sums.get(key, 0) + abs(r)
                    counts[key] = counts.get(key, 0) + 1

    # average correlation per pair
    avg_corr = {k: corr_sums[k] / counts[k] for k in corr_sums}
    return avg_corr


def build_stride_graph_dynamic(stride, joint_names, avg_corr, ceiling=1.0):
    """
    Build a graph for one stride using *dynamic thresholding*.
    Edges are formed if subject's correlation > population-average correlation.

    Parameters:
        stride: [T, 3*njoints] array
        joint_names: list of joint names
        avg_corr: dict of population-average correlations (from compute_population_avg_corr)
        ceiling: optional upper bound for correlations

    Returns:
        NetworkX graph object
    """
    n_joints = len(joint_names)
    G = nx.Graph()

    # Add nodes
    for joint in joint_names:
        G.add_node(joint)

    # Add edges relative to population average
    for i in range(n_joints):
        for j in range(i + 1, n_joints):
            joint_i_xyz = stride[:, i*3:(i+1)*3]
            joint_j_xyz = stride[:, j*3:(j+1)*3]

            # r = max_xyz_correlation(joint_i_xyz, joint_j_xyz)
            # r = rv_coefficient(joint_i_xyz, joint_j_xyz)
            # r = plv_coefficient(joint_i_xyz, joint_j_xyz)
            # r = cca(joint_i_xyz, joint_j_xyz)
            r = mutual_info_regression(joint_i_xyz.flatten().reshape(-1, 1), joint_j_xyz.flatten())[0]

            # r = pearsonr(joint_i_xyz.flatten(), joint_j_xyz.flatten())[0]
            key = (joint_names[i], joint_names[j])

            if not np.isnan(r):
                baseline = avg_corr.get(key, 0)
                if (abs(r) > baseline) and abs(r) < ceiling:
                    G.add_edge(joint_names[i], joint_names[j], weight=1-abs(r))

    return G

def extract_graph_features(G):
    """
    Compute network features from a graph.

    Returns:
        Dictionary of graph features.
    """
    features = {}
    if len(G.nodes) == 0:
        return {k: np.nan for k in ['n_edges', 'mean_degree', 'global_efficiency', 'avg_clustering']}
    
    features['n_edges'] = G.number_of_edges()
    degrees = [d for n, d in G.degree()]
    features['mean_degree'] = np.mean(degrees)
    features['global_efficiency'] = nx.global_efficiency(G)
    features['avg_clustering'] = nx.average_clustering(G)
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

def dynamic_graph_metrics(strides_all, metadata, joint_names, avg_corr, ceiling=1.0):
    """
    Compute graph metrics for every stride.

    Returns:
        pd.DataFrame of graph metrics with metadata
    """
    graph_features = []

    for stride in tqdm(strides_all):
        G = build_stride_graph_dynamic(stride, joint_names, avg_corr, ceiling=ceiling)
        feats = extract_graph_features(G)
        graph_features.append(feats)

    metrics_df = pd.DataFrame(graph_features)
    return pd.concat([metadata.reset_index(drop=True), metrics_df], axis=1)

joint_names =['Trunk1_COG',
 'Trunk2_COG',
 'NeckBase',
 'TrunkPelvis',
 'MidBack',
 'Rshoulder_PROX',
 'Lshoulder_PROX',
 'RHipJnt',
 'RKneeJnt',
 'RAnkJnt',
 'LHipJnt',
 'LKneeJnt',
 'LAnkJnt']

trials_avg_corr = population_avg_corr(trials_all, joint_names)
trial_graph_metrics_dynamic = dynamic_graph_metrics(
    strides_all=trials_all,
    metadata=trials_metadata,
    joint_names=joint_names,
    avg_corr=trials_avg_corr,
    ceiling=1.0
)


metrics =['mean_degree', 'global_efficiency',
               'avg_clustering', 'transitivity', 'avg_shortest_path_length',
               'diameter', 'largest_component_size', 'degree_centrality_std']
for metric in metrics:
    print(metric)
    print(trial_graph_metrics_dynamic[trial_graph_metrics_dynamic['group'] == 'control'][metric].mean(), trial_graph_metrics_dynamic[trial_graph_metrics_dynamic['group'] == 'control'][metric].std())
    print(trial_graph_metrics_dynamic[trial_graph_metrics_dynamic['group'] == 'patient'][metric].mean(), trial_graph_metrics_dynamic[trial_graph_metrics_dynamic['group'] == 'patient'][metric].std())
    print("p-value: ", ttest_ind(
        trial_graph_metrics_dynamic[trial_graph_metrics_dynamic['group'] == 'control'][metric],
        trial_graph_metrics_dynamic[trial_graph_metrics_dynamic['group'] == 'patient'][metric],
        equal_var=False
    )[1])
    # Cohen's d calculation
    pooled_std = np.sqrt(((trial_graph_metrics_dynamic[trial_graph_metrics_dynamic['group'] == 'control'][metric].std() ** 2) + 
                        (trial_graph_metrics_dynamic[trial_graph_metrics_dynamic['group'] == 'patient'][metric].std() ** 2)) / 2)
    cohens_d = abs(trial_graph_metrics_dynamic[trial_graph_metrics_dynamic['group'] == 'control'][metric].mean() - 
                trial_graph_metrics_dynamic[trial_graph_metrics_dynamic['group'] == 'patient'][metric].mean()) / pooled_std
    print(f"Cohen's: {cohens_d:.4f}\n")