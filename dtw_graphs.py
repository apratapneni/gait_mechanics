import numpy as np
import networkx as nx
import pandas as pd
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from tqdm import tqdm
import pickle
from matplotlib import pyplot as plt
import seaborn as sns

# Load trial data
with open('../../out/clustering/trials_all.pkl', 'rb') as f:
    trials_all = pickle.load(f)
with open('../../out/clustering/trials_metadata.pkl', 'rb') as f:
    trials_metadata = pickle.load(f)

def compute_mean_dtw_matrix(all_trials, joint_names):
    """
    Compute mean dtw distance matrix for joints across all trials.
    
    Parameters:
        all_trials (list): list of trial arrays, each [T, 3*n_joints]
        joint_names (list): list of joint names

    Returns:
        mean_dtw_matrix (np.ndarray): mean dtw distance matrix
    """
    n_joints = len(joint_names)
    dtw_matrices = []

    for trial in tqdm(all_trials):
        dtw_matrix = np.zeros((n_joints, n_joints))
        for i in range(n_joints):
            for j in range(n_joints):
                joint_i_xyz = trial[:, i*3:(i+1)*3]
                joint_j_xyz = trial[:, j*3:(j+1)*3]

                r = fastdtw(joint_i_xyz, joint_j_xyz, dist=euclidean, radius=18)[0] / trial.shape[0] # window to 0.1s for speed
                dtw_matrix[i, j] = abs(r) if not np.isnan(r) else 0
                dtw_matrix[j, i] = dtw_matrix[i, j]  # Ensure symmetry
        dtw_matrices.append(dtw_matrix)

    mean_dtw_matrix = np.mean(dtw_matrices, axis=0)
    return mean_dtw_matrix

joint_names =['Trunk1_COG','Trunk2_COG','NeckBase','TrunkPelvis','MidBack','Rshoulder_PROX','Lshoulder_PROX','RHipJnt','RKneeJnt','RAnkJnt','LHipJnt','LKneeJnt','LAnkJnt']

patient_trials = trials_all[trials_metadata['group'] == 'patient']
control_trials = trials_all[trials_metadata['group'] == 'control']

# Compute average correlation for patients and controls
pt_dtw_matrix = compute_mean_dtw_matrix(patient_trials, joint_names)
ct_dtw_matrix = compute_mean_dtw_matrix(control_trials, joint_names)
delta = pt_dtw_matrix - ct_dtw_matrix

plt.subplot(1, 3, 1)
sns.heatmap(pt_dtw_matrix, xticklabels=joint_names, yticklabels=joint_names, cmap='coolwarm', vmin=0, vmax=np.max([pt_dtw_matrix, ct_dtw_matrix]))
plt.title('Patient DTW Matrix')
plt.subplot(1, 3, 2)
sns.heatmap(ct_dtw_matrix, xticklabels=joint_names, yticklabels=joint_names, cmap='coolwarm', vmin=0, vmax=np.max([pt_dtw_matrix, ct_dtw_matrix]))
plt.title('Control DTW Matrix')
plt.subplot(1, 3, 3)
sns.heatmap(delta, xticklabels=joint_names, yticklabels=joint_names, cmap='coolwarm', center=0)
plt.title('Delta DTW Matrix (Patient - Control)')
plt.tight_layout()
plt.show()