import numpy as np
import seaborn as sns
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt

#loading p values
from A01_sim_data import p_values, p_value_fire, p_value_nonfire

p_values
p_value_fire
p_value_nonfire
num_firing = len(p_value_fire)
num_nonfire= len(p_value_nonfire) 

#Define function for bonferroni procedure

def bonferroni(p_values,alpha=0.05, weights = False):
    '''
    Apply Bonferroni correction to a vector of p-values.

    Parameters:
        p_values (list or numpy array): Vector of original p-values.
        alpha: Threshold of significance
        weights: Whether or not to use weighted approach

    Returns:
        corrected_p_values(list): Vector of corrected p-values after Bonferroni correction.
    '''

    if weights == True:
        # Generate hypothesis weights (Need to find a procedure to assign weights)
        random_numbers = np.random.rand(len(p_values))
        weight = random_numbers / np.sum(np.abs(random_numbers)) 

        # Combining p-values and weights into a 2D array
        data = np.column_stack((p_values, weight))

        # Separating p-values and weights
        p_values = data[:, 0]
        weight = data[:, 1]

        # Applying weights to p-values (e.g., multiply by weights)
        weighted_p_values = p_values * weight
        T = sum(weighted_p_values)
        weighted_p_values = weighted_p_values / T
        
        p_values = weighted_p_values

        adj_p = [min(p * len(p_values), 1.0) for p in p_values]
        sig_index = []
        for index, p in enumerate(adj_p):
            if p < alpha:
                sig_index.append(index)

    else:
        # Apply Bonferroni correction to each raw p-value
        adj_p = [min(p * len(p_values), 1.0) for p in p_values]


        sig_index = []
        for index, p in enumerate(adj_p):
            if p < alpha:
                sig_index.append(index)
        
    return sig_index

#Overall significance
sig_indices = bonferroni(p_values,alpha=0.05, weights = False)
sig_p = [p_values[index] for index in sig_indices]
len(sig_p)
sig_p

# True Positives - firing
p_values = p_value_fire 
sig_indices = bonferroni(p_values,alpha=0.05, weights = False)
sig_p = [p_values[index] for index in sig_indices]
len(sig_p)
sig_p

'''
confusion_matrix = bonferroni(p_values,alpha=0.05, weights = False)[1]
power = bonferroni(p_values,alpha=0.05, weights = False)[2]
power

    significant_p_fire =  [p for p in p_value_fire if p < alpha]
    significant_p_nonfire =  [p for p in p_value_nonfire if p < threshold_bon]

    TP = len(significant_p_fire)
    FP = len(significant_p_nonfire)
    TN = num_nonfire - FP
    FN = num_firing - TP

    data = {
        'Actual Positive': [TP, FN],
        'Actual Negative': [FP, TN],
    }
    confusion_matrix = pd.DataFrame(data, index=['Predicted Positive', 'Predicted Negative'])

    significant_p = significant_p_nonfire + significant_p_fire

    sig_p= len(significant_p)
    power = TP/num_firing

'''