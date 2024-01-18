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

def sidak(p_values,alpha=0.05, weights = False):
    '''
    Sidak procedure
    '''
    reject_null,corrected_p_values,_,_ = multipletests(p_values,alpha=alpha,method = 'sidak')
    sidak_p = corrected_p_values
    threshold_sidak = 1 - (1 - alpha) ** (1 / len(p_values))
    #threshold_sidak = alpha
    ###

    significant_p_fire =  [p for p in p_value_fire if p < threshold_sidak]
    significant_p_nonfire =  [p for p in p_value_nonfire if p < threshold_sidak]

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

        significant_p =  [p for p in weighted_p_values if p < threshold_sidak]
        sig_p= len(significant_p)

    else:
        significant_p =  [p for p in p_values if p < threshold_sidak]
        sig_p= len(significant_p)
        
    return sig_p, confusion_matrix, power


sidak(p_values,alpha=0.05, weights = True)

confusion_matrix = sidak(p_values,alpha=0.05, weights = False)[1]
power = sidak(p_values,alpha=0.05, weights = False)[2]
power
