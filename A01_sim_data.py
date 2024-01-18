def simulation_01(seed,num_firing,num_nonfire,threshold=0.05,show_plot=False):
    '''
    This is from t-distribution & uniform
    '''
    import numpy as np
    import seaborn as sns
    import pandas as pd
    import statsmodels.api as sm
    from statsmodels.stats.multitest import multipletests
    import matplotlib.pyplot as plt

    ############################### Simulating t-test (independent samples) ########################################3
    np.random.seed(seed)

    #Control Group Distribution
    m0 = 0
    s0 = 1
    n0 = 100

    #Treatment Group Distribution
    m1 = 0.5
    s1 = 1
    n1 = 100

    p_value_fire = []
    p_value_nonfire = []

    for i in range(num_firing):
        control_group = np.random.normal(m0,s0,size =n0)
        treatment_group = np.random.normal(m1,s1,size=n1)
        p_value = sm.stats.ttest_ind(control_group, treatment_group)[1]
        p_value_fire.append(p_value)

    for i in range(num_nonfire):
        p_value2 = np.random.uniform(0,1)
        p_value_nonfire.append(p_value2)

    p_values = p_value_fire + p_value_nonfire
    #Getting Firing and Non-Firing Indices
    fire_index = [index for index,p in enumerate(p_values) if p_values[index] in p_value_fire]
    nonfire_index = [index for index,p in enumerate(p_values) if p_values[index] in p_value_nonfire]

    print(len(fire_index),len(nonfire_index))

    ################################### Evaluating the Simulation ##################################################
    
    #Pre-requisites
    p_values
    fire_index
    nonfire_index
    threshold

    #significant p values
    significant_p =  [p for p in p_values if p < threshold]
    sig_p= len(significant_p)

    significant_p_fire = [p_values[index] for index in fire_index if p_values[index] < threshold]
    significant_p_nonfire = [p_values[index] for index in nonfire_index if p_values[index] < threshold]

    TP = len(significant_p_fire)
    FP = len(significant_p_nonfire)
    TN = num_nonfire - FP
    FN = num_firing - TP

    data = {
        'Actual Positive': [TP, FN],
        'Actual Negative': [FP, TN],
    }
    confusion_matrix = pd.DataFrame(data, index=['Predicted Positive', 'Predicted Negative'])

    sig_p= len(significant_p)
    power = TP/num_firing
    
    #To find which genes are significant
    TP_index = [index for index in fire_index if p_values[index] < threshold]
    FN_index = [index for index in fire_index if p_values[index] >= threshold]
    FP_index = [index for index in nonfire_index if p_values[index] < threshold]
    TN_index = [index for index in nonfire_index if p_values[index] >= threshold]

    print("TP_index:",len(TP_index))
    print("FN_index:",len(FN_index))
    print("FP_index:",len(FP_index))
    print("TN_index:",len(TN_index))

    #Creating the plot
    hist_data = [p_value_fire, p_value_nonfire]
    plt.hist(hist_data, bins=30,alpha=0.5, label = ['firing','non-firing'],stacked=True)
    plt.title('Distribution of uncorrected p-values')
    plt.xlabel('p-value')
    plt.ylabel('Frequency')
    plt.legend()
    if show_plot:
        plt.show()

    return p_values, confusion_matrix, sig_p, power,fire_index, nonfire_index

#Simulating Dataset for 500 F and 9500 NF 
sim1 = simulation_01(42,500,9500,threshold=0.05,show_plot=False)
p_values, confusion_matrix,sig_p,power,fire_index,nonfire_index = sim1[0],sim1[1],sim1[2],sim1[3],sim1[4],sim1[5]
power
confusion_matrix

'''
#Simulating Dataset for 5000 F and 5000 NF 
sim2 = simulation_01(42,5000,5000,threshold=0.05,show_plot=True)
p_values, confusion_matrix,sig_p,power,fire_index,nonfire_index = sim1[0],sim1[1],sim1[2],sim1[3],sim1[4],sim1[5]
power
fire_index
'''