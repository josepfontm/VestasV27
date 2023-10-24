import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.ndimage import uniform_filter1d

def kl_divergence(regression_model: str, threshold:float):

    """
    Plot divergence between undamaged/repaired and damage scenarios

    Parameters
    ----------
    regression_type : string
        Name folder used to correct data.

    threshold: float
        Threshold selected, based on R2

    Returns
    ----------
    Plot
    """

    path_cdsf_idle_repaired = "../VestasV27/03regression_analysis/02correction/" + regression_model +'/t'+str(threshold)+ "/corrected_dsf_idle_train.csv"
    path_cdsf_parked_repaired = "../VestasV27/03regression_analysis/02correction/" + regression_model +'/t'+str(threshold)+ "/corrected_dsf_parked_train.csv"
    path_cdsf_t1_repaired = "../VestasV27/03regression_analysis/02correction/" + regression_model +'/t'+str(threshold)+ "/corrected_dsf_t1_train.csv"
    path_cdsf_rpm32_repaired = "../VestasV27/03regression_analysis/02correction/" + regression_model +'/t'+str(threshold)+ "/corrected_dsf_rpm32_train.csv"
    path_cdsf_t2_repaired = "../VestasV27/03regression_analysis/02correction/" + regression_model +'/t'+str(threshold)+ "/corrected_dsf_t2_train.csv"
    path_cdsf_rpm43_repaired = "../VestasV27/03regression_analysis/02correction/" + regression_model +'/t'+str(threshold)+ "/corrected_dsf_rpm43_train.csv"

    path_dsf_idle_repaired = "../VestasV27/03regression_analysis/00standarized/dsf_train_idle.csv"
    path_dsf_parked_repaired = "../VestasV27/03regression_analysis/00standarized/dsf_train_parked.csv"
    path_dsf_t1_repaired = "../VestasV27/03regression_analysis/00standarized/dsf_train_t1.csv"
    path_dsf_rpm32_repaired = "../VestasV27/03regression_analysis/00standarized/dsf_train_rpm32.csv"
    path_dsf_t2_repaired = "../VestasV27/03regression_analysis/00standarized/dsf_train_t2.csv"
    path_dsf_rpm43_repaired = "../VestasV27/03regression_analysis/00standarized/dsf_train_rpm43.csv"

    path_scenario_idle_repaired = "../VestasV27/03regression_analysis/00standarized/scenario_train_idle.csv"
    path_scenario_parked_repaired = "../VestasV27/03regression_analysis/00standarized/scenario_train_parked.csv"
    path_scenario_t1_repaired = "../VestasV27/03regression_analysis/00standarized/scenario_train_t1.csv"
    path_scenario_rpm32_repaired = "../VestasV27/03regression_analysis/00standarized/scenario_train_rpm32.csv"
    path_scenario_t2_repaired = "../VestasV27/03regression_analysis/00standarized/scenario_train_t2.csv"
    path_scenario_rpm43_repaired = "../VestasV27/03regression_analysis/00standarized/scenario_train_rpm43.csv"

    cdsf_idle_repaired = pd.read_csv(path_cdsf_idle_repaired,index_col=0,header=0,sep=';')
    cdsf_parked_repaired = pd.read_csv(path_cdsf_parked_repaired,index_col=0,header=0,sep=';')
    cdsf_t1_repaired = pd.read_csv(path_cdsf_t1_repaired,index_col=0,header=0,sep=';')
    cdsf_rpm32_repaired = pd.read_csv(path_cdsf_rpm32_repaired,index_col=0,header=0,sep=';')
    cdsf_t2_repaired = pd.read_csv(path_cdsf_t2_repaired,index_col=0,header=0,sep=';')
    cdsf_rpm43_repaired = pd.read_csv(path_cdsf_rpm43_repaired,index_col=0,header=0,sep=';')

    dsf_idle_repaired = pd.read_csv(path_dsf_idle_repaired,index_col=0,header=0,sep=';')
    dsf_parked_repaired = pd.read_csv(path_dsf_parked_repaired,index_col=0,header=0,sep=';')
    dsf_t1_repaired = pd.read_csv(path_dsf_t1_repaired,index_col=0,header=0,sep=';')
    dsf_rpm32_repaired = pd.read_csv(path_dsf_rpm32_repaired,index_col=0,header=0,sep=';')
    dsf_t2_repaired = pd.read_csv(path_dsf_t2_repaired,index_col=0,header=0,sep=';')
    dsf_rpm43_repaired = pd.read_csv(path_dsf_rpm43_repaired,index_col=0,header=0,sep=';')

    scenario_idle_repaired = pd.read_csv(path_scenario_idle_repaired,index_col=0,sep=';')
    scenario_parked_repaired = pd.read_csv(path_scenario_parked_repaired,index_col=0,sep=';')
    scenario_t1_repaired = pd.read_csv(path_scenario_t1_repaired,index_col=0,sep=';')
    scenario_rpm32_repaired = pd.read_csv(path_scenario_rpm32_repaired,index_col=0,sep=';')
    scenario_t2_repaired = pd.read_csv(path_scenario_t2_repaired,index_col=0,sep=';')
    scenario_rpm43_repaired = pd.read_csv(path_scenario_rpm43_repaired,index_col=0,sep=';')

    path_cdsf_idle = "../VestasV27/03regression_analysis/02correction/" + regression_model +'/t'+str(threshold)+ "/corrected_dsf_idle.csv"
    path_cdsf_parked = "../VestasV27/03regression_analysis/02correction/" + regression_model +'/t'+str(threshold)+ "/corrected_dsf_parked.csv"
    path_cdsf_t1 = "../VestasV27/03regression_analysis/02correction/" + regression_model +'/t'+str(threshold)+ "/corrected_dsf_t1.csv"
    path_cdsf_rpm32 = "../VestasV27/03regression_analysis/02correction/" + regression_model +'/t'+str(threshold)+ "/corrected_dsf_rpm32.csv"
    path_cdsf_t2 = "../VestasV27/03regression_analysis/02correction/" + regression_model +'/t'+str(threshold)+ "/corrected_dsf_t2.csv"
    path_cdsf_rpm43 = "../VestasV27/03regression_analysis/02correction/" + regression_model +'/t'+str(threshold)+ "/corrected_dsf_rpm43.csv"

    path_dsf_idle = "../VestasV27/03regression_analysis/00standarized/dsf_idle.csv"
    path_dsf_parked = "../VestasV27/03regression_analysis/00standarized/dsf_parked.csv"
    path_dsf_t1 = "../VestasV27/03regression_analysis/00standarized/dsf_t1.csv"
    path_dsf_rpm32 = "../VestasV27/03regression_analysis/00standarized/dsf_rpm32.csv"
    path_dsf_t2 = "../VestasV27/03regression_analysis/00standarized/dsf_t2.csv"
    path_dsf_rpm43 = "../VestasV27/03regression_analysis/00standarized/dsf_rpm43.csv"

    path_scenario_idle = "../VestasV27/03regression_analysis/00standarized/scenario_idle.csv"
    path_scenario_parked = "../VestasV27/03regression_analysis/00standarized/scenario_parked.csv"
    path_scenario_t1 = "../VestasV27/03regression_analysis/00standarized/scenario_t1.csv"
    path_scenario_rpm32 = "../VestasV27/03regression_analysis/00standarized/scenario_rpm32.csv"
    path_scenario_t2 = "../VestasV27/03regression_analysis/00standarized/scenario_t2.csv"
    path_scenario_rpm43 = "../VestasV27/03regression_analysis/00standarized/scenario_rpm43.csv"

    cdsf_idle = pd.read_csv(path_cdsf_idle,index_col=0,header=0,sep=';')
    cdsf_parked = pd.read_csv(path_cdsf_parked,index_col=0,header=0,sep=';')
    cdsf_t1 = pd.read_csv(path_cdsf_t1,index_col=0,header=0,sep=';')
    cdsf_rpm32 = pd.read_csv(path_cdsf_rpm32,index_col=0,header=0,sep=';')
    cdsf_t2 = pd.read_csv(path_cdsf_t2,index_col=0,header=0,sep=';')
    cdsf_rpm43 = pd.read_csv(path_cdsf_rpm43,index_col=0,header=0,sep=';')

    dsf_idle = pd.read_csv(path_dsf_idle,index_col=0,header=0,sep=';')
    dsf_parked = pd.read_csv(path_dsf_parked,index_col=0,header=0,sep=';')
    dsf_t1 = pd.read_csv(path_dsf_t1,index_col=0,header=0,sep=';')
    dsf_rpm32 = pd.read_csv(path_dsf_rpm32,index_col=0,header=0,sep=';')
    dsf_t2 = pd.read_csv(path_dsf_t2,index_col=0,header=0,sep=';')
    dsf_rpm43 = pd.read_csv(path_dsf_rpm43,index_col=0,header=0,sep=';')

    scenario_idle = pd.read_csv(path_scenario_idle,index_col=0,sep=';')
    scenario_parked = pd.read_csv(path_scenario_parked,index_col=0,sep=';')
    scenario_t1 = pd.read_csv(path_scenario_t1,index_col=0,sep=';')
    scenario_rpm32 = pd.read_csv(path_scenario_rpm32,index_col=0,sep=';')
    scenario_t2 = pd.read_csv(path_scenario_t2,index_col=0,sep=';')
    scenario_rpm43 = pd.read_csv(path_scenario_rpm43,index_col=0,sep=';')

    # print(dsf_parked_repaired.shape)
    # print(dsf_parked_repaired.iloc[:,0])

    corrected = []
    uncorrected = []

    N = 50

    for pc in range(dsf_idle.shape[1]):
        damage = dsf_parked.iloc[:,pc]
        reference = dsf_parked_repaired.iloc[:,pc]

        result_uncorrected = scipy.stats.ks_2samp(reference, damage, alternative='two-sided')
        uncorrected.append(result_uncorrected.statistic)

        damage = cdsf_parked.iloc[:,pc]
        reference = cdsf_parked_repaired.iloc[:,pc]

        result_corrected = scipy.stats.ks_2samp(reference, damage, alternative='two-sided')
        corrected.append(result_corrected.statistic)

    uncorrected = uniform_filter1d(uncorrected,N)
    corrected = uniform_filter1d(corrected,N)

    plt.figure()
    plt.plot(uncorrected,color='red')
    plt.plot(corrected,color='green')
    plt.xlim([0,719])
    plt.ylim([0,0.5])
    plt.show()

# kl_divergence('adaptive_order1',0.0)
# kl_divergence('adaptive_order1',0.3) 
# kl_divergence('adaptive_order1',0.6)

# kl_divergence('adaptive_order2',0.0)
# kl_divergence('adaptive_order2',0.3) 
# kl_divergence('adaptive_order2',0.6)

# kl_divergence('adaptive_order3',0.0)
# kl_divergence('adaptive_order3',0.3) 
# kl_divergence('adaptive_order3',0.6)

# kl_divergence('adaptive_order4',0.0)
# kl_divergence('adaptive_order4',0.3) 
# kl_divergence('adaptive_order4',0.6) 

# kl_divergence('adaptive_order5',0.0)
# kl_divergence('adaptive_order5',0.3) 
# kl_divergence('adaptive_order5',0.6) 

kl_divergence('adaptive_order6',0.0)
kl_divergence('adaptive_order6',0.3) 
kl_divergence('adaptive_order6',0.6) 