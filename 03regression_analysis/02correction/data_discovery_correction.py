import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import HuberRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def plot_correction(order: int, threshold:float, pc_env: int, dsf: int):
    
    """
    Plot correction of DSFs based on a specific kind of regression and threshold selected.
    
    Parameters
    ----------
    order : integer
        The order of the polynomial regression, if 1 lineal regression.

    threshold: float
        DSFs are only corrected if the R^2 is higher than the threshold defined.
        If DSFs has high correlation with multiple PCs (Env and Op) the highest is used to correct DSF.

    pc_env : int
        Select which PC-ENV to plot

    dsf : int
        Select which DSF to plot

    Returns
    ----------
    corrected_dsf: array
        Array with corrected dsf's based on environmental and operational data.
    """

    #Change paths if necessary
    path_scenario_idle = '../code_6modes/03regression_analysis/00outliers_removed/scenario_idle.csv'
    path_scenario_parked = '../code_6modes/03regression_analysis/00outliers_removed/scenario_parked.csv'
    path_scenario_t1 = '../code_6modes/03regression_analysis/00outliers_removed/scenario_t1.csv'
    path_scenario_rpm32 = '../code_6modes/03regression_analysis/00outliers_removed/scenario_rpm32.csv'
    path_scenario_t2 = '../code_6modes/03regression_analysis/00outliers_removed/scenario_t2.csv'
    path_scenario_rpm43 = '../code_6modes/03regression_analysis/00outliers_removed/scenario_rpm43.csv'

    scenario_idle = pd.read_csv(path_scenario_idle, index_col=0,sep=';')
    scenario_parked = pd.read_csv(path_scenario_parked, index_col=0,sep=';')
    scenario_t1 = pd.read_csv(path_scenario_t1, index_col=0,sep=';')
    scenario_rpm32 = pd.read_csv(path_scenario_rpm32, index_col=0,sep=';')
    scenario_t2 = pd.read_csv(path_scenario_t2, index_col=0,sep=';')
    scenario_rpm43 = pd.read_csv(path_scenario_rpm43, index_col=0,sep=';')

    #Load paths for pc's environemntal data
    path_pca_env_idle = '../code_6modes/03regression_analysis/00outliers_removed/pca_env_idle.csv'
    path_pca_env_parked = '../code_6modes/03regression_analysis/00outliers_removed/pca_env_parked.csv'
    path_pca_env_t1 = '../code_6modes/03regression_analysis/00outliers_removed/pca_env_t1.csv'
    path_pca_env_rpm32 = '../code_6modes/03regression_analysis/00outliers_removed/pca_env_rpm32.csv'
    path_pca_env_t2 = '../code_6modes/03regression_analysis/00outliers_removed/pca_env_t2.csv'
    path_pca_env_rpm43 = '../code_6modes/03regression_analysis/00outliers_removed/pca_env_rpm43.csv'

    pca_env_idle = pd.read_csv(path_pca_env_idle,index_col=0,sep=';')
    pca_env_parked = pd.read_csv(path_pca_env_parked,index_col=0,sep=';')
    pca_env_t1 = pd.read_csv(path_pca_env_t1,index_col=0,sep=';')
    pca_env_rpm32 = pd.read_csv(path_pca_env_rpm32,index_col=0,sep=';')
    pca_env_t2 = pd.read_csv(path_pca_env_t2,index_col=0,sep=';')
    pca_env_rpm43 = pd.read_csv(path_pca_env_rpm43,index_col=0,sep=';')

    path_pca_env_train_idle = '../code_6modes/03regression_analysis/00outliers_removed/pca_env_train_idle.csv'
    path_pca_env_train_parked = '../code_6modes/03regression_analysis/00outliers_removed/pca_env_train_parked.csv'
    path_pca_env_train_t1 = '../code_6modes/03regression_analysis/00outliers_removed/pca_env_train_t1.csv'
    path_pca_env_train_rpm32 = '../code_6modes/03regression_analysis/00outliers_removed/pca_env_train_rpm32.csv'
    path_pca_env_train_t2 = '../code_6modes/03regression_analysis/00outliers_removed/pca_env_train_t2.csv'
    path_pca_env_train_rpm43 = '../code_6modes/03regression_analysis/00outliers_removed/pca_env_train_rpm43.csv'

    pca_env_train_idle = pd.read_csv(path_pca_env_train_idle,index_col=0,sep=';')
    pca_env_train_parked = pd.read_csv(path_pca_env_train_parked,index_col=0,sep=';')
    pca_env_train_t1 = pd.read_csv(path_pca_env_train_t1,index_col=0,sep=';')
    pca_env_train_rpm32 = pd.read_csv(path_pca_env_train_rpm32,index_col=0,sep=';')
    pca_env_train_t2 = pd.read_csv(path_pca_env_train_t2,index_col=0,sep=';')
    pca_env_train_rpm43 = pd.read_csv(path_pca_env_train_rpm43,index_col=0,sep=';')

    #Load paths for pc's operational data
    path_pca_op_idle = '../code_6modes/03regression_analysis/00outliers_removed/pca_op_idle.csv'
    path_pca_op_parked = '../code_6modes/03regression_analysis/00outliers_removed/pca_op_parked.csv'
    path_pca_op_t1 = '../code_6modes/03regression_analysis/00outliers_removed/pca_op_t1.csv'
    path_pca_op_rpm32 = '../code_6modes/03regression_analysis/00outliers_removed/pca_op_rpm32.csv'
    path_pca_op_t2 = '../code_6modes/03regression_analysis/00outliers_removed/pca_op_t2.csv'
    path_pca_op_rpm43 = '../code_6modes/03regression_analysis/00outliers_removed/pca_op_rpm43.csv'

    path_pca_op_train_idle = '../code_6modes/03regression_analysis/00outliers_removed/pca_op_train_idle.csv'
    path_pca_op_train_parked = '../code_6modes/03regression_analysis/00outliers_removed/pca_op_train_parked.csv'
    path_pca_op_train_t1 = '../code_6modes/03regression_analysis/00outliers_removed/pca_op_train_t1.csv'
    path_pca_op_train_rpm32 = '../code_6modes/03regression_analysis/00outliers_removed/pca_op_train_rpm32.csv'
    path_pca_op_train_t2 = '../code_6modes/03regression_analysis/00outliers_removed/pca_op_train_t2.csv'
    path_pca_op_train_rpm43 = '../code_6modes/03regression_analysis/00outliers_removed/pca_op_train_rpm43.csv'

    pca_op_idle = pd.read_csv(path_pca_op_idle,index_col=0,sep=';')
    pca_op_parked = pd.read_csv(path_pca_op_parked,index_col=0,sep=';')
    pca_op_t1 = pd.read_csv(path_pca_op_t1,index_col=0,sep=';')
    pca_op_rpm32 = pd.read_csv(path_pca_op_rpm32,index_col=0,sep=';')
    pca_op_t2 = pd.read_csv(path_pca_op_t2,index_col=0,sep=';')
    pca_op_rpm43 = pd.read_csv(path_pca_op_rpm43,index_col=0,sep=';')

    pca_op_train_idle = pd.read_csv(path_pca_op_train_idle,index_col=0,sep=';')
    pca_op_train_parked = pd.read_csv(path_pca_op_train_parked,index_col=0,sep=';')
    pca_op_train_t1 = pd.read_csv(path_pca_op_train_t1,index_col=0,sep=';')
    pca_op_train_rpm32 = pd.read_csv(path_pca_op_train_rpm32,index_col=0,sep=';')
    pca_op_train_t2 = pd.read_csv(path_pca_op_train_t2,index_col=0,sep=';')
    pca_op_train_rpm43 = pd.read_csv(path_pca_op_train_rpm43,index_col=0,sep=';') 

    #Merge PCs environmental and operational data
    pca_idle = pd.concat([pca_env_idle, pca_op_idle], axis=1)
    pca_parked = pd.concat([pca_env_parked, pca_op_parked], axis=1)
    pca_t1 = pd.concat([pca_env_t1, pca_op_t1], axis=1)
    pca_rpm32 = pd.concat([pca_env_rpm32, pca_op_rpm32], axis=1)
    pca_t2 = pd.concat([pca_env_t2,pca_op_t2],axis=1)
    pca_rpm43 = pd.concat([pca_env_rpm43, pca_op_rpm43], axis = 1)

    pca_train_idle = pd.concat([pca_env_train_idle,pca_op_train_idle],axis=1)
    pca_train_parked = pd.concat([pca_env_train_parked,pca_op_train_parked],axis=1)
    pca_train_t1 = pd.concat([pca_env_train_t1,pca_op_train_t1],axis=1)
    pca_train_rpm32 = pd.concat([pca_env_train_rpm32,pca_op_train_rpm32],axis=1)
    pca_train_t2 = pd.concat([pca_env_train_t2,pca_op_train_t2],axis=1)
    pca_train_rpm43 = pd.concat([pca_env_train_rpm43,pca_op_train_rpm43],axis=1)

    #Paths damage-sensitive features
    path_dsf_idle = '../code_6modes/03regression_analysis/00outliers_removed/dsf_idle.csv'
    path_dsf_parked = '../code_6modes/03regression_analysis/00outliers_removed/dsf_parked.csv'
    path_dsf_t1 = '../code_6modes/03regression_analysis/00outliers_removed/dsf_t1.csv'
    path_dsf_rpm32 = '../code_6modes/03regression_analysis/00outliers_removed/dsf_rpm32.csv'
    path_dsf_t2 = '../code_6modes/03regression_analysis/00outliers_removed/dsf_t2.csv'
    path_dsf_rpm43 = '../code_6modes/03regression_analysis/00outliers_removed/dsf_rpm43.csv'

    dsf_idle = pd.read_csv(path_dsf_idle,index_col=0,sep=';')
    dsf_parked = pd.read_csv(path_dsf_parked,index_col=0,sep=';')
    dsf_t1 = pd.read_csv(path_dsf_t1,index_col=0,sep=';')
    dsf_rpm32 = pd.read_csv(path_dsf_rpm32,index_col=0,sep=';')
    dsf_t2 = pd.read_csv(path_dsf_t2,index_col=0,sep=';')
    dsf_rpm43 = pd.read_csv(path_dsf_rpm43,index_col=0,sep=';')

    path_dsf_train_idle = '../code_6modes/03regression_analysis/00outliers_removed/dsf_train_idle.csv'
    path_dsf_train_parked = '../code_6modes/03regression_analysis/00outliers_removed/dsf_train_parked.csv'
    path_dsf_train_t1 = '../code_6modes/03regression_analysis/00outliers_removed/dsf_train_t1.csv'
    path_dsf_train_rpm32 = '../code_6modes/03regression_analysis/00outliers_removed/dsf_train_rpm32.csv'
    path_dsf_train_t2 = '../code_6modes/03regression_analysis/00outliers_removed/dsf_train_t2.csv'
    path_dsf_train_rpm43 = '../code_6modes/03regression_analysis/00outliers_removed/dsf_train_rpm43.csv'

    dsf_train_idle = pd.read_csv(path_dsf_train_idle,index_col=0,sep=';')
    dsf_train_parked = pd.read_csv(path_dsf_train_parked,index_col=0,sep=';')
    dsf_train_t1 = pd.read_csv(path_dsf_train_t1,index_col=0,sep=';')
    dsf_train_rpm32 = pd.read_csv(path_dsf_train_rpm32,index_col=0,sep=';')
    dsf_train_t2 = pd.read_csv(path_dsf_train_t2,index_col=0,sep=';')
    dsf_train_rpm43 = pd.read_csv(path_dsf_train_rpm43,index_col=0,sep=';')

    path_corrected_dsf_idle = '../code_6modes/03regression_analysis/02correction/poly_order'+str(order)+'/t'+str(threshold)+'/corrected_dsf_idle.csv'
    path_corrected_dsf_parked = '../code_6modes/03regression_analysis/02correction/poly_order'+str(order)+'/t'+str(threshold)+'/corrected_dsf_parked.csv'
    path_corrected_dsf_t1 = '../code_6modes/03regression_analysis/02correction/poly_order'+str(order)+'/t'+str(threshold)+'/corrected_dsf_t1.csv'
    path_corrected_dsf_rpm32 = '../code_6modes/03regression_analysis/02correction/poly_order'+str(order)+'/t'+str(threshold)+'/corrected_dsf_rpm32.csv'
    path_corrected_dsf_t2 = '../code_6modes/03regression_analysis/02correction/poly_order'+str(order)+'/t'+str(threshold)+'/corrected_dsf_t2.csv'
    path_corrected_dsf_rpm43 = '../code_6modes/03regression_analysis/02correction/poly_order'+str(order)+'/t'+str(threshold)+'/corrected_dsf_rpm43.csv'

    corrected_dsf_idle = pd.read_csv(path_corrected_dsf_idle,index_col=0,sep=';')
    corrected_dsf_parked = pd.read_csv(path_corrected_dsf_parked,index_col=0,sep=';')
    corrected_dsf_t1 = pd.read_csv(path_corrected_dsf_t1,index_col=0,sep=';')
    corrected_dsf_rpm32 = pd.read_csv(path_corrected_dsf_rpm32,index_col=0,sep=';')
    corrected_dsf_t2 = pd.read_csv(path_corrected_dsf_t2,index_col=0,sep=';')
    corrected_dsf_rpm43 = pd.read_csv(path_corrected_dsf_rpm43,index_col=0,sep=';')

    path_corrected_dsf_train_idle = '../code_6modes/03regression_analysis/02correction/poly_order'+str(order)+'/t'+str(threshold)+'/corrected_dsf_idle_train.csv'
    path_corrected_dsf_train_parked = '../code_6modes/03regression_analysis/02correction/poly_order'+str(order)+'/t'+str(threshold)+'/corrected_dsf_parked_train.csv'
    path_corrected_dsf_train_t1 = '../code_6modes/03regression_analysis/02correction/poly_order'+str(order)+'/t'+str(threshold)+'/corrected_dsf_t1_train.csv'
    path_corrected_dsf_train_rpm32 = '../code_6modes/03regression_analysis/02correction/poly_order'+str(order)+'/t'+str(threshold)+'/corrected_dsf_rpm32_train.csv'
    path_corrected_dsf_train_t2 = '../code_6modes/03regression_analysis/02correction/poly_order'+str(order)+'/t'+str(threshold)+'/corrected_dsf_t2_train.csv'
    path_corrected_dsf_train_rpm43 = '../code_6modes/03regression_analysis/02correction/poly_order'+str(order)+'/t'+str(threshold)+'/corrected_dsf_rpm43_train.csv'

    corrected_dsf_train_idle = pd.read_csv(path_corrected_dsf_train_idle,index_col=0,sep=';')
    corrected_dsf_train_parked = pd.read_csv(path_corrected_dsf_train_parked,index_col=0,sep=';')
    corrected_dsf_train_t1 = pd.read_csv(path_corrected_dsf_train_t1,index_col=0,sep=';')
    corrected_dsf_train_rpm32 = pd.read_csv(path_corrected_dsf_train_rpm32,index_col=0,sep=';')
    corrected_dsf_train_t2 = pd.read_csv(path_corrected_dsf_train_t2,index_col=0,sep=';')
    corrected_dsf_train_rpm43 = pd.read_csv(path_corrected_dsf_train_rpm43,index_col=0,sep=';')

    colors_uncorrected = {'undamaged':'green',
                            '15cm':'yellow',
                            '30cm':'orange',
                            '45cm':'red',
                            'repaired':'blue'}
    
    colors_corrected = {'undamaged':'green',
                            '15cm':'yellow',
                            '30cm':'orange',
                            '45cm':'red',
                            'repaired':'blue'}

    s = 5 #Size dots scatter

    poly = PolynomialFeatures(degree=order, include_bias=False)

    #Parked
    plt.figure('Train data (Parked)')
    plt.title('Train data (Parked)')
    x_train = pca_env_train_parked.iloc[:,pc_env]
    y_train = dsf_train_parked.iloc[:,dsf]

    plt.scatter(x_train,y_train,color='blue',s=s)
    plt.xlabel('PC-EOV'+str(pc_env))
    plt.ylabel('DSF'+str(dsf))

    plt.figure('Test data (Parked)')
    plt.title('Test data (Parked)')
    x = pca_env_parked.iloc[:,pc_env]
    y = dsf_parked.iloc[:,dsf]
    y_corrected = corrected_dsf_parked.iloc[:,dsf]

    plt.scatter(x,y,c=scenario_parked['DamageScenario'].map(colors_uncorrected),s=s,alpha=0.1)
    plt.scatter(x,y_corrected,c=scenario_parked['DamageScenario'].map(colors_corrected),s=s)
    plt.xlabel('PC-EOV'+str(pc_env))
    plt.ylabel('DSF'+str(dsf))

    print(np.sum(y-y_corrected))

    #Idle
    plt.figure('Train data (Idle)')
    plt.title('Train data (Idle)')
    x_train = pca_env_train_idle.iloc[:,pc_env]
    y_train = dsf_train_idle.iloc[:,dsf]

    plt.scatter(x_train,y_train,color='blue',s=s)
    plt.xlabel('PC-EOV'+str(pc_env))
    plt.ylabel('DSF'+str(dsf))

    plt.figure('Test data (Idle)')
    plt.title('Test data (Idle)')
    x = pca_env_idle.iloc[:,pc_env]
    y = dsf_idle.iloc[:,dsf]
    y_corrected = corrected_dsf_idle.iloc[:,dsf]

    plt.scatter(x,y,c=scenario_idle['DamageScenario'].map(colors_uncorrected),s=s,alpha=0.1)
    plt.scatter(x,y_corrected,c=scenario_idle['DamageScenario'].map(colors_corrected),s=s)
    plt.xlabel('PC-EOV'+str(pc_env))
    plt.ylabel('DSF'+str(dsf))

    print(np.sum(y-y_corrected))

    #T1 (Transient)
    plt.figure('Train data (T1)')
    plt.title('Train data (T1)')
    x_train = pca_env_train_t1.iloc[:,pc_env]
    y_train = dsf_train_t1.iloc[:,dsf]

    plt.scatter(x_train,y_train,color='blue',s=s)
    plt.xlabel('PC-EOV'+str(pc_env))
    plt.ylabel('DSF'+str(dsf))

    plt.figure('Test data (T1)')
    plt.title('Test data (T1)')
    x = pca_env_t1.iloc[:,pc_env]
    y = dsf_t1.iloc[:,dsf]
    y_corrected = corrected_dsf_t1.iloc[:,dsf]

    plt.scatter(x,y,c=scenario_t1['DamageScenario'].map(colors_uncorrected),s=s,alpha=0.1)
    plt.scatter(x,y_corrected,c=scenario_t1['DamageScenario'].map(colors_corrected),s=s)
    plt.xlabel('PC-EOV'+str(pc_env))
    plt.ylabel('DSF'+str(dsf))

    print(np.sum(y-y_corrected))

    #RPM32
    plt.figure('Train data (RPM32)')
    plt.title('Train data (RPM32)')
    x_train = pca_env_train_rpm32.iloc[:,pc_env]
    y_train = dsf_train_rpm32.iloc[:,dsf]

    plt.scatter(x_train,y_train,color='blue',s=s)
    plt.xlabel('PC-EOV'+str(pc_env))
    plt.ylabel('DSF'+str(dsf))

    plt.figure('Test data (RPM32)')
    plt.title('Test data (RPM32)')
    x = pca_env_rpm32.iloc[:,pc_env]
    y = dsf_rpm32.iloc[:,dsf]
    y_corrected = corrected_dsf_rpm32.iloc[:,dsf]

    plt.scatter(x,y,c=scenario_rpm32['DamageScenario'].map(colors_uncorrected),s=s,alpha=0.1)
    plt.scatter(x,y_corrected,c=scenario_rpm32['DamageScenario'].map(colors_corrected),s=s)
    plt.xlabel('PC-EOV'+str(pc_env))
    plt.ylabel('DSF'+str(dsf))

    print(np.sum(y-y_corrected))

    #T2
    plt.figure('Train data (T2)')
    plt.title('Train data (T2)')
    x_train = pca_env_train_t2.iloc[:,pc_env]
    y_train = dsf_train_t2.iloc[:,dsf]

    plt.scatter(x_train,y_train,color='blue',s=s)
    plt.xlabel('PC-EOV'+str(pc_env))
    plt.ylabel('DSF'+str(dsf))

    plt.figure('Test data (T2)')
    plt.title('Test data (T2)')
    x = pca_env_t2.iloc[:,pc_env]
    y = dsf_t2.iloc[:,dsf]
    y_corrected = corrected_dsf_t2.iloc[:,dsf]

    plt.scatter(x,y,c=scenario_t2['DamageScenario'].map(colors_uncorrected),s=s,alpha=0.1)
    plt.scatter(x,y_corrected,c=scenario_t2['DamageScenario'].map(colors_corrected),s=s)
    plt.xlabel('PC-EOV'+str(pc_env))
    plt.ylabel('DSF'+str(dsf))

    print(np.sum(y-y_corrected))

    #RPM43
    plt.figure('Train data (RPM43)')
    plt.title('Train data (RPM43)')
    x_train = pca_env_train_rpm43.iloc[:,pc_env]
    y_train = dsf_train_rpm43.iloc[:,dsf]

    plt.scatter(x_train,y_train,color='blue',s=s)
    plt.xlabel('PC-EOV'+str(pc_env))
    plt.ylabel('DSF'+str(dsf))

    plt.figure('Test data (RPM43)')
    plt.title('Test data (RPM43)')
    x = pca_env_rpm43.iloc[:,pc_env]
    y = dsf_rpm43.iloc[:,dsf]
    y_corrected = corrected_dsf_rpm43.iloc[:,dsf]

    plt.scatter(x,y,c=scenario_rpm43['DamageScenario'].map(colors_uncorrected),s=s,alpha=0.1)
    plt.scatter(x,y_corrected,c=scenario_rpm43['DamageScenario'].map(colors_corrected),s=s)
    plt.xlabel('PC-EOV'+str(pc_env))
    plt.ylabel('DSF'+str(dsf))

    print(np.sum(y-y_corrected))

    plt.show()

    print('done')

plot_correction(order=2, threshold=0.1, pc_env=0, dsf=0)