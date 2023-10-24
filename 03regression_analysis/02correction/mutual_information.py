import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def kl_plot(regression_model: str, threshold:float, n_pcs: int):

    """
    Show in a plot the distributions for healthy and damaged observations (on Y axis) of each PC-DSF (on X axis).

    Parameters
    ----------
    regression_type : string
        Name folder used to correct data.

    threshold: float
        DSFs are only corrected if the R^2 is higher or equal than the threshold defined.

    n_pcs: int
        Number of PCs to plot
    """

    path_cdsf_idle = "../VestasV27/03regression_analysis/02correction/" + regression_model +'/t'+str(threshold)+ "/corrected_dsf_idle.csv"
    path_cdsf_parked = "../VestasV27/03regression_analysis/02correction/" + regression_model +'/t'+str(threshold)+ "/corrected_dsf_parked.csv"
    path_cdsf_t1 = "../VestasV27/03regression_analysis/02correction/" + regression_model +'/t'+str(threshold)+ "/corrected_dsf_t1.csv"
    path_cdsf_rpm32 = "../VestasV27/03regression_analysis/02correction/" + regression_model +'/t'+str(threshold)+ "/corrected_dsf_rpm32.csv"
    path_cdsf_t2 = "../VestasV27/03regression_analysis/02correction/" + regression_model +'/t'+str(threshold)+ "/corrected_dsf_t2.csv"
    path_cdsf_rpm43 = "../VestasV27/03regression_analysis/02correction/" + regression_model +'/t'+str(threshold)+ "/corrected_dsf_rpm43.csv"

    cdsf_idle = pd.read_csv(path_cdsf_idle,index_col=0,header=0,sep=';')
    cdsf_parked = pd.read_csv(path_cdsf_parked,index_col=0,header=0,sep=';')
    cdsf_t1 = pd.read_csv(path_cdsf_t1,index_col=0,header=0,sep=';')
    cdsf_rpm32 = pd.read_csv(path_cdsf_rpm32,index_col=0,header=0,sep=';')
    cdsf_t2 = pd.read_csv(path_cdsf_t2,index_col=0,header=0,sep=';')
    cdsf_rpm43 = pd.read_csv(path_cdsf_rpm43,index_col=0,header=0,sep=';')

    path_scenario_idle = "../VestasV27/03regression_analysis/00standarized/scenario_idle.csv"
    path_scenario_parked = "../VestasV27/03regression_analysis/00standarized/scenario_parked.csv"
    path_scenario_t1 = "../VestasV27/03regression_analysis/00standarized/scenario_t1.csv"
    path_scenario_rpm32 = "../VestasV27/03regression_analysis/00standarized/scenario_rpm32.csv"
    path_scenario_t2 = "../VestasV27/03regression_analysis/00standarized/scenario_t2.csv"
    path_scenario_rpm43 = "../VestasV27/03regression_analysis/00standarized/scenario_rpm43.csv"

    scenario_idle = pd.read_csv(path_scenario_idle,index_col=0,header=0,sep=';')
    scenario_parked = pd.read_csv(path_scenario_parked,index_col=0,header=0,sep=';')
    scenario_t1 = pd.read_csv(path_scenario_t1,index_col=0,header=0,sep=';')
    scenario_rpm32 = pd.read_csv(path_scenario_rpm32,index_col=0,header=0,sep=';')
    scenario_t2 = pd.read_csv(path_scenario_t2,index_col=0,header=0,sep=';')
    scenario_rpm43 = pd.read_csv(path_scenario_rpm43,index_col=0,header=0,sep=';')

    path_dsf_idle = 'D:/VestasV27/03regression_analysis/00standarized/dsf_idle.csv'
    path_dsf_parked = 'D:/VestasV27/03regression_analysis/00standarized/dsf_parked.csv'
    path_dsf_t1 = 'D:/VestasV27/03regression_analysis/00standarized/dsf_t1.csv'
    path_dsf_rpm32 = 'D:/VestasV27/03regression_analysis/00standarized/dsf_rpm32.csv'
    path_dsf_t2 = 'D:/VestasV27/03regression_analysis/00standarized/dsf_t2.csv'
    path_dsf_rpm43 = 'D:/VestasV27/03regression_analysis/00standarized/dsf_rpm43.csv'

    dsf_idle = pd.read_csv(path_dsf_idle,index_col=0,sep=';')
    dsf_parked = pd.read_csv(path_dsf_parked,index_col=0,sep=';')
    dsf_t1 = pd.read_csv(path_dsf_t1,index_col=0,sep=';')
    dsf_rpm32 = pd.read_csv(path_dsf_rpm32,index_col=0,sep=';')
    dsf_t2 = pd.read_csv(path_dsf_t2,index_col=0,sep=';')
    dsf_rpm43 = pd.read_csv(path_dsf_rpm43,index_col=0,sep=';')

    #Print data for boxplots
    index_idle_healthy = scenario_idle[((scenario_idle.DamageScenario == 'undamaged') | (scenario_idle.DamageScenario == 'repaired'))].index
    index_idle_damaged = scenario_idle[((scenario_idle.DamageScenario == '15cm') | (scenario_idle.DamageScenario == '30cm') | (scenario_idle.DamageScenario == '45cm'))].index

    index_parked_healthy = scenario_parked[((scenario_parked.DamageScenario == 'undamaged') | (scenario_parked.DamageScenario == 'repaired'))].index
    index_parked_damaged = scenario_parked[((scenario_parked.DamageScenario == '15cm') | (scenario_parked.DamageScenario == '30cm') | (scenario_parked.DamageScenario == '45cm'))].index

    index_t1_healthy = scenario_t1[((scenario_t1.DamageScenario == 'undamaged') | (scenario_t1.DamageScenario == 'repaired'))].index
    index_t1_damaged = scenario_t1[((scenario_t1.DamageScenario == '15cm') | (scenario_t1.DamageScenario == '30cm') | (scenario_t1.DamageScenario == '45cm'))].index

    index_rpm32_healthy = scenario_rpm32[((scenario_rpm32.DamageScenario == 'undamaged') | (scenario_rpm32.DamageScenario == 'repaired'))].index
    index_rpm32_damaged = scenario_rpm32[((scenario_rpm32.DamageScenario == '15cm') | (scenario_rpm32.DamageScenario == '30cm') | (scenario_rpm32.DamageScenario == '45cm'))].index

    index_t2_healthy = scenario_t2[((scenario_t2.DamageScenario == 'undamaged') | (scenario_t2.DamageScenario == 'repaired'))].index
    index_t2_damaged = scenario_t2[((scenario_t2.DamageScenario == '15cm') | (scenario_t2.DamageScenario == '30cm') | (scenario_t2.DamageScenario == '45cm'))].index

    index_rpm43_healthy = scenario_rpm43[((scenario_rpm43.DamageScenario == 'undamaged') | (scenario_rpm43.DamageScenario == 'repaired'))].index
    index_rpm43_damaged = scenario_rpm43[((scenario_rpm43.DamageScenario == '15cm') | (scenario_rpm43.DamageScenario == '30cm') | (scenario_rpm43.DamageScenario == '45cm'))].index

    cdsf_idle_healthy = cdsf_idle.loc[list(index_idle_healthy)]
    dsf_idle_healthy = dsf_idle.loc[list(index_idle_healthy)]
    cdsf_idle_damaged = cdsf_idle.loc[list(index_idle_damaged)]
    dsf_idle_damaged = dsf_idle.loc[list(index_idle_damaged)]

    print('Idle')
    print(scenario_idle.DamageScenario)
    print(np.shape(cdsf_idle_healthy))
    print(np.shape(dsf_idle_healthy))
    print(np.shape(cdsf_idle_damaged))
    print(np.shape(dsf_idle_damaged))

    cdsf_parked_healthy = cdsf_parked.loc[list(index_parked_healthy)]
    dsf_parked_healthy = dsf_parked.loc[list(index_parked_healthy)]
    cdsf_parked_damaged = cdsf_parked.loc[list(index_parked_damaged)]
    dsf_parked_damaged = dsf_parked.loc[list(index_parked_damaged)]

    print('Parked')
    print(scenario_parked.DamageScenario)
    print(np.shape(cdsf_parked_healthy))
    print(np.shape(dsf_parked_healthy))
    print(np.shape(cdsf_parked_damaged))
    print(np.shape(dsf_parked_damaged))

    cdsf_t1_healthy = cdsf_t1.loc[list(index_t1_healthy)]
    dsf_t1_healthy = dsf_t1.loc[list(index_t1_healthy)]
    cdsf_t1_damaged = cdsf_t1.loc[list(index_t1_damaged)]
    dsf_t1_damaged = dsf_t1.loc[list(index_t1_damaged)]

    cdsf_rpm32_healthy = cdsf_rpm32.loc[list(index_rpm32_healthy)]
    dsf_rpm32_healthy = dsf_rpm32.loc[list(index_rpm32_healthy)]
    cdsf_rpm32_damaged = cdsf_rpm32.loc[list(index_rpm32_damaged)]
    dsf_rpm32_damaged = dsf_rpm32.loc[list(index_rpm32_damaged)]

    cdsf_t2_healthy = cdsf_t2.loc[list(index_t2_healthy)]
    dsf_t2_healthy = dsf_t2.loc[list(index_t2_healthy)]
    cdsf_t2_damaged = cdsf_t2.loc[list(index_t2_damaged)]
    dsf_t2_damaged = dsf_t2.loc[list(index_t2_damaged)]

    # cdsf_rpm43_healthy = cdsf_rpm43.loc[list(index_rpm43_healthy)]
    # dsf_rpm43_healthy = dsf_rpm43.loc[list(index_rpm43_healthy)]
    # cdsf_rpm43_damaged = cdsf_rpm43.loc[index_rpm43_damaged]
    # dsf_rpm43_damaged = dsf_rpm43.loc[list(index_rpm43_damaged)]

    #Merge uncorrected and corrected DSFs
    #IDLE
    dsf_idle_healthy['correction'] = 'uncorrected'
    cdsf_idle_healthy['correction'] = 'corrected'
    dsf_idle_damaged['correction'] = 'uncorrected'
    cdsf_idle_damaged['correction'] = 'corrected'

    idle_healthy = pd.concat([dsf_idle_healthy, cdsf_idle_healthy])
    idle_healthy = pd.melt(idle_healthy, id_vars='correction')
    idle_damaged = pd.concat([dsf_idle_damaged, cdsf_idle_damaged])
    idle_damaged = pd.melt(idle_damaged, id_vars='correction')

    # plt.figure()
    # sns.boxplot(data=idle_healthy,x='variable',y='value', hue='correction',palette = {"uncorrected":"red", "corrected":"green"})
    # sns.boxplot(data=idle_damaged,x='variable',y='value', hue='correction',palette = {"uncorrected":"red", "corrected":"green"})
    # plt.show()

    #PARKED
    print(dsf_parked_healthy)
    dsf_parked_healthy['correction'] = 'uncorrected'
    cdsf_parked_healthy['correction'] = 'corrected'
    dsf_parked_damaged['correction'] = 'uncorrected'
    cdsf_parked_damaged['correction'] = 'corrected'

    parked_healthy = pd.concat([dsf_parked_healthy, cdsf_parked_healthy])
    parked_healthy = pd.melt(parked_healthy, id_vars='correction')
    parked_damaged = pd.concat([dsf_parked_damaged, cdsf_parked_damaged])
    parked_damaged = pd.melt(parked_damaged, id_vars='correction')

    plt.figure()
    sns.boxplot(data=parked_healthy,x='variable',y='value', hue='correction',palette = {"uncorrected":"red", "corrected":"green"})

    plt.figure()
    sns.boxplot(data=parked_damaged,x='variable',y='value', hue='correction',palette = {"uncorrected":"red", "corrected":"green"})
    plt.show()

    #T1
    dsf_t1_healthy['correction'] = 'uncorrected'
    cdsf_t1_healthy['correction'] = 'corrected'
    dsf_t1_damaged['correction'] = 'uncorrected'
    cdsf_t1_damaged['correction'] = 'corrected'

    t1_healthy = pd.concat([dsf_t1_healthy, cdsf_t1_healthy])
    t1_healthy = pd.melt(t1_healthy, id_vars='correction')
    t1_damaged = pd.concat([dsf_t1_damaged, cdsf_t1_damaged])
    t1_damaged = pd.melt(t1_damaged, id_vars='correction')

    plt.figure()
    sns.boxplot(data=t1_healthy,x='variable',y='value', hue='correction',palette = {"uncorrected":"red", "corrected":"green"})
    sns.boxplot(data=t1_damaged,x='variable',y='value', hue='correction',palette = {"uncorrected":"red", "corrected":"green"})
    plt.show()

    #RPM32
    dsf_rpm32_healthy['correction'] = 'uncorrected'
    cdsf_rpm32_healthy['correction'] = 'corrected'
    dsf_rpm32_damaged['correction'] = 'uncorrected'
    cdsf_rpm32_damaged['correction'] = 'corrected'

    rpm32_healthy = pd.concat([dsf_rpm32_healthy, cdsf_rpm32_healthy])
    rpm32_healthy = pd.melt(rpm32_healthy, id_vars='correction')
    rpm32_damaged = pd.concat([dsf_rpm32_damaged, cdsf_rpm32_damaged])
    rpm32_damaged = pd.melt(rpm32_damaged, id_vars='correction')

    plt.figure()
    sns.boxplot(data=rpm32_healthy,x='variable',y='value', hue='correction',palette = {"uncorrected":"red", "corrected":"green"})
    sns.boxplot(data=rpm32_damaged,x='variable',y='value', hue='correction',palette = {"uncorrected":"red", "corrected":"green"})
    plt.show()

    #T2
    dsf_t2_healthy['correction'] = 'uncorrected'
    cdsf_t2_healthy['correction'] = 'corrected'
    dsf_t2_damaged['correction'] = 'uncorrected'
    cdsf_t2_damaged['correction'] = 'corrected'

    t2_healthy = pd.concat([dsf_t2_healthy, cdsf_t2_healthy])
    t2_healthy = pd.melt(t2_healthy, id_vars='correction')
    t2_damaged = pd.concat([dsf_t2_damaged, cdsf_t2_damaged])
    t2_damaged = pd.melt(t2_damaged, id_vars='correction')

    plt.figure()
    # sns.boxplot(data=t2_healthy,x='variable',y='value', hue='correction',palette = {"uncorrected":"red", "corrected":"green"})
    sns.boxplot(data=t2_damaged,x='variable',y='value', hue='correction',palette = {"uncorrected":"red", "corrected":"green"})
    plt.show()

    #RPM43
    # dsf_rpm43_healthy['correction'] = 'uncorrected'
    # cdsf_rpm43_healthy['correction'] = 'corrected'
    # dsf_rpm43_damaged['correction'] = 'uncorrected'
    # cdsf_rpm43_damaged['correction'] = 'corrected'

    # rpm43_healthy = pd.concat([dsf_rpm43_healthy, cdsf_rpm43_healthy])
    # rpm43_healthy = pd.melt(rpm43_healthy, id_vars='correction')
    # rpm43_damaged = pd.concat([dsf_rpm43_damaged, cdsf_rpm43_damaged])
    # rpm43_damaged = pd.melt(rpm43_damaged, id_vars='correction')

    # plt.figure()
    # sns.boxplot(data=rpm43_healthy,x='variable',y='value', hue='correction',palette = {"uncorrected":"red", "corrected":"green"})
    # sns.boxplot(data=rpm43_damaged,x='variable',y='value', hue='correction',palette = {"uncorrected":"red", "corrected":"green"})
    # plt.show()

kl_plot('adaptive_order',0.2,30)