import pandas as pd
import numpy as np
from sklearn.metrics import f1_score,confusion_matrix
import matplotlib.pyplot as plt
from datetime import datetime

def control_chart(regression_model: str, threshold:float):

    """
    Plot control chart.

    Parameters
    ----------
    regression_type : string
        Name folder used to correct data.

    threshold: float
        DSFs are only corrected if the R^2 is higher or equal than the threshold defined.

    """

    path_cdsf_idle = "../VestasV27/03regression_analysis/02correction/" + regression_model +'/t'+str(threshold)+ "/corrected_dsf_idle.csv"
    path_cdsf_parked = "../VestasV27/03regression_analysis/02correction/" + regression_model +'/t'+str(threshold)+ "/corrected_dsf_parked.csv"
    path_cdsf_t1 = "../VestasV27/03regression_analysis/02correction/" + regression_model +'/t'+str(threshold)+ "/corrected_dsf_t1.csv"
    path_cdsf_rpm32 = "../VestasV27/03regression_analysis/02correction/" + regression_model +'/t'+str(threshold)+ "/corrected_dsf_rpm32.csv"
    path_cdsf_t2 = "../VestasV27/03regression_analysis/02correction/" + regression_model +'/t'+str(threshold)+ "/corrected_dsf_t2.csv"
    path_cdsf_rpm43 = "../VestasV27/03regression_analysis/02correction/" + regression_model +'/t'+str(threshold)+ "/corrected_dsf_rpm43.csv"

    path_cdsf_idle_train = "../VestasV27/03regression_analysis/02correction/" + regression_model +'/t'+str(threshold)+ "/corrected_dsf_idle_train.csv"
    path_cdsf_parked_train = "../VestasV27/03regression_analysis/02correction/" + regression_model +'/t'+str(threshold)+ "/corrected_dsf_parked_train.csv"
    path_cdsf_t1_train = "../VestasV27/03regression_analysis/02correction/" + regression_model +'/t'+str(threshold)+ "/corrected_dsf_t1_train.csv"
    path_cdsf_rpm32_train = "../VestasV27/03regression_analysis/02correction/" + regression_model +'/t'+str(threshold)+ "/corrected_dsf_rpm32_train.csv"
    path_cdsf_t2_train = "../VestasV27/03regression_analysis/02correction/" + regression_model +'/t'+str(threshold)+ "/corrected_dsf_t2_train.csv"
    path_cdsf_rpm43_train = "../VestasV27/03regression_analysis/02correction/" + regression_model +'/t'+str(threshold)+ "/corrected_dsf_rpm43_train.csv"

    cdsf_idle = pd.read_csv(path_cdsf_idle,index_col=0,header=0,sep=';')
    cdsf_parked = pd.read_csv(path_cdsf_parked,index_col=0,header=0,sep=';')
    cdsf_t1 = pd.read_csv(path_cdsf_t1,index_col=0,header=0,sep=';')
    cdsf_rpm32 = pd.read_csv(path_cdsf_rpm32,index_col=0,header=0,sep=';')
    cdsf_t2 = pd.read_csv(path_cdsf_t2,index_col=0,header=0,sep=';')
    cdsf_rpm43 = pd.read_csv(path_cdsf_rpm43,index_col=0,header=0,sep=';')

    cdsf_train_idle = pd.read_csv(path_cdsf_idle_train,index_col=0,header=0,sep=';')
    cdsf_train_parked = pd.read_csv(path_cdsf_parked_train,index_col=0,header=0,sep=';')
    cdsf_train_t1 = pd.read_csv(path_cdsf_t1_train,index_col=0,header=0,sep=';')
    cdsf_train_rpm32 = pd.read_csv(path_cdsf_rpm32_train,index_col=0,header=0,sep=';')
    cdsf_train_t2 = pd.read_csv(path_cdsf_t2_train,index_col=0,header=0,sep=';')
    cdsf_train_rpm43 = pd.read_csv(path_cdsf_rpm43_train,index_col=0,header=0,sep=';')

    path_scenario_idle = "../VestasV27/03regression_analysis/00standarized/scenario_idle.csv"
    path_scenario_parked = "../VestasV27/03regression_analysis/00standarized/scenario_parked.csv"
    path_scenario_t1 = "../VestasV27/03regression_analysis/00standarized/scenario_t1.csv"
    path_scenario_rpm32 = "../VestasV27/03regression_analysis/00standarized/scenario_rpm32.csv"
    path_scenario_t2 = "../VestasV27/03regression_analysis/00standarized/scenario_t2.csv"
    path_scenario_rpm43 = "../VestasV27/03regression_analysis/00standarized/scenario_rpm43.csv"

    path_scenario_train_idle = "../VestasV27/03regression_analysis/00standarized/scenario_train_idle.csv"
    path_scenario_train_parked = "../VestasV27/03regression_analysis/00standarized/scenario_train_parked.csv"
    path_scenario_train_t1 = "../VestasV27/03regression_analysis/00standarized/scenario_train_t1.csv"
    path_scenario_train_rpm32 = "../VestasV27/03regression_analysis/00standarized/scenario_train_rpm32.csv"
    path_scenario_train_t2 = "../VestasV27/03regression_analysis/00standarized/scenario_train_t2.csv"
    path_scenario_train_rpm43 = "../VestasV27/03regression_analysis/00standarized/scenario_train_rpm43.csv"

    scenario_idle = pd.read_csv(path_scenario_idle, index_col=0, header=0, sep=';')
    scenario_parked = pd.read_csv(path_scenario_parked, index_col=0, header=0, sep=';')
    scenario_t1 = pd.read_csv(path_scenario_t1, index_col=0, header=0, sep=';')
    scenario_rpm32 = pd.read_csv(path_scenario_rpm32, index_col=0, header=0, sep=';')
    scenario_t2 = pd.read_csv(path_scenario_t2, index_col=0, header=0, sep=';')
    scenario_rpm43 = pd.read_csv(path_scenario_rpm43, index_col=0, header=0, sep=';')

    scenario_idle['Mode'] = 'idle'
    scenario_parked['Mode'] = 'parked'
    scenario_t1['Mode'] = 't1'
    scenario_rpm32['Mode'] = 'rpm32'
    scenario_t2['Mode'] = 't2'
    scenario_rpm43['Mode'] = 'rpm43'

    scenario_train_idle = pd.read_csv(path_scenario_train_idle, index_col=0, header=0, sep=';')
    scenario_train_parked = pd.read_csv(path_scenario_train_parked, index_col=0, header=0, sep=';')
    scenario_train_t1 = pd.read_csv(path_scenario_train_t1, index_col=0, header=0, sep=';')
    scenario_train_rpm32 = pd.read_csv(path_scenario_train_rpm32, index_col=0, header=0, sep=';')
    scenario_train_t2 = pd.read_csv(path_scenario_train_t2, index_col=0, header=0, sep=';')
    scenario_train_rpm43 = pd.read_csv(path_scenario_train_rpm43, index_col=0, header=0, sep=';')

    scenario_train_idle['Mode'] = 'idle'
    scenario_train_parked['Mode'] = 'parked'
    scenario_train_t1['Mode'] = 't1'
    scenario_train_rpm32['Mode'] = 'rpm32'
    scenario_train_t2['Mode'] = 't2'
    scenario_train_rpm43['Mode'] = 'rpm43'

    cov_idle = np.cov(cdsf_train_idle.T)
    cov_parked = np.cov(cdsf_train_parked.T)
    cov_t1 = np.cov(cdsf_train_t1.T)
    cov_rpm32 = np.cov(cdsf_train_rpm32.T)
    cov_t2 = np.cov(cdsf_train_t2.T)
    cov_rpm43 = np.cov(cdsf_train_rpm43.T)

    md_idle = []
    md_train_idle = []
    md_parked = []
    md_train_parked = []
    md_t1 = []
    md_train_t1 = []
    md_rpm32 = []
    md_train_rpm32 = []
    md_t2 = []
    md_train_t2 = []
    md_rpm43 = []
    md_train_rpm43 = []

    #Train data
    for sample in range(np.shape(cdsf_train_idle)[0]):
        distance = cdsf_train_idle.iloc[sample,:] @ cov_idle @ cdsf_train_idle.iloc[sample,:].T
        md_train_idle.append(distance)

    for sample in range(np.shape(cdsf_train_parked)[0]):
        distance = cdsf_train_parked.iloc[sample,:] @ cov_parked @ cdsf_train_parked.iloc[sample,:].T
        md_train_parked.append(distance)

    for sample in range(np.shape(cdsf_train_t1)[0]):
        distance = cdsf_train_t1.iloc[sample,:] @ cov_t1 @ cdsf_train_t1.iloc[sample,:].T
        md_train_t1.append(distance)

    for sample in range(np.shape(cdsf_train_rpm32)[0]):
        distance = cdsf_train_rpm32.iloc[sample,:] @ cov_rpm32 @ cdsf_train_rpm32.iloc[sample,:].T
        md_train_rpm32.append(distance)

    for sample in range(np.shape(cdsf_train_t2)[0]):
        distance = cdsf_train_t2.iloc[sample,:] @ cov_t2 @ cdsf_train_t2.iloc[sample,:].T
        md_train_t2.append(distance)

    for sample in range(np.shape(cdsf_train_rpm43)[0]):
        distance = cdsf_train_rpm43.iloc[sample,:] @ cov_rpm43 @ cdsf_train_rpm43.iloc[sample,:].T
        md_train_rpm43.append(distance)

    idle_threshold = np.percentile(md_train_idle,95)
    parked_threshold = np.percentile(md_train_parked,95)
    t1_threshold = np.percentile(md_train_t1,95)
    rpm32_threshold = np.percentile(md_train_rpm32,95)
    t2_threshold = np.percentile(md_train_t2,95)
    rpm43_threshold = np.percentile(md_train_rpm43,95)

    for sample in range(np.shape(cdsf_idle)[0]):
        distance = cdsf_idle.iloc[sample,:] @ cov_idle @ cdsf_idle.iloc[sample,:].T
        md_idle.append(distance)

    for sample in range(np.shape(cdsf_parked)[0]):
        distance = cdsf_parked.iloc[sample,:] @ cov_parked @ cdsf_parked.iloc[sample,:].T
        md_parked.append(distance)

    for sample in range(np.shape(cdsf_t1)[0]):
        distance = cdsf_t1.iloc[sample,:] @ cov_t1 @ cdsf_t1.iloc[sample,:].T
        md_t1.append(distance)

    for sample in range(np.shape(cdsf_rpm32)[0]):
        distance = cdsf_rpm32.iloc[sample,:] @ cov_rpm32 @ cdsf_rpm32.iloc[sample,:].T
        md_rpm32.append(distance)

    for sample in range(np.shape(cdsf_t2)[0]):
        distance = cdsf_t2.iloc[sample,:] @ cov_t2 @ cdsf_t2.iloc[sample,:].T
        md_t2.append(distance)

    for sample in range(np.shape(cdsf_rpm43)[0]):
        distance = cdsf_rpm43.iloc[sample,:] @ cov_rpm43 @ cdsf_rpm43.iloc[sample,:].T
        md_rpm43.append(distance)

    #Merge lists to preparo control chart
    md_baseline = md_train_idle + md_train_parked + md_train_t1 + md_train_rpm32 + md_train_t2 + md_train_rpm43
    md_baseline = pd.DataFrame(md_baseline, columns=['MD'])

    md_test = md_idle + md_parked + md_t1 + md_rpm32 + md_t2 + md_rpm43
    # md_test = np.array(md_test)
    md_test = pd.DataFrame(md_test, columns=['MD'])
    
    scenario_baseline = pd.concat([scenario_train_idle, scenario_train_parked])
    scenario_baseline = pd.concat([scenario_baseline, scenario_train_t1])
    scenario_baseline = pd.concat([scenario_baseline, scenario_train_rpm32])
    scenario_baseline = pd.concat([scenario_baseline, scenario_train_t2])
    scenario_baseline = pd.concat([scenario_baseline, scenario_train_rpm43])

    scenario_test = pd.concat([scenario_idle, scenario_parked])
    scenario_test = pd.concat([scenario_test, scenario_t1])
    scenario_test = pd.concat([scenario_test, scenario_rpm32])
    scenario_test = pd.concat([scenario_test, scenario_t2])
    scenario_test = pd.concat([scenario_test, scenario_rpm43])

    md_baseline = md_baseline.reset_index(drop=True)
    scenario_baseline = scenario_baseline.reset_index(drop=True)

    md_test = md_test.reset_index(drop=True)
    scenario_test = scenario_test.reset_index(drop=True)

    baseline = pd.concat([md_baseline, scenario_baseline], axis=1)
    test = pd.concat([md_test, scenario_test], axis=1)

    total = pd.concat([baseline, test])
    total['Date'] = pd.to_datetime(total['Date'], format='%d/%m/%y %H:%M')

    print(type(total.Date[0]))
    total = total.sort_values(by='Date')

    plt.figure()
    colors_scenario = {'undamaged': 'green', '15cm': 'yellow','30cm':'orange', '45cm':'red', 'repaired':'blue'}
    colors_mode = {'parked':'grey', 
                   'idle':'#ffa600',
                   't1':'#ff6361',
                   'rpm32':'#bc5090',
                   't2':'#58508d',
                   'rpm43':'#003f5c'}
    
    undamaged = total.loc[total['DamageScenario'] == 'undamaged']
    damage_15cm = total.loc[total['DamageScenario'] == '15cm']
    damage_30cm = total.loc[total['DamageScenario'] == '30cm']
    damage_45cm = total.loc[total['DamageScenario'] == '45cm']
    repaired = total.loc[total['DamageScenario'] == 'repaired']

    median_undamaged = np.median(undamaged['MD'])
    median_damage_15cm = np.median(damage_15cm['MD'])
    median_damage_30cm = np.median(damage_30cm['MD'])
    median_damage_45cm = np.median(damage_45cm['MD'])
    median_repaired = np.median(repaired['MD'])

    plt.scatter(x=total.Date,y=total.MD,s=1,c=total['DamageScenario'].map(colors_scenario))
    plt.yscale('log')
    plt.ylim([100,1000000])
    plt.plot([undamaged['Date'].min(), undamaged['Date'].max()], [median_undamaged, median_undamaged],color='black')
    plt.plot([damage_15cm['Date'].min(), damage_15cm['Date'].max()], [median_damage_15cm, median_damage_15cm],color='black')
    plt.plot([damage_30cm['Date'].min(), damage_30cm['Date'].max()], [median_damage_30cm, median_damage_30cm],color='black')
    plt.plot([damage_45cm['Date'].min(), damage_45cm['Date'].max()], [median_damage_45cm, median_damage_45cm],color='black')
    plt.plot([repaired['Date'].min(), repaired['Date'].max()], [median_repaired, median_repaired],color='black')
    plt.show()

    plt.figure()
    plt.scatter(x=total.Temperature,y=total.MD,s=1,c=total['DamageScenario'].map(colors_scenario))
    plt.yscale('log')
    plt.ylim([100,1000000])
    plt.show()

def control_chart_with_reference(regression_model: str, threshold:float):

    """
    Plot control chart with uncorrected values in grey.

    Parameters
    ----------
    regression_type : string
        Name folder used to correct data.

    threshold: float
        DSFs are only corrected if the R^2 is higher or equal than the threshold defined.

    """

    path_cdsf_idle = "../VestasV27/03regression_analysis/02correction/" + regression_model +'/t'+str(threshold)+ "/corrected_dsf_idle.csv"
    path_cdsf_parked = "../VestasV27/03regression_analysis/02correction/" + regression_model +'/t'+str(threshold)+ "/corrected_dsf_parked.csv"
    path_cdsf_t1 = "../VestasV27/03regression_analysis/02correction/" + regression_model +'/t'+str(threshold)+ "/corrected_dsf_t1.csv"
    path_cdsf_rpm32 = "../VestasV27/03regression_analysis/02correction/" + regression_model +'/t'+str(threshold)+ "/corrected_dsf_rpm32.csv"
    path_cdsf_t2 = "../VestasV27/03regression_analysis/02correction/" + regression_model +'/t'+str(threshold)+ "/corrected_dsf_t2.csv"
    path_cdsf_rpm43 = "../VestasV27/03regression_analysis/02correction/" + regression_model +'/t'+str(threshold)+ "/corrected_dsf_rpm43.csv"

    path_cdsf_idle_train = "../VestasV27/03regression_analysis/02correction/" + regression_model +'/t'+str(threshold)+ "/corrected_dsf_idle_train.csv"
    path_cdsf_parked_train = "../VestasV27/03regression_analysis/02correction/" + regression_model +'/t'+str(threshold)+ "/corrected_dsf_parked_train.csv"
    path_cdsf_t1_train = "../VestasV27/03regression_analysis/02correction/" + regression_model +'/t'+str(threshold)+ "/corrected_dsf_t1_train.csv"
    path_cdsf_rpm32_train = "../VestasV27/03regression_analysis/02correction/" + regression_model +'/t'+str(threshold)+ "/corrected_dsf_rpm32_train.csv"
    path_cdsf_t2_train = "../VestasV27/03regression_analysis/02correction/" + regression_model +'/t'+str(threshold)+ "/corrected_dsf_t2_train.csv"
    path_cdsf_rpm43_train = "../VestasV27/03regression_analysis/02correction/" + regression_model +'/t'+str(threshold)+ "/corrected_dsf_rpm43_train.csv"

    cdsf_idle = pd.read_csv(path_cdsf_idle,index_col=0,header=0,sep=';')
    cdsf_parked = pd.read_csv(path_cdsf_parked,index_col=0,header=0,sep=';')
    cdsf_t1 = pd.read_csv(path_cdsf_t1,index_col=0,header=0,sep=';')
    cdsf_rpm32 = pd.read_csv(path_cdsf_rpm32,index_col=0,header=0,sep=';')
    cdsf_t2 = pd.read_csv(path_cdsf_t2,index_col=0,header=0,sep=';')
    cdsf_rpm43 = pd.read_csv(path_cdsf_rpm43,index_col=0,header=0,sep=';')

    cdsf_train_idle = pd.read_csv(path_cdsf_idle_train,index_col=0,header=0,sep=';')
    cdsf_train_parked = pd.read_csv(path_cdsf_parked_train,index_col=0,header=0,sep=';')
    cdsf_train_t1 = pd.read_csv(path_cdsf_t1_train,index_col=0,header=0,sep=';')
    cdsf_train_rpm32 = pd.read_csv(path_cdsf_rpm32_train,index_col=0,header=0,sep=';')
    cdsf_train_t2 = pd.read_csv(path_cdsf_t2_train,index_col=0,header=0,sep=';')
    cdsf_train_rpm43 = pd.read_csv(path_cdsf_rpm43_train,index_col=0,header=0,sep=';')

    path_cdsf_idle_uncorrected = "../VestasV27/03regression_analysis/02correction/" + 'adaptive_order1' +'/t'+str(1.0)+ "/corrected_dsf_idle.csv"
    path_cdsf_parked_uncorrected = "../VestasV27/03regression_analysis/02correction/" + 'adaptive_order1' +'/t'+str(1.0)+ "/corrected_dsf_parked.csv"
    path_cdsf_t1_uncorrected = "../VestasV27/03regression_analysis/02correction/" + 'adaptive_order1' +'/t'+str(1.0)+ "/corrected_dsf_t1.csv"
    path_cdsf_rpm32_uncorrected = "../VestasV27/03regression_analysis/02correction/" + 'adaptive_order1' +'/t'+str(1.0)+ "/corrected_dsf_rpm32.csv"
    path_cdsf_t2_uncorrected = "../VestasV27/03regression_analysis/02correction/" + 'adaptive_order1' +'/t'+str(1.0)+ "/corrected_dsf_t2.csv"
    path_cdsf_rpm43_uncorrected = "../VestasV27/03regression_analysis/02correction/" + 'adaptive_order1' +'/t'+str(1.0)+ "/corrected_dsf_rpm43.csv"

    path_cdsf_idle_train_uncorrected = "../VestasV27/03regression_analysis/02correction/" + 'adaptive_order1' +'/t'+str(1.0)+ "/corrected_dsf_idle_train.csv"
    path_cdsf_parked_train_uncorrected = "../VestasV27/03regression_analysis/02correction/" + 'adaptive_order1' +'/t'+str(1.0)+ "/corrected_dsf_parked_train.csv"
    path_cdsf_t1_train_uncorrected = "../VestasV27/03regression_analysis/02correction/" + 'adaptive_order1' +'/t'+str(1.0)+ "/corrected_dsf_t1_train.csv"
    path_cdsf_rpm32_train_uncorrected = "../VestasV27/03regression_analysis/02correction/" + 'adaptive_order1' +'/t'+str(1.0)+ "/corrected_dsf_rpm32_train.csv"
    path_cdsf_t2_train_uncorrected = "../VestasV27/03regression_analysis/02correction/" + 'adaptive_order1' +'/t'+str(1.0)+ "/corrected_dsf_t2_train.csv"
    path_cdsf_rpm43_train_uncorrected = "../VestasV27/03regression_analysis/02correction/" + 'adaptive_order1' +'/t'+str(1.0)+ "/corrected_dsf_rpm43_train.csv"

    cdsf_idle_uncorrected = pd.read_csv(path_cdsf_idle_uncorrected,index_col=0,header=0,sep=';')
    cdsf_parked_uncorrected = pd.read_csv(path_cdsf_parked_uncorrected,index_col=0,header=0,sep=';')
    cdsf_t1_uncorrected = pd.read_csv(path_cdsf_t1_uncorrected,index_col=0,header=0,sep=';')
    cdsf_rpm32_uncorrected = pd.read_csv(path_cdsf_rpm32_uncorrected,index_col=0,header=0,sep=';')
    cdsf_t2_uncorrected = pd.read_csv(path_cdsf_t2_uncorrected,index_col=0,header=0,sep=';')
    cdsf_rpm43_uncorrected = pd.read_csv(path_cdsf_rpm43_uncorrected,index_col=0,header=0,sep=';')

    cdsf_train_idle_uncorrected = pd.read_csv(path_cdsf_idle_train_uncorrected,index_col=0,header=0,sep=';')
    cdsf_train_parked_uncorrected = pd.read_csv(path_cdsf_parked_train_uncorrected,index_col=0,header=0,sep=';')
    cdsf_train_t1_uncorrected = pd.read_csv(path_cdsf_t1_train_uncorrected,index_col=0,header=0,sep=';')
    cdsf_train_rpm32_uncorrected = pd.read_csv(path_cdsf_rpm32_train_uncorrected,index_col=0,header=0,sep=';')
    cdsf_train_t2_uncorrected = pd.read_csv(path_cdsf_t2_train_uncorrected,index_col=0,header=0,sep=';')
    cdsf_train_rpm43_uncorrected = pd.read_csv(path_cdsf_rpm43_train_uncorrected,index_col=0,header=0,sep=';')

    path_scenario_idle = "../VestasV27/03regression_analysis/00standarized/scenario_idle.csv"
    path_scenario_parked = "../VestasV27/03regression_analysis/00standarized/scenario_parked.csv"
    path_scenario_t1 = "../VestasV27/03regression_analysis/00standarized/scenario_t1.csv"
    path_scenario_rpm32 = "../VestasV27/03regression_analysis/00standarized/scenario_rpm32.csv"
    path_scenario_t2 = "../VestasV27/03regression_analysis/00standarized/scenario_t2.csv"
    path_scenario_rpm43 = "../VestasV27/03regression_analysis/00standarized/scenario_rpm43.csv"

    path_scenario_train_idle = "../VestasV27/03regression_analysis/00standarized/scenario_train_idle.csv"
    path_scenario_train_parked = "../VestasV27/03regression_analysis/00standarized/scenario_train_parked.csv"
    path_scenario_train_t1 = "../VestasV27/03regression_analysis/00standarized/scenario_train_t1.csv"
    path_scenario_train_rpm32 = "../VestasV27/03regression_analysis/00standarized/scenario_train_rpm32.csv"
    path_scenario_train_t2 = "../VestasV27/03regression_analysis/00standarized/scenario_train_t2.csv"
    path_scenario_train_rpm43 = "../VestasV27/03regression_analysis/00standarized/scenario_train_rpm43.csv"

    scenario_idle = pd.read_csv(path_scenario_idle, index_col=0, header=0, sep=';')
    scenario_parked = pd.read_csv(path_scenario_parked, index_col=0, header=0, sep=';')
    scenario_t1 = pd.read_csv(path_scenario_t1, index_col=0, header=0, sep=';')
    scenario_rpm32 = pd.read_csv(path_scenario_rpm32, index_col=0, header=0, sep=';')
    scenario_t2 = pd.read_csv(path_scenario_t2, index_col=0, header=0, sep=';')
    scenario_rpm43 = pd.read_csv(path_scenario_rpm43, index_col=0, header=0, sep=';')

    scenario_idle['Mode'] = 'idle'
    scenario_parked['Mode'] = 'parked'
    scenario_t1['Mode'] = 't1'
    scenario_rpm32['Mode'] = 'rpm32'
    scenario_t2['Mode'] = 't2'
    scenario_rpm43['Mode'] = 'rpm43'

    scenario_train_idle = pd.read_csv(path_scenario_train_idle, index_col=0, header=0, sep=';')
    scenario_train_parked = pd.read_csv(path_scenario_train_parked, index_col=0, header=0, sep=';')
    scenario_train_t1 = pd.read_csv(path_scenario_train_t1, index_col=0, header=0, sep=';')
    scenario_train_rpm32 = pd.read_csv(path_scenario_train_rpm32, index_col=0, header=0, sep=';')
    scenario_train_t2 = pd.read_csv(path_scenario_train_t2, index_col=0, header=0, sep=';')
    scenario_train_rpm43 = pd.read_csv(path_scenario_train_rpm43, index_col=0, header=0, sep=';')

    scenario_train_idle['Mode'] = 'idle'
    scenario_train_parked['Mode'] = 'parked'
    scenario_train_t1['Mode'] = 't1'
    scenario_train_rpm32['Mode'] = 'rpm32'
    scenario_train_t2['Mode'] = 't2'
    scenario_train_rpm43['Mode'] = 'rpm43'

    cov_idle = np.cov(cdsf_train_idle.T)
    cov_parked = np.cov(cdsf_train_parked.T)
    cov_t1 = np.cov(cdsf_train_t1.T)
    cov_rpm32 = np.cov(cdsf_train_rpm32.T)
    cov_t2 = np.cov(cdsf_train_t2.T)
    cov_rpm43 = np.cov(cdsf_train_rpm43.T)

    cov_idle_uncorrected = np.cov(cdsf_train_idle_uncorrected.T)
    cov_parked_uncorrected = np.cov(cdsf_train_parked_uncorrected.T)
    cov_t1_uncorrected = np.cov(cdsf_train_t1_uncorrected.T)
    cov_rpm32_uncorrected = np.cov(cdsf_train_rpm32_uncorrected.T)
    cov_t2_uncorrected = np.cov(cdsf_train_t2_uncorrected.T)
    cov_rpm43_uncorrected = np.cov(cdsf_train_rpm43_uncorrected.T)

    md_idle = []
    md_train_idle = []
    md_parked = []
    md_train_parked = []
    md_t1 = []
    md_train_t1 = []
    md_rpm32 = []
    md_train_rpm32 = []
    md_t2 = []
    md_train_t2 = []
    md_rpm43 = []
    md_train_rpm43 = []

    md_idle_uncorrected = []
    md_train_idle_uncorrected = []
    md_parked_uncorrected = []
    md_train_parked_uncorrected = []
    md_t1_uncorrected = []
    md_train_t1_uncorrected = []
    md_rpm32_uncorrected = []
    md_train_rpm32_uncorrected = []
    md_t2_uncorrected = []
    md_train_t2_uncorrected = []
    md_rpm43_uncorrected = []
    md_train_rpm43_uncorrected = []

    #Train data
    for sample in range(np.shape(cdsf_train_idle)[0]):
        distance = cdsf_train_idle.iloc[sample,:] @ cov_idle @ cdsf_train_idle.iloc[sample,:].T
        md_train_idle.append(distance)

    for sample in range(np.shape(cdsf_train_parked)[0]):
        distance = cdsf_train_parked.iloc[sample,:] @ cov_parked @ cdsf_train_parked.iloc[sample,:].T
        md_train_parked.append(distance)

    for sample in range(np.shape(cdsf_train_t1)[0]):
        distance = cdsf_train_t1.iloc[sample,:] @ cov_t1 @ cdsf_train_t1.iloc[sample,:].T
        md_train_t1.append(distance)

    for sample in range(np.shape(cdsf_train_rpm32)[0]):
        distance = cdsf_train_rpm32.iloc[sample,:] @ cov_rpm32 @ cdsf_train_rpm32.iloc[sample,:].T
        md_train_rpm32.append(distance)

    for sample in range(np.shape(cdsf_train_t2)[0]):
        distance = cdsf_train_t2.iloc[sample,:] @ cov_t2 @ cdsf_train_t2.iloc[sample,:].T
        md_train_t2.append(distance)

    for sample in range(np.shape(cdsf_train_rpm43)[0]):
        distance = cdsf_train_rpm43.iloc[sample,:] @ cov_rpm43 @ cdsf_train_rpm43.iloc[sample,:].T
        md_train_rpm43.append(distance)

    idle_threshold = np.percentile(md_train_idle,95)
    parked_threshold = np.percentile(md_train_parked,95)
    t1_threshold = np.percentile(md_train_t1,95)
    rpm32_threshold = np.percentile(md_train_rpm32,95)
    t2_threshold = np.percentile(md_train_t2,95)
    rpm43_threshold = np.percentile(md_train_rpm43,95)

    #Uncorrected train
    for sample in range(np.shape(cdsf_train_idle_uncorrected)[0]):
        distance = cdsf_train_idle_uncorrected.iloc[sample,:] @ cov_idle_uncorrected @ cdsf_train_idle_uncorrected.iloc[sample,:].T
        md_train_idle_uncorrected.append(distance)

    for sample in range(np.shape(cdsf_train_parked_uncorrected)[0]):
        distance = cdsf_train_parked_uncorrected.iloc[sample,:] @ cov_parked_uncorrected @ cdsf_train_parked_uncorrected.iloc[sample,:].T
        md_train_parked_uncorrected.append(distance)

    for sample in range(np.shape(cdsf_train_t1_uncorrected)[0]):
        distance = cdsf_train_t1_uncorrected.iloc[sample,:] @ cov_t1_uncorrected @ cdsf_train_t1_uncorrected.iloc[sample,:].T
        md_train_t1_uncorrected.append(distance)

    for sample in range(np.shape(cdsf_train_rpm32_uncorrected)[0]):
        distance = cdsf_train_rpm32_uncorrected.iloc[sample,:] @ cov_rpm32_uncorrected @ cdsf_train_rpm32_uncorrected.iloc[sample,:].T
        md_train_rpm32_uncorrected.append(distance)

    for sample in range(np.shape(cdsf_train_t2_uncorrected)[0]):
        distance = cdsf_train_t2_uncorrected.iloc[sample,:] @ cov_t2_uncorrected @ cdsf_train_t2_uncorrected.iloc[sample,:].T
        md_train_t2_uncorrected.append(distance)

    for sample in range(np.shape(cdsf_train_rpm43_uncorrected)[0]):
        distance = cdsf_train_rpm43_uncorrected.iloc[sample,:] @ cov_rpm43_uncorrected @ cdsf_train_rpm43_uncorrected.iloc[sample,:].T
        md_train_rpm43_uncorrected.append(distance)

    #Test data
    for sample in range(np.shape(cdsf_idle)[0]):
        distance = cdsf_idle.iloc[sample,:] @ cov_idle @ cdsf_idle.iloc[sample,:].T
        md_idle.append(distance)

    for sample in range(np.shape(cdsf_parked)[0]):
        distance = cdsf_parked.iloc[sample,:] @ cov_parked @ cdsf_parked.iloc[sample,:].T
        md_parked.append(distance)

    for sample in range(np.shape(cdsf_t1)[0]):
        distance = cdsf_t1.iloc[sample,:] @ cov_t1 @ cdsf_t1.iloc[sample,:].T
        md_t1.append(distance)

    for sample in range(np.shape(cdsf_rpm32)[0]):
        distance = cdsf_rpm32.iloc[sample,:] @ cov_rpm32 @ cdsf_rpm32.iloc[sample,:].T
        md_rpm32.append(distance)

    for sample in range(np.shape(cdsf_t2)[0]):
        distance = cdsf_t2.iloc[sample,:] @ cov_t2 @ cdsf_t2.iloc[sample,:].T
        md_t2.append(distance)

    for sample in range(np.shape(cdsf_rpm43)[0]):
        distance = cdsf_rpm43.iloc[sample,:] @ cov_rpm43 @ cdsf_rpm43.iloc[sample,:].T
        md_rpm43.append(distance)

    #Uncorrected test data
    for sample in range(np.shape(cdsf_idle_uncorrected)[0]):
        distance = cdsf_idle_uncorrected.iloc[sample,:] @ cov_idle_uncorrected @ cdsf_idle_uncorrected.iloc[sample,:].T
        md_idle_uncorrected.append(distance)

    for sample in range(np.shape(cdsf_parked_uncorrected)[0]):
        distance = cdsf_parked_uncorrected.iloc[sample,:] @ cov_parked_uncorrected @ cdsf_parked_uncorrected.iloc[sample,:].T
        md_parked_uncorrected.append(distance)

    for sample in range(np.shape(cdsf_t1_uncorrected)[0]):
        distance = cdsf_t1_uncorrected.iloc[sample,:] @ cov_t1_uncorrected @ cdsf_t1_uncorrected.iloc[sample,:].T
        md_t1_uncorrected.append(distance)

    for sample in range(np.shape(cdsf_rpm32_uncorrected)[0]):
        distance = cdsf_rpm32_uncorrected.iloc[sample,:] @ cov_rpm32_uncorrected @ cdsf_rpm32_uncorrected.iloc[sample,:].T
        md_rpm32_uncorrected.append(distance)

    for sample in range(np.shape(cdsf_t2_uncorrected)[0]):
        distance = cdsf_t2_uncorrected.iloc[sample,:] @ cov_t2_uncorrected @ cdsf_t2_uncorrected.iloc[sample,:].T
        md_t2_uncorrected.append(distance)

    for sample in range(np.shape(cdsf_rpm43_uncorrected)[0]):
        distance = cdsf_rpm43_uncorrected.iloc[sample,:] @ cov_rpm43_uncorrected @ cdsf_rpm43_uncorrected.iloc[sample,:].T
        md_rpm43_uncorrected.append(distance)

    #Merge lists to preparo control chart
    md_baseline = md_train_idle + md_train_parked + md_train_t1 + md_train_rpm32 + md_train_t2 + md_train_rpm43
    md_baseline = pd.DataFrame(md_baseline, columns=['MD'])

    md_baseline_uncorrected = md_train_idle_uncorrected + md_train_parked_uncorrected + md_train_t1_uncorrected + md_train_rpm32_uncorrected + md_train_t2_uncorrected + md_train_rpm43_uncorrected
    md_baseline_uncorrected = pd.DataFrame(md_baseline_uncorrected, columns=['MD'])

    md_test = md_idle + md_parked + md_t1 + md_rpm32 + md_t2 + md_rpm43
    md_test_uncorrected = md_idle_uncorrected + md_parked_uncorrected + md_t1_uncorrected + md_rpm32_uncorrected + md_t2_uncorrected + md_rpm43_uncorrected
    # md_test = np.array(md_test)
    md_test = pd.DataFrame(md_test, columns=['MD'])
    md_test_uncorrected = pd.DataFrame(md_test_uncorrected, columns=['MD'])

    scenario_baseline = pd.concat([scenario_train_idle, scenario_train_parked])
    scenario_baseline = pd.concat([scenario_baseline, scenario_train_t1])
    scenario_baseline = pd.concat([scenario_baseline, scenario_train_rpm32])
    scenario_baseline = pd.concat([scenario_baseline, scenario_train_t2])
    scenario_baseline = pd.concat([scenario_baseline, scenario_train_rpm43])

    scenario_test = pd.concat([scenario_idle, scenario_parked])
    scenario_test = pd.concat([scenario_test, scenario_t1])
    scenario_test = pd.concat([scenario_test, scenario_rpm32])
    scenario_test = pd.concat([scenario_test, scenario_t2])
    scenario_test = pd.concat([scenario_test, scenario_rpm43])

    md_baseline = md_baseline.reset_index(drop=True)
    md_baseline_uncorrected = md_baseline_uncorrected.reset_index(drop=True)
    scenario_baseline = scenario_baseline.reset_index(drop=True)

    md_test = md_test.reset_index(drop=True)
    md_test_uncorrected = md_test_uncorrected.reset_index(drop=True)
    scenario_test = scenario_test.reset_index(drop=True)

    baseline = pd.concat([md_baseline, scenario_baseline], axis=1)
    baseline_uncorrected = pd.concat([md_baseline_uncorrected, scenario_baseline], axis=1)
    test = pd.concat([md_test, scenario_test], axis=1)
    test_uncorrected = pd.concat([md_test_uncorrected, scenario_test], axis=1)

    total = pd.concat([baseline, test])
    total_uncorrected = pd.concat([baseline_uncorrected, test_uncorrected])
    total['Date'] = pd.to_datetime(total['Date'], format='%d/%m/%y %H:%M')
    total_uncorrected['Date'] = pd.to_datetime(total_uncorrected['Date'], format='%d/%m/%y %H:%M')

    print(type(total.Date[0]))
    total = total.sort_values(by='Date')
    total_uncorrected = total_uncorrected.sort_values(by='Date')

    plt.figure()
    colors_scenario = {'undamaged': 'green', '15cm': 'yellow','30cm':'orange', '45cm':'red', 'repaired':'blue'}
    colors_mode = {'parked':'grey', 
                   'idle':'#ffa600',
                   't1':'#ff6361',
                   'rpm32':'#bc5090',
                   't2':'#58508d',
                   'rpm43':'#003f5c'}
    
    undamaged = total.loc[total['DamageScenario'] == 'undamaged']
    damage_15cm = total.loc[total['DamageScenario'] == '15cm']
    damage_30cm = total.loc[total['DamageScenario'] == '30cm']
    damage_45cm = total.loc[total['DamageScenario'] == '45cm']
    repaired = total.loc[total['DamageScenario'] == 'repaired']

    median_undamaged = np.median(undamaged['MD'])
    median_damage_15cm = np.median(damage_15cm['MD'])
    median_damage_30cm = np.median(damage_30cm['MD'])
    median_damage_45cm = np.median(damage_45cm['MD'])
    median_repaired = np.median(repaired['MD'])

    low = 100
    high = 10000000

    plt.scatter(x=total_uncorrected.Date,y=total_uncorrected.MD,s=1,c='gainsboro')
    plt.scatter(x=total.Date,y=total.MD,s=1,c=total['DamageScenario'].map(colors_scenario))
    plt.yscale('log')
    plt.ylim([low, high])
    plt.plot([undamaged['Date'].min(), undamaged['Date'].max()], [median_undamaged, median_undamaged],color='black')
    plt.plot([damage_15cm['Date'].min(), damage_15cm['Date'].max()], [median_damage_15cm, median_damage_15cm],color='black')
    plt.plot([damage_30cm['Date'].min(), damage_30cm['Date'].max()], [median_damage_30cm, median_damage_30cm],color='black')
    plt.plot([damage_45cm['Date'].min(), damage_45cm['Date'].max()], [median_damage_45cm, median_damage_45cm],color='black')
    plt.plot([repaired['Date'].min(), repaired['Date'].max()], [median_repaired, median_repaired],color='black')
    plt.show()

    plt.figure()
    plt.scatter(x=total_uncorrected.Temperature,y=total_uncorrected.MD,s=1,c='gainsboro')
    plt.scatter(x=total.Temperature,y=total.MD,s=1,c=total['DamageScenario'].map(colors_scenario))
    plt.yscale('log')
    plt.ylim([low, high])
    plt.show()

def control_chart_with_reference_operational_mode(regression_model: str, threshold:float):

    """
    Plot control chart with uncorrected values in grey.

    Parameters
    ----------
    regression_type : string
        Name folder used to correct data.

    threshold: float
        DSFs are only corrected if the R^2 is higher or equal than the threshold defined.

    """

    path_cdsf_idle = "../VestasV27/03regression_analysis/02correction/" + regression_model +'/t'+str(threshold)+ "/corrected_dsf_idle.csv"
    path_cdsf_parked = "../VestasV27/03regression_analysis/02correction/" + regression_model +'/t'+str(threshold)+ "/corrected_dsf_parked.csv"
    path_cdsf_t1 = "../VestasV27/03regression_analysis/02correction/" + regression_model +'/t'+str(threshold)+ "/corrected_dsf_t1.csv"
    path_cdsf_rpm32 = "../VestasV27/03regression_analysis/02correction/" + regression_model +'/t'+str(threshold)+ "/corrected_dsf_rpm32.csv"
    path_cdsf_t2 = "../VestasV27/03regression_analysis/02correction/" + regression_model +'/t'+str(threshold)+ "/corrected_dsf_t2.csv"
    path_cdsf_rpm43 = "../VestasV27/03regression_analysis/02correction/" + regression_model +'/t'+str(threshold)+ "/corrected_dsf_rpm43.csv"

    path_cdsf_idle_train = "../VestasV27/03regression_analysis/02correction/" + regression_model +'/t'+str(threshold)+ "/corrected_dsf_idle_train.csv"
    path_cdsf_parked_train = "../VestasV27/03regression_analysis/02correction/" + regression_model +'/t'+str(threshold)+ "/corrected_dsf_parked_train.csv"
    path_cdsf_t1_train = "../VestasV27/03regression_analysis/02correction/" + regression_model +'/t'+str(threshold)+ "/corrected_dsf_t1_train.csv"
    path_cdsf_rpm32_train = "../VestasV27/03regression_analysis/02correction/" + regression_model +'/t'+str(threshold)+ "/corrected_dsf_rpm32_train.csv"
    path_cdsf_t2_train = "../VestasV27/03regression_analysis/02correction/" + regression_model +'/t'+str(threshold)+ "/corrected_dsf_t2_train.csv"
    path_cdsf_rpm43_train = "../VestasV27/03regression_analysis/02correction/" + regression_model +'/t'+str(threshold)+ "/corrected_dsf_rpm43_train.csv"

    cdsf_idle = pd.read_csv(path_cdsf_idle,index_col=0,header=0,sep=';')
    cdsf_parked = pd.read_csv(path_cdsf_parked,index_col=0,header=0,sep=';')
    cdsf_t1 = pd.read_csv(path_cdsf_t1,index_col=0,header=0,sep=';')
    cdsf_rpm32 = pd.read_csv(path_cdsf_rpm32,index_col=0,header=0,sep=';')
    cdsf_t2 = pd.read_csv(path_cdsf_t2,index_col=0,header=0,sep=';')
    cdsf_rpm43 = pd.read_csv(path_cdsf_rpm43,index_col=0,header=0,sep=';')

    cdsf_train_idle = pd.read_csv(path_cdsf_idle_train,index_col=0,header=0,sep=';')
    cdsf_train_parked = pd.read_csv(path_cdsf_parked_train,index_col=0,header=0,sep=';')
    cdsf_train_t1 = pd.read_csv(path_cdsf_t1_train,index_col=0,header=0,sep=';')
    cdsf_train_rpm32 = pd.read_csv(path_cdsf_rpm32_train,index_col=0,header=0,sep=';')
    cdsf_train_t2 = pd.read_csv(path_cdsf_t2_train,index_col=0,header=0,sep=';')
    cdsf_train_rpm43 = pd.read_csv(path_cdsf_rpm43_train,index_col=0,header=0,sep=';')

    path_cdsf_idle_uncorrected = "../VestasV27/03regression_analysis/02correction/" + 'adaptive_order1' +'/t'+str(1.0)+ "/corrected_dsf_idle.csv"
    path_cdsf_parked_uncorrected = "../VestasV27/03regression_analysis/02correction/" + 'adaptive_order1' +'/t'+str(1.0)+ "/corrected_dsf_parked.csv"
    path_cdsf_t1_uncorrected = "../VestasV27/03regression_analysis/02correction/" + 'adaptive_order1' +'/t'+str(1.0)+ "/corrected_dsf_t1.csv"
    path_cdsf_rpm32_uncorrected = "../VestasV27/03regression_analysis/02correction/" + 'adaptive_order1' +'/t'+str(1.0)+ "/corrected_dsf_rpm32.csv"
    path_cdsf_t2_uncorrected = "../VestasV27/03regression_analysis/02correction/" + 'adaptive_order1' +'/t'+str(1.0)+ "/corrected_dsf_t2.csv"
    path_cdsf_rpm43_uncorrected = "../VestasV27/03regression_analysis/02correction/" + 'adaptive_order1' +'/t'+str(1.0)+ "/corrected_dsf_rpm43.csv"

    path_cdsf_idle_train_uncorrected = "../VestasV27/03regression_analysis/02correction/" + 'adaptive_order1' +'/t'+str(1.0)+ "/corrected_dsf_idle_train.csv"
    path_cdsf_parked_train_uncorrected = "../VestasV27/03regression_analysis/02correction/" + 'adaptive_order1' +'/t'+str(1.0)+ "/corrected_dsf_parked_train.csv"
    path_cdsf_t1_train_uncorrected = "../VestasV27/03regression_analysis/02correction/" + 'adaptive_order1' +'/t'+str(1.0)+ "/corrected_dsf_t1_train.csv"
    path_cdsf_rpm32_train_uncorrected = "../VestasV27/03regression_analysis/02correction/" + 'adaptive_order1' +'/t'+str(1.0)+ "/corrected_dsf_rpm32_train.csv"
    path_cdsf_t2_train_uncorrected = "../VestasV27/03regression_analysis/02correction/" + 'adaptive_order1' +'/t'+str(1.0)+ "/corrected_dsf_t2_train.csv"
    path_cdsf_rpm43_train_uncorrected = "../VestasV27/03regression_analysis/02correction/" + 'adaptive_order1' +'/t'+str(1.0)+ "/corrected_dsf_rpm43_train.csv"

    cdsf_idle_uncorrected = pd.read_csv(path_cdsf_idle_uncorrected,index_col=0,header=0,sep=';')
    cdsf_parked_uncorrected = pd.read_csv(path_cdsf_parked_uncorrected,index_col=0,header=0,sep=';')
    cdsf_t1_uncorrected = pd.read_csv(path_cdsf_t1_uncorrected,index_col=0,header=0,sep=';')
    cdsf_rpm32_uncorrected = pd.read_csv(path_cdsf_rpm32_uncorrected,index_col=0,header=0,sep=';')
    cdsf_t2_uncorrected = pd.read_csv(path_cdsf_t2_uncorrected,index_col=0,header=0,sep=';')
    cdsf_rpm43_uncorrected = pd.read_csv(path_cdsf_rpm43_uncorrected,index_col=0,header=0,sep=';')

    cdsf_train_idle_uncorrected = pd.read_csv(path_cdsf_idle_train_uncorrected,index_col=0,header=0,sep=';')
    cdsf_train_parked_uncorrected = pd.read_csv(path_cdsf_parked_train_uncorrected,index_col=0,header=0,sep=';')
    cdsf_train_t1_uncorrected = pd.read_csv(path_cdsf_t1_train_uncorrected,index_col=0,header=0,sep=';')
    cdsf_train_rpm32_uncorrected = pd.read_csv(path_cdsf_rpm32_train_uncorrected,index_col=0,header=0,sep=';')
    cdsf_train_t2_uncorrected = pd.read_csv(path_cdsf_t2_train_uncorrected,index_col=0,header=0,sep=';')
    cdsf_train_rpm43_uncorrected = pd.read_csv(path_cdsf_rpm43_train_uncorrected,index_col=0,header=0,sep=';')

    path_scenario_idle = "../VestasV27/03regression_analysis/00standarized/scenario_idle.csv"
    path_scenario_parked = "../VestasV27/03regression_analysis/00standarized/scenario_parked.csv"
    path_scenario_t1 = "../VestasV27/03regression_analysis/00standarized/scenario_t1.csv"
    path_scenario_rpm32 = "../VestasV27/03regression_analysis/00standarized/scenario_rpm32.csv"
    path_scenario_t2 = "../VestasV27/03regression_analysis/00standarized/scenario_t2.csv"
    path_scenario_rpm43 = "../VestasV27/03regression_analysis/00standarized/scenario_rpm43.csv"

    path_scenario_train_idle = "../VestasV27/03regression_analysis/00standarized/scenario_train_idle.csv"
    path_scenario_train_parked = "../VestasV27/03regression_analysis/00standarized/scenario_train_parked.csv"
    path_scenario_train_t1 = "../VestasV27/03regression_analysis/00standarized/scenario_train_t1.csv"
    path_scenario_train_rpm32 = "../VestasV27/03regression_analysis/00standarized/scenario_train_rpm32.csv"
    path_scenario_train_t2 = "../VestasV27/03regression_analysis/00standarized/scenario_train_t2.csv"
    path_scenario_train_rpm43 = "../VestasV27/03regression_analysis/00standarized/scenario_train_rpm43.csv"

    scenario_idle = pd.read_csv(path_scenario_idle, index_col=0, header=0, sep=';')
    scenario_parked = pd.read_csv(path_scenario_parked, index_col=0, header=0, sep=';')
    scenario_t1 = pd.read_csv(path_scenario_t1, index_col=0, header=0, sep=';')
    scenario_rpm32 = pd.read_csv(path_scenario_rpm32, index_col=0, header=0, sep=';')
    scenario_t2 = pd.read_csv(path_scenario_t2, index_col=0, header=0, sep=';')
    scenario_rpm43 = pd.read_csv(path_scenario_rpm43, index_col=0, header=0, sep=';')

    scenario_idle['Mode'] = 'idle'
    scenario_parked['Mode'] = 'parked'
    scenario_t1['Mode'] = 't1'
    scenario_rpm32['Mode'] = 'rpm32'
    scenario_t2['Mode'] = 't2'
    scenario_rpm43['Mode'] = 'rpm43'

    scenario_train_idle = pd.read_csv(path_scenario_train_idle, index_col=0, header=0, sep=';')
    scenario_train_parked = pd.read_csv(path_scenario_train_parked, index_col=0, header=0, sep=';')
    scenario_train_t1 = pd.read_csv(path_scenario_train_t1, index_col=0, header=0, sep=';')
    scenario_train_rpm32 = pd.read_csv(path_scenario_train_rpm32, index_col=0, header=0, sep=';')
    scenario_train_t2 = pd.read_csv(path_scenario_train_t2, index_col=0, header=0, sep=';')
    scenario_train_rpm43 = pd.read_csv(path_scenario_train_rpm43, index_col=0, header=0, sep=';')

    scenario_train_idle['Mode'] = 'idle'
    scenario_train_parked['Mode'] = 'parked'
    scenario_train_t1['Mode'] = 't1'
    scenario_train_rpm32['Mode'] = 'rpm32'
    scenario_train_t2['Mode'] = 't2'
    scenario_train_rpm43['Mode'] = 'rpm43'

    cov_idle = np.cov(cdsf_train_idle.T)
    cov_parked = np.cov(cdsf_train_parked.T)
    cov_t1 = np.cov(cdsf_train_t1.T)
    cov_rpm32 = np.cov(cdsf_train_rpm32.T)
    cov_t2 = np.cov(cdsf_train_t2.T)
    cov_rpm43 = np.cov(cdsf_train_rpm43.T)

    cov_idle_uncorrected = np.cov(cdsf_train_idle_uncorrected.T)
    cov_parked_uncorrected = np.cov(cdsf_train_parked_uncorrected.T)
    cov_t1_uncorrected = np.cov(cdsf_train_t1_uncorrected.T)
    cov_rpm32_uncorrected = np.cov(cdsf_train_rpm32_uncorrected.T)
    cov_t2_uncorrected = np.cov(cdsf_train_t2_uncorrected.T)
    cov_rpm43_uncorrected = np.cov(cdsf_train_rpm43_uncorrected.T)

    md_idle = []
    md_train_idle = []
    md_parked = []
    md_train_parked = []
    md_t1 = []
    md_train_t1 = []
    md_rpm32 = []
    md_train_rpm32 = []
    md_t2 = []
    md_train_t2 = []
    md_rpm43 = []
    md_train_rpm43 = []

    md_idle_uncorrected = []
    md_train_idle_uncorrected = []
    md_parked_uncorrected = []
    md_train_parked_uncorrected = []
    md_t1_uncorrected = []
    md_train_t1_uncorrected = []
    md_rpm32_uncorrected = []
    md_train_rpm32_uncorrected = []
    md_t2_uncorrected = []
    md_train_t2_uncorrected = []
    md_rpm43_uncorrected = []
    md_train_rpm43_uncorrected = []

    #Train data
    for sample in range(np.shape(cdsf_train_idle)[0]):
        distance = cdsf_train_idle.iloc[sample,:] @ cov_idle @ cdsf_train_idle.iloc[sample,:].T
        md_train_idle.append(distance)

    for sample in range(np.shape(cdsf_train_parked)[0]):
        distance = cdsf_train_parked.iloc[sample,:] @ cov_parked @ cdsf_train_parked.iloc[sample,:].T
        md_train_parked.append(distance)

    for sample in range(np.shape(cdsf_train_t1)[0]):
        distance = cdsf_train_t1.iloc[sample,:] @ cov_t1 @ cdsf_train_t1.iloc[sample,:].T
        md_train_t1.append(distance)

    for sample in range(np.shape(cdsf_train_rpm32)[0]):
        distance = cdsf_train_rpm32.iloc[sample,:] @ cov_rpm32 @ cdsf_train_rpm32.iloc[sample,:].T
        md_train_rpm32.append(distance)

    for sample in range(np.shape(cdsf_train_t2)[0]):
        distance = cdsf_train_t2.iloc[sample,:] @ cov_t2 @ cdsf_train_t2.iloc[sample,:].T
        md_train_t2.append(distance)

    for sample in range(np.shape(cdsf_train_rpm43)[0]):
        distance = cdsf_train_rpm43.iloc[sample,:] @ cov_rpm43 @ cdsf_train_rpm43.iloc[sample,:].T
        md_train_rpm43.append(distance)

    idle_threshold = np.percentile(md_train_idle,95)
    parked_threshold = np.percentile(md_train_parked,95)
    t1_threshold = np.percentile(md_train_t1,95)
    rpm32_threshold = np.percentile(md_train_rpm32,95)
    t2_threshold = np.percentile(md_train_t2,95)
    rpm43_threshold = np.percentile(md_train_rpm43,95)

    #Uncorrected train
    for sample in range(np.shape(cdsf_train_idle_uncorrected)[0]):
        distance = cdsf_train_idle_uncorrected.iloc[sample,:] @ cov_idle_uncorrected @ cdsf_train_idle_uncorrected.iloc[sample,:].T
        md_train_idle_uncorrected.append(distance)

    for sample in range(np.shape(cdsf_train_parked_uncorrected)[0]):
        distance = cdsf_train_parked_uncorrected.iloc[sample,:] @ cov_parked_uncorrected @ cdsf_train_parked_uncorrected.iloc[sample,:].T
        md_train_parked_uncorrected.append(distance)

    for sample in range(np.shape(cdsf_train_t1_uncorrected)[0]):
        distance = cdsf_train_t1_uncorrected.iloc[sample,:] @ cov_t1_uncorrected @ cdsf_train_t1_uncorrected.iloc[sample,:].T
        md_train_t1_uncorrected.append(distance)

    for sample in range(np.shape(cdsf_train_rpm32_uncorrected)[0]):
        distance = cdsf_train_rpm32_uncorrected.iloc[sample,:] @ cov_rpm32_uncorrected @ cdsf_train_rpm32_uncorrected.iloc[sample,:].T
        md_train_rpm32_uncorrected.append(distance)

    for sample in range(np.shape(cdsf_train_t2_uncorrected)[0]):
        distance = cdsf_train_t2_uncorrected.iloc[sample,:] @ cov_t2_uncorrected @ cdsf_train_t2_uncorrected.iloc[sample,:].T
        md_train_t2_uncorrected.append(distance)

    for sample in range(np.shape(cdsf_train_rpm43_uncorrected)[0]):
        distance = cdsf_train_rpm43_uncorrected.iloc[sample,:] @ cov_rpm43_uncorrected @ cdsf_train_rpm43_uncorrected.iloc[sample,:].T
        md_train_rpm43_uncorrected.append(distance)

    #Test data
    for sample in range(np.shape(cdsf_idle)[0]):
        distance = cdsf_idle.iloc[sample,:] @ cov_idle @ cdsf_idle.iloc[sample,:].T
        md_idle.append(distance)

    for sample in range(np.shape(cdsf_parked)[0]):
        distance = cdsf_parked.iloc[sample,:] @ cov_parked @ cdsf_parked.iloc[sample,:].T
        md_parked.append(distance)

    for sample in range(np.shape(cdsf_t1)[0]):
        distance = cdsf_t1.iloc[sample,:] @ cov_t1 @ cdsf_t1.iloc[sample,:].T
        md_t1.append(distance)

    for sample in range(np.shape(cdsf_rpm32)[0]):
        distance = cdsf_rpm32.iloc[sample,:] @ cov_rpm32 @ cdsf_rpm32.iloc[sample,:].T
        md_rpm32.append(distance)

    for sample in range(np.shape(cdsf_t2)[0]):
        distance = cdsf_t2.iloc[sample,:] @ cov_t2 @ cdsf_t2.iloc[sample,:].T
        md_t2.append(distance)

    for sample in range(np.shape(cdsf_rpm43)[0]):
        distance = cdsf_rpm43.iloc[sample,:] @ cov_rpm43 @ cdsf_rpm43.iloc[sample,:].T
        md_rpm43.append(distance)

    #Uncorrected test data
    for sample in range(np.shape(cdsf_idle_uncorrected)[0]):
        distance = cdsf_idle_uncorrected.iloc[sample,:] @ cov_idle_uncorrected @ cdsf_idle_uncorrected.iloc[sample,:].T
        md_idle_uncorrected.append(distance)

    for sample in range(np.shape(cdsf_parked_uncorrected)[0]):
        distance = cdsf_parked_uncorrected.iloc[sample,:] @ cov_parked_uncorrected @ cdsf_parked_uncorrected.iloc[sample,:].T
        md_parked_uncorrected.append(distance)

    for sample in range(np.shape(cdsf_t1_uncorrected)[0]):
        distance = cdsf_t1_uncorrected.iloc[sample,:] @ cov_t1_uncorrected @ cdsf_t1_uncorrected.iloc[sample,:].T
        md_t1_uncorrected.append(distance)

    for sample in range(np.shape(cdsf_rpm32_uncorrected)[0]):
        distance = cdsf_rpm32_uncorrected.iloc[sample,:] @ cov_rpm32_uncorrected @ cdsf_rpm32_uncorrected.iloc[sample,:].T
        md_rpm32_uncorrected.append(distance)

    for sample in range(np.shape(cdsf_t2_uncorrected)[0]):
        distance = cdsf_t2_uncorrected.iloc[sample,:] @ cov_t2_uncorrected @ cdsf_t2_uncorrected.iloc[sample,:].T
        md_t2_uncorrected.append(distance)

    for sample in range(np.shape(cdsf_rpm43_uncorrected)[0]):
        distance = cdsf_rpm43_uncorrected.iloc[sample,:] @ cov_rpm43_uncorrected @ cdsf_rpm43_uncorrected.iloc[sample,:].T
        md_rpm43_uncorrected.append(distance)

    #Merge lists to preparo control chart
    md_baseline = md_train_idle + md_train_parked + md_train_t1 + md_train_rpm32 + md_train_t2 + md_train_rpm43
    md_baseline = pd.DataFrame(md_baseline, columns=['MD'])

    md_baseline_uncorrected = md_train_idle_uncorrected + md_train_parked_uncorrected + md_train_t1_uncorrected + md_train_rpm32_uncorrected + md_train_t2_uncorrected + md_train_rpm43_uncorrected
    md_baseline_uncorrected = pd.DataFrame(md_baseline_uncorrected, columns=['MD'])

    md_test = md_idle + md_parked + md_t1 + md_rpm32 + md_t2 + md_rpm43
    md_test_uncorrected = md_idle_uncorrected + md_parked_uncorrected + md_t1_uncorrected + md_rpm32_uncorrected + md_t2_uncorrected + md_rpm43_uncorrected
    # md_test = np.array(md_test)
    md_test = pd.DataFrame(md_test, columns=['MD'])
    md_test_uncorrected = pd.DataFrame(md_test_uncorrected, columns=['MD'])

    scenario_baseline = pd.concat([scenario_train_idle, scenario_train_parked])
    scenario_baseline = pd.concat([scenario_baseline, scenario_train_t1])
    scenario_baseline = pd.concat([scenario_baseline, scenario_train_rpm32])
    scenario_baseline = pd.concat([scenario_baseline, scenario_train_t2])
    scenario_baseline = pd.concat([scenario_baseline, scenario_train_rpm43])

    scenario_test = pd.concat([scenario_idle, scenario_parked])
    scenario_test = pd.concat([scenario_test, scenario_t1])
    scenario_test = pd.concat([scenario_test, scenario_rpm32])
    scenario_test = pd.concat([scenario_test, scenario_t2])
    scenario_test = pd.concat([scenario_test, scenario_rpm43])

    md_baseline = md_baseline.reset_index(drop=True)
    md_baseline_uncorrected = md_baseline_uncorrected.reset_index(drop=True)
    scenario_baseline = scenario_baseline.reset_index(drop=True)

    md_test = md_test.reset_index(drop=True)
    md_test_uncorrected = md_test_uncorrected.reset_index(drop=True)
    scenario_test = scenario_test.reset_index(drop=True)

    baseline = pd.concat([md_baseline, scenario_baseline], axis=1)
    baseline_uncorrected = pd.concat([md_baseline_uncorrected, scenario_baseline], axis=1)
    test = pd.concat([md_test, scenario_test], axis=1)
    test_uncorrected = pd.concat([md_test_uncorrected, scenario_test], axis=1)

    total = pd.concat([baseline, test])
    total_uncorrected = pd.concat([baseline_uncorrected, test_uncorrected])
    total['Date'] = pd.to_datetime(total['Date'], format='%d/%m/%y %H:%M')
    total_uncorrected['Date'] = pd.to_datetime(total_uncorrected['Date'], format='%d/%m/%y %H:%M')

    print(type(total.Date[0]))
    total = total.sort_values(by='Date')
    total_uncorrected = total_uncorrected.sort_values(by='Date')

    plt.figure()
    colors_scenario = {'undamaged': 'green', '15cm': 'yellow','30cm':'orange', '45cm':'red', 'repaired':'blue'}
    colors_mode = {'parked':'grey', 
                   'idle':'#ffa600',
                   't1':'#ff6361',
                   'rpm32':'#bc5090',
                   't2':'#58508d',
                   'rpm43':'#003f5c'}
    
    undamaged = total.loc[total['DamageScenario'] == 'undamaged']
    damage_15cm = total.loc[total['DamageScenario'] == '15cm']
    damage_30cm = total.loc[total['DamageScenario'] == '30cm']
    damage_45cm = total.loc[total['DamageScenario'] == '45cm']
    repaired = total.loc[total['DamageScenario'] == 'repaired']

    idle = total.loc[total['Mode'] == 'idle']
    parked = total.loc[total['Mode'] == 'parked']
    t1 = total.loc[total['Mode'] == 't1']
    rpm32 = total.loc[total['Mode'] == 'rpm32']
    t2 = total.loc[total['Mode'] == 't2']
    rpm43 = total.loc[total['Mode'] == 'rpm43']

    idle_uncorrected = total_uncorrected.loc[total['Mode'] == 'idle']
    parked_uncorrected = total_uncorrected.loc[total['Mode'] == 'parked']
    t1_uncorrected = total_uncorrected.loc[total['Mode'] == 't1']
    rpm32_uncorrected = total_uncorrected.loc[total['Mode'] == 'rpm32']
    t2_uncorrected = total_uncorrected.loc[total['Mode'] == 't2']
    rpm43_uncorrected = total_uncorrected.loc[total['Mode'] == 'rpm43']

    low = 0.01
    high = 10e18

    #CHANGE OPERATIONAL MODE
    rpm43_undamaged = rpm43.loc[rpm43['DamageScenario'] == 'undamaged']
    rpm43_15cm = rpm43.loc[rpm43['DamageScenario'] == '15cm']
    rpm43_30cm = rpm43.loc[rpm43['DamageScenario'] == '30cm']
    rpm43_45cm = rpm43.loc[rpm43['DamageScenario'] == '45cm']
    rpm43_repaired = rpm43.loc[rpm43['DamageScenario'] == 'repaired']

    median_undamaged = np.median(rpm43_undamaged['MD'])
    median_damage_15cm = np.median(rpm43_15cm['MD'])
    median_damage_30cm = np.median(rpm43_30cm['MD'])
    median_damage_45cm = np.median(rpm43_45cm['MD'])
    median_repaired = np.median(rpm43_repaired['MD'])

    plt.scatter(x=rpm43_uncorrected.Date,y=rpm43_uncorrected.MD,s=1,c='gainsboro')
    plt.scatter(x=rpm43.Date,y=rpm43.MD,s=1,c=rpm43['DamageScenario'].map(colors_scenario))
    plt.yscale('log')
    plt.ylim([low, high])
    plt.xlim([datetime(2014,11,25),datetime(2015,3,20)])
    plt.plot([rpm43_undamaged['Date'].min(), rpm43_undamaged['Date'].max()], [median_undamaged, median_undamaged],color='black')
    plt.plot([rpm43_15cm['Date'].min(), rpm43_15cm['Date'].max()], [median_damage_15cm, median_damage_15cm],color='black')
    plt.plot([rpm43_30cm['Date'].min(), rpm43_30cm['Date'].max()], [median_damage_30cm, median_damage_30cm],color='black')
    plt.plot([rpm43_45cm['Date'].min(), rpm43_45cm['Date'].max()], [median_damage_45cm, median_damage_45cm],color='black')
    plt.plot([rpm43_repaired['Date'].min(), rpm43_repaired['Date'].max()], [median_repaired, median_repaired],color='black')
    plt.show()

    plt.figure()
    plt.scatter(x=rpm43_uncorrected.Temperature,y=rpm43_uncorrected.MD,s=1,c='gainsboro')
    plt.scatter(x=rpm43.Temperature,y=rpm43.MD,s=1,c=rpm43['DamageScenario'].map(colors_scenario))
    plt.yscale('log')
    plt.ylim([low, high])
    plt.xlim([-10,15])
    plt.show()

control_chart_with_reference_operational_mode('adaptive_order6',0.0)