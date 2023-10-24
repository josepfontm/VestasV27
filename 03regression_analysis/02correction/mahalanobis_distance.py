import pandas as pd
import numpy as np
from sklearn.metrics import f1_score,confusion_matrix

def metrics(regression_model: str, threshold:float):

    """
    Apply Mahalnobis Distance and quantify performance of a set of corrected DSFs.

    Parameters
    ----------
    regression_type : string
        Name folder used to correct data.

    threshold: float
        DSFs are only corrected if the R^2 is higher or equal than the threshold defined.

    Returns
    ----------
    metrics: array
        Metrics used to objectively quantify the permonace of the migitation procedure.
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

    #True values: 1 (undamaged/repaired) and 0 (damaged)
    scenario_idle = pd.read_csv(path_scenario_idle,sep=';').DamageScenario.to_numpy()
    scenario_parked = pd.read_csv(path_scenario_parked,sep=';').DamageScenario.to_numpy()
    scenario_t1 = pd.read_csv(path_scenario_t1,sep=';').DamageScenario.to_numpy()
    scenario_rpm32 = pd.read_csv(path_scenario_rpm32,sep=';').DamageScenario.to_numpy()
    scenario_t2 = pd.read_csv(path_scenario_t2,sep=';').DamageScenario.to_numpy()
    scenario_rpm43 = pd.read_csv(path_scenario_rpm43,sep=';').DamageScenario.to_numpy()

    print(np.shape(scenario_parked))
    print(np.shape(md_parked))

    scenario_train_idle = pd.read_csv(path_scenario_train_idle,sep=';').DamageScenario.to_numpy()
    scenario_train_parked = pd.read_csv(path_scenario_train_parked,sep=';').DamageScenario.to_numpy()
    scenario_train_t1 = pd.read_csv(path_scenario_train_t1,sep=';').DamageScenario.to_numpy()
    scenario_train_rpm32 = pd.read_csv(path_scenario_train_rpm32,sep=';').DamageScenario.to_numpy()
    scenario_train_t2 = pd.read_csv(path_scenario_train_t2,sep=';').DamageScenario.to_numpy()
    scenario_train_rpm43 = pd.read_csv(path_scenario_train_rpm43,sep=';').DamageScenario.to_numpy()

    labels_idle = np.where((scenario_idle == 'undamaged') | (scenario_idle == 'repaired'),0,1)
    labels_parked = np.where((scenario_parked == 'undamaged') | (scenario_parked == 'repaired'),0,1)
    labels_t1 = np.where((scenario_t1 == 'undamaged') | (scenario_t1 == 'repaired'),0,1)
    labels_rpm32 = np.where((scenario_rpm32 == 'undamaged') | (scenario_rpm32 == 'repaired'),0,1)
    labels_t2 = np.where((scenario_t2 == 'undamaged') | (scenario_t2 == 'repaired'),0,1)
    labels_rpm43 = np.where((scenario_rpm43 == 'undamaged') | (scenario_rpm43 == 'repaired'),0,1)

    labels_train_idle = np.where((scenario_train_idle == 'undamaged') | (scenario_train_idle == 'repaired'),0,1)
    labels_train_parked = np.where((scenario_train_parked == 'undamaged') | (scenario_train_parked == 'repaired'),0,1)
    labels_train_t1 = np.where((scenario_train_t1 == 'undamaged') | (scenario_train_t1 == 'repaired'),0,1)
    labels_train_rpm32 = np.where((scenario_train_rpm32 == 'undamaged') | (scenario_train_rpm32 == 'repaired'),0,1)
    labels_train_t2 = np.where((scenario_train_t2 == 'undamaged') | (scenario_train_t2 == 'repaired'),0,1)
    labels_train_rpm43 = np.where((scenario_train_rpm43 == 'undamaged') | (scenario_train_rpm43 == 'repaired'),0,1)

    #Predicted values
    md_idle = (md_idle > idle_threshold).astype(int)
    md_parked = (md_parked > parked_threshold).astype(int)
    md_t1 = (md_t1 > t1_threshold).astype(int)
    md_rpm32 = (md_rpm32 > rpm32_threshold).astype(int)
    md_t2 = (md_t2 > t2_threshold).astype(int)
    md_rpm43 = (md_rpm43 > rpm43_threshold).astype(int)

    labels = np.hstack((labels_idle, labels_parked))
    labels = np.hstack((labels, labels_t1))
    labels = np.hstack((labels, labels_rpm32))
    labels = np.hstack((labels, labels_t2))
    labels = np.hstack((labels, labels_rpm43))

    predicted = np.hstack((md_idle, md_parked))
    predicted = np.hstack((predicted, md_t1))
    predicted = np.hstack((predicted, md_rpm32))
    predicted = np.hstack((predicted, md_t2))
    predicted = np.hstack((predicted, md_rpm43))

    print(np.shape(labels))
    print(np.shape(predicted))

    print(np.shape(labels_idle))
    print(np.shape(md_idle))

    print(np.shape(labels_parked))
    print(np.shape(md_parked))

    print(np.shape(labels_t1))
    print(np.shape(md_t1))

    print(threshold)
    print('---TOTAL---')
    print(f1_score(labels,predicted,average='weighted')) 

    print('---PARKED---')
    print(f1_score(labels_parked,md_parked,average='weighted')) 

    print('---IDLE---')
    print(f1_score(labels_idle,md_idle,average='weighted')) 

    print('---T1---')
    print(f1_score(labels_t1,md_t1,average='weighted')) 

    print('---RPM32---')
    print(f1_score(labels_rpm32,md_rpm32,average='weighted')) 

    print('---T2---')
    print(f1_score(labels_t2,md_t2,average='weighted')) 

    print('---RPM43---')
    print(f1_score(labels_rpm43,md_rpm43,average='weighted')) 

    print('---------------------------------------')

# metrics('svr', 0.0)
# metrics('svr', 0.05)
# metrics('svr', 0.1)
# metrics('svr', 0.15)
# metrics('svr', 0.2)
# metrics('svr', 0.25)
# metrics('svr', 0.3)
metrics('svr', 0.35)
# metrics('svr', 0.4)
# metrics('svr', 0.45)
# metrics('svr', 0.5)
# metrics('svr', 0.55)
metrics('svr', 0.6)
# metrics('svr', 0.65)
# metrics('svr', 0.7)
# metrics('svr', 0.75)
# metrics('svr', 0.8)
metrics('svr', 0.85)
# metrics('svr', 0.90)
# metrics('svr', 0.95)
# metrics('svr', 1.0)