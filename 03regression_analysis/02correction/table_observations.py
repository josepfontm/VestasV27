import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

path_scenario_idle = "../code_6modes/03regression_analysis/00standarized/scenario_idle.csv"
path_scenario_parked = "../code_6modes/03regression_analysis/00standarized/scenario_parked.csv"
path_scenario_t1 = "../code_6modes/03regression_analysis/00standarized/scenario_t1.csv"
path_scenario_rpm32 = "../code_6modes/03regression_analysis/00standarized/scenario_rpm32.csv"
path_scenario_t2 = "../code_6modes/03regression_analysis/00standarized/scenario_t2.csv"
path_scenario_rpm43 = "../code_6modes/03regression_analysis/00standarized/scenario_rpm43.csv"

scenario_idle = pd.read_csv(path_scenario_idle,index_col=0,header=0,sep=';')
scenario_parked = pd.read_csv(path_scenario_parked,index_col=0,header=0,sep=';')
scenario_t1 = pd.read_csv(path_scenario_t1,index_col=0,header=0,sep=';')
scenario_rpm32 = pd.read_csv(path_scenario_rpm32,index_col=0,header=0,sep=';')
scenario_t2 = pd.read_csv(path_scenario_t2,index_col=0,header=0,sep=';')
scenario_rpm43 = pd.read_csv(path_scenario_rpm43,index_col=0,header=0,sep=';')

path_scenario_train_idle = "../code_6modes/03regression_analysis/00standarized/scenario_train_idle.csv"
path_scenario_train_parked = "../code_6modes/03regression_analysis/00standarized/scenario_train_parked.csv"
path_scenario_train_t1 = "../code_6modes/03regression_analysis/00standarized/scenario_train_t1.csv"
path_scenario_train_rpm32 = "../code_6modes/03regression_analysis/00standarized/scenario_train_rpm32.csv"
path_scenario_train_t2 = "../code_6modes/03regression_analysis/00standarized/scenario_train_t2.csv"
path_scenario_train_rpm43 = "../code_6modes/03regression_analysis/00standarized/scenario_train_rpm43.csv"

scenario_train_idle = pd.read_csv(path_scenario_train_idle,index_col=0,header=0,sep=';')
scenario_train_parked = pd.read_csv(path_scenario_train_parked,index_col=0,header=0,sep=';')
scenario_train_t1 = pd.read_csv(path_scenario_train_t1,index_col=0,header=0,sep=';')
scenario_train_rpm32 = pd.read_csv(path_scenario_train_rpm32,index_col=0,header=0,sep=';')
scenario_train_t2 = pd.read_csv(path_scenario_train_t2,index_col=0,header=0,sep=';')
scenario_train_rpm43 = pd.read_csv(path_scenario_train_rpm43,index_col=0,header=0,sep=';')

parked_undamaged = 0 #Gives error because no observations are present for this case
parked_15cm = scenario_parked['DamageScenario'].value_counts()['15cm']
parked_30cm = scenario_parked['DamageScenario'].value_counts()['30cm']
parked_45cm = scenario_parked['DamageScenario'].value_counts()['45cm']
parked_repaired = scenario_train_parked['DamageScenario'].value_counts()['repaired']

idle_undamaged = scenario_idle['DamageScenario'].value_counts()['undamaged'] 
idle_15cm = 0 #Gives error because no observations are present for this case
idle_30cm = 0 #Gives error because no observations are present for this case
idle_45cm = 0 #Gives error because no observations are present for this case
idle_repaired = scenario_train_idle['DamageScenario'].value_counts()['repaired']

t1_undamaged = scenario_t1['DamageScenario'].value_counts()['undamaged'] 
t1_15cm = scenario_t1['DamageScenario'].value_counts()['15cm']
t1_30cm = scenario_t1['DamageScenario'].value_counts()['30cm']
t1_45cm = scenario_t1['DamageScenario'].value_counts()['45cm']
t1_repaired = scenario_train_t1['DamageScenario'].value_counts()['repaired']

rpm32_undamaged = scenario_rpm32['DamageScenario'].value_counts()['undamaged'] 
rpm32_15cm = scenario_rpm32['DamageScenario'].value_counts()['15cm']
rpm32_30cm = scenario_rpm32['DamageScenario'].value_counts()['30cm']
rpm32_45cm = scenario_rpm32['DamageScenario'].value_counts()['45cm']
rpm32_repaired = scenario_train_rpm32['DamageScenario'].value_counts()['repaired']

t2_undamaged = scenario_t2['DamageScenario'].value_counts()['undamaged'] 
t2_15cm = scenario_t2['DamageScenario'].value_counts()['15cm']
t2_30cm = scenario_t2['DamageScenario'].value_counts()['30cm']
t2_45cm = scenario_t2['DamageScenario'].value_counts()['45cm']
t2_repaired = scenario_train_t2['DamageScenario'].value_counts()['repaired']

rpm43_undamaged = scenario_rpm43['DamageScenario'].value_counts()['undamaged'] 
rpm43_15cm = scenario_rpm43['DamageScenario'].value_counts()['15cm']
rpm43_30cm = scenario_rpm43['DamageScenario'].value_counts()['30cm']
rpm43_45cm = scenario_rpm43['DamageScenario'].value_counts()['45cm']
rpm43_repaired = scenario_train_rpm43['DamageScenario'].value_counts()['repaired']

damage_scenarios = ['undamaged','15cm','30cm','45cm','repaired']
operational_modes = ['Parked','Idle','T1','RPM32','T2','RPM43']

data = np.array([[parked_undamaged, idle_undamaged,t1_undamaged,rpm32_undamaged,t2_undamaged,rpm43_undamaged],
                 [parked_15cm, idle_15cm,t1_15cm,rpm32_15cm,t2_15cm,rpm43_15cm],
                 [parked_30cm,idle_30cm,t1_30cm,rpm32_30cm,t2_30cm,rpm43_30cm],
                 [parked_45cm,idle_45cm,t1_45cm,rpm32_45cm,t2_45cm,rpm43_45cm],
                 [parked_repaired,idle_repaired,t1_repaired,rpm32_repaired,t2_repaired,rpm43_repaired]])

table = pd.DataFrame(data, damage_scenarios,operational_modes)
print(table)