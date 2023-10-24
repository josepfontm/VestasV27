import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#Change paths if necessary
path_metadata_idle = '../VestasV27/01clean_data/results/metadata_idle.csv'
path_metadata_parked = '../VestasV27/01clean_data/results/metadata_parked.csv'
path_metadata_t1 = '../VestasV27/01clean_data/results/metadata_t1.csv'
path_metadata_rpm32 = '../VestasV27/01clean_data/results/metadata_rpm32.csv'
path_metadata_t2 = '../VestasV27/01clean_data/results/metadata_t2.csv'
path_metadata_rpm43 = '../VestasV27/01clean_data/results/metadata_rpm43.csv'

path_psd_idle = '../VestasV27/01clean_data/results/psd_idle.csv'
path_psd_parked = '../VestasV27/01clean_data/results/psd_parked.csv'
path_psd_t1 = '../VestasV27/01clean_data/results/psd_t1.csv'
path_psd_rpm32 = '../VestasV27/01clean_data/results/psd_rpm32.csv'
path_psd_t2 = '../VestasV27/01clean_data/results/psd_t2.csv'
path_psd_rpm43 = '../VestasV27/01clean_data/results/psd_rpm43.csv'

#Generate Panda DataFrames and convert to numpy arrays
metadata_idle = pd.read_csv(path_metadata_idle,index_col=0,sep=';').reset_index()
metadata_parked = pd.read_csv(path_metadata_parked,index_col=0,sep=';').reset_index()
metadata_t1 = pd.read_csv(path_metadata_t1,index_col=0,sep=';').reset_index()
metadata_rpm32 = pd.read_csv(path_metadata_rpm32,index_col=0,sep=';').reset_index()
metadata_t2 = pd.read_csv(path_metadata_t2,index_col=0,sep=';').reset_index()
metadata_rpm43 = pd.read_csv(path_metadata_rpm43,index_col=0,sep=';').reset_index()

psd_idle = pd.read_csv(path_psd_idle,index_col=0,sep=';').reset_index()
psd_parked = pd.read_csv(path_psd_parked,index_col=0,sep=';').reset_index()
psd_t1 = pd.read_csv(path_psd_t1,index_col=0,sep=';').reset_index()
psd_rpm32 = pd.read_csv(path_psd_rpm32,index_col=0,sep=';').reset_index()
psd_t2 = pd.read_csv(path_psd_t2,index_col=0,sep=';').reset_index()
psd_rpm43 = pd.read_csv(path_psd_rpm43,index_col=0,sep=';').reset_index()

psd_idle = psd_idle.iloc[:,1:]
psd_parked = psd_parked.iloc[:,1:]
psd_t1 = psd_t1.iloc[:,1:]
psd_rpm32 = psd_rpm32.iloc[:,1:]
psd_t2 = psd_t2.iloc[:,1:]
psd_rpm43 = psd_rpm43.iloc[:,1:]

psd_idle = np.log10(psd_idle)
psd_parked = np.log10(psd_parked)
psd_t1 = np.log10(psd_t1)
psd_rpm32 = np.log10(psd_rpm32)
psd_t2 = np.log10(psd_t2)
psd_rpm43 = np.log10(psd_rpm43)

#Extract train data which corresponds to repaired damage scenario
metadata_idle_repaired = metadata_idle[metadata_idle.DamageScenario == 'repaired']
psd_idle_repaired = psd_idle[metadata_idle.DamageScenario == 'repaired']

metadata_parked_repaired = metadata_parked[metadata_parked.DamageScenario == 'repaired']
psd_parked_repaired = psd_parked[metadata_parked.DamageScenario == 'repaired']

metadata_t1_repaired = metadata_t1[metadata_t1.DamageScenario == 'repaired']
psd_t1_repaired = psd_t1[metadata_t1.DamageScenario == 'repaired']

metadata_rpm32_repaired = metadata_rpm32[metadata_rpm32.DamageScenario == 'repaired']
psd_rpm32_repaired = psd_rpm32[metadata_rpm32.DamageScenario == 'repaired']

metadata_t2_repaired = metadata_t2[metadata_t2.DamageScenario == 'repaired']
psd_t2_repaired = psd_t2[metadata_t2.DamageScenario == 'repaired']

metadata_rpm43_repaired = metadata_rpm43[metadata_rpm43.DamageScenario == 'repaired']
psd_rpm43_repaired = psd_rpm43[metadata_rpm43.DamageScenario == 'repaired']

#Test split (does not contazin training data)
metadata_idle = metadata_idle.drop(metadata_idle[metadata_idle.DamageScenario == 'repaired'].index)
psd_idle = psd_idle.loc[metadata_idle.index]

metadata_parked = metadata_parked.drop(metadata_parked[metadata_parked.DamageScenario == 'repaired'].index)
psd_parked = psd_parked.loc[metadata_parked.index]

metadata_t1 = metadata_t1.drop(metadata_t1[metadata_t1.DamageScenario == 'repaired'].index)
psd_t1 = psd_t1.loc[metadata_t1.index]

metadata_rpm32 = metadata_rpm32.drop(metadata_rpm32[metadata_rpm32.DamageScenario == 'repaired'].index)
psd_rpm32 = psd_rpm32.loc[metadata_rpm32.index]

metadata_t2 = metadata_t2.drop(metadata_t2[metadata_t2.DamageScenario == 'repaired'].index)
psd_t2 = psd_t2.loc[metadata_t2.index]

metadata_rpm43 = metadata_rpm43.drop(metadata_rpm43[metadata_rpm43.DamageScenario == 'repaired'].index)
psd_rpm43 = psd_rpm43.loc[metadata_rpm43.index]

ds_idle = pd.concat([metadata_idle.DamageScenario, metadata_idle.Date, metadata_idle.Temperature], axis=1)
ds_parked = pd.concat([metadata_parked.DamageScenario, metadata_parked.Date, metadata_parked.Temperature], axis=1)
ds_t1 = pd.concat([metadata_t1.DamageScenario, metadata_t1.Date, metadata_t1.Temperature], axis=1)
ds_rpm32 = pd.concat([metadata_rpm32.DamageScenario, metadata_rpm32.Date, metadata_rpm32.Temperature], axis=1)
ds_t2 = pd.concat([metadata_t2.DamageScenario, metadata_t2.Date, metadata_t2.Temperature], axis=1)
ds_rpm43 = pd.concat([metadata_rpm43.DamageScenario, metadata_rpm43.Date, metadata_rpm43.Temperature], axis=1)

ds_idle.to_csv('../VestasV27/02dimensionality_reduction/results/scenario_idle.csv',sep=';')
ds_parked.to_csv('../VestasV27/02dimensionality_reduction/results/scenario_parked.csv',sep=';')
ds_t1.to_csv('../VestasV27/02dimensionality_reduction/results/scenario_t1.csv',sep=';')
ds_rpm32.to_csv('../VestasV27/02dimensionality_reduction/results/scenario_rpm32.csv',sep=';')
ds_t2.to_csv('../VestasV27/02dimensionality_reduction/results/scenario_t2.csv',sep=';')
ds_rpm43.to_csv('../VestasV27/02dimensionality_reduction/results/scenario_rpm43.csv',sep=';')

print('Shapes')
print(np.shape(psd_parked))
print(np.shape(psd_parked_repaired))
print(np.shape(psd_idle))
print(np.shape(psd_idle_repaired))
print(np.shape(psd_t1))
print(np.shape(psd_t1_repaired))
print(np.shape(psd_rpm32))
print(np.shape(psd_rpm32_repaired))
print(np.shape(psd_t2))
print(np.shape(psd_t2_repaired))
print(np.shape(psd_rpm43))
print(np.shape(psd_rpm43_repaired))

# plot = psd_rpm43_repaired.to_numpy()
# print('rpm43 repaired')
# plt.figure()
# plt.plot(plot.T)
# plt.show()

ds_train_idle = pd.concat([metadata_idle_repaired.DamageScenario, metadata_idle_repaired.Date, metadata_idle_repaired.Temperature], axis=1)
ds_train_parked = pd.concat([metadata_parked_repaired.DamageScenario, metadata_parked_repaired.Date, metadata_parked_repaired.Temperature], axis=1)
ds_train_t1 = pd.concat([metadata_t1_repaired.DamageScenario, metadata_t1_repaired.Date, metadata_t1_repaired.Temperature], axis=1)
ds_train_rpm32 = pd.concat([metadata_rpm32_repaired.DamageScenario, metadata_rpm32_repaired.Date, metadata_rpm32_repaired.Temperature], axis=1)
ds_train_t2 = pd.concat([metadata_t2_repaired.DamageScenario, metadata_t2_repaired.Date, metadata_t2_repaired.Temperature], axis=1)
ds_train_rpm43 = pd.concat([metadata_rpm43_repaired.DamageScenario, metadata_rpm43_repaired.Date, metadata_rpm43_repaired.Temperature], axis=1)

ds_train_idle.to_csv('../VestasV27/02dimensionality_reduction/results/scenario_train_idle.csv',sep=';')
ds_train_parked.to_csv('../VestasV27/02dimensionality_reduction/results/scenario_train_parked.csv',sep=';')
ds_train_t1.to_csv('../VestasV27/02dimensionality_reduction/results/scenario_train_t1.csv',sep=';')
ds_train_rpm32.to_csv('../VestasV27/02dimensionality_reduction/results/scenario_train_rpm32.csv',sep=';')
ds_train_t2.to_csv('../VestasV27/02dimensionality_reduction/results/scenario_train_t2.csv',sep=';')
ds_train_rpm43.to_csv('../VestasV27/02dimensionality_reduction/results/scenario_train_rpm43.csv',sep=';')

#Training data only using samples from repaired data (Unsupervised manner)
psd_train = pd.concat([psd_idle_repaired, psd_parked_repaired], axis=0)
psd_train = pd.concat([psd_train, psd_t1_repaired], axis=0)
psd_train = pd.concat([psd_train, psd_rpm32_repaired], axis = 0)
psd_train = pd.concat([psd_train, psd_t2_repaired],axis=0)
psd_train = pd.concat([psd_train, psd_rpm43_repaired], axis=0)

metadata_train = pd.concat([metadata_idle_repaired, metadata_parked_repaired], axis=0)
metadata_train = pd.concat([metadata_train, metadata_t1_repaired], axis=0)
metadata_train = pd.concat([metadata_train, metadata_rpm32_repaired], axis=0)
metadata_train = pd.concat([metadata_train, metadata_t2_repaired], axis=0)
metadata_train = pd.concat([metadata_train, metadata_rpm43_repaired], axis=0)

#Reference state used for DSF extraction (PCA)
psd_ref = psd_rpm43_repaired
metadata_ref = metadata_rpm43_repaired

#PCA FROM ENVIRONMENTAL DATA
#Extract some columns to use in PCA (These variables are continuous, good for linear PCA)
env_ref = metadata_ref.loc[:,["Wind","WindDirection","Temperature"]].to_numpy()
env_train = metadata_train.loc[:,["Wind","WindDirection","Temperature"]].to_numpy()

env_idle = metadata_idle.loc[:,["Wind","WindDirection","Temperature"]].to_numpy()
env_parked = metadata_parked.loc[:,["Wind","WindDirection","Temperature"]].to_numpy()
env_t1 = metadata_t1.loc[:,["Wind","WindDirection","Temperature"]].to_numpy()
env_rpm32 = metadata_rpm32.loc[:,["Wind","WindDirection","Temperature"]].to_numpy()
env_t2 = metadata_t2.loc[:,["Wind","WindDirection","Temperature"]].to_numpy()
env_rpm43 = metadata_rpm43.loc[:,["Wind","WindDirection","Temperature"]].to_numpy()

np.savetxt('../VestasV27/02dimensionality_reduction/results/env_idle.csv', env_idle, delimiter = ';')
np.savetxt('../VestasV27/02dimensionality_reduction/results/env_parked.csv', env_parked, delimiter = ';')
np.savetxt('../VestasV27/02dimensionality_reduction/results/env_t1.csv', env_t1, delimiter = ';')
np.savetxt('../VestasV27/02dimensionality_reduction/results/env_rpm32.csv', env_rpm32, delimiter = ';')
np.savetxt('../VestasV27/02dimensionality_reduction/results/env_t2.csv', env_t2, delimiter = ';')
np.savetxt('../VestasV27/02dimensionality_reduction/results/env_rpm43.csv', env_rpm43, delimiter = ';')

env_train_idle = metadata_idle_repaired.loc[:,["Wind","WindDirection","Temperature"]].to_numpy()
env_train_parked = metadata_parked_repaired.loc[:,["Wind","WindDirection","Temperature"]].to_numpy()
env_train_t1 = metadata_t1_repaired.loc[:,["Wind","WindDirection","Temperature"]].to_numpy()
env_train_rpm32 = metadata_rpm32_repaired.loc[:,["Wind","WindDirection","Temperature"]].to_numpy()
env_train_t2 = metadata_t2_repaired.loc[:,["Wind","WindDirection","Temperature"]].to_numpy()
env_train_rpm43 = metadata_rpm43_repaired.loc[:,["Wind","WindDirection","Temperature"]].to_numpy()

env_scaler = StandardScaler()
env_train_norm = env_scaler.fit_transform(env_train)

env_train_idle_norm = env_scaler.transform(env_train_idle)
env_train_parked_norm = env_scaler.transform(env_train_parked)
env_train_t1_norm = env_scaler.transform(env_train_t1)
env_train_rpm32_norm = env_scaler.transform(env_train_rpm32)
env_train_t2_norm = env_scaler.transform(env_train_t2)
env_train_rpm43_norm = env_scaler.transform(env_train_rpm43)

env_ref_norm = env_scaler.transform(env_ref)
env_idle_norm = env_scaler.transform(env_idle)
env_parked_norm = env_scaler.transform(env_parked)
env_t1_norm = env_scaler.transform(env_t1)
env_rpm32_norm = env_scaler.transform(env_rpm32)
env_t2_norm = env_scaler.transform(env_t2)
env_rpm43_norm = env_scaler.transform(env_rpm43)

pca_env = PCA(n_components=0.95)
pca_env.fit(env_train_norm)
print(pca_env.n_components_)
print(pca_env.explained_variance_ratio_)

pca_env_train = pca_env.transform(env_train_norm)

pca_env_train_idle = pca_env.transform(env_train_idle_norm)
pca_env_train_parked = pca_env.transform(env_train_parked_norm)
pca_env_train_t1 = pca_env.transform(env_train_t1_norm)
pca_env_train_rpm32 = pca_env.transform(env_train_rpm32_norm)
pca_env_train_t2 = pca_env.transform(env_train_t2_norm)
pca_env_train_rpm43 = pca_env.transform(env_train_rpm43_norm)

pca_env_ref = pca_env.transform(env_ref_norm)
pca_env_idle = pca_env.transform(env_idle_norm)
pca_env_parked = pca_env.transform(env_parked_norm)
pca_env_t1 = pca_env.transform(env_t1_norm)
pca_env_rpm32 = pca_env.transform(env_rpm32_norm)
pca_env_t2 = pca_env.transform(env_t2_norm)
pca_env_rpm43 = pca_env.transform(env_rpm43_norm)

#PCA FROM OPERATIONAL DATA
op_ref = metadata_ref.loc[:,["PitchMean","PitchStd","PitchMin","PitchMax","RPM"]].to_numpy() #43rpm repaired
op_ref_idle = metadata_idle_repaired.loc[:,["PitchMean","PitchStd","PitchMin","PitchMax","RPM"]].to_numpy()
op_ref_parked = metadata_parked_repaired.loc[:,["PitchMean","PitchStd","PitchMin","PitchMax","RPM"]].to_numpy()
op_ref_t1 = metadata_t1_repaired.loc[:,["PitchMean","PitchStd","PitchMin","PitchMax","RPM"]].to_numpy()
op_ref_rpm32 = metadata_rpm32_repaired.loc[:,["PitchMean","PitchStd","PitchMin","PitchMax","RPM"]].to_numpy()
op_ref_t2 = metadata_t2_repaired.loc[:,["PitchMean","PitchStd","PitchMin","PitchMax","RPM"]].to_numpy()
op_ref_rpm43 = op_ref

op_idle = metadata_idle.loc[:,["PitchMean","PitchStd","PitchMin","PitchMax","RPM"]].to_numpy()
op_parked = metadata_parked.loc[:,["PitchMean","PitchStd","PitchMin","PitchMax","RPM"]].to_numpy()
op_t1 = metadata_t1.loc[:,["PitchMean","PitchStd","PitchMin","PitchMax","RPM"]].to_numpy()
op_rpm32 = metadata_rpm32.loc[:,["PitchMean","PitchStd","PitchMin","PitchMax","RPM"]].to_numpy()
op_t2 = metadata_t2.loc[:,["PitchMean","PitchStd","PitchMin","PitchMax","RPM"]].to_numpy()
op_rpm43 = metadata_rpm43.loc[:,["PitchMean","PitchStd","PitchMin","PitchMax","RPM"]].to_numpy()

np.savetxt('../VestasV27/02dimensionality_reduction/results/op_idle.csv', op_idle, delimiter = ';')
np.savetxt('../VestasV27/02dimensionality_reduction/results/op_parked.csv', op_parked, delimiter = ';')
np.savetxt('../VestasV27/02dimensionality_reduction/results/op_t1.csv', op_t1, delimiter = ';')
np.savetxt('../VestasV27/02dimensionality_reduction/results/op_rpm32.csv', op_rpm32, delimiter = ';')
np.savetxt('../VestasV27/02dimensionality_reduction/results/op_t2.csv', op_t2, delimiter = ';')
np.savetxt('../VestasV27/02dimensionality_reduction/results/op_rpm43.csv', op_rpm43, delimiter = ';')

op_scaler = StandardScaler()

op_ref_idle_norm = op_scaler.fit_transform(op_ref_idle)
op_idle_norm = op_scaler.transform(op_idle)
#PCA Analysis for each operational mode
pca_op = PCA(n_components=0.95)
pca_op_train_idle = pca_op.fit_transform(op_ref_idle_norm)

print('Weights PCA Operational')
print(pca_op.components_)
pca_op_idle = pca_op.transform(op_idle_norm)

op_ref_parked_norm = op_scaler.fit_transform(op_ref_parked)
op_parked_norm = op_scaler.transform(op_parked)
#PCA Analysis for each operational mode
pca_op = PCA(n_components=0.95)
pca_op_train_parked = pca_op.fit_transform(op_ref_parked_norm)
pca_op_parked = pca_op.transform(op_parked_norm)

op_ref_t1_norm = op_scaler.fit_transform(op_ref_t1)
op_t1_norm = op_scaler.transform(op_t1)
#PCA Analysis for each operational mode
pca_op = PCA(n_components=0.95)
pca_op_train_t1 = pca_op.fit_transform(op_ref_t1_norm)
pca_op_t1 = pca_op.transform(op_t1_norm)

op_ref_rpm32_norm = op_scaler.fit_transform(op_ref_rpm32)
op_rpm32_norm = op_scaler.transform(op_rpm32)
#PCA Analysis for each operational mode
pca_op = PCA(n_components=0.95)
pca_op_train_rpm32 = pca_op.fit_transform(op_ref_rpm32_norm)
pca_op_rpm32 = pca_op.transform(op_rpm32_norm)

op_ref_t2_norm = op_scaler.fit_transform(op_ref_t2)
op_t2_norm = op_scaler.transform(op_t2)
#PCA Analysis for each operational mode
pca_op = PCA(n_components=0.95)
pca_op_train_t2 = pca_op.fit_transform(op_ref_t2_norm)
pca_op_t2 = pca_op.transform(op_t2_norm)

op_ref_rpm43_norm = op_scaler.fit_transform(op_ref_rpm43)
op_rpm43_norm = op_scaler.transform(op_rpm43)
#PCA Analysis for each operational mode
pca_op = PCA(n_components=0.95)
pca_op_train_rpm43 = pca_op.fit_transform(op_ref_rpm43_norm)
pca_op_rpm43 = pca_op.transform(op_rpm43_norm)

#Damage Sensitive Features extracted from PSDs using PCA
#Normalize data (each feature should have a mean of 0 and a standard deviation of 1)
psd_scaler = StandardScaler()

psd_ref_norm = psd_scaler.fit_transform(psd_ref)

psd_idle_train_norm = psd_scaler.fit_transform(psd_idle_repaired)
psd_idle_norm = psd_scaler.transform(psd_idle)

psd_parked_train_norm = psd_scaler.fit_transform(psd_parked_repaired)
psd_parked_norm = psd_scaler.transform(psd_parked)

psd_t1_train_norm = psd_scaler.fit_transform(psd_t1_repaired)
psd_t1_norm = psd_scaler.transform(psd_t1)

psd_rpm32_train_norm = psd_scaler.fit_transform(psd_rpm32_repaired)
psd_rpm32_norm = psd_scaler.transform(psd_rpm32)

psd_t2_train_norm = psd_scaler.fit_transform(psd_t2_repaired)
psd_t2_norm = psd_scaler.transform(psd_t2)

psd_rpm43_train_norm = psd_scaler.fit_transform(psd_rpm43_repaired)
psd_rpm43_norm = psd_scaler.transform(psd_rpm43)

# print('normalized')
# plt.figure('Idle')
# plt.plot(psd_idle_norm.T,color='blue',linewidth=1)
# plt.show()
# print(np.shape(psd_idle_norm))

# plt.figure('T1')
# plt.plot(psd_t1_norm.T,color='blue',linewidth=1)
# plt.show()
# print(np.shape(psd_t1_norm))

# plt.figure('32 RPM')
# plt.plot(psd_rpm32_norm.T,color='blue',linewidth=1)
# plt.show()
# print(np.shape(psd_rpm32_norm))

# plt.figure('T2')
# plt.plot(psd_t2_norm.T,color='blue',linewidth=1)
# plt.show()
# print(np.shape(psd_t2_norm))

# plt.figure('43 RPM')
# plt.plot(psd_rpm43_norm.T,color='blue',linewidth=1)
# plt.show()
# print(np.shape(psd_rpm43_norm))

pca_psd = PCA(n_components=0.95)
pca_psd.fit(psd_ref_norm)

dsf_idle = pca_psd.transform(psd_idle_norm)
dsf_parked = pca_psd.transform(psd_parked_norm)
dsf_t1 = pca_psd.transform(psd_t1_norm)
dsf_rpm32 = pca_psd.transform(psd_rpm32_norm)
dsf_t2 = pca_psd.transform(psd_t2_norm)
dsf_rpm43 = pca_psd.transform(psd_rpm43_norm)

dsf_train_idle = pca_psd.transform(psd_idle_train_norm)
dsf_train_parked = pca_psd.transform(psd_parked_train_norm)
dsf_train_t1 = pca_psd.transform(psd_t1_train_norm)
dsf_train_rpm32 = pca_psd.transform(psd_rpm32_train_norm)
dsf_train_t2 = pca_psd.transform(psd_t2_train_norm)
dsf_train_rpm43 = pca_psd.transform(psd_rpm43_train_norm)

#Store Panda dataframes as csv files
#---IDLE---
#damage sensitive features
list_pc = list(range(np.shape(dsf_idle)[1]))
list_pc = list(map(lambda x: "PC" + str(x), list_pc))
dsf_idle = pd.DataFrame(dsf_idle, columns=list_pc)
dsf_idle.to_csv('../VestasV27/02dimensionality_reduction/results/dsf_idle.csv', sep=';')

list_pc = list(range(np.shape(dsf_train_idle)[1]))
list_pc = list(map(lambda x: "PC" + str(x), list_pc))
dsf_train_idle = pd.DataFrame(dsf_train_idle, columns=list_pc)
dsf_train_idle.to_csv('../VestasV27/02dimensionality_reduction/results/dsf_train_idle.csv', sep=';')

#operational_data
list_pc = list(range(np.shape(pca_op_idle)[1]))
list_pc = list(map(lambda x: "OP" + str(x), list_pc))
pca_op_idle = pd.DataFrame(pca_op_idle, columns=list_pc)
pca_op_idle.to_csv('../VestasV27/02dimensionality_reduction/results/pca_op_idle.csv', sep=';')

list_pc = list(range(np.shape(pca_op_train_idle)[1]))
list_pc = list(map(lambda x: "OP" + str(x), list_pc))
pca_op_train_idle = pd.DataFrame(pca_op_train_idle, columns=list_pc)
pca_op_train_idle.to_csv('../VestasV27/02dimensionality_reduction/results/pca_op_train_idle.csv', sep=';')

#envrionmental data
list_pc = list(range(np.shape(pca_env_idle)[1]))
list_pc = list(map(lambda x: "ENV" + str(x), list_pc))
pca_env_idle = pd.DataFrame(pca_env_idle, columns=list_pc)
pca_env_idle.to_csv('../VestasV27/02dimensionality_reduction/results/pca_env_idle.csv', sep=';')

list_pc = list(range(np.shape(pca_env_train_idle)[1]))
list_pc = list(map(lambda x: "ENV" + str(x), list_pc))
pca_env_train_idle = pd.DataFrame(pca_env_train_idle, columns=list_pc)
pca_env_train_idle.to_csv('../VestasV27/02dimensionality_reduction/results/pca_env_train_idle.csv', sep=';')

#---PARKED---
#damage sensitive features
list_pc = list(range(np.shape(dsf_parked)[1]))
list_pc = list(map(lambda x: "PC" + str(x), list_pc))
dsf_parked = pd.DataFrame(dsf_parked, columns=list_pc)
dsf_parked.to_csv('../VestasV27/02dimensionality_reduction/results/dsf_parked.csv', sep=';')

list_pc = list(range(np.shape(dsf_train_parked)[1]))
list_pc = list(map(lambda x: "PC" + str(x), list_pc))
dsf_train_parked = pd.DataFrame(dsf_train_parked, columns=list_pc)
dsf_train_parked.to_csv('../VestasV27/02dimensionality_reduction/results/dsf_train_parked.csv', sep=';')

#operational_data
list_pc = list(range(np.shape(pca_op_parked)[1]))
list_pc = list(map(lambda x: "OP" + str(x), list_pc))
pca_op_parked = pd.DataFrame(pca_op_parked, columns=list_pc)
pca_op_parked.to_csv('../VestasV27/02dimensionality_reduction/results/pca_op_parked.csv', sep=';')

list_pc = list(range(np.shape(pca_op_train_parked)[1]))
list_pc = list(map(lambda x: "OP" + str(x), list_pc))
pca_op_train_parked = pd.DataFrame(pca_op_train_parked, columns=list_pc)
pca_op_train_parked.to_csv('../VestasV27/02dimensionality_reduction/results/pca_op_train_parked.csv', sep=';')

#envrionmental data
list_pc = list(range(np.shape(pca_env_parked)[1]))
list_pc = list(map(lambda x: "ENV" + str(x), list_pc))
pca_env_parked = pd.DataFrame(pca_env_parked, columns=list_pc)
pca_env_parked.to_csv('../VestasV27/02dimensionality_reduction/results/pca_env_parked.csv', sep=';')

list_pc = list(range(np.shape(pca_env_train_parked)[1]))
list_pc = list(map(lambda x: "ENV" + str(x), list_pc))
pca_env_train_parked = pd.DataFrame(pca_env_train_parked, columns=list_pc)
pca_env_train_parked.to_csv('../VestasV27/02dimensionality_reduction/results/pca_env_train_parked.csv', sep=';')

#---T1---
#damage sensitive features
list_pc = list(range(np.shape(dsf_t1)[1]))
list_pc = list(map(lambda x: "PC" + str(x), list_pc))
dsf_t1 = pd.DataFrame(dsf_t1, columns=list_pc)
dsf_t1.to_csv('../VestasV27/02dimensionality_reduction/results/dsf_t1.csv', sep=';')

list_pc = list(range(np.shape(dsf_train_t1)[1]))
list_pc = list(map(lambda x: "PC" + str(x), list_pc))
dsf_train_t1 = pd.DataFrame(dsf_train_t1, columns=list_pc)
dsf_train_t1.to_csv('../VestasV27/02dimensionality_reduction/results/dsf_train_t1.csv', sep=';')

#operational_data
list_pc = list(range(np.shape(pca_op_t1)[1]))
list_pc = list(map(lambda x: "OP" + str(x), list_pc))
pca_op_t1 = pd.DataFrame(pca_op_t1, columns=list_pc)
pca_op_t1.to_csv('../VestasV27/02dimensionality_reduction/results/pca_op_t1.csv', sep=';')

list_pc = list(range(np.shape(pca_op_train_t1)[1]))
list_pc = list(map(lambda x: "OP" + str(x), list_pc))
pca_op_train_t1 = pd.DataFrame(pca_op_train_t1, columns=list_pc)
pca_op_train_t1.to_csv('../VestasV27/02dimensionality_reduction/results/pca_op_train_t1.csv', sep=';')

#envrionmental data
list_pc = list(range(np.shape(pca_env_t1)[1]))
list_pc = list(map(lambda x: "ENV" + str(x), list_pc))
pca_env_t1 = pd.DataFrame(pca_env_t1, columns=list_pc)
pca_env_t1.to_csv('../VestasV27/02dimensionality_reduction/results/pca_env_t1.csv', sep=';')

list_pc = list(range(np.shape(pca_env_train_t1)[1]))
list_pc = list(map(lambda x: "ENV" + str(x), list_pc))
pca_env_train_t1 = pd.DataFrame(pca_env_train_t1, columns=list_pc)
pca_env_train_t1.to_csv('../VestasV27/02dimensionality_reduction/results/pca_env_train_t1.csv', sep=';')

#RPM32
#damage sensitive features
list_pc = list(range(np.shape(dsf_rpm32)[1]))
list_pc = list(map(lambda x: "PC" + str(x), list_pc))
dsf_rpm32 = pd.DataFrame(dsf_rpm32, columns=list_pc)
dsf_rpm32.to_csv('../VestasV27/02dimensionality_reduction/results/dsf_rpm32.csv', sep=';')

list_pc = list(range(np.shape(dsf_train_rpm32)[1]))
list_pc = list(map(lambda x: "PC" + str(x), list_pc))
dsf_train_rpm32 = pd.DataFrame(dsf_train_rpm32, columns=list_pc)
dsf_train_rpm32.to_csv('../VestasV27/02dimensionality_reduction/results/dsf_train_rpm32.csv', sep=';')

#operational_data
list_pc = list(range(np.shape(pca_op_rpm32)[1]))
list_pc = list(map(lambda x: "OP" + str(x), list_pc))
pca_op_rpm32 = pd.DataFrame(pca_op_rpm32, columns=list_pc)
pca_op_rpm32.to_csv('../VestasV27/02dimensionality_reduction/results/pca_op_rpm32.csv', sep=';')

list_pc = list(range(np.shape(pca_op_train_rpm32)[1]))
list_pc = list(map(lambda x: "OP" + str(x), list_pc))
pca_op_train_rpm32 = pd.DataFrame(pca_op_train_rpm32, columns=list_pc)
pca_op_train_rpm32.to_csv('../VestasV27/02dimensionality_reduction/results/pca_op_train_rpm32.csv', sep=';')

#envrionmental data
list_pc = list(range(np.shape(pca_env_rpm32)[1]))
list_pc = list(map(lambda x: "ENV" + str(x), list_pc))
pca_env_rpm32 = pd.DataFrame(pca_env_rpm32, columns=list_pc)
pca_env_rpm32.to_csv('../VestasV27/02dimensionality_reduction/results/pca_env_rpm32.csv', sep=';')

list_pc = list(range(np.shape(pca_env_train_rpm32)[1]))
list_pc = list(map(lambda x: "ENV" + str(x), list_pc))
pca_env_train_rpm32 = pd.DataFrame(pca_env_train_rpm32, columns=list_pc)
pca_env_train_rpm32.to_csv('../VestasV27/02dimensionality_reduction/results/pca_env_train_rpm32.csv', sep=';')

#T2
#damage sensitive features
list_pc = list(range(np.shape(dsf_t2)[1]))
list_pc = list(map(lambda x: "PC" + str(x), list_pc))
dsf_t2 = pd.DataFrame(dsf_t2, columns=list_pc)
dsf_t2.to_csv('../VestasV27/02dimensionality_reduction/results/dsf_t2.csv', sep=';')

list_pc = list(range(np.shape(dsf_train_t2)[1]))
list_pc = list(map(lambda x: "PC" + str(x), list_pc))
dsf_train_t2 = pd.DataFrame(dsf_train_t2, columns=list_pc)
dsf_train_t2.to_csv('../VestasV27/02dimensionality_reduction/results/dsf_train_t2.csv', sep=';')

#operational_data
list_pc = list(range(np.shape(pca_op_t2)[1]))
list_pc = list(map(lambda x: "OP" + str(x), list_pc))
pca_op_t2 = pd.DataFrame(pca_op_t2, columns=list_pc)
pca_op_t2.to_csv('../VestasV27/02dimensionality_reduction/results/pca_op_t2.csv', sep=';')

list_pc = list(range(np.shape(pca_op_train_t2)[1]))
list_pc = list(map(lambda x: "OP" + str(x), list_pc))
pca_op_train_t2 = pd.DataFrame(pca_op_train_t2, columns=list_pc)
pca_op_train_t2.to_csv('../VestasV27/02dimensionality_reduction/results/pca_op_train_t2.csv', sep=';')

#envrionmental data
list_pc = list(range(np.shape(pca_env_t2)[1]))
list_pc = list(map(lambda x: "ENV" + str(x), list_pc))
pca_env_t2 = pd.DataFrame(pca_env_t2, columns=list_pc)
pca_env_t2.to_csv('../VestasV27/02dimensionality_reduction/results/pca_env_t2.csv', sep=';')

list_pc = list(range(np.shape(pca_env_train_t2)[1]))
list_pc = list(map(lambda x: "ENV" + str(x), list_pc))
pca_env_train_t2 = pd.DataFrame(pca_env_train_t2, columns=list_pc)
pca_env_train_t2.to_csv('../VestasV27/02dimensionality_reduction/results/pca_env_train_t2.csv', sep=';')

#RPM43
#damage sensitive features
list_pc = list(range(np.shape(dsf_rpm43)[1]))
list_pc = list(map(lambda x: "PC" + str(x), list_pc))
dsf_rpm43 = pd.DataFrame(dsf_rpm43, columns=list_pc)
dsf_rpm43.to_csv('../VestasV27/02dimensionality_reduction/results/dsf_rpm43.csv', sep=';')

list_pc = list(range(np.shape(dsf_train_rpm43)[1]))
list_pc = list(map(lambda x: "PC" + str(x), list_pc))
dsf_train_rpm43 = pd.DataFrame(dsf_train_rpm43, columns=list_pc)
dsf_train_rpm43.to_csv('../VestasV27/02dimensionality_reduction/results/dsf_train_rpm43.csv', sep=';')

#operational_data
list_pc = list(range(np.shape(pca_op_rpm43)[1]))
list_pc = list(map(lambda x: "OP" + str(x), list_pc))
pca_op_rpm43 = pd.DataFrame(pca_op_rpm43, columns=list_pc)
pca_op_rpm43.to_csv('../VestasV27/02dimensionality_reduction/results/pca_op_rpm43.csv', sep=';')

list_pc = list(range(np.shape(pca_op_train_rpm43)[1]))
list_pc = list(map(lambda x: "OP" + str(x), list_pc))
pca_op_train_rpm43 = pd.DataFrame(pca_op_train_rpm43, columns=list_pc)
pca_op_train_rpm43.to_csv('../VestasV27/02dimensionality_reduction/results/pca_op_train_rpm43.csv', sep=';')

#envrionmental data
list_pc = list(range(np.shape(pca_env_rpm43)[1]))
list_pc = list(map(lambda x: "ENV" + str(x), list_pc))
pca_env_rpm_43 = pd.DataFrame(pca_env_rpm43, columns=list_pc)
pca_env_rpm_43.to_csv('../VestasV27/02dimensionality_reduction/results/pca_env_rpm43.csv', sep=';')

list_pc = list(range(np.shape(pca_env_train_rpm43)[1]))
list_pc = list(map(lambda x: "ENV" + str(x), list_pc))
pca_env_train_rpm_43 = pd.DataFrame(pca_env_train_rpm43, columns=list_pc)
pca_env_train_rpm_43.to_csv('../VestasV27/02dimensionality_reduction/results/pca_env_train_rpm43.csv', sep=';')

print('Done')