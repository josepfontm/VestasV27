import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

#Colors used for plotting
colors = {'undamaged':'green',
              'repaired':'blue',
              '15cm':'yellow',
              '30cm':'orange',
              '45cm':'red'}

#Change paths if necessary
path_scenario_idle = '../code_6modes/02dimensionality_reduction/results/scenario_idle.csv'
path_scenario_parked = '../code_6modes/02dimensionality_reduction/results/scenario_parked.csv'
path_scenario_t1 = '../code_6modes/02dimensionality_reduction/results/scenario_t1.csv'
path_scenario_rpm32 = '../code_6modes/02dimensionality_reduction/results/scenario_rpm32.csv'
path_scenario_t2 = '../code_6modes/02dimensionality_reduction/results/scenario_t2.csv'
path_scenario_rpm43 = '../code_6modes/02dimensionality_reduction/results/scenario_rpm43.csv'

path_scenario_train_idle = '../code_6modes/02dimensionality_reduction/results/scenario_train_idle.csv'
path_scenario_train_parked = '../code_6modes/02dimensionality_reduction/results/scenario_train_parked.csv'
path_scenario_train_t1 = '../code_6modes/02dimensionality_reduction/results/scenario_train_t1.csv'
path_scenario_train_rpm32 = '../code_6modes/02dimensionality_reduction/results/scenario_train_rpm32.csv'
path_scenario_train_t2 = '../code_6modes/02dimensionality_reduction/results/scenario_train_t2.csv'
path_scenario_train_rpm43 = '../code_6modes/02dimensionality_reduction/results/scenario_train_rpm43.csv'

path_metadata_idle = '../code_6modes/01clean_data/results/metadata_idle.csv'
path_metadata_parked = '../code_6modes/01clean_data/results/metadata_parked.csv'
path_metadata_t1 = '../code_6modes/01clean_data/results/metadata_t1.csv'
path_metadata_rpm32 = '../code_6modes/01clean_data/results/metadata_rpm32.csv'
path_metadata_t2 = '../code_6modes/01clean_data/results/metadata_t2.csv'
path_metadata_rpm43 = '../code_6modes/01clean_data/results/metadata_rpm43.csv'

scenario_idle = pd.read_csv(path_scenario_idle, index_col=0,sep=';')
scenario_parked = pd.read_csv(path_scenario_parked, index_col=0,sep=';')
scenario_t1 = pd.read_csv(path_scenario_t1, index_col=0,sep=';')
scenario_rpm32 = pd.read_csv(path_scenario_rpm32, index_col=0,sep=';')
scenario_t2 = pd.read_csv(path_scenario_t2, index_col=0,sep=';')
scenario_rpm43 = pd.read_csv(path_scenario_rpm43, index_col=0,sep=';')

scenario_train_idle = pd.read_csv(path_scenario_train_idle, index_col=0,sep=';')
scenario_train_parked = pd.read_csv(path_scenario_train_parked, index_col=0,sep=';')
scenario_train_t1 = pd.read_csv(path_scenario_train_t1, index_col=0,sep=';')
scenario_train_rpm32 = pd.read_csv(path_scenario_train_rpm32, index_col=0,sep=';')
scenario_train_t2 = pd.read_csv(path_scenario_train_t2, index_col=0,sep=';')
scenario_train_rpm43 = pd.read_csv(path_scenario_train_rpm43, index_col=0,sep=';')

metadata_idle = pd.read_csv(path_metadata_idle, index_col=0,sep=';')
metadata_parked = pd.read_csv(path_metadata_parked, index_col=0,sep=';')
metadata_t1 = pd.read_csv(path_metadata_t1, index_col=0,sep=';')
metadata_rpm32 = pd.read_csv(path_metadata_rpm32, index_col=0,sep=';')
metadata_t2 = pd.read_csv(path_metadata_t2, index_col=0,sep=';')
metadata_rpm43 = pd.read_csv(path_metadata_rpm43, index_col=0,sep=';')

#Load paths for pc's environemntal data
path_pca_env_idle = '../code_6modes/02dimensionality_reduction/results/pca_env_idle.csv'
path_pca_env_parked = '../code_6modes/02dimensionality_reduction/results/pca_env_parked.csv'
path_pca_env_t1 = '../code_6modes/02dimensionality_reduction/results/pca_env_t1.csv'
path_pca_env_rpm32 = '../code_6modes/02dimensionality_reduction/results/pca_env_rpm32.csv'
path_pca_env_t2 = '../code_6modes/02dimensionality_reduction/results/pca_env_t2.csv'
path_pca_env_rpm43 = '../code_6modes/02dimensionality_reduction/results/pca_env_rpm43.csv'

pca_env_idle = pd.read_csv(path_pca_env_idle,index_col=0,sep=';')
pca_env_parked = pd.read_csv(path_pca_env_parked,index_col=0,sep=';')
pca_env_t1 = pd.read_csv(path_pca_env_t1,index_col=0,sep=';')
pca_env_rpm32 = pd.read_csv(path_pca_env_rpm32,index_col=0,sep=';')
pca_env_t2 = pd.read_csv(path_pca_env_t2,index_col=0,sep=';')
pca_env_rpm43 = pd.read_csv(path_pca_env_rpm43,index_col=0,sep=';')

path_pca_env_train_idle = '../code_6modes/02dimensionality_reduction/results/pca_env_train_idle.csv'
path_pca_env_train_parked = '../code_6modes/02dimensionality_reduction/results/pca_env_train_parked.csv'
path_pca_env_train_t1 = '../code_6modes/02dimensionality_reduction/results/pca_env_train_t1.csv'
path_pca_env_train_rpm32 = '../code_6modes/02dimensionality_reduction/results/pca_env_train_rpm32.csv'
path_pca_env_train_t2 = '../code_6modes/02dimensionality_reduction/results/pca_env_train_t2.csv'
path_pca_env_train_rpm43 = '../code_6modes/02dimensionality_reduction/results/pca_env_train_rpm43.csv'

pca_env_train_idle = pd.read_csv(path_pca_env_train_idle,index_col=0,sep=';')
pca_env_train_parked = pd.read_csv(path_pca_env_train_parked,index_col=0,sep=';')
pca_env_train_t1 = pd.read_csv(path_pca_env_train_t1,index_col=0,sep=';')
pca_env_train_rpm32 = pd.read_csv(path_pca_env_train_rpm32,index_col=0,sep=';')
pca_env_train_t2 = pd.read_csv(path_pca_env_train_t2,index_col=0,sep=';')
pca_env_train_rpm43 = pd.read_csv(path_pca_env_train_rpm43,index_col=0,sep=';')

#Load paths for pc's operational data
path_pca_op_idle = '../code_6modes/02dimensionality_reduction/results/pca_op_idle.csv'
path_pca_op_parked = '../code_6modes/02dimensionality_reduction/results/pca_op_parked.csv'
path_pca_op_t1 = '../code_6modes/02dimensionality_reduction/results/pca_op_t1.csv'
path_pca_op_rpm32 = '../code_6modes/02dimensionality_reduction/results/pca_op_rpm32.csv'
path_pca_op_t2 = '../code_6modes/02dimensionality_reduction/results/pca_op_t2.csv'
path_pca_op_rpm43 = '../code_6modes/02dimensionality_reduction/results/pca_op_rpm43.csv'

path_pca_op_train_idle = '../code_6modes/02dimensionality_reduction/results/pca_op_train_idle.csv'
path_pca_op_train_parked = '../code_6modes/02dimensionality_reduction/results/pca_op_train_parked.csv'
path_pca_op_train_t1 = '../code_6modes/02dimensionality_reduction/results/pca_op_train_t1.csv'
path_pca_op_train_rpm32 = '../code_6modes/02dimensionality_reduction/results/pca_op_train_rpm32.csv'
path_pca_op_train_t2 = '../code_6modes/02dimensionality_reduction/results/pca_op_train_t2.csv'
path_pca_op_train_rpm43 = '../code_6modes/02dimensionality_reduction/results/pca_op_train_rpm43.csv'

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
# pca_idle = pd.concat([pca_env_idle, pca_op_idle], axis=1)
# pca_t1 = pd.concat([pca_env_t1, pca_op_t1], axis=1)
# pca_rpm32 = pd.concat([pca_env_rpm32, pca_op_rpm32], axis=1)
# pca_t2 = pd.concat([pca_env_t2,pca_op_t2],axis=1)
# pca_rpm43 = pd.concat([pca_env_rpm43, pca_op_rpm43], axis = 1)

# pca_train_idle = pd.concat([pca_env_train_idle,pca_op_train_idle],axis=1)
# pca_train_t1 = pd.concat([pca_env_train_t1,pca_op_train_t1],axis=1)
# pca_train_rpm32 = pd.concat([pca_env_train_rpm32,pca_op_train_rpm32],axis=1)
# pca_train_t2 = pd.concat([pca_env_train_t2,pca_op_train_t2],axis=1)
# pca_train_rpm43 = pd.concat([pca_env_train_rpm43,pca_op_train_rpm43],axis=1)

#Paths damage-sensitive features
path_dsf_idle = '../code_6modes/02dimensionality_reduction/results/dsf_idle.csv'
path_dsf_parked = '../code_6modes/02dimensionality_reduction/results/dsf_parked.csv'
path_dsf_t1 = '../code_6modes/02dimensionality_reduction/results/dsf_t1.csv'
path_dsf_rpm32 = '../code_6modes/02dimensionality_reduction/results/dsf_rpm32.csv'
path_dsf_t2 = '../code_6modes/02dimensionality_reduction/results/dsf_t2.csv'
path_dsf_rpm43 = '../code_6modes/02dimensionality_reduction/results/dsf_rpm43.csv'

dsf_idle = pd.read_csv(path_dsf_idle,index_col=0,sep=';')
dsf_parked = pd.read_csv(path_dsf_parked,index_col=0,sep=';')
dsf_t1 = pd.read_csv(path_dsf_t1,index_col=0,sep=';')
dsf_rpm32 = pd.read_csv(path_dsf_rpm32,index_col=0,sep=';')
dsf_t2 = pd.read_csv(path_dsf_t2,index_col=0,sep=';')
dsf_rpm43 = pd.read_csv(path_dsf_rpm43,index_col=0,sep=';')

path_dsf_train_idle = '../code_6modes/02dimensionality_reduction/results/dsf_train_idle.csv'
path_dsf_train_parked = '../code_6modes/02dimensionality_reduction/results/dsf_train_parked.csv'
path_dsf_train_t1 = '../code_6modes/02dimensionality_reduction/results/dsf_train_t1.csv'
path_dsf_train_rpm32 = '../code_6modes/02dimensionality_reduction/results/dsf_train_rpm32.csv'
path_dsf_train_t2 = '../code_6modes/02dimensionality_reduction/results/dsf_train_t2.csv'
path_dsf_train_rpm43 = '../code_6modes/02dimensionality_reduction/results/dsf_train_rpm43.csv'

dsf_train_idle = pd.read_csv(path_dsf_train_idle,index_col=0,sep=';')
dsf_train_parked = pd.read_csv(path_dsf_train_parked,index_col=0,sep=';')
dsf_train_t1 = pd.read_csv(path_dsf_train_t1,index_col=0,sep=';')
dsf_train_rpm32 = pd.read_csv(path_dsf_train_rpm32,index_col=0,sep=';')
dsf_train_t2 = pd.read_csv(path_dsf_train_t2,index_col=0,sep=';')
dsf_train_rpm43 = pd.read_csv(path_dsf_train_rpm43,index_col=0,sep=';')

#Normalize before applying regression
scaler = StandardScaler()

#DSFs
scaler.fit(dsf_train_idle)
dsf_train_idle.values[:] = scaler.transform(dsf_train_idle)
dsf_idle.values[:] = scaler.transform(dsf_idle)

scaler.fit(dsf_train_parked)
dsf_train_parked.values[:] = scaler.transform(dsf_train_parked)
dsf_parked.values[:] = scaler.transform(dsf_parked)

scaler.fit(dsf_train_t1)
dsf_train_t1.values[:] = scaler.transform(dsf_train_t1)
dsf_t1.values[:] = scaler.transform(dsf_t1)

scaler.fit(dsf_train_rpm32)
dsf_train_rpm32.values[:] = scaler.transform(dsf_train_rpm32)
dsf_rpm32.values[:] = scaler.transform(dsf_rpm32)

scaler.fit(dsf_train_t2)
dsf_train_t2.values[:] = scaler.transform(dsf_train_t2)
dsf_t2.values[:] = scaler.transform(dsf_t2)

scaler.fit(dsf_train_rpm43)
dsf_train_rpm43.values[:] = scaler.transform(dsf_train_rpm43)
dsf_rpm43.values[:] = scaler.transform(dsf_rpm43)

#PCA-ENV
scaler.fit(pca_env_train_idle)
pca_env_train_idle.values[:] = scaler.transform(pca_env_train_idle)
pca_env_idle.values[:] = scaler.transform(pca_env_idle)

scaler.fit(pca_env_train_parked)
pca_env_train_parked.values[:] = scaler.transform(pca_env_train_parked)
pca_env_parked.values[:] = scaler.transform(pca_env_parked)

scaler.fit(pca_env_train_t1)
pca_env_train_t1.values[:] = scaler.transform(pca_env_train_t1)
pca_env_t1.values[:] = scaler.transform(pca_env_t1)

scaler.fit(pca_env_train_rpm32)
pca_env_train_rpm32.values[:] = scaler.transform(pca_env_train_rpm32)
pca_env_rpm32.values[:] = scaler.transform(pca_env_rpm32)

scaler.fit(pca_env_train_t2)
pca_env_train_t2.values[:] = scaler.transform(pca_env_train_t2)
pca_env_t2.values[:] = scaler.transform(pca_env_t2)

scaler.fit(pca_env_train_rpm43)
pca_env_train_rpm43.values[:] = scaler.transform(pca_env_train_rpm43)
pca_env_rpm43.values[:] = scaler.transform(pca_env_rpm43)

#PCA-OP
scaler.fit(pca_op_train_idle)
pca_op_train_idle.values[:] = scaler.transform(pca_op_train_idle)
pca_op_idle.values[:] = scaler.transform(pca_op_idle)

scaler.fit(pca_op_train_parked)
pca_op_train_parked.values[:] = scaler.transform(pca_op_train_parked)
pca_op_parked.values[:] = scaler.transform(pca_op_parked)

scaler.fit(pca_op_train_t1)
pca_op_train_t1.values[:] = scaler.transform(pca_op_train_t1)
pca_op_t1.values[:] = scaler.transform(pca_op_t1)

scaler.fit(pca_op_train_rpm32)
pca_op_train_rpm32.values[:] = scaler.transform(pca_op_train_rpm32)
pca_op_rpm32.values[:] = scaler.transform(pca_op_rpm32)

scaler.fit(pca_op_train_t2)
pca_op_train_t2.values[:] = scaler.transform(pca_op_train_t2)
pca_op_t2.values[:] = scaler.transform(pca_op_t2)

scaler.fit(pca_op_train_rpm43)
pca_op_train_rpm43.values[:] = scaler.transform(pca_op_train_rpm43)
pca_op_rpm43.values[:] = scaler.transform(pca_op_rpm43)

#Save data
dsf_idle.to_csv('../code_6modes/03regression_analysis/00standarized/dsf_idle.csv',sep=';')
dsf_parked.to_csv('../code_6modes/03regression_analysis/00standarized/dsf_parked.csv',sep=';')
dsf_t1.to_csv('../code_6modes/03regression_analysis/00standarized/dsf_t1.csv',sep=';')
dsf_rpm32.to_csv('../code_6modes/03regression_analysis/00standarized/dsf_rpm32.csv',sep=';')
dsf_t2.to_csv('../code_6modes/03regression_analysis/00standarized/dsf_t2.csv',sep=';')
dsf_rpm43.to_csv('../code_6modes/03regression_analysis/00standarized/dsf_rpm43.csv',sep=';')

dsf_train_idle.to_csv('../code_6modes/03regression_analysis/00standarized/dsf_train_idle.csv', sep=';')
dsf_train_parked.to_csv('../code_6modes/03regression_analysis/00standarized/dsf_train_parked.csv', sep=';')
dsf_train_t1.to_csv('../code_6modes/03regression_analysis/00standarized/dsf_train_t1.csv', sep=';')
dsf_train_rpm32.to_csv('../code_6modes/03regression_analysis/00standarized/dsf_train_rpm32.csv', sep=';')
dsf_train_t2.to_csv('../code_6modes/03regression_analysis/00standarized/dsf_train_t2.csv', sep=';')
dsf_train_rpm43.to_csv('../code_6modes/03regression_analysis/00standarized/dsf_train_rpm43.csv', sep=';')

print('Shapes')
print(np.shape(dsf_idle))
print(np.shape(dsf_parked))
print(np.shape(dsf_t1))
print(np.shape(dsf_rpm32))
print(np.shape(dsf_t2))
print(np.shape(dsf_rpm43))

pca_env_idle.to_csv('../code_6modes/03regression_analysis/00standarized/pca_env_idle.csv', sep=';')
pca_env_parked.to_csv('../code_6modes/03regression_analysis/00standarized/pca_env_parked.csv', sep=';')
pca_env_t1.to_csv('../code_6modes/03regression_analysis/00standarized/pca_env_t1.csv',sep=';')
pca_env_rpm32.to_csv('../code_6modes/03regression_analysis/00standarized/pca_env_rpm32.csv', sep=';')
pca_env_t2.to_csv('../code_6modes/03regression_analysis/00standarized/pca_env_t2.csv', sep=';')
pca_env_rpm43.to_csv('../code_6modes/03regression_analysis/00standarized/pca_env_rpm43.csv', sep=';')

pca_env_train_idle.to_csv('../code_6modes/03regression_analysis/00standarized/pca_env_train_idle.csv', sep=';')
pca_env_train_parked.to_csv('../code_6modes/03regression_analysis/00standarized/pca_env_train_parked.csv', sep=';')
pca_env_train_t1.to_csv('../code_6modes/03regression_analysis/00standarized/pca_env_train_t1.csv', sep=';')
pca_env_train_rpm32.to_csv('../code_6modes/03regression_analysis/00standarized/pca_env_train_rpm32.csv', sep=';')
pca_env_train_t2.to_csv('../code_6modes/03regression_analysis/00standarized/pca_env_train_t2.csv', sep=';')
pca_env_train_rpm43.to_csv('../code_6modes/03regression_analysis/00standarized/pca_env_train_rpm43.csv', sep=';')

metadata_idle.to_csv('../code_6modes/03regression_analysis/00standarized/metadata_idle.csv', sep=';')
metadata_parked.to_csv('../code_6modes/03regression_analysis/00standarized/metadata_parked.csv', sep=';')
metadata_t1.to_csv('../code_6modes/03regression_analysis/00standarized/metadata_t1.csv', sep=';')
metadata_rpm32.to_csv('../code_6modes/03regression_analysis/00standarized/metadata_rpm32.csv', sep=';')
metadata_t2.to_csv('../code_6modes/03regression_analysis/00standarized/metadata_t2.csv', sep=';')
metadata_rpm43.to_csv('../code_6modes/03regression_analysis/00standarized/metadata_rpm43.csv', sep=';')

pca_op_idle.to_csv('../code_6modes/03regression_analysis/00standarized/pca_op_idle.csv', sep=';')
pca_op_parked.to_csv('../code_6modes/03regression_analysis/00standarized/pca_op_parked.csv', sep=';')
pca_op_t1.to_csv('../code_6modes/03regression_analysis/00standarized/pca_op_t1.csv',sep=';')
pca_op_rpm32.to_csv('../code_6modes/03regression_analysis/00standarized/pca_op_rpm32.csv', sep=';')
pca_op_t2.to_csv('../code_6modes/03regression_analysis/00standarized/pca_op_t2.csv', sep=';')
pca_op_rpm43.to_csv('../code_6modes/03regression_analysis/00standarized/pca_op_rpm43.csv', sep=';')

pca_op_train_idle.to_csv('../code_6modes/03regression_analysis/00standarized/pca_op_train_idle.csv', sep=';')
pca_op_train_parked.to_csv('../code_6modes/03regression_analysis/00standarized/pca_op_train_parked.csv', sep=';')
pca_op_train_t1.to_csv('../code_6modes/03regression_analysis/00standarized/pca_op_train_t1.csv', sep=';')
pca_op_train_rpm32.to_csv('../code_6modes/03regression_analysis/00standarized/pca_op_train_rpm32.csv', sep=';')
pca_op_train_t2.to_csv('../code_6modes/03regression_analysis/00standarized/pca_op_train_t2.csv', sep=';')
pca_op_train_rpm43.to_csv('../code_6modes/03regression_analysis/00standarized/pca_op_train_rpm43.csv', sep=';')

scenario_idle.to_csv('../code_6modes/03regression_analysis/00standarized/scenario_idle.csv',sep=';')
scenario_parked.to_csv('../code_6modes/03regression_analysis/00standarized/scenario_parked.csv',sep=';')
scenario_t1.to_csv('../code_6modes/03regression_analysis/00standarized/scenario_t1.csv',sep=';')
scenario_rpm32.to_csv('../code_6modes/03regression_analysis/00standarized/scenario_rpm32.csv',sep=';')
scenario_t2.to_csv('../code_6modes/03regression_analysis/00standarized/scenario_t2.csv',sep=';')
scenario_rpm43.to_csv('../code_6modes/03regression_analysis/00standarized/scenario_rpm43.csv',sep=';')

scenario_train_idle.to_csv('../code_6modes/03regression_analysis/00standarized/scenario_train_idle.csv', sep=';')
scenario_train_parked.to_csv('../code_6modes/03regression_analysis/00standarized/scenario_train_parked.csv', sep=';')
scenario_train_t1.to_csv('../code_6modes/03regression_analysis/00standarized/scenario_train_t1.csv', sep=';')
scenario_train_rpm32.to_csv('../code_6modes/03regression_analysis/00standarized/scenario_train_rpm32.csv', sep=';')
scenario_train_t2.to_csv('../code_6modes/03regression_analysis/00standarized/scenario_train_t2.csv', sep=';')
scenario_train_rpm43.to_csv('../code_6modes/03regression_analysis/00standarized/scenario_train_rpm43.csv', sep=';')

print('Done')

