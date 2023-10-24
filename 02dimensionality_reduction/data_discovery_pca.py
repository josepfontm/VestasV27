import pandas as pd
import matplotlib.pyplot as plt

path_metadata_idle = '../code_6modes/01clean_data/results/metadata_idle.csv'
path_metadata_parked = '../code_6modes/01clean_data/results/metadata_parked.csv'
path_metadata_t1 = '../code_6modes/01clean_data/results/metadata_t1.csv'
path_metadata_rpm32 = '../code_6modes/01clean_data/results/metadata_rpm32.csv'
path_metadata_t2= '../code_6modes/01clean_data/results/metadata_t2.csv'
path_metadata_rpm43 = '../code_6modes/01clean_data/results/metadata_rpm43.csv'

metadata_idle = pd.read_csv(path_metadata_idle,index_col=0,sep=';')
metadata_parked = pd.read_csv(path_metadata_parked,index_col=0,sep=';')
metadata_t1 = pd.read_csv(path_metadata_t1,index_col=0,sep=';')
metadata_rpm32 = pd.read_csv(path_metadata_rpm32,index_col=0,sep=';')
metadata_t2 = pd.read_csv(path_metadata_t2,index_col=0,sep=';')
metadata_rpm43 = pd.read_csv(path_metadata_rpm43,index_col=0,sep=';')

path_pca_op_idle = '../code_6modes/02dimensionality_reduction/results/pca_op_idle.csv'
path_pca_op_parked = '../code_6modes/02dimensionality_reduction/results/pca_op_parked.csv'
path_pca_op_t1 = '../code_6modes/02dimensionality_reduction/results/pca_op_t1.csv'
path_pca_op_rpm32 = '../code_6modes/02dimensionality_reduction/results/pca_op_rpm32.csv'
path_pca_op_t2 = '../code_6modes/02dimensionality_reduction/results/pca_op_t2.csv'
path_pca_op_rpm43 = '../code_6modes/02dimensionality_reduction/results/pca_op_rpm43.csv'

pca_op_idle = pd.read_csv(path_pca_op_idle,index_col=0,sep=';').reset_index()
pca_op_parked = pd.read_csv(path_pca_op_parked,index_col=0,sep=';').reset_index()
pca_op_t1 = pd.read_csv(path_pca_op_t1,index_col=0,sep=';').reset_index()
pca_op_rpm32 = pd.read_csv(path_pca_op_rpm32,index_col=0,sep=';').reset_index()
pca_op_t2 = pd.read_csv(path_pca_op_t2,index_col=0,sep=';').reset_index()
pca_op_rpm43 = pd.read_csv(path_pca_op_rpm43,index_col=0,sep=';').reset_index()

path_pca_env_idle = '../code_6modes/02dimensionality_reduction/results/pca_env_idle.csv'
path_pca_env_parked = '../code_6modes/02dimensionality_reduction/results/pca_env_parked.csv'
path_pca_env_t1 = '../code_6modes/02dimensionality_reduction/results/pca_env_t1.csv'
path_pca_env_rpm32 = '../code_6modes/02dimensionality_reduction/results/pca_env_rpm32.csv'
path_pca_env_t2 = '../code_6modes/02dimensionality_reduction/results/pca_env_t2.csv'
path_pca_env_rpm43 = '../code_6modes/02dimensionality_reduction/results/pca_env_rpm43.csv'

pca_env_idle = pd.read_csv(path_pca_env_idle,index_col=0,sep=';').reset_index()
pca_env_parked = pd.read_csv(path_pca_env_parked,index_col=0,sep=';').reset_index()
pca_env_t1 = pd.read_csv(path_pca_env_t1,index_col=0,sep=';').reset_index()
pca_env_rpm32 = pd.read_csv(path_pca_env_rpm32,index_col=0,sep=';').reset_index()
pca_env_t2 = pd.read_csv(path_pca_env_t2,index_col=0,sep=';').reset_index()
pca_env_rpm43 = pd.read_csv(path_pca_env_rpm43,index_col=0,sep=';').reset_index()

size = 3

# plt.figure('PitchMean - OP0')
# plt.scatter(metadata_idle['PitchMean'], pca_op_idle['OP0'],color='#ffa600',s=size)
# plt.scatter(metadata_parked['PitchMean'], pca_op_parked['OP0'],color='grey',s=size)
# plt.scatter(metadata_t1['PitchMean'], pca_op_t1['OP0'],color='#ff6361',s=size)
# plt.scatter(metadata_rpm32['PitchMean'], pca_op_rpm32['OP0'],color='#bc5090',s=size)
# plt.scatter(metadata_t2['PitchMean'], pca_op_t2['OP0'],color='#58508d',s=size)
# plt.scatter(metadata_rpm43['PitchMean'], pca_op_rpm43['OP0'],color='#003f5c',s=size)
# plt.xlabel('Pitch Mean')
# plt.ylabel('PC-OP0')
# plt.show()

# plt.figure('PitchMean - OP1')
# plt.scatter(metadata_idle['PitchMean'], pca_op_idle['OP1'],color='#ffa600',s=size)
# plt.scatter(metadata_parked['PitchMean'], pca_op_parked['OP1'],color='grey',s=size)
# plt.scatter(metadata_t1['PitchMean'], pca_op_t1['OP1'],color='#ff6361',s=size)
# plt.scatter(metadata_rpm32['PitchMean'], pca_op_rpm32['OP1'],color='#bc5090',s=size)
# plt.scatter(metadata_t2['PitchMean'], pca_op_t2['OP1'],color='#58508d',s=size)
# plt.scatter(metadata_rpm43['PitchMean'], pca_op_rpm43['OP1'],color='#003f5c',s=size)
# plt.xlabel('Pitch Mean')
# plt.ylabel('PC-OP1')
# plt.show()

# plt.figure('PitchStd - OP0')
# plt.scatter(metadata_idle['PitchStd'], pca_op_idle['OP0'],color='#ffa600',s=size)
# plt.xlabel('Pitch Std')
# plt.ylabel('PC-OP0')

# plt.figure('PitchStd - OP1')
# plt.scatter(metadata_idle['PitchStd'], pca_op_idle['OP1'],color='#ffa600',s=size)
# plt.xlabel('Pitch Std')
# plt.ylabel('PC-OP1')
# plt.show()

# plt.figure('PitchMin - OP0')
# plt.scatter(metadata_idle['PitchMin'], pca_op_idle['OP0'],color='#ffa600',s=size)
# plt.xlabel('Pitch Min')
# plt.ylabel('PC-OP0')

# plt.figure('PitchMin - OP1')
# plt.scatter(metadata_idle['PitchMin'], pca_op_idle['OP1'],color='#ffa600',s=size)
# plt.xlabel('Pitch Min')
# plt.ylabel('PC-OP1')
# plt.show()

# plt.figure('RPM - OP0')
# plt.scatter(metadata_idle['RPM'], pca_op_idle['OP0'],color='#ffa600',s=size)
# plt.scatter(metadata_parked['RPM'], pca_op_parked['OP0'],color='grey',s=size)
# plt.scatter(metadata_t1['RPM'], pca_op_t1['OP0'],color='#ff6361',s=size)
# plt.scatter(metadata_rpm32['RPM'], pca_op_rpm32['OP0'],color='#bc5090',s=size)
# plt.scatter(metadata_t2['RPM'], pca_op_t2['OP0'],color='#58508d',s=size)
# plt.scatter(metadata_rpm43['RPM'], pca_op_rpm43['OP0'],color='#003f5c',s=size)
# plt.xlabel('RPM')
# plt.ylabel('PC-OP0')

# plt.figure('RPM - OP1')
# plt.scatter(metadata_idle['RPM'], pca_op_idle['OP1'],color='#ffa600',s=size)
# plt.scatter(metadata_parked['RPM'], pca_op_parked['OP1'],color='grey',s=size)
# plt.scatter(metadata_t1['RPM'], pca_op_t1['OP1'],color='#ff6361',s=size)
# plt.scatter(metadata_rpm32['RPM'], pca_op_rpm32['OP1'],color='#bc5090',s=size)
# plt.scatter(metadata_t2['RPM'], pca_op_t2['OP1'],color='#58508d',s=size)
# plt.scatter(metadata_rpm43['RPM'], pca_op_rpm43['OP1'],color='#003f5c',s=size)
# plt.xlabel('RPM')
# plt.ylabel('PC-OP1')

plt.figure('Parked PCA-ENV TEMPERATURE')
plt.scatter(pca_env_parked['ENV0'],metadata_parked['Temperature'])

plt.figure('Parked PCA-ENV Wind')
plt.scatter(pca_env_parked['ENV0'],metadata_parked['Wind'])

plt.figure('Parked PCA-ENV WindDirection')
plt.scatter(pca_env_parked['ENV0'],metadata_parked['WindDirection'])
plt.show()