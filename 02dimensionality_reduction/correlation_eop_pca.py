import pandas as pd
from sklearn.metrics import r2_score
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt

path_env_idle = '../code_6modes/02dimensionality_reduction/results/env_idle.csv'
path_env_parked = '../code_6modes/02dimensionality_reduction/results/env_parked.csv'
path_env_t1 = '../code_6modes/02dimensionality_reduction/results/env_t1.csv'
path_env_rpm32 = '../code_6modes/02dimensionality_reduction/results/env_rpm32.csv'
path_env_t2= '../code_6modes/02dimensionality_reduction/results/env_t2.csv'
path_env_rpm43 = '../code_6modes/02dimensionality_reduction/results/env_rpm43.csv'

env_idle = pd.read_csv(path_env_idle,index_col=None,header=None,sep=';')
env_parked = pd.read_csv(path_env_parked,index_col=None,header=None,sep=';')
env_t1 = pd.read_csv(path_env_t1,index_col=None,header=None,sep=';')
env_rpm32 = pd.read_csv(path_env_rpm32,index_col=None,header=None,sep=';')
env_t2 = pd.read_csv(path_env_t2,index_col=None,header=None,sep=';')
env_rpm43 = pd.read_csv(path_env_rpm43,index_col=None,header=None,sep=';')

path_op_idle = '../code_6modes/02dimensionality_reduction/results/op_idle.csv'
path_op_parked = '../code_6modes/02dimensionality_reduction/results/op_parked.csv'
path_op_t1 = '../code_6modes/02dimensionality_reduction/results/op_t1.csv'
path_op_rpm32 = '../code_6modes/02dimensionality_reduction/results/op_rpm32.csv'
path_op_t2= '../code_6modes/02dimensionality_reduction/results/op_t2.csv'
path_op_rpm43 = '../code_6modes/02dimensionality_reduction/results/op_rpm43.csv'

op_idle = pd.read_csv(path_op_idle,index_col=None,header=None,sep=';')
op_parked = pd.read_csv(path_op_parked,index_col=None,header=None,sep=';')
op_t1 = pd.read_csv(path_op_t1,index_col=None,header=None,sep=';')
op_rpm32 = pd.read_csv(path_op_rpm32,index_col=None,header=None,sep=';')
op_t2 = pd.read_csv(path_op_t2,index_col=None,header=None,sep=';')
op_rpm43 = pd.read_csv(path_op_rpm43,index_col=None,header=None,sep=';')

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

pca_env = pd.concat([pca_env_parked, pca_env_idle])
pca_env = pd.concat([pca_env, pca_env_t1])
pca_env = pd.concat([pca_env, pca_env_rpm32])
pca_env = pd.concat([pca_env, pca_env_t2])
pca_env = pd.concat([pca_env, pca_env_rpm43])

pca_op = pd.concat([pca_op_parked, pca_op_idle])
pca_op = pd.concat([pca_op, pca_op_t1])
pca_op = pd.concat([pca_op, pca_op_rpm32])
pca_op = pd.concat([pca_op, pca_op_t2])
pca_op = pd.concat([pca_op, pca_op_rpm43])

env = pd.concat([env_parked,env_idle])
env = pd.concat([env, env_t1])
env = pd.concat([env, env_rpm32])
env = pd.concat([env, env_t2])
env = pd.concat([env, env_rpm43])

print(pca_op)
print(op_idle)

print('Mutual Information - EV')
print(mutual_info_regression(pca_env['ENV0'].values.reshape(-1,1),env.iloc[:,0]))
print(mutual_info_regression(pca_env['ENV0'].values.reshape(-1,1),env.iloc[:,1]))
print(mutual_info_regression(pca_env['ENV0'].values.reshape(-1,1),env.iloc[:,2]))

print(mutual_info_regression(pca_env['ENV1'].values.reshape(-1,1),env.iloc[:,0]))
print(mutual_info_regression(pca_env['ENV1'].values.reshape(-1,1),env.iloc[:,1]))
print(mutual_info_regression(pca_env['ENV1'].values.reshape(-1,1),env.iloc[:,2]))

print(mutual_info_regression(pca_env['ENV2'].values.reshape(-1,1),env.iloc[:,0]))
print(mutual_info_regression(pca_env['ENV2'].values.reshape(-1,1),env.iloc[:,1]))
print(mutual_info_regression(pca_env['ENV2'].values.reshape(-1,1),env.iloc[:,2]))

# print('Mutual Information - OP')
# print(mutual_info_regression(pca_env['ENV0'].values.reshape(-1,1),env.iloc[:,0]))
# print(mutual_info_regression(pca_env['ENV0'].values.reshape(-1,1),env.iloc[:,1]))
# print(mutual_info_regression(pca_env['ENV0'].values.reshape(-1,1),env.iloc[:,2]))

# print(mutual_info_regression(pca_env['ENV1'].values.reshape(-1,1),env.iloc[:,0]))
# print(mutual_info_regression(pca_env['ENV1'].values.reshape(-1,1),env.iloc[:,1]))
# print(mutual_info_regression(pca_env['ENV1'].values.reshape(-1,1),env.iloc[:,2]))

# print(mutual_info_regression(pca_env['ENV2'].values.reshape(-1,1),env.iloc[:,0]))
# print(mutual_info_regression(pca_env['ENV2'].values.reshape(-1,1),env.iloc[:,1]))
# print(mutual_info_regression(pca_env['ENV2'].values.reshape(-1,1),env.iloc[:,2]))

plt.figure()
plt.scatter(env.iloc[:,0],pca_env['ENV0'],s=2)

plt.figure()
plt.scatter(env.iloc[:,1],pca_env['ENV0'],s=2)

plt.figure()
plt.scatter(env.iloc[:,2],pca_env['ENV0'],s=2)
plt.show()
