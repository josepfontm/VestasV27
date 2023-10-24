#Rutina
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Chnage paths if necessary
path_psd_u = '../code_6modes/original_data/df_psd_u.csv'
path_psd_15 = '../code_6modes/original_data/df_psd_15.csv'
path_psd_30 = '../code_6modes/original_data/df_psd_30.csv'
path_psd_45 = '../code_6modes/original_data/df_psd_45.csv'
path_psd_r = '../code_6modes/original_data/df_psd_r.csv'

path_metadata = '../code_6modes/original_data/metadata.csv'

psd_u = pd.read_csv(path_psd_u,index_col=0)
psd_15 = pd.read_csv(path_psd_15,index_col=0)
psd_30 = pd.read_csv(path_psd_30,index_col=0)
psd_45 = pd.read_csv(path_psd_45,index_col=0)
psd_r = pd.read_csv(path_psd_r,index_col=0)

df_psd = pd.concat([psd_u,psd_15],axis=0)
df_psd = pd.concat([df_psd,psd_30],axis=0)
df_psd = pd.concat([df_psd,psd_45],axis=0)
df_psd = pd.concat([df_psd,psd_r],axis=0)

df_metadata = pd.read_csv(path_metadata,sep=';')

df_psd = df_psd.reset_index()
df_metadata = df_metadata.reset_index()

df = pd.concat([df_psd.iloc[:,1:],df_metadata.iloc[:,1:]],axis=1,join='inner')

#CLASSIFY IN DIFFERENT MODES OF OPERATION
df_idle = df[df.RPM <= 0.5]
df_t1 = df[(df.RPM > 0.5) & (df.RPM < 31.5)]
df_rpm32 = df[(df.RPM >= 31.5) & (df.RPM <= 32.5)]
df_t2 = df[(df.RPM > 32.5) & (df.RPM < 42)]
df_rpm43 = df[df.RPM >= 42]

keys_psd = list(psd_u.columns.values) #Extract keys to access psd at different frequencies
keys_metadata = list(df_metadata.columns.values)

psd_t1 = df_t1.loc[:,keys_psd]

psd_t1 = psd_t1.to_numpy()

variance_psd = np.var(psd_t1,axis=1)
rms_psd = np.mean(psd_t1,axis=1)

#Colors used for plotting
colors = {'undamaged':'green',
              'repaired':'blue',
              '15cm':'yellow',
              '30cm':'orange',
              '45cm':'red'}

# plt.figure()
# plt.scatter(variance_psd.reshape(-1,1), rms_psd.reshape(-1,1), c=df_rpm32.DamageScenario.map(colors),alpha=0.3)
# plt.scatter(np.mean(variance_psd), np.mean(rms_psd), color='black')
# plt.xlabel('Variance')
# plt.ylabel('Mean (RMS)')
# plt.show()

# plt.figure()
# plt.semilogy(psd_rpm32)

low = np.percentile(variance_psd,0)
high = np.percentile(variance_psd,100)

#Outlier analysis
print(df_t1.shape)
df_t1 = df_t1[((variance_psd < high) & (variance_psd > low))]
#df_t1 = df_t1[(variance_psd < high_variance) & (variance_psd > low_variance)]
print(df_t1.shape)

psd_t1 = df_t1.loc[:,keys_psd]
metadata_t1 = df_t1.loc[:,keys_metadata[1:]]

#Save results
psd_t1.to_csv('../code_6modes/01clean_data/results/psd_t1.csv',sep=';')
metadata_t1.to_csv('../code_6modes/01clean_data/results/metadata_t1.csv',sep=';')

psd_t1 = psd_t1.to_numpy()

variance_psd = np.var(psd_t1,axis=1)
rms_psd = np.mean(psd_t1,axis=1)

plt.figure('Removed outliers from PSDs T1')
plt.semilogy(psd_t1.T,color='blue',linewidth = 1)
plt.show()

print('Done')