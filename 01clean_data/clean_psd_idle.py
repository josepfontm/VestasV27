#Rutina
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Chnage paths if necessary
path_psd_u = '../code_methodology/original_data/df_psd_u.csv'
path_psd_15 = '../code_methodology/original_data/df_psd_15.csv'
path_psd_30 = '../code_methodology/original_data/df_psd_30.csv'
path_psd_45 = '../code_methodology/original_data/df_psd_45.csv'
path_psd_r = '../code_methodology/original_data/df_psd_r.csv'

path_metadata = '../code_methodology/original_data/metadata.csv'

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
df_t2 = df[(df.RPM > 32.5) & (df.RPM < 42.5)]
df_rpm43 = df[df.RPM >= 42.5]

keys_psd = list(psd_u.columns.values) #Extract keys to access psd at different frequencies
keys_metadata = list(df_metadata.columns.values)

psd_idle = df_idle.loc[:,keys_psd]

psd_idle = psd_idle.to_numpy()

variance_psd = np.var(psd_idle,axis=1)
rms_psd = np.mean(psd_idle,axis=1)

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

low = np.percentile(variance_psd,5)
high = np.percentile(variance_psd,95)

#Outlier analysis
print(df_idle.shape)
df_idle = df_idle[((variance_psd < high) & (variance_psd > low))]
#df_idle = df_idle[(variance_psd < high_variance) & (variance_psd > low_variance)]
print(df_idle.shape)

psd_idle = df_idle.loc[:,keys_psd]
metadata_idle = df_idle.loc[:,keys_metadata[1:]]

#Save results
psd_idle.to_csv('../code_methodology/01clean_data/results/psd_idle.csv',sep=';')
metadata_idle.to_csv('../code_methodology/01clean_data/results/metadata_idle.csv',sep=';')

psd_idle = psd_idle.to_numpy()

variance_psd = np.var(psd_idle,axis=1)
rms_psd = np.mean(psd_idle,axis=1)

plt.figure('Clean PSDs Idle')
plt.semilogy(psd_idle.T,color='blue',linewidth = 1)
plt.show()

print('Done')