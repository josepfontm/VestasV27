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
df_rpm43 = df[df.RPM >= 42]

keys_psd = list(psd_u.columns.values) #Extract keys to access psd at different frequencies
keys_metadata = list(df_metadata.columns.values)

psd_rpm43 = df_rpm43.loc[:,keys_psd]

psd_rpm43 = psd_rpm43.to_numpy()

variance_psd = np.var(psd_rpm43,axis=1)
rms_psd = np.mean(psd_rpm43,axis=1)

#Colors used for plotting
colors = {'undamaged':'green',
              'repaired':'blue',
              '15cm':'yellow',
              '30cm':'orange',
              '45cm':'red'}

# plt.figure()
# plt.scatter(variance_psd.reshape(-1,1), rms_psd.reshape(-1,1), c=df_rpm43.DamageScenario.map(colors),alpha=0.3)
# plt.scatter(np.mean(variance_psd), np.mean(rms_psd), color='black')
# plt.xlabel('Variance')
# plt.ylabel('Mean (RMS)')
# plt.show()

# plt.figure()
# plt.semilogy(psd_rpm43)

low = np.percentile(variance_psd,0)
high = np.percentile(variance_psd,100)

#Outlier analysis
print(df_rpm43.shape)
df_rpm43 = df_rpm43[((variance_psd < high) & (variance_psd > low))]
#df_rpm43 = df_rpm43[(variance_psd < high_variance) & (variance_psd > low_variance)]
print(df_rpm43.shape)

psd_rpm43 = df_rpm43.loc[:,keys_psd]
metadata_rpm43 = df_rpm43.loc[:,keys_metadata[1:]]

#Save results
psd_rpm43.to_csv('../code_6modes/01clean_data/results/psd_rpm43.csv',sep=';')
metadata_rpm43.to_csv('../code_6modes/01clean_data/results/metadata_rpm43.csv',sep=';')

print('Files saved')

psd_rpm43 = psd_rpm43.to_numpy()

variance_psd = np.var(psd_rpm43,axis=1)
rms_psd = np.mean(psd_rpm43,axis=1)

plt.figure()
plt.loglog(psd_rpm43.T,color='blue',linewidth = 1)
plt.show()

print('Done')