import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Chnage paths if necessary

file_PSD_u = '../VestasV27/psd_sum/df_psd_u.csv'
file_PSD_15 = '../VestasV27/psd_sum/df_psd_15.csv'
file_PSD_30 = '../VestasV27/psd_sum/df_psd_30.csv'
file_PSD_45 = '../VestasV27/psd_sum/df_psd_45.csv'
file_PSD_r = '../VestasV27/psd_sum/df_psd_r.csv'

file_metadata = '../VestasV27/original_data/metadata.csv'

psd_u = pd.read_csv(file_PSD_u, index_col = None, header = None)
psd_15 = pd.read_csv(file_PSD_15, index_col = None, header = None)
psd_30 = pd.read_csv(file_PSD_30, index_col = None, header = None)
psd_45 = pd.read_csv(file_PSD_45, index_col = None, header = None)
psd_r = pd.read_csv(file_PSD_r, index_col = None, header = None)

metadata = pd.read_csv(file_metadata, index_col = None, sep=';')

psd_u = psd_u.to_numpy()
psd_15 = psd_15.to_numpy()
psd_30 = psd_30.to_numpy()
psd_45 = psd_45.to_numpy()
psd_r = psd_r.to_numpy()

psd = np.vstack((psd_u, psd_15))
psd = np.vstack((psd, psd_30))
psd = np.vstack((psd, psd_45))
psd = np.vstack((psd, psd_r))

rpm = metadata.RPM_Class
print(rpm[0])
print(metadata.shape)

print(np.shape(psd))

for i in range(np.shape(psd)[0]):
    if rpm[i] == 'T2':
        color_line = 'gainsboro'
    elif rpm[i] == '32 RPM':
        color_line = 'gainsboro'
    elif rpm[i] == 'Idle':
        color_line = 'gainsboro'
    elif rpm[i] == 'T1':
        color_line = 'gainsboro'
    else:
        continue

    plt.loglog(np.linspace(0,64,np.shape(psd)[1]),psd[i,:],color = color_line,linewidth=0.05)
    plt.ylim([10e-9,10])

for i in range(np.shape(psd)[0]):
    print(rpm[i])
    if rpm[i] == '43 RPM':
        color_line = '#003f5c'
        plt.loglog(np.linspace(0,64,np.shape(psd)[1]),psd[i,:],color = color_line,linewidth=0.05)
        plt.ylim([10e-9,10])

#43 RPM 
plt.plot([0.716666,0.716666],[100,10e-10], color='black')

#Harmonics from rotation 43 RPM
plt.plot([2*0.716666,2*0.716666],[100,10e-10], color='black', linestyle='dashed', linewidth = 0.9)
plt.plot([3*0.716666,3*0.716666],[100,10e-10], color='black', linestyle='dashed', linewidth = 0.9)
plt.plot([4*0.716666,4*0.716666],[100,10e-10], color='black', linestyle='dashed', linewidth = 0.9)
plt.plot([5*0.716666,5*0.716666],[100,10e-10], color='black', linestyle='dashed', linewidth = 0.9)
plt.plot([6*0.716666,6*0.716666],[100,10e-10], color='black', linestyle='dashed', linewidth = 0.9)
plt.plot([7*0.716666,7*0.716666],[100,10e-10], color='black', linestyle='dashed', linewidth = 0.9)
plt.plot([8*0.716666,8*0.716666],[100,10e-10], color='black', linestyle='dashed', linewidth = 0.9)

plt.text(0.75,10E-6,'43 RPM')

#32 RPM
plt.plot([0.533333,0.533333],[100,10e-10], color='black')

plt.text(0.3,10E-6,'32 RPM')
plt.ylim([10e-9,10])

plt.xlabel('Frequency [Hz]')
plt.ylabel('Power Spectral Density (PSD)')
plt.show()

print('done')