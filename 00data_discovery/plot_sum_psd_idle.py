import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Chnage paths if necessary

file_PSD_u = '../code_methodology/psd_sum/df_psd_u.csv'
file_PSD_15 = '../code_methodology/psd_sum/df_psd_15.csv'
file_PSD_30 = '../code_methodology/psd_sum/df_psd_30.csv'
file_PSD_45 = '../code_methodology/psd_sum/df_psd_45.csv'
file_PSD_r = '../code_methodology/psd_sum/df_psd_r.csv'

file_metadata = '../code_methodology/original_data/metadata.csv'

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

print(np.shape(psd))
print(np.shape(metadata))

rpm = metadata.RPM
pitch = metadata.PitchMean

#Chose variable to use to color log-log plot in idle
var = metadata.WindDirection
vmin = var.min()
vmax = var.max()
mean = var.mean()
mode = var.mode()

plt.figure()
plt.ylim([10e-9,10])
for i in range(np.shape(psd)[0]):
    if rpm[i] >= 42.5: #43 RPM 
        color_line = 'gainsboro' #003f5c
        plt.semilogy(np.linspace(0,64,np.shape(psd)[1]),psd[i,:],color = color_line ,linewidth=0.01)
    elif rpm[i] > 32.5 and rpm[i] < 42.5: #T2
        color_line = 'gainsboro' #58508d
        plt.semilogy(np.linspace(0,64,np.shape(psd)[1]),psd[i,:],color = color_line ,linewidth=0.01)
    elif rpm[i] >= 31.5 and rpm[i] <= 32.5: #32 RPM
        color_line = 'gainsboro' #bc5090
        plt.semilogy(np.linspace(0,64,np.shape(psd)[1]),psd[i,:],color = color_line ,linewidth=0.01)
    elif rpm[i] > 0.5 and rpm[i] < 31.5: #T1
        color_line = 'gainsboro' #ff6361
        plt.semilogy(np.linspace(0,64,np.shape(psd)[1]),psd[i,:],color = color_line ,linewidth=0.01)
    elif rpm[i] <= 0.5 and pitch[i] >= 60: #Parked
        color_line = 'gainsboro' 
        plt.semilogy(np.linspace(0,64,np.shape(psd)[1]),psd[i,:],color = color_line ,linewidth=0.01)


for i in range(np.shape(psd)[0]):
    if rpm[i] <= 0.5 and pitch[i] < 60: #Idle
        color_line = '#ffa600'
        plt.semilogy(np.linspace(0,64,np.shape(psd)[1]),psd[i,:],color = color_line ,linewidth=0.01)
        plt.ylim([10e-9,10])

plt.xlim([0,64])
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power Spectral Density (PSD)')
plt.show()

print('done')

# for i in range(np.shape(psd_u)[0]):
#     if rpm[i] == '43 RPM':
#         continue #color_line = '#003f5c'
#     elif rpm[i] == '32 RPM':
#         continue #color_line = '#bc5090'
#     elif rpm[i] == 'Idle':
#         if var[i] < 90:
#             color_line = 'red'
#         elif var[i] < 180:
#             color_line = 'black'
#         elif var[i] < 270:
#             color_line = 'green'
#         else:
#             color_line = 'green'
#     else:
#         continue

    # plt.loglog(np.linspace(0,64,np.shape(psd_u)[1]),psd_u[i,:],color = color_line,linewidth=0.01)
    # plt.ylim([10e-9,10])

#43 RPM 
# plt.plot([0.716666,0.716666],[100,10e-10], color='black')

#Harmonics from rotation 43 RPM
# plt.plot([2*0.716666,2*0.716666],[100,10e-10], color='black', linestyle='dashed',linewidth = 0.9)
# plt.plot([3*0.716666,3*0.716666],[100,10e-10], color='black', linestyle='dashed',linewidth = 0.9)
# plt.plot([4*0.716666,4*0.716666],[100,10e-10], color='black', linestyle='dashed',linewidth = 0.9)
# plt.plot([5*0.716666,5*0.716666],[100,10e-10], color='black', linestyle='dashed',linewidth = 0.9)
# plt.plot([6*0.716666,6*0.716666],[100,10e-10], color='black', linestyle='dashed',linewidth = 0.9)
# plt.plot([7*0.716666,7*0.716666],[100,10e-10], color='black', linestyle='dashed',linewidth = 0.9)
# plt.plot([8*0.716666,8*0.716666],[100,10e-10], color='black', linestyle='dashed',linewidth = 0.9)

# plt.text(0.75,10E-6,'43 RPM')

#32 RPM
# plt.plot([0.533333,0.533333],[100,10e-10], color='black')

#Harmonics from rotation 43 RPM
# plt.plot([2*0.533333,2*0.533333],[100,10e-10], color='black', linestyle='dashed',linewidth = 0.9)
# plt.plot([3*0.533333,3*0.533333],[100,10e-10], color='black', linestyle='dashed',linewidth = 0.9)
# plt.plot([4*0.533333,4*0.533333],[100,10e-10], color='black', linestyle='dashed',linewidth = 0.9)
# plt.plot([5*0.533333,5*0.533333],[100,10e-10], color='black', linestyle='dashed',linewidth = 0.9)
# plt.plot([6*0.533333,6*0.533333],[100,10e-10], color='black', linestyle='dashed',linewidth = 0.9)
# plt.plot([7*0.533333,7*0.533333],[100,10e-10], color='black', linestyle='dashed',linewidth = 0.9)
# plt.plot([8*0.533333,8*0.533333],[100,10e-10], color='black', linestyle='dashed',linewidth = 0.9)
# plt.plot([9*0.533333,9*0.533333],[100,10e-10], color='black', linestyle='dashed',linewidth = 0.9)
# plt.plot([10*0.533333,10*0.533333],[100,10e-10], color='black', linestyle='dashed',linewidth = 0.9)
# plt.plot([11*0.533333,11*0.533333],[100,10e-10], color='black', linestyle='dashed',linewidth = 0.9)
# plt.plot([12*0.533333,12*0.533333],[100,10e-10], color='black', linestyle='dashed',linewidth = 0.9)
# plt.text(0.3,10E-6,'32 RPM')
# plt.ylim([10e-9,10])

