import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

path_psd_idle = '../VestasV27/01clean_data/results/psd_idle.csv'
path_psd_parked = '../VestasV27/01clean_data/results/psd_parked.csv'
path_psd_t1 = '../VestasV27/01clean_data/results/psd_t1.csv'
path_psd_rpm32 = '../VestasV27/01clean_data/results/psd_rpm32.csv'
path_psd_t2= '../VestasV27/01clean_data/results/psd_t2.csv'
path_psd_rpm43 = '../VestasV27/01clean_data/results/psd_rpm43.csv'

psd_idle = pd.read_csv(path_psd_idle,index_col=0,sep=';')
psd_parked = pd.read_csv(path_psd_parked,index_col=0,sep=';')
psd_t1 = pd.read_csv(path_psd_t1,index_col=0,sep=';')
psd_rpm32 = pd.read_csv(path_psd_rpm32,index_col=0,sep=';')
psd_t2 = pd.read_csv(path_psd_t2,index_col=0,sep=';')
psd_rpm43 = pd.read_csv(path_psd_rpm43,index_col=0,sep=';')

psd_idle = psd_idle.to_numpy()
psd_parked = psd_parked.to_numpy()
psd_t1 = psd_t1.to_numpy()
psd_rpm32 = psd_rpm32.to_numpy()
psd_t2 = psd_t2.to_numpy()
psd_rpm43 = psd_rpm43.to_numpy()

#Sum all PSDs from different acceleormteres into one
psd_idle1 = psd_idle[:,0::8]
psd_idle2 = psd_idle[:,513:1025]
psd_idle3 = psd_idle[:,1026:1538]
psd_idle4 = psd_idle[:,1539:2051]
psd_idle5 = psd_idle[:,2052:2564]
psd_idle6 = psd_idle[:,2565:3077]
psd_idle7 = psd_idle[:,3078:3590]
psd_idle8 = psd_idle[:,3591:4103]
psd_idle = psd_idle1 + psd_idle2 + psd_idle3 + psd_idle4 + psd_idle5 + psd_idle6 + psd_idle7 + psd_idle8

psd_parked1 = psd_parked[:,0:512]
psd_parked2 = psd_parked[:,513:1025]
psd_parked3 = psd_parked[:,1026:1538]
psd_parked4 = psd_parked[:,1539:2051]
psd_parked5 = psd_parked[:,2052:2564]
psd_parked6 = psd_parked[:,2565:3077]
psd_parked7 = psd_parked[:,3078:3590]
psd_parked8 = psd_parked[:,3591:4103]
psd_parked = psd_parked1 + psd_parked2 + psd_parked3 + psd_parked4 + psd_parked5 + psd_parked6 + psd_parked7 + psd_parked8

psd_t1_1 = psd_t1[:,0:512]
psd_t1_2 = psd_t1[:,513:1025]
psd_t1_3 = psd_t1[:,1026:1538]
psd_t1_4 = psd_t1[:,1539:2051]
psd_t1_5 = psd_t1[:,2052:2564]
psd_t1_6 = psd_t1[:,2565:3077]
psd_t1_7 = psd_t1[:,3078:3590]
psd_t1_8 = psd_t1[:,3591:4103]
psd_t1 = psd_t1_1 + psd_t1_2 + psd_t1_3 + psd_t1_4 + psd_t1_5 + psd_t1_6 + psd_t1_7 + psd_t1_8

psd_rpm32_1 = psd_rpm32[:,0:512]
psd_rpm32_2 = psd_rpm32[:,513:1025]
psd_rpm32_3 = psd_rpm32[:,1026:1538]
psd_rpm32_4 = psd_rpm32[:,1539:2051]
psd_rpm32_5 = psd_rpm32[:,2052:2564]
psd_rpm32_6 = psd_rpm32[:,2565:3077]
psd_rpm32_7 = psd_rpm32[:,3078:3590]
psd_rpm32_8 = psd_rpm32[:,3591:4103]
psd_rpm32 = psd_rpm32_1 + psd_rpm32_2 + psd_rpm32_3 + psd_rpm32_4 + psd_rpm32_5 + psd_rpm32_6 + psd_rpm32_7 + psd_rpm32_8

psd_t2_1 = psd_t2[:,0:512]
psd_t2_2 = psd_t2[:,513:1025]
psd_t2_3 = psd_t2[:,1026:1538]
psd_t2_4 = psd_t2[:,1539:2051]
psd_t2_5 = psd_t2[:,2052:2564]
psd_t2_6 = psd_t2[:,2565:3077]
psd_t2_7 = psd_t2[:,3078:3590]
psd_t2_8 = psd_t2[:,3591:4103]
psd_t2 = psd_t2_1 + psd_t2_2 + psd_t2_3 + psd_t2_4 + psd_t2_5 + psd_t2_6 + psd_t2_7 + psd_t2_8

psd_rpm43_1 = psd_rpm43[:,0:512]
psd_rpm43_2 = psd_rpm43[:,513:1025]
psd_rpm43_3 = psd_rpm43[:,1026:1538]
psd_rpm43_4 = psd_rpm43[:,1539:2051]
psd_rpm43_5 = psd_rpm43[:,2052:2564]
psd_rpm43_6 = psd_rpm43[:,2565:3077]
psd_rpm43_7 = psd_rpm43[:,3078:3590]
psd_rpm43_8 = psd_rpm43[:,3591:4103]
psd_rpm43 = psd_rpm43_1 + psd_rpm43_2 + psd_rpm43_3 + psd_rpm43_4 + psd_rpm43_5 + psd_rpm43_6 + psd_rpm43_7 + psd_rpm43_8


size = 3

print(np.shape(psd_idle))
print(np.shape(psd_rpm32))

plt.figure()
plt.rcParams.update({'font.size': 25})
plt.loglog(psd_parked.T,color='blue')
plt.show()