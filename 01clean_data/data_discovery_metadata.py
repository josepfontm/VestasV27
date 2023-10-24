import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

path_metadata_idle = '../VestasV27/01clean_data/results/metadata_idle.csv'
path_metadata_parked = '../VestasV27/01clean_data/results/metadata_parked.csv'
path_metadata_t1 = '../VestasV27/01clean_data/results/metadata_t1.csv'
path_metadata_rpm32 = '../VestasV27/01clean_data/results/metadata_rpm32.csv'
path_metadata_t2= '../VestasV27/01clean_data/results/metadata_t2.csv'
path_metadata_rpm43 = '../VestasV27/01clean_data/results/metadata_rpm43.csv'

path_metadata_original = '../VestasV27/original_data\metadata.csv'

metadata_idle = pd.read_csv(path_metadata_idle,index_col=0,sep=';')
metadata_parked = pd.read_csv(path_metadata_parked,index_col=0,sep=';')
metadata_t1 = pd.read_csv(path_metadata_t1,index_col=0,sep=';')
metadata_rpm32 = pd.read_csv(path_metadata_rpm32,index_col=0,sep=';')
metadata_t2 = pd.read_csv(path_metadata_t2,index_col=0,sep=';')
metadata_rpm43 = pd.read_csv(path_metadata_rpm43,index_col=0,sep=';')
metadata_original = pd.read_csv(path_metadata_original,index_col=0,sep=';')

fig,ax = plt.subplots(2,3)
plt.suptitle('Temperature')
ax[0,0].hist(metadata_parked['Temperature'],range=[-9,13],color='grey')
ax[0,0].set_xticks([])
ax[0,1].hist(metadata_idle['Temperature'],range=[-9,13],color='#ffa600')                                                                                                                 
ax[0,1].set_xticks([])
ax[0,2].hist(metadata_t1['Temperature'],range=[-9,13],color='#ff6361')
ax[0,2].set_xticks([])
ax[1,0].hist(metadata_rpm32['Temperature'],range=[-9,13],color='#bc5090')
ax[1,0].set_xlabel('Temperature')
ax[1,1].hist(metadata_t2['Temperature'],range=[-9,13],color='#58508d')
ax[1,1].set_xlabel('Temperature')
ax[1,2].hist(metadata_rpm43['Temperature'],range=[-9,13],color='#003f5c')
ax[1,2].set_xlabel('Temperature')
plt.figure()
plt.hist(metadata_original['Temperature'])

fig,ax = plt.subplots(2,3)
plt.suptitle('Wind')
ax[0,0].hist(metadata_parked['Wind'],range=[0,30],color='grey')
ax[0,0].set_xticks([])
ax[0,1].hist(metadata_idle['Wind'],range=[0,30],color='#ffa600')                                                                                                                 
ax[0,1].set_xticks([])
ax[0,2].hist(metadata_t1['Wind'],range=[0,30],color='#ff6361')
ax[0,2].set_xticks([])
ax[1,0].hist(metadata_rpm32['Wind'],range=[0,30],color='#bc5090')
ax[1,0].set_xlabel('Wind')
ax[1,1].hist(metadata_t2['Wind'],range=[0,30],color='#58508d')
ax[1,1].set_xlabel('Wind')
ax[1,2].hist(metadata_rpm43['Wind'],range=[0,30],color='#003f5c')
ax[1,2].set_xlabel('Wind')
plt.figure()
plt.hist(metadata_original['Wind'])

fig,ax = plt.subplots(2,3)
plt.suptitle('WindDirection')
ax[0,0].hist(metadata_parked['WindDirection'],range=[0,360],color='grey')
ax[0,0].set_xticks([])
ax[0,1].hist(metadata_idle['WindDirection'],range=[0,360],color='#ffa600')                                                                                                                 
ax[0,1].set_xticks([])
ax[0,2].hist(metadata_t1['WindDirection'],range=[0,360],color='#ff6361')
ax[0,2].set_xticks([])
ax[1,0].hist(metadata_rpm32['WindDirection'],range=[0,360],color='#bc5090')
ax[1,0].set_xlabel('WindDirection')
ax[1,1].hist(metadata_t2['WindDirection'],range=[0,360],color='#58508d')
ax[1,1].set_xlabel('WindDirection')
ax[1,2].hist(metadata_rpm43['WindDirection'],range=[0,360],color='#003f5c')
ax[1,2].set_xlabel('WindDirection')
plt.figure()
plt.hist(metadata_original['WindDirection'])

plt.show()



 

