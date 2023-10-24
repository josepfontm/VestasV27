import pandas as pd
import matplotlib.pyplot as plt

path_metadata_idle = '../VestasV27/01clean_data/results/metadata_idle.csv'
path_metadata_parked = '../VestasV27/01clean_data/results/metadata_parked.csv'
path_metadata_t1 = '../VestasV27/01clean_data/results/metadata_t1.csv'
path_metadata_rpm32 = '../VestasV27/01clean_data/results/metadata_rpm32.csv'
path_metadata_t2= '../VestasV27/01clean_data/results/metadata_t2.csv'
path_metadata_rpm43 = '../VestasV27/01clean_data/results/metadata_rpm43.csv'

metadata_idle = pd.read_csv(path_metadata_idle,index_col=0,sep=';')
metadata_parked = pd.read_csv(path_metadata_parked,index_col=0,sep=';')
metadata_t1 = pd.read_csv(path_metadata_t1,index_col=0,sep=';')
metadata_rpm32 = pd.read_csv(path_metadata_rpm32,index_col=0,sep=';')
metadata_t2 = pd.read_csv(path_metadata_t2,index_col=0,sep=';')
metadata_rpm43 = pd.read_csv(path_metadata_rpm43,index_col=0,sep=';')

size = 3

plt.figure()
plt.rcParams.update({'font.size': 25})
plt.scatter(metadata_idle['Wind'], metadata_idle['RPM'],color='#ffa600',s=size)
plt.scatter(metadata_parked['Wind'], metadata_parked['RPM'],color='grey',s=size)
plt.scatter(metadata_t1['Wind'], metadata_t1['RPM'],color='#ff6361',s=size)
plt.scatter(metadata_rpm32['Wind'], metadata_rpm32['RPM'],color='#bc5090',s=size)
plt.scatter(metadata_t2['Wind'], metadata_t2['RPM'],color='#58508d',s=size)
plt.scatter(metadata_rpm43['Wind'], metadata_rpm43['RPM'],color='#003f5c',s=size)
plt.legend()

plt.xlabel('Wind [m/s]')
plt.ylabel('RPM')
plt.show()

plt.figure()
plt.scatter(metadata_idle['Wind'], metadata_idle['PitchMean'],color='#ffa600',s=size)
plt.scatter(metadata_parked['Wind'], metadata_parked['PitchMean'],color='grey',s=size)
plt.scatter(metadata_t1['Wind'], metadata_t1['PitchMean'],color='#ff6361',s=size)
plt.scatter(metadata_rpm32['Wind'], metadata_rpm32['PitchMean'],color='#bc5090',s=size)
plt.scatter(metadata_t2['Wind'], metadata_t2['PitchMean'],color='#58508d',s=size)
plt.scatter(metadata_rpm43['Wind'], metadata_rpm43['PitchMean'],color='#003f5c',s=size)
plt.legend()

plt.xlabel('Wind [m/s]')
plt.ylabel('Pitch Mean [degrees]')
plt.show()

plt.figure()
plt.scatter(metadata_idle['PitchMean'], metadata_idle['RPM'],color='#ffa600',s=size)
plt.scatter(metadata_parked['PitchMean'], metadata_parked['RPM'],color='grey',s=size)
plt.scatter(metadata_t1['PitchMean'], metadata_t1['RPM'],color='#ff6361',s=size)
plt.scatter(metadata_rpm32['PitchMean'], metadata_rpm32['RPM'],color='#bc5090',s=size)
plt.scatter(metadata_t2['PitchMean'], metadata_t2['RPM'],color='#58508d',s=size)
plt.scatter(metadata_rpm43['PitchMean'], metadata_rpm43['RPM'],color='#003f5c',s=size)
plt.legend()

plt.xlabel('Pitch Mean [degrees]')
plt.ylabel('RPM')
plt.show()