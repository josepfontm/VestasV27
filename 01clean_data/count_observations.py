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

idle_ds = metadata_idle.DamageScenario.values
parked_ds = metadata_parked.DamageScenario.values
t1_ds = metadata_t1.DamageScenario.values
rpm32_ds = metadata_rpm32.DamageScenario.values
t2_ds = metadata_t2.DamageScenario.values
rpm43_ds = metadata_rpm43.DamageScenario.values

print('Parked')
print(list(parked_ds).count('undamaged'))
print(list(parked_ds).count('15cm'))
print(list(parked_ds).count('30cm'))
print(list(parked_ds).count('45cm'))
print(list(parked_ds).count('repaired'))

print('Idle')
print(list(idle_ds).count('undamaged'))
print(list(idle_ds).count('15cm'))
print(list(idle_ds).count('30cm'))
print(list(idle_ds).count('45cm'))
print(list(idle_ds).count('repaired'))

print('T1')
print(list(t1_ds).count('undamaged'))
print(list(t1_ds).count('15cm'))
print(list(t1_ds).count('30cm'))
print(list(t1_ds).count('45cm'))
print(list(t1_ds).count('repaired'))

print('RPM32')
print(list(rpm32_ds).count('undamaged'))
print(list(rpm32_ds).count('15cm'))
print(list(rpm32_ds).count('30cm'))
print(list(rpm32_ds).count('45cm'))
print(list(rpm32_ds).count('repaired'))

print('T2')
print(list(t2_ds).count('undamaged'))
print(list(t2_ds).count('15cm'))
print(list(t2_ds).count('30cm'))
print(list(t2_ds).count('45cm'))
print(list(t2_ds).count('repaired'))

print('RPM43')
print(list(rpm43_ds).count('undamaged'))
print(list(rpm43_ds).count('15cm'))
print(list(rpm43_ds).count('30cm'))
print(list(rpm43_ds).count('45cm'))
print(list(rpm43_ds).count('repaired'))