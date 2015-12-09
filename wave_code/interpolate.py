import pandas as pd 
import numpy as np 
import scipy as scs
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt 
from datetime import datetime


#data['energy_output'] = f(data['WVHT'],data['DPD'])

def load_conv_matrix():
	matrix = pd.read_csv('energy_conversion_matrix.csv')
	return matrix

def load_data(file):
	data =  pd.read_csv(file, low_memory=False).iloc[1:,]
	data['datetime'] =  pd.to_datetime(data['datetime'], format='%m/%d/%Y %H:%M')
	for i in data.columns:
		if type(data[i][1]) != pd.tslib.Timestamp:
			data[i] = data[i].astype(float)

	return data 

def get_col():
	data = load_conv_matrix()
	x = data['Sig_Wave_Height']
	y = data['Power_Period']
	z = data['Elec_Output']
	return x, y, z

def interpolate(height,period,output, kind='linear'):
	energy = []
	f = scs.interpolate.interp2d(height, period, output, kind=kind)
	xnew = np.arange(.5,8.05,.05)
	ynew = np.arange(5,13.05,.05)

	for i in xnew:
		for j in ynew:
			if f(i,j) < 0:
				energy.append(0)
			else:
				energy.append(float(f(i,j)))
	return energy, xnew, ynew

def replace_nan(data, column, value_to_replace):
	no_nan = data[column].replace(value_to_replace,np.nan)
	interp_data = no_nan.interpolate()
	data[column] = interp_data
	return data


def energy_output(x,y,output, data):
	g = scs.interpolate.interp2d(x, y, output, kind='linear')
	data['energy_output'] = g(data['WVHT'], data['DPD'])
	return data 

if __name__ == '__main__':

	matrix = load_conv_matrix()
	data = load_data('46013_00_14_116.4M/46013_BodBayCA.csv')
	cols = get_col()
	x1 = cols[0]
	y1 = cols[1]
	z1 = cols[2]
	xnew = interpolate(x1,y1,z1)[1]
	ynew = interpolate(x1,y1,z1)[2]
	energy = interpolate(x1,y1,z1)[0]
	data_new = replace_nan(data,'WVHT',99)
	data_new1 = replace_nan(data_new,'DPD',99)
	data_new2 = replace_nan(data_new1,'APD',99)
	data_new3 = replace_nan(data_new2,'MWD',999)
	data_new4 = replace_nan(data_new3,'GST',99)
	data_new5 = replace_nan(data_new4,'WTMP',999)
	data_new6 = replace_nan(data_new5,'PRES',99)
	data_new7 = replace_nan(data_new6,'PRES',9999)


	data_new = energy_output(xnew,ynew, energy,data_new7)



