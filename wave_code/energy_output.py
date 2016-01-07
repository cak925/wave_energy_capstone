import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates



def load_data(data):
	data  = pd.read_csv(data)
	data = data.set_index('datetime')
	data.index = pd.to_datetime(data.index)
	return data


def groupby(data_new):
	energy_only = data_new[['energy_per_hour','#YY','MM']]
	year = energy_only.groupby(['#YY','MM'])['energy_per_hour'].sum()
	return year
	


def plot_energy(energy_data):
	energy_data = pd.read_csv(energy_data)
	energy_data = energy_data.set_index('datetime')
	energy_data.index = pd.to_datetime(energy_data.index)
	
	fig, ax = plt.subplots()
	
	index = energy_data['Month']
	bar_width = 0.2
	opacity = 0.4
	rects1 = plt.bar(index, energy_data['bb'], bar_width,
	         alpha=opacity,
	         color='b',
	         label='Bodega Bay: Average Per Month 14,152 households\n')
	rects2 = plt.bar(index+bar_width, energy_data['cc'], bar_width,
	         alpha=opacity,
	         color='r',
	         label='Crescent City: Average Per Month 14,175 households\n')
	rects3 = plt.bar(index+2*bar_width, energy_data['cm'], bar_width,
	         alpha=opacity,
	         color='g',
	         label='Cape Mendicino: Average Per Month 20,842 households')
	plt.legend(loc='best')
	plt.xlabel('Month', fontsize=20)
	plt.hlines(14152, linestyles='dashed', color='b',xmin=0,xmax=25)
	plt.hlines(14175, linestyles='dashed', color='r',xmin=0,xmax=25)
	plt.hlines(20842, linestyles='dashed', color='g',xmin=0,xmax=25)
	plt.ylabel('# of households', fontsize = 20)
	plt.title('Total Households Powered Per Month', fontsize = 24)
	plt.show()
