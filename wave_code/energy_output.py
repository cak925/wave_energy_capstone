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
	energy_data =energy_data.set_index('datetime')
	dates = [energy_data.index[i] for i in range(len(energy_data))]
	fig, ax = plt.subplots()

	bar_width = 4
	opacity = 0.6
	error_config = {'ecolor': '0.3'}
	plt.bar(pd.to_datetime(dates), energy_data['households_40'], bar_width,
                 alpha=opacity,
                 color='r',
                 label='Average per month: 13,635 households')
	plt.legend(loc='best')
	plt.xlabel('Date')
	plt.ylabel('# of households')
	plt.title('Total Households Powered Per Month : \n Crescent City, CA - Jan. 2012 : Dec. 2014 : 40 Converters')
	plt.savefig('output_40_cc.png')
	plt.show()
