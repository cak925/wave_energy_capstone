import seaborn as sns 
import matplotlib.pyplot as plt 
from itertools import izip

def correlations(data):
    plt.figure(figsize=(15,10))
    sns.corrplot(data.iloc[:,6:])
    plt.savefig('heatmap1.png')

def plot_year(data, column):
	years = [2009,2010,2011,2012,2013,2014]
	ran = range(6)
	f, ax = plt.subplots(figsize=(5,10))
	for i, j in izip(years,ran):
		plt.subplot(3,2,j+1)
		data_i = data.loc[data['#YY'] == i]
		data_i[column].plot()

def plot_month(data, column):
	months = [1,2,3,4,5,6,7,8,9,10,11,12]
	ran = range(12)
	f, ax = plt.subplots(figsize=(12,8))
	for i, j in izip(months,ran):
		plt.subplot(6,2,j+1)
		data_i = data.loc[data['MM'] == i]
		plt.plot(data_i['DD'],data_i[column])


