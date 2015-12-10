
import pandas as pd
import numpy as np
import statsmodels as sm
from scipy import stats
import matplotlib.pyplot as plt

def time_model(data):

	y = data['WVHT'].values
	no_wvht = data.drop('WVHT', axis=1)
	no_wvht.set_index('datetime')
	no_wvht['time'] = range(1,no_wvht.shape[0]+1)
	X = no_wvht.drop(['#YY','MM','DD','mm','FLUX1_mean','FLUX2_peak','energy_output'], axis=1)
	X_new = sm.add_constant(X).set_index('datetime')
	model = sm.OLS(y, X_new).fit()

	data['WVHT'].plot(figsize=(12,8))
	pd.Series(model.fittedvalues).plot()



y = dat2012['WVHT'].values
no_wvht = dat2012.drop('WVHT', axis=1)
no_wvht['time'] = range(1,no_wvht.shape[0]+1)
no_wvht['time2'] = no_wvht['time']**2
no_wvht['time3'] = no_wvht['time']**3
X_new = no_wvht[['time','time2', 'time3']]
# X = no_wvht.drop(['#YY','MM','DD','mm','FLUX1_mean','FLUX2_peak','energy_output', 'APD', 'WSPD','GST','WTMP','PRES'], axis=1)
X_new = sm.add_constant(X_new)
model = sm.OLS(y, X_new).fit()
model.summary()

dat2012['WVHT'].plot(figsize=(12,8))
pd.Series(model.fittedvalues).plot()


def mess_with_month(data):
	jantest = pd.read_csv('clean_output.csv')
	data_Jan = jantest.loc[jantest['MM'] == 1]
	c = data_Jan.set_index('datetime')
	plt.xticks(rotation=45)
	c['WVHT'].plot(figsize=(1y0,5))



	dat2010 = jantest.loc[jantest['#YY'] == 2010]