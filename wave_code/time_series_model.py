import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.metrics import mean_squared_error

def ready_data(file,pred_matrix):
	X = pd.read_csv(file)
	pred_matrix = pd.read_csv(pred_matrix)
	pred_matrix = pred_matrix.set_index('datetime')
	pred_matrix.index = pd.to_datetime(pred_matrix.index)
	X = X.set_index('datetime')
	X.index = pd.to_datetime(X.index)
	y1 = X['WVHT']
	y2 = X['DPD']
	X = X[['#YY','MM','DD','hh']]
	X['Q1']=0
	X['Q2']=0
	X['Q3']=0
	X['Q4']=0
	return X, y1, y2, pred_matrix


def quarter_dummies(data):

	for index, i in enumerate(data['MM']):
		if i == 1 or i == 2 or i == 3:
			data['Q1'][index] = 1
		elif i == 4 or i == 5 or i == 6:
			data['Q2'][index] = 1
		elif i == 7 or i == 8 or i == 9:
			data['Q3'][index] = 1
		elif i == 10 or i == 11 or i == 12:
			data['Q4'][index] = 1
	return data

def ols(X,wvht, dpd, pred_matrix):
	X = quarter_dummies(X)
	drop_mo = X.drop('MM', axis=1)
	drop_mo['time'] = range(1,drop_mo.shape[0]+1)
	drop_mo['time2'] = drop_mo['time']**2
	drop_mo['time3'] = drop_mo['time']**3

	X_train = drop_mo[['time','time2', 'time3','Q1','Q2','Q3','Q4']]
	model_wvht = sm.OLS(wvht, sm.add_constant(X_train)).fit()
	model_dpd = sm.OLS(dpd, sm.add_constant(X_train)).fit()

	forecast_wvht = model_wvht.predict(pred_matrix)
	forecast_dpd = model_dpd.predict(pred_matrix)

	resid_wvht = model_wvht.resid
	resid_dpd = model_dpd.resid

	y_pred_wvht = model_wvht.predict()
	y_pred_dpd = model_dpd.predict()

	return resid_wvht, resid_dpd, y_pred_wvht, y_pred_dpd, forecast_wvht, forecast_dpd

def arima(resid_wvht, resid_dpd):  # pass in ols(X1,y1,y2)[0], ols(X1,y1,y2)[1]for resid 
	ts_wvht=sm.tsa.SARIMAX(resid_wvht.values, order=(4,1,0),seasonal_order = (0,1,2,4),freq='H').fit() 
	ts_dpd=sm.tsa.ARIMA(resid_dpd.values, order=(5,1,2),  freq='H').fit()

	predict_insample_wvht = ts_wvht.predict(start=4,typ='levels')
	predict_insample_dpd = ts_dpd.predict(start=5,typ='levels')

	predict_oos_wvht = ts_wvht.forecast(steps=5)
	predict_oos_dpd = ts_dpd.forecast(steps=5)

	mse_wvht = mean_absolute_error(resid_wvht[4:],predict_insample_wvht) 
	mse_dpd = mean_absolute_error(resid_dpd[5:],predict_insample_dpd) 

	err_pred = mean_absolute_error(resid_wvht[-5:], predict_oos_wvht)

	return predict_insample_wvht, predict_insample_dpd, predict_oos_wvht, predict_oos_dpd, mse_wvht, mse_dpd, err_pred

def prediction_plots(X1,y1,y2,y_pred_wvht, y_pred_dpd, resid_wvht, predict_insample_wvht, 
					pred_insample_dpd, forecast_wvht,forecast_dpd):

	predicted_values_wvht = list(y_pred_wvht[4:]+predict_insample_wvht)
	predicted_values_dpd = list(y_pred_dpd[5:]+predict_insample_dpd)
	predicted_values_wvht.extend(list(forecast_wvht))
	predicted_values_dpd.extend(list(forecast_dpd))
	# zeros = list(np.zeros(len(y1.index[len(y1)-100:])))
	# zeros.extend(list(forecast_wvht))
	# zeros.extend(list(forecast_dpd))

	fig, ax = plt.subplots(2,1)
	#plt.subplot(2, 1, 1)
	ax[0].plot(y1.index[len(y1)-305:]+timedelta(hours=5),predicted_values_wvht[len(y1)-304:], color='r',label='Predicted Wave height')
	ax[0].plot(y1.index[len(y1)-300:],y1[len(y1)-300:],color='g', label = "Observed Values") #last 2 months
	ax[0].set_title('300 hourly Waveheight observations: Seasonal ARIMA: Bodega Bay, CA \n \n Mean Absolute Error : .12146',fontsize=18)
	ax[0].set_ylabel('Meters')

	ax[0].legend(loc='best')

	ax[1].plot(y2.index[len(y2)-304:]+timedelta(hours=5),predicted_values_dpd[len(y2)-304:],color='r',label='Predicted Wave Period')
	#plt.plot([1,2,3,4,5], forecast_dpd, color='b')
	ax[1].plot(y2.index[len(y2)-300:],y2[len(y2)-300:],color='g', label = "Observed Values") #last 2 months
	ax[1].set_title('300 hourly Dominant Wave Period observations: Seasonal ARIMA :  Bodega Bay, CA \n \n  Mean Absolute Error : 1.0553',fontsize=18)
	ax[1].set_xlabel('Date')
	ax[1].set_ylabel('Seconds')

	ax[1].legend(loc='best')
	plt.tight_layout()
	plt.show()

	return predicted_values_wvht[-5:], predicted_values_dpd[-5:]
