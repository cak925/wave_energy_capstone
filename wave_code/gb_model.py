from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn import grid_search
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np 
from datetime import timedelta

def grid_search(X, y, split, max_features=[4,6,8,None], learning_rate=[.002,.005,.05,.1]):
		for feat in max_features:
			for learn in learning_rate:
				model = GradientBoostingRegressor(n_estimators=2000,
													learning_rate=learn,
													max_features=feat,
													subsample = .3,
													min_samples_leaf=50,
													random_state=3)
				model.fit(X[:split], y[:split])
				in_samp_score = mean_squared_error(model.predict(X[:split]), y[:split])
				out_samp_score = mean_squared_error(model.predict(X[split:]), y[split:])
				print 'learn, max_features: {},{}'.format(learn,feat)
				print 'in-sample score, out-sample score: {}, {}'.format(in_samp_score, out_samp_score)
			


def cross_val_ts(X,y, model,time):
	 in_sample_score = []
	 oos_score = []
	 for i in range(1,len(y)/time):
	 	y_train = y[0:i*time]
	 	y_test = y[time*i:]
	 	X_train = X.iloc[0:i*time,]
	 	X_test = X.iloc[i*time:,]
	 	model.fit(X_train,y_train)
	 	in_sample_score.append(mean_squared_error(y_train, model.predict(X_train)))
	 	oos_score.append(mean_squared_error(y_test, model.predict(X_test)))
	 	print 'Time Window', i 
	 	print 'in sample score', in_sample_score[-1]
	 	print 'out of sample score', oos_score[-1]

	 return model, np.mean(in_sample_score), np.mean(oos_score)


def grad_boost_model1(X,y, time, n_estimators=2000, learning_rate = .002, min_samples_leaf=50,
					  subsample=.5, max_features=8):
	model = GradientBoostingRegressor(n_estimators=n_estimators, 
									  learning_rate = learning_rate,
									  min_samples_leaf = min_samples_leaf,
									  max_features=max_features,
									  subsample = subsample,
									  random_state = 1)
									  
	return cross_val_ts(X,y,model, time)	
	return model.fit(X,y)

def future_predict(X1_wvht,y1,X1_dpd,y2, pred_wvht, pred_dpd):
	model = GradientBoostingRegressor(n_estimators=2000, 
									  learning_rate = .002,
									  min_samples_leaf = 50,
									  max_features=8,
									  subsample = .5,
									  random_state = 1)

	for i in range(1,5):
		random_state=2
		w = model.fit(X1_wvht[:-i],y1[:-i])
		fore_wvht = w.predict(X1_wvht[-i:])
		mae = mean_absolute_error(y1[-5:], fore_wvht)
		avg_resid = np.mean(np.array(y1[-i:]) - fore_wvht)
		print mae, fore_wvht, y1[-i:]

	for i in range(1,5):
		random_state=2
		w = model.fit(X1_dpd[:-i],y2[:-i])
		fore_dpd = w.predict(X1_dpd[-i:])
		mae = mean_absolute_error(y2[-5:], fore_dpd)
		avg_resid = np.mean(np.array(y2[-i:]) - fore_dpd)
		print mae, fore_dpd, y2[-i:]



def prediction_plots(X1_wvht,y1,X1_dpd,y2, pred_wvht, pred_dpd):
	model = GradientBoostingRegressor(n_estimators=2000, 
									  learning_rate = .002,
									  min_samples_leaf = 50,
									  max_features=8,
									  subsample = .5,
									  random_state = 1)
	w = model.fit(X1_wvht,y1)
	ypred_wvht= list(w.predict(X1_wvht))
	fore_wvht = w.predict(pred_wvht)
	ypred_wvht.extend(list(fore_wvht))

	d = model.fit(X1_dpd,y2)
	ypred_dpd= list(d.predict(X1_dpd))
	fore_dpd = d.predict(pred_dpd)
	ypred_dpd.extend(list(fore_dpd))

	fig, ax = plt.subplots(2,1)
	ax[0].plot(y1.index[len(y1)-300:]+timedelta(hours=5),ypred_wvht[len(y1)-295:], color='r',label='Predicted Wave Height')
	ax[0].plot(y1.index[len(y1)-295:],y1[len(y1)-295:],color='g', label = "Observed Values") #last 2 months
	ax[0].set_title('300 hourly Waveheight observations: Cape Mendicino, CA : Gradient Boosted')
	ax[0].set_xlabel('Date')
	ax[0].set_ylabel('Meters')

	ax[0].legend(loc='best')

	ax[1].plot(y2.index[len(y2)-300:]+timedelta(hours=5),ypred_dpd[len(y2)-295:], color='r',label='Predicted Wave Period')
	ax[1].plot(y2.index[len(y2)-287:],y2[len(y2)-287:],color='g', label = "Observed Values") #last 2 months
	ax[1].set_title('300 hourly Wave Period observations: Cape Mendicino, CA : Gradient Boosted')
	ax[1].set_xlabel('Date')
	ax[1].set_ylabel('Seconds')

	ax[1].legend(loc='best')
	plt.savefig('cm_gb.png')
	plt.tight_layout()
	plt.show()
