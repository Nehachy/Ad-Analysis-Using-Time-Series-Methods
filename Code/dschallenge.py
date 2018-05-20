import numpy as np
from numpy.linalg import inv
import csv
import matplotlib.pyplot as plt
import collections
import pprint
import os

#fetch current directory
#/Users/nehachoudhary/Desktop/DS_Challenge
#fold = os.getcwd()
fold = '/Users/nehachoudhary/Desktop/DS_Challenge/'
print ("current directory", fold)

def makehash():
	return collections.defaultdict(makehash)

def read_data(url):
	data = np.genfromtxt(url, delimiter=",", dtype=None)
	header_list = data[0,:]
	data = np.delete(data, 0, 0)
	return data, header_list

def data_partition(group, data):
	mask=[]
	for i in range(data.shape[0]):
		if data[i,6]==group:
			mask.append(i)
		else:
			pass
	ad_data = data[mask]
	return ad_data

def plot_data(y,group):
	x = np.arange(1,y.shape[0]+1,1)
	plt.plot(list(x),y, label= group)
	plt.title("Ad Group Shown over time")
	plt.xlabel("Time")
	plt.ylabel("No. of time shown")
	plt.legend()
	plt.savefig( fold + "/Figures/time" + str(group) + ".png")
	plt.close()

def train_validation_set(data,c):
	C = data.shape[0]-c
	train, test = data[:C], data[C:]
	return train, test

def compute_mse(test, pred):
	return np.mean(np.abs((test - pred)/test))

def moving_average(series, n):
	series = list(series)
	pred = sum(series[-n:])/n
	return pred

def get_c_prediction(series,n,c):
	pred_mv = []
	X = list(series)
	for i in range(c):
		X = X + [moving_average(X, n)]
	return X[-c:]

def compute_coeff(x,y):
	beta=np.matmul((inv(np.matmul(x.T,x))),np.matmul(x.T,y))
	#print (beta=np.matmul((inv(np.matmul(x.T,x))),np.matmul(x.T,y)))
	#print (beta.shapeč
	return beta
	
url = fold+'/Data/ad_table (1).csv'
full_data, header_list= read_data(url)

#Get the ad groups
ad_groups = list(np.unique(full_data[:,6]))
size_of_test = 14 

###Q1 - Top 5

myhash = makehash()
#myhash = {}
metrics = [1,2,3,5] #select metrics
for j in metrics:
	for i in range(len(ad_groups)):
		data = data_partition(ad_groups[i], full_data)
		myhash[header_list[j]][ad_groups[i]]=np.mean(data[:,j].astype(float))

top5 = {}

#top5 = {header_list[i]:sorted(myhash[header_list[i]].items(), key= lambda(k,v): v, reverse=True)[0:5] for i in metrics}
top5 = {header_list[i]:sorted(myhash[header_list[i]].items(), key= lambda kv: kv[1], reverse=True)[0:5] for i in metrics}
print("Print Top 5 Ad Group by Average of Metrics")
print("\n")
pp = pprint.PrettyPrinter(width=20, compact=False)
pp.pprint(top5)
print("Total Revenue and Converted are the metrics that are most preferable to identify the top5 groups.")
print("It is observed that Total Revenue and Converted have the same groups as well as the order in which they appear.")
###Q2 - Forecast
#shown
Forecast ={}

for i in range(len(ad_groups)):
	ad_data = data_partition(ad_groups[i], full_data)
	shown = ad_data[:,1].astype(float)
	missing_substitution = np.mean(shown)
	y = shown==0
	shown[y]= missing_substitution 
	plot_data(shown,ad_groups[i])
	#Training, Validation split
	train, test = train_validation_set(shown,size_of_test)
	###Q2 - Forecast
	pred = get_c_prediction(train,7,size_of_test)
	#print(pred)
	print("Ad Group =", ad_groups[i],"Validation MAPE % = ",100.0*compute_mse(test, np.array(pred)))
	#Get prediction
	Forecast[ad_groups[i]] = get_c_prediction(shown,7,23)

for k,v in Forecast.items():
	print("Forecast for Dec 15 for ad group ", k, v[-1])

###Q3 - Trend
#avg_cost_per_click
Trend = {}

for i in range(len(ad_groups)):
	ad_data = data_partition(ad_groups[i], full_data)
	y=ad_data[:,4].astype(float)
	x=np.arange(1,y.shape[0]+1)
	b=np.ones(y.shape[0])#adding bias
	xin = (np.append([b],[x],axis=0)).T
	beta = compute_coeff(xin,y)
	Trend[ad_groups[i]] = beta[1]
	#print("Trend Coeff = ", beta[1])

cluster_1 = []
cluster_2 = []
cluster_3 = []
for k,v in Trend.items():
	if v > 0:
		cluster_1.append(k)
	elif v < 0 :
		cluster_2.append(k)
	else:
		cluster_3.append(k)

print("cluster_1: avg_cost_per_click is going up", cluster_1)
print("cluster_2: avg_cost_per_click is flat", cluster_3)
print("cluster_3: avg_cost_per_click is going down", cluster_2)





