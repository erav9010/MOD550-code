#!/usr/bin/env python
# coding: utf-8

# In[327]:


#Excercise 2
#Author - Erik William Ravndal


# In[328]:


import sklearn.metrics as skm
import sklearn as sk
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import timeit as it


# In[329]:


#1 Task

def mean_squared_error_van(observed, predicted):
    if len(observed) != len(predicted):
        raise ValueError("The lengths of input lists are" + f"not equal {len(observed)} {len(predicted)}.")
    # Initialize the sum of squared errors
    sum_squared_error = 0
    # Loop through all observations
    for obs, pred in zip(observed, predicted):
    # Calculate the square difference, and add it to the sum
        sum_squared_error += (obs - pred) ** 2
    # Calculate the mean squared error
        mse = sum_squared_error / len(observed)
    return mse

def mean_squared_error_np(observed, predicted):
    observed_np = np.array(observed)
    predicted_np = np.array(predicted)
    mse = np.mean((observed_np - predicted_np) ** 2)
    return mse

observed = [2, 4, 6, 8]
predicted = [2.5, 3.5, 5.5, 7.5]


mse_vanilla = mean_squared_error_van(observed, predicted)
time_v = it.timeit('mean_squared_error_van(observed, predicted)', globals=globals(), number=10) / 100

mse_numpy = mean_squared_error_np(observed, predicted)
time_np = it.timeit('mean_squared_error_np(observed, predicted)',globals=globals(), number=10) / 100

sk_mse = skm.mean_squared_error(observed, predicted)
time_sk = it.timeit('skm.mean_squared_error(observed, predicted)',globals=globals(), number=10) / 100


for mse, mse_type, time in zip([mse_vanilla, mse_numpy, sk_mse],['vanilla', 'numpy', 'sklearn'],[time_v, time_np, time_sk]):
    print(f"Mean Squared Error, {mse_type}:", mse,f"Average execution time: {time} seconds")
    assert(mse_vanilla == mse_numpy == sk_mse)
    print("Task completed sucessfully")


# In[330]:


#Task 2
class task2:
    def __init__(self,npoints,freq,timestart,timestop,amplitude):
        self.npoints=npoints
        self.freq=freq
        self.timestart=timestart
        self.timestop=timestop
        self.amplitude=amplitude
        self.dataset=None
        return
    
    def oneDimoscillatory(self):
        self.dataset=pd.DataFrame()
        #we do the same as last time round in exercise 1.
        time=np.linspace(self.timestart,self.timestop,self.npoints)
        self.dataset["time"]=time
        self.dataset["true_curve"]=self.amplitude*np.sin(2*np.pi*self.freq*time)
        ranger=[self.timestart,self.timestop]
        noise=np.random.rand(len(time))*self.amplitude
        self.dataset["noisy_data"]=self.dataset["true_curve"]+noise
        #also print the dataframe output so that the dataset can be assessed - i.e. stats as an output (pandas df stats)
        plt.plot(time,self.dataset["true_curve"],color="black")
        plt.scatter(time,self.dataset["noisy_data"],color="red",alpha=0.3)
        plt.plot()
        
        print(f"Data generated: points: {self.npoints}, range: {ranger}, amplitude {self.amplitude}, frequency {self.freq}")
        return self.dataset


# In[331]:


a=task2(npoints=300,freq=0.01,timestart=0,timestop=100,amplitude=9)
dataset=a.oneDimoscillatory()


# In[332]:


#Task 3: Data Clustering
#sources:
#https://www.youtube.com/watch?v=5w5iUbTlpMQ
#https://en.wikipedia.org/wiki/K-means_clustering
#https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
#Unsupervised_base.ipynb from example files
    
class kmean_cluster:
    def __init__(self,max_k,data,N):
        #N=number of iterations
        self.N=N
        #K=number of clusters
        self.max_k=max_k
        self.data=data
        self.X=None
        self.means=None
        self.output=None
        

        return
    
    def optimizer(self):
        #will use the cluster centre as the mean (hence name k-means)
        #NOTE : you MUST run optimizer - prior to running sort...
        #we know the data and for reproducivity the data is preselected here - however should one genrealize then there would be an input function here or some table of features
        x2=self.data["noisy_data"].values
        x1=self.data["time"].values
        
        self.X=np.column_stack((x1,x2))
        num_k=[]
        variance=[]
        for i in range(1, self.max_k):
            kmeans=KMeans(n_clusters=i,init="random",max_iter=self.N)
            kmeans.fit(self.X)
            num_k.append(i)
            
            inertia=kmeans.inertia_
            #as there is a "K" number of means hence and the inertia is the squared distance from the centre the variance should be something equal to this...
            variance.append(inertia/self.N)
        
        self.output=pd.DataFrame()
        self.output["variance"]=variance
        self.output["number of clusters"]=num_k
        plt.plot(num_k,variance,color="black",label="variance")
        plt.xlabel("number of clusters")
        plt.ylabel("variance")
        plt.legend()
        plt.show()
        print(self.output)
        return self.output
    
    def sort(self,k):
        #this module is basically repeating the above, however we are utilzing the findings from previus function to find the optimal number k
        self.centroids=[]        
        
        kmeans=KMeans(n_clusters=k,init="random",max_iter=self.N)
        kmeans.fit(self.X)
        labels=kmeans.labels_
        self.centroids=kmeans.cluster_centers_
        variance=self.output["variance"][4]
        plt.figure(figsize=(15,6))
        plt.scatter(self.data["time"],self.data["noisy_data"],c=labels)
        
        for i in range(0,len(self.centroids[:,0])):
            plt.scatter(self.centroids[i,0],self.centroids[i,1],color="red",label=f"center of cluster {i+1}",marker="*")
        plt.legend()
        plt.show()
        
        print(f"number of means: {5}", "\n", f"location of means: {self.centroids}","\n", f"avg. variance: {variance}")
        
        return
    
        


# In[333]:


#Task 3: Data Clustering Output:
b=kmean_cluster(max_k=18,data=dataset,N=400)
b.optimizer()
print("\n")
print("We can see that there greatest change in variance occurs at 5 clusters, hence the selection of this number for number of means")
print("\n")
b.sort(k=5)
print("\n")
print("Selected method for initiation of k-means was randomized centroids although kmeans++ gives this capability,"\
      "\n the latter was better understood by myself. Also it is understood from relevant articles that this was more accurate than randomly selecting from dataset")



# In[366]:


#Task 4: LR NN PINN - regression of dataset:
#Task 5 Plot regression as a function of iteration:

#sources:
#Supervised_base.ipynb - from coursework
#https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression
#https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor


class regression:
    
    def __init__(self,data):
        self.data=data
        self.feat=None
        self.LR_out=None
        self.NN_out=None
        self.PINN_out=None
        return
    
    def linreg(self):
        x=self.data["time"].values
        y=self.data["noisy_data"].values
        x_2=x.reshape(-1,1)
        reg = LinearRegression().fit(x_2, y)
        y_2=reg.predict(x_2)
        self.LR_out=y_2
        plt.scatter(x_2,y,color="blue",label="data",alpha=0.3)
        plt.plot(x_2,y_2,color="red",label="predicted")
        plt.legend()
        plt.show()
        #skm.mean_squared_error(observed, predicted)
        print("Task completed Linear Regression")
        return
        
    def NeuralNet(self,max_N):
        self.NN_out=pd.DataFrame()
        x=self.data["time"].values
        y=self.data["noisy_data"].values
        x_2=x.reshape(-1,1)
        #after some trial and error it appears that as I adjusted the number of parameters and adaptive training step and using the tanh and adam activation function were sufficent
        for i in range (100,max_N+1,100):
            if i%100==0:
                reg2=MLPRegressor(activation="tanh",learning_rate="adaptive",solver="adam",max_iter=i)
                reg2.fit(x_2,y)
                y_2=reg2.predict(x_2)
                self.NN_out[f"{i}"]=y_2
                plt.scatter(x_2,y,color="blue",label="data",alpha=0.3)
                plt.plot(x_2,y_2,color="red",label=f"predicted at {i} No. iterations")
                plt.legend()
                plt.show()
        print("Task completed NN Regression")
        return
    
    def PINN(self):
        #use PINN from example file torch can be used?
        #DNF --- did not finish ---
        return

    #Task 6:
    def true_error(self):
        #comparison between true value and the approximate function
        tru=self.data["true_curve"].values
        x=self.data["time"].values
        x_2=x.reshape(-1,1)
        v1=tru-self.LR_out
        e1=abs(v1)
        
        R1=np.dot(v1.T,v1)
        
        headers=self.NN_out.columns
        
        R2=pd.DataFrame(columns=headers)
        e2=pd.DataFrame(columns=headers)
        
        #error estimate NN
        for i in headers:
            val_NN=self.NN_out[i].values
            v2=tru-val_NN
            e2[i]=abs(v2)
            r=np.dot(v2.T,v2)
            R2.at[0,i]=r
        
        plt.figure(figsize=(15,6))
        plt.plot(x_2,e1,label="Linear Regression error")
        
        for i in headers:
            plt.plot(x_2,e2[i],label=f"NN Regression error @ {i} iterations")
        plt.xlabel("Time")
        plt.ylabel("error")
        plt.legend()
        plt.show()
        
        #plot the deviation between 
        plt.figure(figsize=(15,6))
        #For some reason I cannot plot the LR residual...
        #plt.scatter([1],[float(R1)],"Linear Regression Residual")
        plt.plot(headers,R2.iloc[0],label="NN Regression Residual")
        plt.xlabel("No. Iterations")
        plt.ylabel("Residual")
        plt.legend()
        plt.show()
        
        return
    
    #Task 7:
    def error_estimate(self,max_N):
        #we will here split the datgaset into two categories - training and test:
        self.NN_out=pd.DataFrame()
        x=self.data["time"].values
        y=self.data["noisy_data"].values
        N=len(x)
        #splitting the data into the following: training and test dataset - in order to quantify deviance of the model relative to data:
        #NOTE: Experience with this kind of work on unsupervised models in a monitoring environment has led to overfitting and underpeformance...
        n_train=int((N/100)*70)
        
        #owing to the nature of the dataset (timeseries) - the dataset must be sampled along the time axis
        x_train=x[:n_train]
        x_test=x[n_train:]
        y_train=y[:n_train]
        y_test=y[n_train:]
        
        #convert into usable format (2D array)
        x_2te=x_test.reshape(-1,1)
        x_2tr=x_train.reshape(-1,1)
        
        #after some trial and error it appears that as I adjusted the number of parameters and adaptive training step and using the tanh and adam activation function were sufficent
        iterations=[]
        R2_score=[]
        
        for i in range (100,max_N+1,100):
            if i%100==0:
                iterations.append(i)
                reg2=MLPRegressor(activation="tanh",learning_rate="adaptive",solver="adam",max_iter=i)
                reg2.fit(x_2tr,y_train)
                reg2.predict(x_2te)
                r2=reg2.score(x_2te,y_test)
                R2_score.append(r2)
        plt.figure(figsize=(15,6))
        plt.plot(iterations,R2_score,color="red",label="Residual")
        plt.ylabel("Residual [R^2]")
        plt.xlabel("iterations")
        plt.legend()
        plt.show()
        return
        


# In[367]:


#initializing with dataset from Task 1.
c=regression(data=dataset)


# In[368]:


#Task 4: LR NN PINN - regression of dataset (output):
#Task 5 Plot regression as a function of iteration (output):
c.linreg()
c.NeuralNet(500)


# In[369]:


#Task 6 output:
c.true_error()
print("The only really interesting value is the residual score per iteration as that would indicate the total performance of the model relative to the totality of the actual dataset")


# In[373]:


#Task 7 output:
c.error_estimate(500)


# In[ ]:





# In[ ]:




