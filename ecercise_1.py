#MOD 550 Assignment 1 Author: Erik William Ravndal

#sources: https://www.geeksforgeeks.org/python-pandas-dataframe-append/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#1. Construct a class that is able to generate a 2D dataset. 
#2. The 2d dataset can be: 2d random, noise around a FUNCTION, your truth (not a line). 


class datagen:
    def __init__(self,length,rangeval):
        self.length=length
        self.rangeval=rangeval
        self.independent=None
        self.dependent=None
        self.true_curve=None
        self.approx_curve=None
        self.dataset1=None
        self.dataset2=None
        self.data_comb=None
        return
    
    def data_2d_curve(self):
        #we will here use
        self.dataset1=pd.DataFrame()
        
        midpoint=(self.rangeval[1]-self.rangeval[0])/2
        
        a=3
        b=2
        
        self.independent=midpoint+np.random.randn(self.length)*self.rangeval[1]/6
        noise=np.random.randn(self.length)*100
        #to create a curve that is somewhat accurate a 
        self.true_curve=a*np.sort(self.independent)**2+b
        self.dependent=(a*self.independent**2+b)+noise
        self.dataset1["random_independent_val"]=self.independent
        self.dataset1["random_dependent_val"]=self.dependent
        self.dataset1["True_curve"]=self.true_curve

        
        plt.figure(figsize=(15,6))
        plt.scatter(self.independent,self.dependent,label="random data",color="blue",alpha=0.3)
        plt.plot(np.sort(self.independent),self.true_curve,label="curve function",color="black")
        plt.title("dataset_1")
        plt.legend()
        plt.savefig("2d_data_structured.jpeg")
        plt.show()
        return 
    
    def data_2d_random(self):
        #this will only output a random dataset in 2D
        self.dataset2=pd.DataFrame()
        midpoint=(self.rangeval[1]-self.rangeval[0])/2
        data=np.random.rand(2,self.length)*midpoint
        self.dataset2["First dimension"]=data[0,:]
        self.dataset2["Second dimension"]=data[1,:]
        
        #Plotting the data
        plt.figure(figsize=(15,6))
        plt.scatter(data[0,:],data[1,:],label="random data",color="blue",alpha=0.3)
        plt.title("dataset_2")
        plt.legend()
        plt.savefig('random2d_data.jpeg')
        plt.show()
        return

#2. The 2d dataset can be: 2d random, noise around a FUNCTION, your truth (not a line). 

    def combinator(self):
        self.dataset1=None
        self.dataset2=None
        self.data_2d_curve()
        self.data_2d_random()
        
        self.data_comb=pd.concat({"dataset_1":self.dataset1,"dataset_2":self.dataset2},axis=1)#,keys=['Dataset 1', 'Dataset 2'], axis=0)
        self.data_comb.to_csv('exercise_1_data.csv')
        plt.figure(figsize=(15,6))
        plt.scatter(self.dataset1.iloc[:,0],self.dataset1.iloc[:,1],label="dataset1",color="red",alpha=0.5)
        plt.scatter(self.dataset2.iloc[:,0],self.dataset2.iloc[:,1],label="dataset2",color="blue",alpha=0.5)
        plt.title("combined plot")
        plt.legend()
        plt.savefig('combined_data.jpeg')
        plt.show()
        return

#run the code here
a=datagen(200,[0,20])

a.combinator()
