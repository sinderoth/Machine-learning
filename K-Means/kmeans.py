import math
import random
import numpy as np
import copy

random.seed(5710414)

class KMeans:
    def __init__(self, X, n_clusters, max_iterations=1000, epsilon=0.01, distance_metric="manhattan"):        
        self.X = X
        self.n_clusters = n_clusters
        self.distance_metric = distance_metric
        self.clusters = []
        self.cluster_centers = []
        self.epsilon = epsilon
        self.max_iterations = max_iterations

    def fit(self):
        center = []
        for i in range(self.n_clusters):
            color = ()
            color = self.generate_random_color()
            center.append(color)
        
        mind = self.findClusterIndex(self.X,center,self.distance_metric)
        mind = np.asarray(mind)
        eps = center

        for i in range(self.max_iterations):
            print ("KMeans iteration: "+ str(i+1))
            center = []
            for j in range(self.n_clusters):
                if len(self.X[mind==j])> 0:
                    average = np.mean(self.X[mind==j],axis=0)
                else:
                    average = (0,0,0)
                center.append(average)

            mind = self.findClusterIndex(self.X,center,self.distance_metric)
            mind = np.asarray(mind)   
            
            self.cluster = mind
            self.cluster_centers = center
            
            if self.epsilonCheck(center,eps,self.n_clusters,self.epsilon) == True:
                print ("Epsilon boundary reached! Halting...")
                return
            eps = center
        print ("Max iterations reached! Halting...")

    # predict cluter for given (rgb)
    def predict(self, instance):
        if instance!=list:
            instance = [instance]
            
        a = self.findClusterIndex(instance,self.cluster_centers,self.distance_metric)
        return a[0]

    # generate random color for initialization of kmeans
    def generate_random_color(self):
        return int(random.uniform(0, 256)), int(random.uniform(0, 256)), int(random.uniform(0, 256))

    # distance between two rgb
    def calculateDistance(self,a,b,metric):
        result = []
        a = np.asarray(a)
        b = np.asarray(b)
        
        if metric == "manhattan":
            return result
        else:
            dist = np.linalg.norm(a - b[0,:],axis=1).reshape(-1,1)
            for i in range(1,self.n_clusters):
                dist = np.append(dist,np.linalg.norm(a - b[i,:],axis=1).reshape(-1,1),axis=1)
            return dist
    
    # return cluster indices
    def findClusterIndex(self,x,center,metric):
        dist = self.calculateDistance(x,center,metric)
        ls = np.argmin(dist,axis=1)
        
        return ls

    # naive epsilon check for halt
    def epsilonCheck(self,c,e,k,epsilon):
        for i in range(k):
            if   math.sqrt( ( (c[i][0]-e[i][0])**2 + (c[i][1]-e[i][1])**2 + (c[i][2]-e[i][2])**2 )) > epsilon:
                return False
        return True