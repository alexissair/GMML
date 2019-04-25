import numpy as np
import sys
import os
PATH = os.getcwd() + '/src'
sys.path.append(PATH)
from floydwarshall import *
from mmds import *

class Isomap() :
    def __init__(self, n_components, method, n_iter, k_neighbors = None, epsilon = None) :
        self.n_components = n_components
        self.method = method
        self.niter = n_iter
        assert self.method in ['epsilon', 'knn']
        if self.method == 'epsilon' :
            assert epsilon is not None
            self.epsilon = epsilon
        else :
            assert isinstance(k_neighbors, int)
            self.k = k_neighbors
        return
        
    def _build_graph(self, D) :
        if self.method == 'knn' :
            self._knn(D)
        else :
            self._epsilon(D)
        return

    def _knn(self, D) :
        '''A sample is linked to another if it is among its kth neighbors '''
        self.adjacency = np.zeros(D.shape)
        for i in range(D.shape[0]) :
            argsorted_i = np.argsort(D[i,:])
            argsorted_i = argsorted_i[:self.k]
            self.adjacency[i,argsorted_i] = 1
        self.graph = self.adjacency * D
        self.graph[self.adjacency == 0] = np.inf
        for i in range(self.graph.shape[0]):
            self.graph[i,i] = 0
        return

    def _epsilon(self, D) :
        '''Two samples are linked iff their dissimilarity is < eps * mean, 
        where mean is the mean dissimilarity'''
        self.adjacency = np.zeros(D.shape)
        mean = np.mean(D)
        self.adjacency[D < self.epsilon * mean] = 1
        self.graph = self.adjacency * D
        self.graph[self.adjacency == 0] = np.inf
        for i in range(self.graph.shape[0]):
            self.graph[i,i] = 0
        return
    
    def _call_FW(self) :
        return FloydWarshall(self.graph)
    
    def _fill_inf(self, X) :
        m = max(X[X != np.inf])
        X[X == np.inf] = m
        return
    
    
    def fit(self, D, fill_inf = True, verbose = False) :
        'D is the matrix of initial pairwise distances '
        self._build_graph(D)
        self.geodesic = self._call_FW()
        if fill_inf :
            self._fill_inf(self.geodesic)
        m = MMDS(self.n_components, self.niter)
        m.fit(self.geodesic, verbose=verbose)
        self.Zs, self.stress = m.Zs, m.stress
        self.X_transformed = np.array(self.Zs[-1])
        return