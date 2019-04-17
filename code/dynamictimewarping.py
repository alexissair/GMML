import numpy as np
import pandas as pd
from utils import *

class DTW() :
    def __init__(self, distance) :
        '''Note :
            - the distance takes two arguments x and y and returns a dissimilarity measure 
            (l2, l1, - cosine ...)'''
        self.dist = distance
        return
    
    def _distance_warping(self, X, i, j) :
        '''Compute the dtw between timeseries i and timeseries j and store it in self.dtw
        the data is supposed to be a numpy array, of shape (n_timeseries, length of the timeseries)'''
        loc_dtw = np.zeros((X.shape[1] + 1, X.shape[1] + 1))
        loc_dtw.fill(np.infty)
        loc_dtw[0, 0] = 0
        for k in range(1, X.shape[1] + 1) :
            for l in range(1, X.shape[1] + 1) :
                c = self.dist(X[i, k - 1], X[j, l - 1])
                loc_dtw[k, l] = c + min(loc_dtw[k -1, l], loc_dtw[k, l-1], loc_dtw[k - 1, l - 1])
        return loc_dtw[-1, -1]


    def fit(self, X, verbose = False) :
        self.X = X
        self.dtw_ = np.zeros((self.X.shape[0], self.X.shape[0]))
        for i in range(self.X.shape[0]) :
            if verbose :
                if i%10 == 0 :
                    print(i/self.X.shape[0] * 100, '% done')
            for j in range(i, self.X.shape[0]) :
                val = self._distance_warping(self.X, i, j)
                self.dtw_[i, j] = val
                self.dtw_[j, i] = val
        return self.dtw_

if __name__ == "__main__":
    data = pd.read_csv('./data/Sales_Transactions_Dataset_Weekly.csv')
    data = data.drop(columns = ['Product_Code'])
    data = data.values[:, :51]
    dtw = DTW(distance = l1)
    dtw.fit(data)
    print(dtw.dtw_)