import numpy as np 

class KMeans() :
    def __init__(self, k, max_iter = 1000) :
        self.k = k
        # max_iter is the maximum number of iterations the algo does if it has not converged yet
        self.max_iter = max_iter
        return

    def kmeans_forgy(self, X):
        '''Forgy initialization consists in randomly chosing centroids at initialization'''
        nsamples, _ = X.shape[0], X.shape[1]
        self.clusters = np.zeros(nsamples)
        centroids = np.random.choice(range(nsamples), replace = True, size = self.k)
        self.centroids = X[centroids, :]
        same = False
        j = 0
        while (not same) and (j < self.max_iter):
            j += 1
            for i in range(nsamples) :
                self.clusters[i] = self._get_closest_centroid(X[i, :])
            same = self._update_centroids(X)
        return
    
    def _get_closest_centroid(self, sample) :
        mini, ind = np.inf, 0
        for i in range(self.k) :
            r = np.linalg.norm(sample - self.centroids[i])
            if  r < mini :
                mini, ind =  r, i
        return ind

    def _update_centroids(self, X) :
        prev_centroids = self.centroids.copy()
        same = False
        for l in range(self.k) :
            n_l = len(self.clusters[self.clusters == l])
            self.centroids[l] = np.sum(X[self.clusters == l, :], axis = 0)/n_l
        same = np.array_equal(prev_centroids, self.centroids)
        # same is a boolean which checks if the centroids have changed
        return same

    def fit(self, X) :
        self.kmeans_forgy(X)
        return
    
        
    
