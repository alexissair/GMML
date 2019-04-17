import numpy as np 


'Implementation of Metric Multi-Dimensionnal scaling'


class MMDS():
    def __init__(self, n_components, n_iter) :
        self.niter = n_iter
        self.ncomponents = n_components
        return


    def _dissimilarity(self, Z):
        Z_sq = np.sum(Z**2, 1)
        D = Z_sq + Z_sq[:, None] - 2 * Z.dot(Z.T)
        return D

    def _dissimilarity_sqrt(self, Z):
        D = self._dissimilarity(Z)
        D = np.maximum(D, 0)
        return np.sqrt(D)
    
    def _get_smacof_B(self, Z, D):
        '''Return the matrix B in SMACOF iteration.'''
        B = -np.divide(D, self._dissimilarity_sqrt(Z)+1e-20) # Add 1e-20 for stability issues
        B = B - np.diag(np.sum(B,0))
        return B
    
    def fit(self, D, verbose = False) :
        # D is our dissimilarity matrix , e.g. the shortest path returned by the FW algorithm
        self.stress = []
        print('test')
        n = D.shape[0]
        self.Zs = []
        Z = np.random.normal(size=self.ncomponents*n).reshape([n,self.ncomponents])
        print('initial', Z)
        for i in range(self.niter):
            if verbose :
                print('Iter {} out of {} iterations'.format(i, self.niter))
            Z = self._get_smacof_B(Z,D).dot(Z)/n
            print(Z, 'ds boucle')
            self.Zs.append(Z)
            print(np.sum(np.sum((D-self._dissimilarity_sqrt(Z))**2)))
            self.stress.append(np.sum(np.sum((D-self._dissimilarity_sqrt(Z))**2)))
        self.stress[:] = [x/n**2 for x in self.stress]
        return
    

if __name__ == "__main__":
    mmds = MMDS(2, n_iter = 50)
    z = np.random.normal(size = (10, 2))
    D = np.random.rand(10, 10)
    z2 = mmds._get_smacof_B(z, D).dot(z)/10
    print(mmds._get_smacof_B(z2, D))
