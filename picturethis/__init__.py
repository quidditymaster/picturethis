__version__ = '0.0.0'

import numpy as np

class GaussianImager(object):

    def __init__(
        self,
        im_shape,
        adjacent_cov=0.6,
        max_iter=20,
        verbose=False,
    ):
        self.im_shape = im_shape
        self.adjacent_cov=adjacent_cov
        self.max_iter = max_iter
        self.verbose = verbose
    
    def _renormalize_projection(self):
        self.P /= np.sqrt(np.sum(self.P**2, axis=-1, keepdims=True))
    
    def fit(self, x):
        ndim_x = x.shape[1]
        
        #initialize a random projection matrix
        P = np.random.normal(
            size=(np.prod(self.im_shape), ndim_x)
        )
        self.P = P
        self._renormalize_projection()
        
        pixel_indexes = np.meshgrid(
            *[range(k) for k in self.im_shape],
            indexing="ij",
        )
        pixel_indexes = [pi.reshape((-1,)) for pi in pixel_indexes]
        
        n1 = np.ravel_multi_index(
            (np.maximum(0, pixel_indexes[0]-1), pixel_indexes[1]),
            dims=self.im_shape
        )
        n2 = np.ravel_multi_index(
            (np.minimum(self.im_shape[0]-1, pixel_indexes[0]+1), pixel_indexes[1]),
            dims=self.im_shape
        )
        n3 = np.ravel_multi_index(
            (pixel_indexes[0], np.maximum(0, pixel_indexes[1]-1)),
            dims=self.im_shape
        )
        n4 = np.ravel_multi_index(
            (pixel_indexes[0], np.minimum(self.im_shape[1]-1, pixel_indexes[1]+1)),
            dims=self.im_shape
        )
        
        for iter_idx in range(self.max_iter):
            update = np.zeros_like(P)
            for neighbor_ix in [n1, n2, n3, n4]:
                #grab the neighbors
                nvals = P[neighbor_ix]
                #evaluate the update 
                dprod = np.sum(nvals*P, axis=1, keepdims=True)
                if self.verbose:
                    print("dprod", np.mean(dprod), np.std(dprod))                
                update += nvals*(self.adjacent_cov - dprod)
                #update += nvals
            P += 0.25*update
            self._renormalize_projection()
    
    def transform(self, x):
        res = np.dot(x, self.P.transpose())
        return res.reshape([-1] + list(self.im_shape))
