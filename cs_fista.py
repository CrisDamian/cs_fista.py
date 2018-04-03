# -*- coding: utf-8 -*-

import numpy as np
from skimage.restoration import denoise_tv_chambolle

    
class Fista:
    """
    Class for making Compressive Sensing reconstrunctions using the  
    Fast Shrinkage Thresholding Algorithm (FISTA).   
    
    """
    
    def __init__(self, 
                 forward,
                 proximal=None,
                 backward=None, 
                 weight = 1,
                 tol=2.0e-4,
                 maxiter=200):
        """
        Initalize parameters for the reconstruction algorithm:
            
        Parameters
        ----------
        forward : 2d ndarray, callable (ndarray -> ndarray)
            The measurement operator. A function that takes a latent image 
            and returns its noseless measurements.
            
        backward : callable (ndarray -> ndarray)
            The adjoint of the measurement operator. Necesary if forward is 
            given a callable.
            
        prox_op: string, callable (ndarray y, float lmb -> ndarray x)
            Proximal operator used for regularisation. The proximal operator
            solves::
                
                argmin x of sum((x-y)**2) + lmb*f(x)
            
            - If 'l1' or None (default) then L1-norm regularisation is used.
            
            - If 'tv' then Total Variation regularisation is used using 
              Chambolle's algorithm.
            
        tol : float, optional
            Difference between succesive iterates at witch to stop the algorithm.
            The algorithm stops when: 
            
                max(abs(X1-X)) < eps
                
        maxiter : int, optional
            Maximum number of iterations of the algorithm.
            
        References
        ----------
            Beck, A. & Teboulle, M. 
            A Fast Iterative Shrinkage-Thresholding Algorithm for 
            Linear Inverse Problems. 
            SIAM Journal on Imaging Sciences, 2009, 2, 183-202
                    
        """
        
        if forward is None:
            raise TypeError("Forward can not be `None`")
            
        elif np.ndim(forward) == 2 and all(np.isfinite(forward).flat):
            forward = np.asarray(forward)
            self.forward = lambda x: forward @ x
            self.backward = lambda x: forward.T @ x
            
        elif callable(forward):
            if not callable(backward):
                raise ValueError(
                    "if `forward` is a function then its adjoint must be given as `backward`.")
            self.forward = forward
            self.backward = backward
            
        else:
            raise ValueError(
                "`forward` must be a matrix with real numbers or callable.")   
            
        if proximal is None or proximal == 'l1':
            self.proximal = lambda x, w: x - np.clip(x,-w,w)
        
        elif proximal == 'tv':
            self.proximal = denoise_tv_chambolle
            
        elif callable(proximal):
            self.proximal = proximal
            
        else:
            raise ValueError(
                "`proximal` must be a callable or a keword string.")
            
        self.weight = weight
        self.tol = tol
        self.maxiter = maxiter
        
        
    def reconstruct(self, data, weight=None, guess=None):
        """
        Reconstructs the image from the given measurements.
        
        Parameters
        ----------
        data : ndarray
            The measurements from witch to reconstruct the image.
        
        weight : float, optional
            The regularisation weight. If none then 
        
        guess : numpy.ndarray, optional
            The initial guess of the reconstruction.
            
        Returns
        -------
            X : ndarray
            The reconstructed image.
        
        """
        
        phi = self.forward
        phit = self.backward
        D = np.asarray(data)
        
        if weight is None:
            weight = self.weight
        
        if guess is None: 
            X = phit(D)
        else:
            X = np.asarray(guess)
        
        laststep = 1    
        m = 0
        t = 1
        Y = X
        
        while laststep > self.tol and m < self.maxiter:
            m +=1
            
            # Gradient descent
            r = -phit(phi(Y) - D)
            curve = np.sum(r*phit(phi(r)))
            mu = np.sum(r**2)/curve if curve else 1
            V = Y + mu * r
            
            X1 = self.proximal(V, 2*mu*weight)
            
            # Nesterov acceleration 
            t1 = (1 + (1+ t**2)**.5)/2
            Y = X1 + (t-1)/t1 * (X1 - X)
            
            laststep = np.max(abs(X1-X))
            X = X1
            
        return X

    
    def project(self, X):
        """
        Generate noiseless measrements from an image X.
        """
        return self.froward(X)
    
    def backproject(self, D):
        """
        Generate a backprojetion of the image using the measurements D. 
        """
        return self.backward(D)
       
    
