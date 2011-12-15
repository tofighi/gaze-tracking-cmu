import numpy as np
import pickle
import cv
from scipy.cluster.vq import vq, kmeans2, whiten, kmeans

import random

from pylab import plot, show, scatter

import time

'''
seeds = np.array([[ 0.26364343,  0.5783477 ],
                  [ 0.21797107,  0.43560179],
                  [ 0.27246221,  0.88984877],
                  [ 0.2655625 ,  0.25490294],
                  [ 0.25171378,  1.15914697]])
'''

def dxySegment(d_full, nClusters=5, graphics=False, seeds=None, skip=4, verbose=False):
    if seeds is not None:
        nClusters = seeds.shape[0]

    if verbose:
        st = time.time()
        stst = time.time()
    
    d = d_full.copy()
    
    d[:50,:] = 0
    d = d[:-200,:] # snip off the last 100 rows
    width, height = d.shape    
    
    X,Y = np.meshgrid(np.arange(width),np.arange(height))
    Y = Y.T.flatten()
    X = X.T.flatten()
    d = d.flatten()
    
    X_unscaled = X.copy() # in image coordinates
    Y_unscaled = Y.copy()
    d_unscaled = d.copy()

    d_scaled = d.copy() 
    Y_scaled = Y.copy()
    
    # scaled, though maybe using "whiten" would produce better properties here 
    maxD = np.max(d_scaled)
    d_scaled = 0.5*d_scaled/maxD      
    Y_scaled = 1.5*Y_scaled/height 
    if seeds is not None: 
        seeds = seeds * [0.5/maxD, 1.5/height]
    
    d_filtered = d_scaled.copy()
    Y_filtered = Y_scaled.copy()
    d_filtered = d_filtered[::skip]
    Y_filtered = Y_filtered[::skip]
    Y_filtered = Y_filtered[d_filtered!=0] 
    d_filtered = d_filtered[d_filtered!=0]
    
    # should use a proper distance threshold here
    Y_filtered = Y_filtered[d_filtered<0.4] 
    d_filtered = d_filtered[d_filtered<0.4]
    
    if verbose: print 'preparing data', time.time() - st; st = time.time()
    
    dataset = np.vstack((d_filtered,Y_filtered))
    
    if seeds is not None:
        centroids, idx = kmeans2(dataset.T,seeds)
    else:
        centroids, idx = kmeans2(dataset.T,nClusters)
    
    if verbose: print 'clustered', time.time() - st; st = time.time()
    
    full_data = np.vstack((d_scaled,Y_scaled)).T
    codes, dist = vq(full_data,centroids)
    
    if verbose: print 'projected back to image', time.time() - st; st = time.time()
    
    if graphics:
        colors = [[random.randint(0,255)/255.0,random.randint(0,255)/255.0,random.randint(0,255)/255.0] for i in range(nClusters)]
        
        im = np.zeros((width*height,3))
        for k in range(nClusters):
            im[codes==k] = colors[k]
    
        if verbose: print 'coloring', time.time() - st; st = time.time()
    
        im[d_unscaled == 0] = (0,0,0)
        im.shape = (width,height,3)  
        cv_im = cv.fromarray(im)
    
    if verbose: print 'gathering again', time.time() - st; st = time.time()
    
    rects = []
    face_height = 50
    codes_im = codes.copy()
    codes_im[d_scaled==0] = -1
    for k in range(nClusters):
        X_cluster = X_unscaled[codes_im==k]
        Y_cluster = Y_unscaled[codes_im==k]
        try:
            xmin = np.min(X_cluster)
            Y_face = Y_cluster[X_cluster < xmin+face_height]
            ymin, ymax = np.min(Y_face), np.max(Y_face)
            rects.append([ymin, xmin, ymax, xmin+face_height])
            if graphics:
                cv.Rectangle(cv_im, (ymin, xmin), (ymax, xmin+60), (0, 255, 255), 2)
        except: pass
        
    if verbose: print 'finding bounding boxes', time.time() - st; st = time.time()
    if verbose: print rects
    if verbose: print 'total time', time.time() - stst
    if graphics:
        cv.ShowImage("blah",cv_im)
        #cv.WaitKey(0)
        
        ### Will not work with current code!
        #colorList = [colors[j] for j in codes]
        #scatter(d,Y, c=colorList)
        #randColor = [random.randint(0,255)/255.0,random.randint(0,255)/255.0,random.randint(0,255)/255.0]
        #scatter(centroids[:,0],centroids[:,1], marker='o', s = 500, linewidths=2, c=randColor)
        #scatter(centroids[:,0],centroids[:,1], marker='x', s = 500, linewidths=2, c=randColor)
        #show()
    
    return rects, centroids * [maxD/0.5, height/1.5]
        
    
if __name__ == '__main__':
    f = open('/home/ben/Desktop/ros/gaze-tracking-cmu/src/It2_depth.dat','r')
    im1 = pickle.load(f)
    
    f = open('/home/ben/Desktop/ros/gaze-tracking-cmu/src/It_depth.dat','r')
    im2 = pickle.load(f)

    cv.NamedWindow("blah")
    
    #cv.ShowImage("blah",cv.fromarray(im1))
    #cv.WaitKey(0)
    
    seeds = np.array([[ 2213.42758621,   191.77011494],
       [ 2726.21090387,   251.42037303],
       [ 2740.46336207,   379.4612069 ],
       [ 2698.85686465,   108.26679649],
       [ 2562.12815126,   496.15546218]])
    
    for i in range(100):
        if i % 2 == 0:
            im = im1
        else:
            im = im2
            
        if i == 0:
            r, seeds = dxySegment(im, graphics=False, seeds=None, skip=10, verbose=True, nClusters=5)
        else:
            dxySegment(im, graphics=False, seeds=seeds, skip=100, verbose=True, nClusters=5)
        cv.WaitKey(3)
