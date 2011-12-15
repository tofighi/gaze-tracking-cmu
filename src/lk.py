import numpy as np
from scipy import ndimage
import scipy.linalg
import cv
inv = scipy.linalg.inv

import pickle

#It and It1 need to be single channel, floating, grayscale
def lk(It, It1, rect, graphics=False):

    '''f = open('/home/ben/Desktop/It_depth.dat','w')
    f2 = open('/home/ben/Desktop/It2_depth.dat','w')
    f3 = open('/home/ben/Desktop/rect.dat','w')
    
    pickle.dump(It, f)
    pickle.dump(It1, f2)
    pickle.dump(rect, f3)'''
    
    #raw_input(">>>>")

    u, v = 0, 0
    uvDelt = [1,1]
    
    It = np.asarray(It, dtype=np.float64)
    It1 = np.asarray(It1, dtype=np.float64)
    
    X,Y = np.meshgrid(np.arange(rect[0],rect[2]),np.arange(rect[1],rect[3]))
    Itwindow = ndimage.map_coordinates(It,[Y.flatten(),X.flatten()], order=1)
    Itwindow.shape = X.shape
    
    window_size = X.shape
    
    numIter = 1
    while numIter < 20: # or do max iterations , uvDelt[0]**2 + uvDelt[1]**2 > 0.001 or
        numIter += 1
    
        if graphics:
            disp_img = cv.CreateImage((640,480),64,1)
            cv.Copy(cv.fromarray(It1), disp_img)

        # translate Window along estimated u,v
        x1t = rect[0] + u
        x2t = rect[2] + u
        y1t = rect[1] + v
        y2t = rect[3] + v        
        
        while len(np.arange(x1t,x2t)) < window_size[1]:
            x2t += 1
        while len(np.arange(x1t,x2t)) > window_size[1]:
            x2t -= 1
        while len(np.arange(y1t,y2t)) < window_size[0]:
            y2t += 1
        while len(np.arange(y1t,y2t)) > window_size[0]:
            y2t -= 1
        
        X,Y = np.meshgrid(np.arange(x1t,x2t),np.arange(y1t,y2t))
        It1guess = ndimage.map_coordinates(It1,[Y.flatten(),X.flatten()], order=1)
        It1guess.shape = X.shape
        
        if graphics:
            cv.Rectangle(disp_img, (int(x1t), int(y1t)), (int(x2t), int(y2t)), 127, 2)
        
        # compute error
        err = Itwindow - It1guess 
        
        if graphics:
            cv.ShowImage("blah", disp_img)
            cv.ShowImage("blah2", cv.fromarray(It1guess))
            cv.ShowImage("blah3", cv.fromarray(err))
            
            cv.WaitKey(0)

        # compute (translated) gradient
        Ix, Iy = np.gradient(It1guess) 
        A = np.vstack((Iy.flatten(1),Ix.flatten(1))).T # Ix, Iy
        b = err.flatten(1)
        
        #A = [Ix(:) Iy(:)];
        #b = err(:);

        #uvDelt = inv(A'*A)*A'*b;

        try:
            uvDelt = np.dot(np.dot(inv(np.dot(A.T,A)),A.T),b)
        except:
            return 0, 0

        u += uvDelt[0]
        v += uvDelt[1]

    return u, v
    
if __name__ == '__main__':
    rect = (369, 135, 408, 190)
    
    f = open('/home/ben/Desktop/ros/gaze-tracking-cmu/src/It_depth.dat','r')
    f2 = open('/home/ben/Desktop/ros/gaze-tracking-cmu/src/It2_depth.dat','r')
    f3 = open('/home/ben/Desktop/ros/gaze-tracking-cmu/src/rect.dat','r')

    im1 = pickle.load(f)
    im2 = pickle.load(f2)
    rect = pickle.load(f3)
    
    f.close()
    f2.close()
    f3.close()
    
    print rect
    
    cv.NamedWindow("blah")
    cv.NamedWindow("blah2")
    cv.NamedWindow("blah3")
    
    
    '''cv.ShowImage("blah",cv.fromarray(im1))
    
    
    cv.ShowImage("blah2",cv.fromarray(im2))
    
    cv.WaitKey(0)'''
    

    print lk(im1,im2,rect, graphics=True)
     
    
