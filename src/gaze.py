#!/usr/bin/env python
import roslib; roslib.load_manifest('gaze-tracking-cmu')
import rospy
from sensor_msgs.msg import PointCloud2, PointField, Image, RegionOfInterest
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
import numpy as np
from point_cloud import read_points, create_cloud
from scipy.linalg import eig, norm
import cv
from cv_bridge import CvBridge, CvBridgeError
from time import time

currIm = None
currRoi = None

bridge = CvBridge()

def callback(topic, *args):
    cv.NamedWindow("test", cv.CV_NORMAL)
    cv.ResizeWindow("test", 640, 480)
    global currIm
    global currRoi
    if type(topic) == Image:
        currIm = convert_depth_image(topic)
    if type(topic) == RegionOfInterest:
        currRoi = topic
        
    if currIm and currRoi:
        x1, y1 = currRoi.x_offset, currRoi.y_offset
        x2, y2 = x1 + currRoi.width, y1 + currRoi.height
        u,v = np.mgrid[x1:x2,y1:y2]
        d = np.asarray(currIm, dtype=np.float32)[y1:y2,x1:x2]
        xyz = makeCloud(u, v, d)
        
        n = len(xyz)
        mu = np.sum(xyz, axis=0)/n
        xyz -= mu
        cov = np.dot(xyz.T, xyz)/n
        e, v = eig(cov)

        print v[2]
        
def makeCloud(u, v, d):       
    # Build a 3xN matrix of the d,u,v data  
    C = np.vstack((u.flatten(), v.flatten(), d.flatten(), 0*u.flatten()+1))  

    # Project the duv matrix into xyz using xyz_matrix()  
    X,Y,Z,W = np.dot(xyz_matrix(),C)  
    X,Y,Z = X/W, Y/W, Z/W  
    xyz = np.vstack((X,Y,Z)).transpose()  
    xyz = xyz[Z<0,:]
    return xyz          
        
def xyz_matrix():  
    fx = 594.21  
    fy = 591.04  
    a = -0.0030711  
    b = 3.3309495  
    cx = 339.5  
    cy = 242.7  
    mat = np.array([[1/fx, 0, 0, -cx/fx],                  
                    [0, -1/fy, 0, cy/fy],                  
                    [0,   0, 0,    -1],                  
                    [0,   0, a,     b]])  
    return mat         

    
def makeMarker(pos, vec, idNum=1, color=(1,0,0)):
    m = Marker()
    m.id = idNum
    m.ns = "test"
    m.header.frame_id = "/openni_rgb_optical_frame"
    m.type = Marker.ARROW
    m.action = Marker.ADD
    
    markerPt1 = Point()
    markerPt2 = Point()
    markerPt1.x = pos[0];
    markerPt1.y = pos[1];
    markerPt1.z = pos[2];
    markerPt2.x = markerPt1.x + vec[0];
    markerPt2.y = markerPt1.y + vec[1];
    markerPt2.z = markerPt1.z + vec[2];
    
    m.points = [markerPt1, markerPt2]
    
    m.scale.x = .1;
    m.scale.y = .1;
    m.scale.z = .1;
    m.color.a = 1;
    m.color.r = color[0];
    m.color.g = color[1];
    m.color.b = color[2];
    #m.lifetime = rospy.Duration()
    return m
    
def convert_depth_image(ros_image):
    try:
        global bridge
        depth_image = bridge.imgmsg_to_cv(ros_image, "32FC1") #32fc1
        return depth_image

    except CvBridgeError, e:
        print e

def listener():
    rospy.init_node('gaze-tracking-cmu', anonymous=True)
    
    pubFaceCloud = rospy.Publisher('gaze_cloud', PointCloud2)
    pubFaceNormals = rospy.Publisher('gaze_markers', Marker)
    
    rospy.Subscriber("/camera/depth/image", Image, callback, 
                     queue_size=1, callback_args=(pubFaceCloud,pubFaceNormals)) 
    rospy.Subscriber("/roi", RegionOfInterest, callback, 
                     queue_size=1, callback_args=(pubFaceCloud,pubFaceNormals)) 
                        
    rospy.spin()

if __name__ == '__main__':
    listener()
