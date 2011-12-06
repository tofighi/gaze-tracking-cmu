#!/usr/bin/env python
import roslib; roslib.load_manifest('gaze-tracking-cmu')
import rospy
from sensor_msgs.msg import PointCloud2, PointField, Image, RegionOfInterest
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
import numpy as np
from point_cloud import read_points, create_cloud, create_cloud_xyz32
from scipy.linalg import eig, norm
import cv
from cv_bridge import CvBridge, CvBridgeError
import time
import sys

class Gaze:
    def __init__(self, node_name):
        rospy.init_node(node_name)
        
        rospy.on_shutdown(self.cleanup)
    
        self.node_name = node_name
        self.input_rgb_image = "input_rgb_image"
        self.input_depth_image = "input_depth_image"
        
        self.pubFaceCloud = rospy.Publisher('gaze_cloud', PointCloud2)
        self.pubFaceNormals = rospy.Publisher('gaze_markers', Marker)
        
        """ Initialize a number of global variables """
        self.image = None
        self.image_size = None
        self.depth_image = None
        self.grey = None
        self.small_image = None
        
        self.show_text = True

        """ Create the display window """
        self.cv_window_name = self.node_name
        cv.NamedWindow(self.cv_window_name, cv.CV_NORMAL)
        cv.ResizeWindow(self.cv_window_name, 640, 480)
        
        """ Create the cv_bridge object """
        self.bridge = CvBridge()     
        
        """ Subscribe to the raw camera image topic and set the image processing callback """
        self.image_sub = rospy.Subscriber(self.input_rgb_image, Image, self.image_callback)
        self.depth_sub = rospy.Subscriber(self.input_depth_image, Image, self.depth_callback)
        
        #rospy.Subscriber("/camera/depth/image", Image, callback, 
        #                 queue_size=1, callback_args=(pubFaceCloud,pubFaceNormals)) 
        #rospy.Subscriber("/roi", RegionOfInterest, callback, 
        #                 queue_size=1, callback_args=(pubFaceCloud,pubFaceNormals)) 
    
        """ Set up the face detection parameters """
        self.cascade_frontal_alt = rospy.get_param("~cascade_frontal_alt", "")
        self.cascade_frontal_alt2 = rospy.get_param("~cascade_frontal_alt2", "")
        self.cascade_profile = rospy.get_param("~cascade_profile", "")
        
        self.cascade_frontal_alt = cv.Load(self.cascade_frontal_alt)
        self.cascade_frontal_alt2 = cv.Load(self.cascade_frontal_alt2)
        self.cascade_profile = cv.Load(self.cascade_profile)
        
        self.camera_frame_id = "kinect_depth_optical_frame"
        
        # viola jones parameters
        self.min_size = (20, 20)
        self.image_scale = 2
        self.haar_scale = 1.5
        self.min_neighbors = 1
        self.haar_flags = cv.CV_HAAR_DO_CANNY_PRUNING
        
        self.cps = 0 # Cycles per second = number of processing loops per second.
        self.cps_values = list()
        self.cps_n_values = 20
                        
        """ Wait until the image topics are ready before starting """
        rospy.wait_for_message(self.input_rgb_image, Image)
        rospy.wait_for_message(self.input_depth_image, Image)
            
        rospy.loginfo("Starting " + self.node_name)
            

    def detect_faces(self, cv_image):
        if self.grey is None:
            """ Allocate temporary images """      
            self.grey = cv.CreateImage(self.image_size, 8, 1)
            
        if self.small_image is None:
            self.small_image = cv.CreateImage((cv.Round(self.image_size[0] / self.image_scale),
                       cv.Round(self.image_size[1] / self.image_scale)), 8, 1)

        """ Convert color input image to grayscale """
        cv.CvtColor(cv_image, self.grey, cv.CV_BGR2GRAY)
        
        """ Equalize the histogram to reduce lighting effects. """
        cv.EqualizeHist(self.grey, self.grey)

        """ Scale input image for faster processing """
        cv.Resize(self.grey, self.small_image, cv.CV_INTER_LINEAR)

        """ First check one of the frontal templates """
        frontal_faces = cv.HaarDetectObjects(self.small_image, self.cascade_frontal_alt, cv.CreateMemStorage(0),
                                             self.haar_scale, self.min_neighbors, self.haar_flags, self.min_size)
                                         
        """ Now check the profile template """
        profile_faces = cv.HaarDetectObjects(self.small_image, self.cascade_profile, cv.CreateMemStorage(0),
                                             self.haar_scale, self.min_neighbors, self.haar_flags, self.min_size)

        """ Lastly, check a different frontal profile """
        #faces = cv.HaarDetectObjects(self.small_image, self.cascade_frontal_alt2, cv.CreateMemStorage(0),
        #                                 self.haar_scale, self.min_neighbors, self.haar_flags, self.min_size)
        #profile_faces.extend(faces)
            
        '''if not frontal_faces and not profile_faces:
            if self.show_text:
                text_font = cv.InitFont(cv.CV_FONT_VECTOR0, 3, 2, 0, 3)
                cv.PutText(self.marker_image, "NO FACES!", (50, int(self.image_size[1] * 0.9)), text_font, cv.RGB(255, 255, 0))'''
             
        faces_boxes = []   
        for ((x, y, w, h), n) in frontal_faces + profile_faces:
            """ The input to cv.HaarDetectObjects was resized, so scale the 
                bounding box of each face and convert it to two CvPoints """
            pt1 = (int(x * self.image_scale), int(y * self.image_scale))
            pt2 = (int((x + w) * self.image_scale), int((y + h) * self.image_scale))
            #face_width = pt2[0] - pt1[0]
            #face_height = pt2[1] - pt1[1]
                
            face_box = (pt1[0], pt1[1], pt2[0], pt2[1])
            faces_boxes.append(face_box)
        return faces_boxes


    def process_faces(self, boxes):     
        if not self.depth_image: 
            print 'whoops! no depth image!'
            return
          
        skip = 1     
        xyz_all = []
        
        idNum = 1
        for (x1,y1,x2,y2) in boxes:
            #x1,x2,y1,y2 = 0,640,0,480
            xpad = 10
            ypad = 40
            ypad2 = -10

            u,v = np.mgrid[y1-ypad:y2:skip,x1-xpad:x2+xpad:skip]
            u,v = u.flatten(), v.flatten()
            d = np.asarray(self.depth_image[y1-ypad:y2+ypad2,x1-xpad:x2+xpad])
            d = d[::skip,::skip]
            d = d.flatten()
            
            u = u[np.isfinite(d)]
            v = v[np.isfinite(d)]
            d = d[np.isfinite(d)]
            
            ### only if from dat
            u = u[d!=1023]
            v = v[d!=1023]
            d = d[d!=1023]
            
            
            #u = u[d < alpha*median]
            #v = v[d < alpha*median] 
            #d = d[d < alpha*median]
            
            xyz = self.makeCloud_correct(u,v,d)
            
            '''median = sorted(xyz[:,2])[len(d)//2]
            alpha = 1.2
            xyz = xyz[xyz[:,2] < alpha*median,:]'''
            
            xyz_all.extend(xyz)
            
            
            n = len(xyz)
            mu = np.sum(xyz, axis=0)/n
            xyz_norm = xyz - mu
            cov = np.dot(xyz_norm.T, xyz_norm)/n
            e, v = eig(cov)
            
            #print v[2]
            
            if v[2][2] > 0: v[2] = -v[2]
            
            # publish marker here
            m = self.makeMarker(mu, v[2], idNum=idNum, color=(1,0,0))
            idNum += 1
            self.pubFaceNormals.publish(m)

        pc = PointCloud2()
        pc.header.frame_id = "/camera_rgb_optical_frame"
        pc.header.stamp = rospy.Time()
        pc = create_cloud_xyz32(pc.header, xyz_all)
        self.pubFaceCloud.publish(pc)
            
    def depth_callback(self, data):
        depth_image = self.convert_depth_image(data)
            
        if not self.depth_image:
            (cols, rows) = cv.GetSize(depth_image)
            self.depth_image = cv.CreateMat(rows, cols, cv.CV_16UC1) # cv.CV_32FC1
            
        cv.Copy(depth_image, self.depth_image)
        
        #self.depth_image = depth_image

    def image_callback(self, data):       
        start = time.time()
    
        """ Convert the raw image to OpenCV format using the convert_image() helper function """
        cv_image = self.convert_image(data)
          
        """ Create a few images we will use for display """
        if not self.image:
            self.image_size = cv.GetSize(cv_image)
            self.image = cv.CreateImage(self.image_size, 8, 3)
            self.display_image = cv.CreateImage(self.image_size, 8, 3)

        """ Copy the current frame to the global image in case we need it elsewhere"""
        cv.Copy(cv_image, self.image)
        cv.Copy(cv_image, self.display_image)
        
        """ Process the image to detect and track objects or features """
        faces = self.detect_faces(cv_image)
        #faces = ((0,0,640,480),)
        self.process_faces(faces)
            
        for (x,y,x2,y2) in faces:
            cv.Rectangle(self.display_image, (x, y), (x2, y2), cv.RGB(255, 0, 0), 2, 8, 0)
        
        """ Handle keyboard events """
        self.keystroke = cv.WaitKey(5)
            
        end = time.time()
        duration = end - start
        fps = int(1.0 / duration)
        self.cps_values.append(fps)
        if len(self.cps_values) > self.cps_n_values:
            self.cps_values.pop(0)
        self.cps = int(sum(self.cps_values) / len(self.cps_values))
        
        if self.show_text:
            text_font = cv.InitFont(cv.CV_FONT_VECTOR0, 1, 1, 0, 2, 8)
            """ Print cycles per second (CPS) and resolution (RES) at top of the image """
            cv.PutText(self.display_image, "CPS: " + str(self.cps), (10, int(self.image_size[1] * 0.1)), text_font, cv.RGB(255, 255, 0))
            cv.PutText(self.display_image, "RES: " + str(self.image_size[0]) + "X" + str(self.image_size[1]), (int(self.image_size[0] * 0.6), int(self.image_size[1] * 0.1)), text_font, cv.RGB(255, 255, 0))

        # Now display the image.
        cv.ShowImage(self.node_name, self.display_image)
        
        """ Process any keyboard commands """
        if 32 <= self.keystroke and self.keystroke < 128:
            cc = chr(self.keystroke).lower()
            if cc == 't':
                self.show_text = not self.show_text
            elif cc == 'q':
                """ user has press the q key, so exit """
                rospy.signal_shutdown("User hit q key to quit.")      

    def convert_image(self, ros_image):
        try:
            cv_image = self.bridge.imgmsg_to_cv(ros_image, "bgr8")
            return cv_image
        except CvBridgeError, e:
          print e
          
    def convert_depth_image(self, ros_image):
        try:
            
            ros_image.step = 1280 # weird bug here -- only needed for dat?
            depth_image = self.bridge.imgmsg_to_cv(ros_image, "16UC1") #"32FC1"
            
            return depth_image
    
        except CvBridgeError, e:
            print e    
 
    def makeCloud_correct(self, u, v, d): # for depth values, from pi_vision
        # only if from dat
        d = 1000.0 * .1236 * np.tan((d / 2842.5) + 1.1863) - 0.0370
        
        const = .001/575.8157348632812
        
        xp = d*const*(v-314.5)
        yp = d*const*(u-235.5)
        zp = d*0.001
        
        return np.vstack((xp,yp,zp)).T
            
    def makeCloud(self, u, v, d): # for depth values, from pi_vision
        xyz = []
        C = np.vstack((u.flatten(), v.flatten(), d.flatten()))  

        for i in range(C.shape[1]):
            
            y, x, z = C[0,i], C[1,i], C[2,i]
            
            zp = z #1.0 / (z * -0.0030711016 + 3.3309495161)
            #print z, zp
            
            #550,.1 or 0,600 (saw 0,603.3)
            xp = (zp-.0) * 1.094 * (x - 640 / 2.0) / float(603.3) 
            yp = (zp-.0) * 1.094 * (y - 480 / 2.0) / float(603.3) #550
            
            if not np.isnan(xp) and not np.isnan(yp) and not np.isnan(zp):
                xyz.append((xp,yp,zp))
        return xyz
                    

    
    def makeMarker(self, pos, vec, idNum=1, color=(1,0,0)):
        m = Marker()
        m.id = idNum
        m.ns = "test"
        m.header.frame_id = "/camera_rgb_optical_frame"
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
        
    def cleanup(self):
        print "Shutting down vision node."
        cv.DestroyAllWindows()  
    
def main(args):
    """ Display a help message if appropriate """
    '''help_message =  "Hot keys: \n" \
          "\tq - quit the program\n" \
          "\tc - delete current features\n" \
          "\tt - toggle text captions on/off\n" \
          "\tf - toggle display of features on/off\n" \
          "\tn - toggle \"night\" mode on/off\n" \
          "\ta - toggle auto face tracking on/off\n"

    print help_message'''
    
    """ Fire up the Face Tracker node """
    g = Gaze("gaze")

    try:
      rospy.spin()
    except KeyboardInterrupt:
      print "Shutting down face tracker node."
      cv.DestroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
