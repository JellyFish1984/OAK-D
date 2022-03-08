import numpy as np
from numpy.core import uint16
import cv2
from cv2 import aruco
import depthai as dai

import math
from imutils.video import VideoStream
import argparse
import time
import cv2
from cv2 import aruco
import sys
import numpy as np


# Calculates Rotation Matrix given euler angles.
def eulerAnglesToRotationMatrix(theta):
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]])
    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]])
    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R


# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])










def main():


    pipeline = dai.Pipeline() #pipeline object we only need one.
    img_counter=0
    # here we define source and output connection
    camRGB = pipeline.create(dai.node.ColorCamera)  # this is the node for the RGB camera (the one in the midle)
    xoutVideo = pipeline.create(dai.node.XLinkOut)

    #now, use your connection to stream stuff.
    xoutVideo.setStreamName("video")

    #properties for the camera
    camRGB.setBoardSocket(dai.CameraBoardSocket.RGB)
    camRGB.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRGB.setVideoSize(1920,1080)
    xoutVideo.input.setBlocking(False)
    xoutVideo.input.setQueueSize(1)

    #linking the cam node with the pipeline
    camRGB.video.link(xoutVideo.input)

    
    # Get coeffecients and camera matrix from yaml calibration file
    cv_file = cv2.FileStorage("calibration_chessboard.yaml", cv2.FileStorage_READ)
    camera_matrix = cv_file.getNode('K').mat()
    distortion_coeffs = cv_file.getNode('D').mat()
    cv_file.release()
        



    # define the type of aruco marker to detect
    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    arucoParams = cv2.aruco.DetectorParameters_create()



    #load all of the above to the cam
    with dai.Device(pipeline) as device:
        
        #establish queue
        video = device.getOutputQueue(name="video", maxSize=1, blocking=False)
        # Main loop
        while True:

            videoIn = video.get()            
            frame = videoIn.getCvFrame()
            # detect ArUco markers in the input frame
            (corners, ids, rejected) = cv2.aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)
            # draw borders

            
            if len(corners) > 0:
                aruco.drawDetectedMarkers(frame, corners, ids)

                # Get the rotation and translation vectors
                rvecs, tvecs, obj_points = cv2.aruco.estimatePoseSingleMarkers(
                    corners,
                    7,
                    camera_matrix,
                    distortion_coeffs)

                # get eular angles
                rotmatrix = np.zeros(shape=(3,3))
                rotmatrix, _ = cv2.Rodrigues(rvecs)
                # if isRotationMatrix(rotmatrix):
                #      eular= rotationMatrixToEulerAngles(rotmatrix)
                #      print(tvecs,eular*180/np.pi)
                #correction_ratio = 50/78  # Will be different for each camera
                #print(tvecs*correction_ratio)
                aruco.drawAxis(frame, camera_matrix, distortion_coeffs, rvecs, tvecs, 3)
                # use 0.043 with the aruco of 7.3cm.
                print(tvecs)
               
                
           
                
           
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break


                





     



if __name__ == '__main__':
    main()





