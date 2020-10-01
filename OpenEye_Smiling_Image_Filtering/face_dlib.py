from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
from scipy.spatial import distance as dist
import shutil, os
from imutils import paths

def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	# compute the euclidean distance between the horizontal eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])
	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)
	# return the eye aspect ratio
	return ear

def mouth_aspect_ratio(mouth):
    # compute the euclidean distances between the three sets of near-vertical mouth landmarks (x, y)-coordinates
    A = dist.euclidean(mouth[14], mouth[18])
    B = dist.euclidean(mouth[13], mouth[19])
    C = dist.euclidean(mouth[15], mouth[17])
    # compute the euclidean distances between the two sets of far-vertical mouth landmarks (x, y)-coordinates
    D = dist.euclidean(mouth[2], mouth[10])
    E = dist.euclidean(mouth[3], mouth[9])
    F = dist.euclidean(mouth[4], mouth[8])
	# compute the mouth aspect ratio
    mar = (A + B + C) / (D + E + F)
	# return the mouth aspect ratio
    return mar

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", default="shape_predictor_68_face_landmarks.dat",
                help="path to facial landmark predictor")
ap.add_argument("-d", "--dataset", default="dataset/examples",
                help="path to image dataset")
ap.add_argument('-w', '--weights', default='mmod_human_face_detector.dat',
                help='path to weights file')
args = vars(ap.parse_args())


print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))

# Define two constants, one for the eye aspect ratio to indicate closed eye below the threshold
# and then a second constant for the smile aspect ratio to indicate smile above the threshold
EYE_AR_THRESH = 0.2
SMILE_AR_THRESH = 0.25

# initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# grab the indexes of the facial landmarks for the mouth
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]



for imagePath in imagePaths:
    
    # initialize the counters for total no of closed pair of eyes, smiling faces and detected faces respectively
    CLOSED = 0
    SMILE = 0
    TOTAL = 0
    
    # load the input image, resize it, and convert it to grayscale
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # detect faces in the grayscale image
    rects = detector(gray, 1)
    
    # loop over the face detections
    for rect in rects:
        
    	# determine the facial landmarks for the face region, then
    	# convert the facial landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
    	# extract the left and right eye coordinates, then use the coordinates to compute 
        # the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        
        # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0 
        
        # compute the convex hull for the left and right eye, then visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(image, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(image, [rightEyeHull], -1, (0, 255, 0), 1)
        
        # similarly extract mouth coordinates, then use the coordinates to compute the mouth aspect ratio
        mouth = shape[mStart:mEnd]
        mar = mouth_aspect_ratio(mouth)
    	
        # compute the convex hull for the mouth, then visualize it
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(image, [mouthHull], -1, (255, 0, 0), 1)
        
    	# check to see if the eye aspect ratio is below the given threshold, and if so, increment the closed eyes counter
        if ear < EYE_AR_THRESH:
            CLOSED += 1
        
        # check to see if the mouth aspect ratio is above the given threshold, and if so, increment the smiling counter
        if mar >= SMILE_AR_THRESH:
            SMILE += 1
        
        # increase the number for faces counter for each detected face
        TOTAL += 1
        
        # calculate the total percentage of closed pair of eyes and smiling faces
        closedEye_percentage = CLOSED/TOTAL*100
        smile_percentage = SMILE/TOTAL*100
        
        # draw all eyes and mouth on the image along with the computed eye aspect ratio and mouth aspect ratio
        cv2.putText(image, "EAR: {:.2f}".format(ear), (shape[rStart][0], shape[rStart][1]-40),
        	cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
        cv2.putText(image, "MAR: {:.2f}".format(mar), (shape[rStart][0], shape[rStart][1]-30),
        	cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
        
    # show both the percentage in top left corner of image
    if TOTAL>0:      
        cv2.putText(image, "Closed Eye: {}%".format(closedEye_percentage), (10, 20),
        	cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(image, "Smiling Face: {}%".format(smile_percentage), (10, 40),
        	cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # copy desired image based on the requirement. We will select all image without any face detection
    # or atleast one smiling face with less than 25% closed eye 
    if TOTAL==0 or (closedEye_percentage<=35 and smile_percentage!=0):
        shutil.copy(imagePath, 'dataset/desired_dataset')
    
    # save analysed images with visualisation in another folder
    print("[INFO] saving images...")
    filename = 'dataset/analysed_dataset/'+imagePath.split(os.path.sep)[-1]
    cv2.imwrite(filename, image)

