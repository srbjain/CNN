# IMAGE FILTERING

![Python 3.6](https://img.shields.io/badge/Python-3.6-brightgreen.svg) ![Framework](https://img.shields.io/badge/Framework-Keras/TensorFlow-orange.svg)

# PROJECT OVERVIEW

• Created a model to filter out images with people with open eyes and smiling faces from an photograph shoot/album.<br/>
• Used facial landmark to detect the landmarks of eyes and mouth and applied some equation to determine the state of facial regions.

# PROJECT DETAILS

The dlib library uses a pre-trained face detector which is based on a modification to the Histogram of Oriented Gradients + Linear SVM method for object detection.

The facial landmarks produced by dlib follow an indexable list, as I describe in this tutorial:

![The full set of facial landmarks that can be detected via dlib](resources/facial_landmarks.jpg)


Examining the image, we can see that facial regions can be accessed via simple Python indexing (assuming zero-indexing with Python since the image above is one-indexed):<br/>
• The mouth can be accessed through points [48, 68].<br/>
• The right eyebrow through points [17, 22].<br/>
• The left eyebrow through points [22, 27].<br/>
• The right eye using [36, 42].<br/>
• The left eye with [42, 48].<br/>
• The nose using [27, 35].<br/>
• And the jaw via [0, 17].<br/>

# EYE

Based on the work by Soukupová and Čech in their 2016 paper, Real-Time Eye Blink Detection using Facial Landmarks, we can then derive an equation that reflects this relation called the eye aspect ratio (EAR):

![DL](resources/eye_landmark.png)

![DL](resources/eye_detection_equation.png)

# SMILE

Similarly we have applied an equation to predict smile <br/>
NUM = ||P51-P59||+||P52-P58||+||P53-P57|| <br/>
DEN = ||P62-P68||+||P63-P67||+||P64-P66|| <br/>
MAR = NUM/DEN

# THRESHOLD

We have selected threshold for defining whether person is smiling or not on the basis of MAR. Are there eyes open or not on the basis of EAR. In this project we have chosen following values for threshold:<br/>
EAR_Threshold = 0.2<br/>
MAR_Threshold = 0.25

# RESULT

Our main focus was to filter out desired images from a set of marriage album collection. Our equation and threshold did pretty good on detected face. Detector model however was not able to detect distant and blurr images. So keeping that in mind below is one of the images of marriage

![DL](dataset/examples/example_1.jpg)

So, Lets visualise what our model is doing

![DL](dataset/analysed_dataset/example_1.jpg)




