# Deep Learning Projects

![Python 3.6](https://img.shields.io/badge/Python-3.6-brightgreen.svg) ![Framework](https://img.shields.io/badge/Framework-Keras/TensorFlow-orange.svg)




The dlib library uses a pre-trained face detector which is based on a modification to the Histogram of Oriented Gradients + Linear SVM method for object detection.

The facial landmarks produced by dlib follow an indexable list, as I describe in this tutorial:

![DL](blink_detection_plot.jpg)

Examining the image, we can see that facial regions can be accessed via simple Python indexing (assuming zero-indexing with Python since the image above is one-indexed):

The mouth can be accessed through points [48, 68].
The right eyebrow through points [17, 22].
The left eyebrow through points [22, 27].
The right eye using [36, 42].
The left eye with [42, 48].
The nose using [27, 35].
And the jaw via [0, 17].

Based on the work by Soukupová and Čech in their 2016 paper, Real-Time Eye Blink Detection using Facial Landmarks, we can then derive an equation that reflects this relation called the eye aspect ratio (EAR):

![DL](blink_detection_equation.png)




![DL](dataset/examples/7L2A7078.jpg)


