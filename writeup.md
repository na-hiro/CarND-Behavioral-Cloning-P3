# **Behavioral Cloning**

## Writeup

---

** Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

The my first personal goal was to generate a generalized model using Track1 only learning data. Stable running of only Track 1 was achieved without difficulty. However, because the stable running of Track2 was very difficult, the target was changed. The final goal was changed as follows.

The goal is to use both Track1(Lake) and 2(Jungle) as learning data and realize high speed and stable running in both courses.

[//]: # (Image References)

[image1]: ./examples/nvidia.png "Nvidia Model"
[image2]: ./examples/mymodel.jpg "My Model"
[image3]: ./examples/cam_vali.png "cam variation"
[image4]: ./examples/recovery.png "recovery"
[image5]: ./examples/rev_bridge.png "bridge and rev"
[image6]: ./examples/add_trk2.png "fliped"
[image7]: ./examples/histgram.png "fliped"
[image8]: ./examples/fliped.png "fliped"
[image9]: ./examples/brightness.png "brightness"
[image11]: ./examples/rotate.png "rotate.png"
[image10]: ./examples/shadow.png "shadow.png"
[image12]: ./examples/crop_resize.png "input.png"
[image13]: ./examples/log_fig.png "los.png"



## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup.md summarizing the results
* video.mp4 (a video recording of my vehicle driving autonomously around the track1(Lake) for at least one full lap)
* video_advanced.mp4 (a video recording of my vehicle driving autonomously around the track2(jungle) for at least one full lap)

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. My Model Architecture

 I adopted the modified one based on the following model proposed by [Nvidia](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/). The detail of model is as shown in other section.

![alt text][image1]

The base of the adopted model is the model proposed by Nvidia. That model has been modified as follows.

###### *My Model Architecture*
![alt text][image2]

In the base model architecture, there was no dropout. By adding Dropout to the fully conection(fc) layer, over learning was suppressed and good results were obtained.
Especially, it was possible to obtain good results by setting the keep ratio in the higher fc layer to a large value and setting the keep rate of the lower fc layer to a smaller value. Furthermore, Adam optimizer is used.

The set value of Dropout tried many variations. As a result, the values shown in My model architecture were adopted.

#### 2. Creation of the Training Set

##### 2.1 Acquisition of training data Set

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. In addition, the image of the left and right camera, the image when running on a bridge, and the image acquired during the reverse run were adopted.

![alt text][image3]
![alt text][image4]
![alt text][image5]

In addition, I attempted generalization of the model by using learning data of Track1 only at the beginning, but it was very difficult. Therefore, I decided to add image data of Track2 as a new strategy. Especially in the addition of the learning image, the caution points are as follows.In the trial result, failed scenes were added as learning data. The images added by focusing are as follows.

![alt text][image6]

##### 2.2 Preprocessing of training data Set

This section describes preprocessing of learning data.

I showed that accuracy is improved by improving the uneven distribution of learning data in P2 labeling classification task. Therefore, a histogram of acquired data was acquired. It is understood that the acquired data has many straight runs.

In order to process the data, the average number of data in the histogram was calculated. The number of target data is obtained by multiplying the calculated average number of data by a coefficient. When the number of data is larger than the number of data obtained, data was randomly thinned out to suppress uneven distribution of learning data.

The histograms before and after data unevenness suppression processing are shown below.

![alt text][image7]

##### 2.3 augmentation of training data Set

In order to increase variations of training data, the following augmenting process was performed.
* Flipped Image
* Change in lightness
* Add shadow
* Image rotation

The processing result image is as follows.

・fliped image
![alt text][image8]

・randam brightness
![alt text][image9]

・randam shadow
![alt text][image10]

・randam rodation
![alt text][image11]

##### 2.4 Final processing of input data

In this section, the input image to the model is explained.

The input to the model is the data that was removed from the part unnecessary for learning and resized.

Resizing was done to reduce the processing cost and the lane was set to be approximately 45 degrees at the time of straight ahead.

The images before and after processing are shown below.

![alt text][image12]

I finally randomly shuffled the data set and put 20% of the data into a validation set.

#### 3. Trainning and Result

In this section, I will describe the learning results and simulation results.

Loss values during learning are as shown. I adopted 20 epochs.

![alt text][image13]

The stability at the convergence of learning was greatly influenced by the dropout value. As a result of trying several cases, this set value was adopted.

It was shown that stable autonomous running is possible for both Track1 and Track2 by applying the model generated by the above learning process.

#### Conclusion

I estimated the steering value from the image by using CNN and realized autonomous running on the simulator. What I learned through this assignment is as follows.

In the acquisition method of learning data, we added a failed scene as a learning data after applying the model, thereby making it possible to generate a model that enables stable traveling. However, I guessed that the stability of straight running in Track 1 has been compromised because I focused too much on the sharp curve. Furthermore, it is possible to generate a more generalized model by adding more shear, shift processing and the like.

Stability in a sharp curve is thought to stabilize by considering speed. This task, a simple function to change the target speed was added to the drive.py by the steering value. In order to achieve higher accuracy, I think that the method adopted at the time of learning is interesting.

I speculate that more generalized model generation requires more data and more information.

#### End of file
