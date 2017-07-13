# iOS-CoreML-Yolo

This is the implementation of Object Detection using Tiny YOLO v1 model on Apple's CoreML Framework.

The app fetches image from your camera and perform object detection @ (average) 17.8 FPS.

## Requirements

- Xcode 9 
- iOS 11
- For training: Python 2.7 (Keras 1.2.2, TensorFlow 1.1, CoreMLTools 0.4.0)

## Usage

To use this app, open **iOS-CoreML-MNIST.xcodeproj** in Xcode 9 and run it on a device with iOS 11. (You can also use simulator)

## Model conversion

In this project, I am not training YOLO from scratch but converting the already existing model to CoreML model. If you want to create model on your own. 
- Create Anaconda environment. Open your terminal and type the following commands.
```
$ conda create -n coreml python=2.7
$ source activate coreml
(coreml) $ conda install pandas matplotlib jupyter notebook scipy scikit-learn opencv
(coreml) $ pip install tensorflow==1.1
(coreml) $ pip install keras==1.2.2
(coreml) $ pip install h5py
(coreml) $ pip install coremltools
```
- Download the weights from the following [link](https://drive.google.com/file/d/0B1tW_VtY7onibmdQWE1zVERxcjQ/view?usp=sharing) and move it into `./nnet` directory.
- Enter the environment and run the following commands in terminal with `./nnet` as master directory.
```
(coreml) $ sudo python convert.py
```

I also included a jupyter notebook for better understanding the above code. You need to use it with root permissions for mainly converting the keras model to CoreML model. Initialise the jupyter notebook instance with the following command:

```
(coreml) $ jupyter notebook --allow-root
```

## Tutorial

If you are interested in creating the Tiny YOLO v1 model on your own, a **step-by-step tutorial** is available at - [**Link**](https://sriraghu.com/2017/07/12/computer-vision-in-ios-object-detection/) 

## Results

These are the results of the app when tested on iPhone 7. 

<img src="https://github.com/r4ghu/iOS-CoreML-MNIST/blob/master/Screenshots/IMG_0029.PNG" alt="Result 1" width="280"> <img src="https://github.com/r4ghu/iOS-CoreML-MNIST/blob/master/Screenshots/IMG_0030.PNG" alt="Result 1" width="280"> <img src="https://github.com/r4ghu/iOS-CoreML-MNIST/blob/master/Screenshots/IMG_20170710_171328.jpg" alt="Result 1" width="280"> <img src="https://github.com/r4ghu/iOS-CoreML-MNIST/blob/master/Screenshots/IMG_20170710_171336.jpg" alt="Result 1" width="280"> <img src="https://github.com/r4ghu/iOS-CoreML-MNIST/blob/master/Screenshots/IMG_20170710_171359.jpg" alt="Result 1" width="280"> <img src="https://github.com/r4ghu/iOS-CoreML-MNIST/blob/master/Screenshots/IMG_20170710_171433.jpg" alt="Result 1" width="280"> 

## Author

Sri Raghu Malireddi / [@r4ghu](https://sriraghu.com)
