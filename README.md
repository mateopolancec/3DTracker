# Object Tracking

Capstone project of Cpp Udacity Nanodegree. Main goal is to build visual based tracker. 

Tracker consists of four parts:
1. First, for each received image object detection is performed by YoloV3 algorithm
2. Second, received lidar data is croped by given ROI's from object detector
3. Third, parallel execution of keypoint detection, description and matching for specified detector, matcher and few different descriptor types
4. TTC for each detected cluster in current frame (*Note: lidar data are extracted just for one object (needed better combination of lidar/camera data))

<img src="images/course_code_structure.png" width="779" height="414" />

Further work will be add Kalman Filter and track management. This is just begining of autonomous vehicles code base.

## Dependencies for Running Locally
* Note: Tested on ubuntu 18
* cmake >= 3.10.2
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* Git LFS
  * Weight files are handled using [LFS](https://git-lfs.github.com/)
* OpenCV >= 4.1
  * https://www.learnopencv.com/install-opencv-4-on-ubuntu-18-04/
* Yaml-cpp for ubuntu 18 
  * https://github.com/jbeder/yaml-cpp.git  
  * from source (cmake .., make, make install)
* gcc/g++ >= 7
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

Desired output of algorithm:
* Visualization of Cluster outputs with class_id and confidence factors (class_ids ../dat/yolo/coco.names (int class_id corresponds with line number, starts with zero, in file coco.names)) 
* TTC for every cluster calculated and visualized (Performance: because of bad lidar data visualization will be made just for one/few objects of interest)

Structure of algorithm:
* Tracker - main function, run program
* ReceiveData - load images and lidar data and through for loop run execution
* EgoVehicle - main holder data class related to ego vehicle which runs all main algorithms and store data
* ObjectDetection - base class for object detection algorithms
* YoloV3 - Object Detection class  (input = image, output = clusters)
* FeaturePointMatcher2D - Calculate keypoint detection, description and matching based on two subsequent images and return best keypoint matches 
* dataStructures - contain all classes and structs used for data storage

## Basic Build Instructions

1. Clone this repo.
2. ./get_yolov3_weights
3. Make a build directory in the top level project directory: `mkdir build && cd build`
4. Compile: `cmake .. && make`
5. Run it: `./tracker`

*Note - build process fail during cmake .. or make call just repeat the process again. Some problems with flag pthread set in CMakeLists.
