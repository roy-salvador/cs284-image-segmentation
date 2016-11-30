# cs284-image-segmentation

A python application which performs image segmentation of the [Weizmann Horse Dataset](http://www.msri.org/people/members/eranb/) using Conditional Random Fields (CRF). A Mini Project requirement for CS 284 course at University of the Philippines Diliman AY 2016-2017 under Sir Prospero Naval.


## Requirements
* [pystruct](https://pystruct.github.io/)
* [OpenCV](http://docs.opencv.org/3.1.0/d5/de5/tutorial_py_setup_in_windows.html#gsc.tab=0) Computer Vision Library
* [Scikit Image](http://scikit-image.org/)
* [Scikit Learn](http://scikit-learn.org/)
* Python 2.7.12


## Instructions
1. Clone and download the repository.
2. Train the CRF model. 
  
  ```  
  python -W ignore imagesegmentation.py
  ```

3. Update the horseModelLogger variable with the generated model file. Run the demo application to visualize segmentation.   
  
  ```  
  python -W ignore imagesegmentationDemo.py WeizmannSingleScale/horses/training/images/image-xxx.png
  
  python -W ignore imagesegmentationDemo.py WeizmannSingleScale/horses/test/images/image-xxx.png
  ```
  
4. Measure the pixelwise performance of the model.
 ```  
  python -W ignore imagesegmentationPerf.py
 ```
