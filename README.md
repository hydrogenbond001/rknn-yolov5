Video Object Detection Using RKNN and OpenCV
This project demonstrates how to perform object detection on a video file using an RKNN (Rockchip Neural Network) model and OpenCV. The RKNN model is loaded, and inference is performed on each frame of the video. The detected objects are drawn on the video frames and displayed in real-time.

Features
RKNN Model Loading: Load a pre-trained RKNN model for object detection.
Video Input: Read video files for processing.
Inference: Perform inference using the RKNN model on each video frame.
Visualization: Draw bounding boxes and labels around detected objects on the video frames.
Performance Measurement: Measure and print the inference time for each frame.
Requirements
C++ Compiler: You need a C++ compiler to compile the code.
OpenCV: Make sure OpenCV is installed on your system.
RKNN Toolkit: Ensure that the RKNN Toolkit is properly installed and configured.
Additional Libraries: The project might depend on other libraries like dlfcn, sys/time.h, etc., which should be available on your system.
Installation
Clone the repository:
bash
复制代码
git clone https://github.com/hydrogenbond001/rknn-yolov5.git
build
cd rknn-yolov5
./build-linux_RK3588.sh

Usage
bash
复制代码
./rknn_yolov5_demo <rknn_model.rknn> <video_file.mp4>
Example
bash
复制代码
cd install/rknn_yolov5_demo_Linux
./rknn_yolov5_demo ./model/RK3588yolov5s-640-640.rknn ../../xxx.mp4


Notes
Ensure that the video file exists and the RKNN model is correctly formatted.
The application will display the video with detected objects in real-time. Press any key in the display window to stop the processing.
License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
This project uses OpenCV for video processing.
RKNN Toolkit is provided by Rockchip for model inference on their hardware.
