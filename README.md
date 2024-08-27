# Video Object Detection Using RKNN and OpenCV

This project demonstrates how to perform object detection on a video file using an RKNN (Rockchip Neural Network) model and OpenCV. The RKNN model is loaded, and inference is performed on each frame of the video. The detected objects are drawn on the video frames and displayed in real-time.

## Features

- **RKNN Model Loading**: Load a pre-trained RKNN model for object detection.
- **Video Input**: Read video files for processing.
- **Inference**: Perform inference using the RKNN model on each video frame.
- **Visualization**: Draw bounding boxes and labels around detected objects on the video frames.
- **Performance Measurement**: Measure and print the inference time for each frame.

## Requirements

- **C++ Compiler**: You need a C++ compiler to compile the code.
- **OpenCV**: Make sure OpenCV is installed on your system.
- **RKNN Toolkit**: Ensure that the RKNN Toolkit is properly installed and configured.
- **Additional Libraries**: The project might depend on other libraries like `dlfcn`, `sys/time.h`, etc., which should be available on your system.

## Installation

1. **Clone the repository**:

    ```bash
    git clone https://github.com/hydrogenbond001/rknn-yolov5.git
    ```

2. **Build the project**:

    ```bash
    cd rknn-yolov5
    ./build-linux_RK3588.sh
    ```

## Usage

```bash
./rknn_yolov5_demo <rknn_model.rknn> <video_file.mp4>
```
## Example
```bash
cd install/rknn_yolov5_demo_Linux
./rknn_yolov5_demo ./model/RK3588yolov5s-640-640.rknn ../../video.mp4
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
