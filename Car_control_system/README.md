# Data Acquisition Package

This package is designed for data acquisition tasks, including lidar data collection, audio data publishing, video frame publishing, and more. It is built using ROS (Robot Operating System) and provides various scripts and nodes for handling sensor data.

## Directory Structure


src/data_acquisition/
├── CMakeLists.txt
├── package.xml
├── launch/
│   ├── AC.launch
├── scripts/
│   ├── audio_publisher.py
│   ├── audio_sub.py
│   ├── lidar_reader.py
│   ├── video_play.py
│   ├── video_publisher.py
├── src/
│   ├── lidar_acquisition.cpp



## Nodes and Scripts

### 1. **Lidar Data Acquisition**
   - **File**: [`src/lidar_acquisition.cpp`](src/lidar_acquisition.cpp)
   - **Description**: This node subscribes to the `/scan` topic to collect lidar data and saves it to a ROS bag file (`lidar_data.bag`).
   - **Usage**:
     ```bash
     rosrun data_acquisition lidar_acquisition
     ```
   - **Key Features**:
     - Subscribes to `/scan` topic.
     - Saves data to a `.bag` file.
     - Handles graceful shutdown on `Ctrl+C`.



### 2. **Audio Publisher**
   - **File**: audio_publisher.py
   - **Description**: Publishes audio data from a microphone to the `/audio_data` topic in base64-encoded format.
   - **Usage**:
     ```bash
     rosrun data_acquisition audio_publisher.py
     ```
   - **Launch**: Use the script `audio_pub.sh` to run this node.



### 3. **Audio Subscriber**
   - **File**: audio_sub.py
   - **Description**: Subscribes to the `/audio_data` topic, decodes the base64-encoded audio data, and plays it using PyAudio.
   - **Usage**:
     ```bash
     rosrun data_acquisition audio_sub.py
     ```



### 4. **Video Frame Publisher**
   - **File**: video_publisher.py
   - **Description**: Captures video frames from a camera and publishes them to the `/output_video_frame` topic.
   - **Usage**:
     ```bash
     rosrun data_acquisition video_publisher.py
     ```
   - **Launch**: This node is included in the `AC.launch` file.



### 5. **Video Frame Subscriber**
   - **File**: video_play.py
   - **Description**: Subscribes to the `/output_video_frame` topic and displays the video frames using OpenCV.
   - **Usage**:
     ```bash
     rosrun data_acquisition video_play.py
     ```



## Launch Files

### 1. **AC.launch**
   - **File**: AC.launch
   - **Description**: Launches multiple nodes, including:
     - video_publisher.py
     - video_play.py
     - audio_publisher.py
   - **Usage**:
     ```bash
     roslaunch data_acquisition AC.launch
     ```

## Dependencies

This package depends on the following ROS packages and system libraries:
- `geometry_msgs`
- `rosbag`
- `roscpp`
- `rospy`
- `std_msgs`
- `sensor_msgs`
- `cv_bridge`
- `turtlesim`
- `pyaudio`
- `OpenCV`

Make sure to install these dependencies before running the nodes.


## How to Build

1. Navigate to the root of your catkin workspace:
   ```bash
   cd ~/mycar_ws
   ```
2. Build the workspace:
   ```bash
   catkin_make
   ```
3. Source the workspace:
   ```bash
   source devel/setup.bash
   ```

## Shell Scripts Overview

### 1. **data_pub.sh**
   - **File**: [data_pub.sh](http://_vscodecontentref_/3)
   - **Description**: Launches the [AC.launch](http://_vscodecontentref_/4) file to start multiple data acquisition nodes.
   - **Usage**:
     ```bash
     ./data_pub.sh
     ```
   - **Details**:
     - Launches nodes for video publishing, video playback, and audio publishing.


### 2. **lidar_acquisition.sh**
   - **File**: [lidar_acquisition.sh](http://_vscodecontentref_/5)
   - **Description**: Records data from multiple topics, including lidar, audio, and video, into a ROS bag file.
   - **Usage**:
     ```bash
     ./lidar_acquisition.sh
     ```
   - **Details**:
     - Records topics such as `/scan`, `/car_speed`, `/audio_data`, and `/output_video_frame`.

### 3. **main_program.sh**
   - **File**: [main_program.sh](http://_vscodecontentref_/8)
   - **Description**: Starts the main Python program for controlling the car using the [rosmaster_main.py](http://_vscodecontentref_/9) script.
   - **Usage**:
     ```bash
     ./main_program.sh
     ```
   - **Details**:
     - Waits for 15 seconds before starting the program.
     - Executes the [rosmaster_main.py](http://_vscodecontentref_/10) script.


### 4. **my_start.sh**
   - **File**: [my_start.sh](http://_vscodecontentref_/14)
   - **Description**: Starts multiple processes, including data publishing, the main program, and lidar data acquisition.
   - **Usage**:
     ```bash
     ./my_start.sh
     ```
   - **Details**:
     - Kills any existing [rosmaster_main.py](http://_vscodecontentref_/15) process.
     - Opens new terminal tabs to run:
       - [data_pub.sh](http://_vscodecontentref_/16)
       - [main_program.sh](http://_vscodecontentref_/17)
       - [lidar_acquisition.sh](http://_vscodecontentref_/18)


## How to Run

1. Start the shell script my_start.sh
   ```bash
   ./my_start.sh
   ```


## Notes

- Ensure that all scripts have executable permissions. Use the following command if needed:
  ```bash
  chmod +x <script_name>.sh
- Ensure that your camera and microphone are properly connected for video and audio nodes.
- Modify the parameters in the launch files as needed for your specific setup.
- Use `Ctrl+C` to gracefully shut down nodes.


