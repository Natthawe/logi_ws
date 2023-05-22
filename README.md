# logi_ws

### open webcam
    ros2 launch usb_cam demo_launch.py

### generate_marker
    ros2 run ros2_aruco aruco_generate_marker --id 100 --dictionary DICT_4X4_1000

### camera_calibration
    ros2 run camera_calibration cameracalibrator --size 8x6 --square 0.108 --ros-args --remap image:=/camera/image_raw camera:=/camera