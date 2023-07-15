#!/usr/bin/env python3

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
import math


class LineDetection(Node):
    def __init__(self):
        super().__init__('line_detection')
        self.bridge = CvBridge()
        self.publisher = self.create_publisher(Float32, 'line_distance', 10)
        self.subscription = self.create_subscription(Image, 'camera/image_raw', self.image_callback, 10)
        self.subscription  # prevent unused variable warning
        self.declare_parameter('grid_shape', [2, 2])
        self.grid_shape = self.get_parameter('grid_shape').value
        self.cx = 320
        self.cy = 90

    def draw_grid(self, img, grid_shape, color=(0, 255, 0), thickness=1):
        h, w, _ = img.shape
        rows, cols = grid_shape
        dy, dx = h / rows, w / cols

        # draw vertical lines
        for x in np.linspace(start=dx, stop=w - dx, num=cols - 1):
            x = int(round(x))
            cv2.line(img, (x, 0), (x, h), color=color, thickness=thickness)

        # draw horizontal lines
        for y in np.linspace(start=dy, stop=h - dy, num=rows - 1):
            y = int(round(y))
            cv2.line(img, (0, y), (w, y), color=color, thickness=thickness)

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        frame = frame[300:480, ::]
        self.draw_grid(frame, self.grid_shape)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv = cv2.medianBlur(hsv, 9)
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 30])
        mask = cv2.inRange(hsv, lower_black, upper_black)
        result = cv2.bitwise_and(frame, frame, mask=mask)

        contours, hierarchy = cv2.findContours(
            mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE
        )
        image_copy = mask.copy()

        if len(contours) != 0:
            cv2.drawContours(
                image=result,
                contours=contours,
                contourIdx=-1,
                color=(0, 255, 0),
                thickness=1,
                lineType=cv2.LINE_AA,
            )
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                d = 0
                self.get_logger().info(f"Centroid: {cX}, {cY}")
                if cX < self.cx:
                    d = math.sqrt((self.cy - cY) ** 2 + (self.cx - cX) ** 2)
                    self.get_logger().info("Turn left")
                    self.get_logger().info(f"Distance: {d}")
                elif cX > self.cx:
                    d = -math.sqrt((self.cy - cY) ** 2 + (self.cx - cX) ** 2)
                    self.get_logger().info("Turn right")
                    self.get_logger().info(f"Distance: {d}")
                elif cX == self.cx:
                    self.get_logger().info("You are on the right path")
                if d == 0:
                    self.get_logger().info("You are right")
            else:
                cX, cY = 0, 0

            cv2.circle(frame, (cX, cY), 5, (255, 255, 255), -1)
            cv2.putText(
                frame,
                "centroid",
                (cX - 25, cY - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

        self.publisher.publish(Float32(data=d))
        cv2.imshow('Cropped', frame)
        cv2.imshow('Result', result)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    line_detection = LineDetection()
    rclpy.spin(line_detection)
    line_detection.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
