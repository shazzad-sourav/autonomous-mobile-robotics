"""ROS 2 node: find four ArUco markers (DICT_4X4_50, IDs 1-4), record their
positions from wheel odometry, then drive to the centre of the four markers.

States:
  SEARCH_MARKER  rotate in place until an unvisited marker is roughly centred
  GO_TO          drive toward it, steering on the marker's x-offset, until it
                 is within distance_threshold; store the odometry position
  GO_TO_CENTER   once all four are stored: turn until the marker diagonally
                 opposite the last one is centred, then drive until the robot
                 has covered half the diagonal, then stop

Topic names and camera intrinsics are set for the TurtleBot3 Waffle Pi in
Gazebo (TURTLEBOT3_MODEL=waffle_pi) or the AgileX LIMO (any other value).
A live 2D map of markers, robot path and centroid is shown and saved as
robot_map.png.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import math
import numpy as np
import cv2.aruco as aruco
from scipy.spatial.transform import Rotation as R
from nav_msgs.msg import Odometry
import os

class ArucoNavigator(Node):
    def __init__(self):
        # Initialize the node with the name 'turtlebot_controller'
        super().__init__('aruco_navigator')

        # Movement control setup
        # Create a publisher for sending velocity commands
        # This publisher will send messages of type Twist to the 'cmd_vel' topic,
        # which is commonly used for controlling robot motion. The queue size of 10
        # ensures that up to 10 messages can be buffered for sending if necessary,
        # managing the flow of commands under varying system loads.
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        # Configure QoS profile for publishing and subscribing
        # Quality of Service (qos) policies that allow you to tune communication between nodes.
        # QoS policies for subscriber must match the publisher
        # For more info, see: https://docs.ros.org/en/humble/Concepts/Intermediate/About-Quality-of-Service-Settings.html
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,  # BEST_EFFORT: attempt to deliver samples, but may lose them if the network is not robust.
            durability=DurabilityPolicy.VOLATILE,  # VOLATILE: no attempt is made to persist samples.
            history=HistoryPolicy.KEEP_LAST,  # KEEP_LAST: only store up to N samples, configurable via the queue depth option.
            depth=10,  # a queue size of 10 to buffer messages if they arrive faster than they can be processed
        )

        # Turtlebot3(Gazebo)/LIMO specific parameters
        robot_env = os.getenv('TURTLEBOT3_MODEL')
        if(robot_env == 'waffle_pi'):
            camera_topic_name = "/camera/image_raw"
            odometry_topic_name = "/odom"
            self.camera_matrix = np.array([[530.4669406576809, 0.0, 320.5],[0.0,530.4669406576809, 240.5],[0.0, 0.0, 1.0]], dtype=np.float32)
            self.forward_speed = 0.08
            self.angular_speed = 0.09
            self.rotation_gain = 0.3
            self.distance_threshold = 0.5
        else:
            camera_topic_name = "/camera/color/image_raw"
            odometry_topic_name = "/wheel/odom"
            self.camera_matrix = np.array([[453.54339599609375, 0.0, 329.1346130371094], [0.0, 453.54339599609375, 241.37791442871094], [0.0, 0.0, 1.0]], dtype=np.float32)
            self.forward_speed = 0.08
            self.angular_speed = 0.13
            self.rotation_gain = 0.3
            self.distance_threshold = 0.5
        # Global
        self.alignment_threshold = 0.02
        self.initial_forward_speed = self.forward_speed
        self.forward_speed_gain = 0.22

        # Subscribe to the camera topic to receive image messages
        # Create a subscription to listen for messages on the '/camera/image_raw' topic,
        # using the Image message type. The 'image_callback' function is called for each new message
        self.subscription = self.create_subscription(
            Image,
            camera_topic_name,
            self.image_callback,
            qos_profile=qos_profile,
        )
        self.bridge = (
            CvBridge()
        )  # Initialize a CvBridge to convert ROS images to OpenCV format
        self.odom_sub = self.create_subscription(
            Odometry,
            odometry_topic_name,
            self.odom_callback,
            qos_profile=qos_profile
        )
        # Stop margin (m) subtracted from the camera distance while approaching the centre
        self.center_stop_margin = 0.4
        # Aruco
        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
        self.parameters = aruco.DetectorParameters_create()
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)
        self.marker_length = 0.16
        # Odometer
        self.odom = Odometry()
        self.last_pose_x = 0.0
        self.last_pose_y = 0.0
        self.last_pose_theta = 0.0
        self.goal_pose_x = 0.0
        self.goal_pose_y = 0.0
        self.init_odom_state = False
        self.step = 1
        # Initialize State
        self.state = "SEARCH_MARKER"
        # Update State Periodically
        self.update_timer = self.create_timer(0.010, self.update_state)  # unit: s
        self.marker_coords = {}
        self.relative_marker_coords = {}
        # +1 for clockwise, -1 for anti
        self.rotation_direction = +1

        # for movement
        self.z_distance = 0.0
        self.x_offset = 0.0
        self.pitch = 0.0

        # map
        self.map_size = 1000  # pixels
        self.scale = 80     # pixels per metre
        self.map_center = (self.map_size // 2, self.map_size // 2)
        self.robot_trail = []  # list of (x, y) positions
        self.map_image = 255 * np.ones((self.map_size, self.map_size, 3), dtype=np.uint8)  # white background

        # Relative marker coordinate
        self.x_ref = 0.0
        self.y_ref = 0.0

        self.last_marker = 1
        self.last_distance = 0.0

    def world_to_map_coords(self, x, y):
        # Convert to image coordinates (centered origin, y-axis flipped for image)
        mx = int(self.map_center[0] + x * self.scale)
        my = int(self.map_center[1] - y * self.scale)
        return mx, my

    def draw_marker(self, marker_id, x, y):
        mx, my = self.world_to_map_coords(x, y)
        cv2.circle(self.map_image, (mx, my), 5, (0, 255, 0), -1)
        cv2.putText(self.map_image, f"M{marker_id} (X={self.relative_marker_coords[marker_id]['x']}, Y={self.relative_marker_coords[marker_id]['y']})", (mx + 6, my - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    def draw_robot_trail(self):
        for x, y in self.robot_trail:
            mx, my = self.world_to_map_coords(x, y)
            cv2.circle(self.map_image, (mx, my), 1, (255, 0, 0), -1)

    def draw_centroid(self):
        count = 0
        x_total = 0.0
        y_total = 0.0
        for m in self.marker_coords.values():
            if m['x'] is not None and m['y'] is not None:
                x_total += m['x']
                y_total += m['y']
                count += 1
        if count == 4:
            cx = x_total / 4
            cy = y_total / 4
            mx, my = self.world_to_map_coords(cx, cy)
            cv2.line(self.map_image, (mx - 10, my - 10), (mx + 10, my + 10), (0, 0, 255), 2)
            cv2.line(self.map_image, (mx + 10, my - 10), (mx - 10, my + 10), (0, 0, 255), 2)
            cv2.putText(self.map_image, "Centroid", (mx + 12, my - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)

    def draw_grid(self, spacing_m=0.5, color=(200, 200, 200), thickness=1):
        spacing_px = int(spacing_m * self.scale)

        # Vertical lines
        for x in range(0, self.map_size, spacing_px):
            cv2.line(self.map_image, (x, 0), (x, self.map_size), color, thickness)

        # Horizontal lines
        for y in range(0, self.map_size, spacing_px):
            cv2.line(self.map_image, (0, y), (self.map_size, y), color, thickness)

        # Draw center axes in darker color
        cv2.line(self.map_image, (self.map_center[0], 0), (self.map_center[0], self.map_size), (150, 150, 150), 2)
        cv2.line(self.map_image, (0, self.map_center[1]), (self.map_size, self.map_center[1]), (150, 150, 150), 2)

    def update_map(self):
        self.map_image = 255 * np.ones((self.map_size, self.map_size, 3), dtype=np.uint8)  # Reset to white

        self.draw_grid(spacing_m=0.5)  # Grid every 0.5 meters

        # Draw markers
        for marker_id, m in self.marker_coords.items():
            if m['x'] is not None and m['y'] is not None:
                self.draw_marker(marker_id, m['x'], m['y'])

        # Draw robot trail
        self.draw_robot_trail()

        # Draw centroid if all found
        if len(self.marker_coords) == 4:
            self.draw_centroid()
            sorted_ids = sorted(self.marker_coords.keys())  # consistent ordering
            points = [self.world_to_map_coords(self.marker_coords[mid]['x'], self.marker_coords[mid]['y']) for mid in sorted_ids]

            for i in range(len(points)):
                pt1 = points[i]
                pt2 = points[(i + 1) % len(points)]  # wrap around to form a closed shape
                cv2.line(self.map_image, pt1, pt2, (0, 0, 0), 1)

            cv2.imwrite("robot_map.png", self.map_image)

        cv2.imshow("VSLAM", self.map_image)
        cv2.waitKey(1)

    def odom_callback(self, msg):
        self.last_pose_x = msg.pose.pose.position.x
        self.last_pose_y = msg.pose.pose.position.y
        orientation_arr = msg.pose.pose.orientation
        _, _, self.last_pose_theta = self.euler_from_quaternion(orientation_arr.x, orientation_arr.y, orientation_arr.z, orientation_arr.w)
        self.init_odom_state = True
        self.robot_trail.append((self.last_pose_x, self.last_pose_y))
        self.update_map()

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f'Image conversion failed: {e}')
            return

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers
        corners, ids, rejected = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)
        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, self.marker_length, self.camera_matrix, self.dist_coeffs)
            for i, corner in enumerate(corners):
                pts = corner[0].astype(int)
                top_left, top_right, bottom_right, bottom_left = pts

                # Store the id
                id = int(ids[i][0])

                # Draw green bounding box
                cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

                # Center of the marker
                center_x = int(np.mean(pts[:, 0]))
                center_y = int(np.mean(pts[:, 1]))
                center = (center_x, center_y)

                # Draw red dot at top-left and blue dot at center
                cv2.circle(frame, tuple(top_left), 4, (0, 0, 255), -1)
                cv2.circle(frame, center, 4, (255, 0, 0), -1)

                # Draw axis on each marker
                cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvecs[i], tvecs[i], 0.15)

                # Compute distance
                tvec = tvecs[i][0]

                # Store the rotation information
                rotation_matrix = np.eye(4)
                rotation_matrix[0:3, 0:3] = cv2.Rodrigues(np.array(rvecs[i][0]))[0]
                r = R.from_matrix(rotation_matrix[0:3, 0:3])
                quat = r.as_quat()

                # Quaternion format
                transform_rotation_x = quat[0]
                transform_rotation_y = quat[1]
                transform_rotation_z = quat[2]
                transform_rotation_w = quat[3]

                # Euler angle format in radians
                roll_x, pitch_y, yaw_z = self.euler_from_quaternion(transform_rotation_x, transform_rotation_y, transform_rotation_z, transform_rotation_w)
                roll_x = math.degrees(roll_x)
                pitch_y = math.degrees(pitch_y)
                yaw_z = math.degrees(yaw_z)

                z_distance = tvec[2]
                x_offset = tvec[0]

                cv2.putText(frame,
                    f"ID: {id} Z_Dist: {z_distance: .2f} X_OFFSET: {tvec[0]: .2f} Roll: {roll_x:.2f} Pitch: {pitch_y:.2f} Yaw: {yaw_z:.2f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 2)

                if self.state == "SEARCH_MARKER":
                    if id not in self.marker_coords and abs(x_offset) < 0.45:
                        self.state = "GO_TO"
                        self.z_distance = z_distance
                        self.x_offset = x_offset
                        self.pitch = pitch_y
                elif self.state == "GO_TO":
                    if id not in self.marker_coords:
                        self.z_distance = z_distance
                        self.x_offset = x_offset
                        self.pitch = pitch_y

                        # move forward faster if far from marker
                        if self.z_distance > self.distance_threshold + 0.3:
                            self.forward_speed = self.initial_forward_speed + self.forward_speed_gain
                        else:
                            self.forward_speed = self.initial_forward_speed

                        if z_distance < self.distance_threshold:
                            if(len(self.marker_coords)) == 0:
                                if id == 1 or id == 3:
                                    self.rotation_direction = self.rotation_direction * -1
                            if(len(self.marker_coords)) == 2:
                                self.rotation_direction = self.rotation_direction * -1
                            self.set_marker_position(id, self.last_pose_x, self.last_pose_y)


                            if (len(self.marker_coords)) != 4:
                                self.state = "SEARCH_MARKER"
                            else:
                                # Calculate the centroid
                                self.get_centroid()
                                self.state = "GO_TO_CENTER"
                                print(f"Goal position set: x={self.goal_pose_x}, y={self.goal_pose_y}")
                                self.last_marker = id

                                diagonal_pairs = {1: 3, 2: 4, 3: 1, 4: 2}
                                self.last_diagonal_marker = diagonal_pairs.get(self.last_marker)

                                self.last_distance = round((math.sqrt((self.marker_coords[self.last_diagonal_marker]['x'] - self.marker_coords[self.last_marker]['x'])**2 +
                                    (self.marker_coords[self.last_diagonal_marker]['y'] - self.marker_coords[self.last_marker]['y'])**2)) / 2, 2)

                            self.robot_trail.clear()  # Clear path when marker detected
                elif self.state == "GO_TO_CENTER":
                    z_distance_rounded = round(round(z_distance, 2) - self.center_stop_margin, 2)
                    if self.step == 1 and id == self.last_diagonal_marker and abs(x_offset) < 0.25:
                        self.step += 1
                    if self.step == 2 and (z_distance_rounded - self.last_distance) < 0:
                        self.step += 1

        # Show the output
        cv2.imshow("ArUco Detection", frame)
        cv2.waitKey(1)

    def update_state(self):
        twist = Twist()
        if self.state == 'SEARCH_MARKER':
            # Continuous rotation until a marker is found
            twist.linear.x = 0.0
            twist.angular.z = self.angular_speed * self.rotation_direction
        elif self.state == "GO_TO":
            twist.linear.x = self.forward_speed
            # Align to marker
            if abs(self.x_offset) > self.alignment_threshold:
                twist.angular.z = -self.x_offset * self.rotation_gain
            else:
                twist.angular.z = 0.0
        elif self.state == "GO_TO_CENTER":
            # Final Step - GO TO CENTER
            if self.init_odom_state is True:
                # Step 1: Turn
                if self.step == 1:
                    twist.linear.x = 0.0
                    twist.angular.z = self.angular_speed * self.rotation_direction
                # Step 2: Drive straight
                elif self.step == 2:
                    twist.linear.x = self.forward_speed
                    twist.angular.z = 0.0
                # Step 3: Stop at the centre
                elif self.step == 3:
                    twist.linear.x = 0.0
                    twist.linear.y = 0.0

        self.publisher_.publish(twist)

    def euler_from_quaternion(self, x, y, z, w):
        """
        Convert quaternion (w in last place) to euler roll, pitch, yaw.
        """
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(sinp)

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def set_marker_position(self, marker_id, x, y):
        if(len(self.marker_coords)) == 0:
            self.x_ref = round(self.last_pose_x, 2)
            self.y_ref = round(self.last_pose_y, 2)

        if marker_id not in self.marker_coords:
            x_rounded = round(x, 2)
            y_rounded = round(y, 2)
            self.marker_coords[marker_id] = {'x': x_rounded, 'y': y_rounded}
            self.relative_marker_coords[marker_id] = {'x': round(x_rounded - self.x_ref, 2), 'y': round(y_rounded - self.y_ref, 2)}
            print(f"Marker {marker_id} position set: x={x_rounded}, y={y_rounded}")

    def get_centroid(self):
        x_total = 0.0
        y_total = 0.0
        count = 0

        for marker in self.marker_coords.values():
            x = marker['x']
            y = marker['y']
            if x is not None and y is not None:
                x_total += x
                y_total += y
                count += 1

        if count == 0:
            return None, None  # Avoid division by zero

        centroid_x = round((x_total / count), 2)
        centroid_y = round((y_total / count), 2)

        self.goal_pose_x = centroid_x
        self.goal_pose_y = centroid_y

        return centroid_x, centroid_y

def main(args=None):
    rclpy.init(args=args)
    node = ArucoNavigator()
    rclpy.spin(node)
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()