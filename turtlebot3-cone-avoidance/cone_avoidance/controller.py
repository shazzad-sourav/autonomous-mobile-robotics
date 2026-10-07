"""Camera-only obstacle and boundary avoidance for a TurtleBot3 Waffle Pi in Gazebo.

Reconstructed from the Assignment 3 report (the original code was lost):

1. Keep the bottom half of the camera frame (the camera sits close to the ground).
2. Convert to HSV and build two colour masks:
   - cones: two orange/red ranges (cone body and darker top/base), combined
   - boundary: grey floor outside the field, plus yellow and blue goal posts
3. Clean each mask with a morphological close, then open.
4. If enough cone pixels are visible, turn clockwise. Otherwise, if enough boundary
   pixels are visible, turn clockwise. Otherwise drive forward.

Thresholds are fractions of the region of interest, so they do not depend on the
camera resolution. The defaults equal the report's pixel counts (18,000 and
160,000) on the Waffle Pi's 1920 x 1080 camera, whose bottom half is 1,036,800 px.
All colour ranges and thresholds are ROS parameters.
"""

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))


def hsv_mask(hsv, ranges):
    """OR of cv2.inRange over a list of (lower, upper) HSV triples."""
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lower, upper in ranges:
        mask |= cv2.inRange(hsv, np.array(lower, np.uint8), np.array(upper, np.uint8))
    return mask


def clean(mask):
    """Close small gaps, then remove specks left over by the closing."""
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)
    return cv2.morphologyEx(closed, cv2.MORPH_OPEN, KERNEL)


def as_ranges(flat):
    """[h1, s1, v1, h2, s2, v2, ...] -> [((h1, s1, v1), (h2, s2, v2)), ...]"""
    values = [int(v) for v in flat]
    if len(values) % 6:
        raise ValueError("HSV range lists need 6 numbers per range")
    return [(values[i:i + 3], values[i + 3:i + 6]) for i in range(0, len(values), 6)]


class ConeAvoidance(Node):
    def __init__(self):
        super().__init__("cone_avoidance")
        params = {
            "linear_speed": 0.15,             # m/s
            "turn_speed": 0.6,                # rad/s, applied clockwise
            "cone_fraction": 0.0174,          # 18,000 / 1,036,800
            "boundary_fraction": 0.154,       # 160,000 / 1,036,800
            # Cone body (orange) and top/base (red-orange, wraps around hue 0)
            "cone_hsv": [5, 120, 90, 22, 255, 255,
                         0, 120, 70, 5, 255, 255,
                         170, 120, 70, 180, 255, 255],
            # Grey floor outside the field, yellow goal, blue goal
            "boundary_hsv": [0, 0, 60, 180, 35, 200,
                             22, 100, 100, 35, 255, 255,
                             100, 120, 60, 130, 255, 255],
            "show_debug": True,
        }
        for name, default in params.items():
            self.declare_parameter(name, default)
        get = lambda n: self.get_parameter(n).value
        self.linear_speed = get("linear_speed")
        self.turn_speed = get("turn_speed")
        self.cone_fraction = get("cone_fraction")
        self.boundary_fraction = get("boundary_fraction")
        self.cone_ranges = as_ranges(get("cone_hsv"))
        self.boundary_ranges = as_ranges(get("boundary_hsv"))
        self.show_debug = get("show_debug")

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.bridge = CvBridge()
        self.cmd_pub = self.create_publisher(Twist, "cmd_vel", 10)
        self.create_subscription(Image, "/camera/image_raw", self.on_image, qos)
        self.state = None

    def decide(self, frame):
        """Return ('cone' | 'boundary' | 'clear', cone mask, boundary mask)."""
        roi = frame[frame.shape[0] // 2:, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        cones = clean(hsv_mask(hsv, self.cone_ranges))
        boundary = clean(hsv_mask(hsv, self.boundary_ranges))
        area = roi.shape[0] * roi.shape[1]
        if cv2.countNonZero(cones) > self.cone_fraction * area:
            return "cone", cones, boundary
        if cv2.countNonZero(boundary) > self.boundary_fraction * area:
            return "boundary", cones, boundary
        return "clear", cones, boundary

    def on_image(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as exc:  # noqa: BLE001 - log and skip bad frames
            self.get_logger().error(f"Image conversion failed: {exc}")
            return

        state, cones, boundary = self.decide(frame)
        twist = Twist()
        if state == "clear":
            twist.linear.x = self.linear_speed
        else:
            twist.angular.z = -self.turn_speed     # clockwise
        self.cmd_pub.publish(twist)

        if state != self.state:
            self.get_logger().info(f"state: {state}")
            self.state = state
        if self.show_debug:
            small = lambda img: cv2.resize(img, (480, 135))
            cv2.imshow("cone mask | boundary mask", np.vstack([small(cones), small(boundary)]))
            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = ConeAvoidance()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_pub.publish(Twist())   # stop the robot
        node.destroy_node()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
