"""Start Gazebo with the cone world, spawn a TurtleBot3 and run the avoidance node.

Requires TURTLEBOT3_MODEL=waffle_pi (the model with a camera).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory("cone_avoidance")
    tb3 = os.path.join(get_package_share_directory("turtlebot3_gazebo"), "launch")
    gazebo = os.path.join(get_package_share_directory("gazebo_ros"), "launch")

    use_sim_time = LaunchConfiguration("use_sim_time", default="true")
    x_pose = LaunchConfiguration("x_pose", default="-2.5")
    y_pose = LaunchConfiguration("y_pose", default="0.0")
    world = os.path.join(pkg, "worlds", "new_world.world")

    return LaunchDescription([
        DeclareLaunchArgument("x_pose", default_value="-2.5"),
        DeclareLaunchArgument("y_pose", default_value="0.0"),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(gazebo, "gzserver.launch.py")),
            launch_arguments={"world": world}.items()),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(gazebo, "gzclient.launch.py"))),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(tb3, "robot_state_publisher.launch.py")),
            launch_arguments={"use_sim_time": use_sim_time}.items()),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(tb3, "spawn_turtlebot3.launch.py")),
            launch_arguments={"x_pose": x_pose, "y_pose": y_pose}.items()),
        Node(package="cone_avoidance", executable="cone_avoidance", output="screen",
             parameters=[{"use_sim_time": True}]),
    ])
