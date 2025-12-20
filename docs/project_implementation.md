# Project Implementation Guide

## Overview

This guide provides step-by-step instructions for implementing the complete Physical AI & Humanoid Robotics project. Following the modules covered earlier, we'll integrate all components to create a functional humanoid robot control system.

## Prerequisites

Before beginning the implementation, ensure you have:

- Ubuntu 22.04 LTS installed
- ROS 2 Humble Hawksbill (or Iron Irwini) installed
- Python 3.10+
- Gazebo Garden or Fortress
- Unity 2022.3 LTS (for visualization)
- NVIDIA Isaac Sim (optional, for advanced simulation)
- Git for version control

## Step 1: Project Setup

### 1.1 Creating the Workspace

First, let's create our ROS 2 workspace structure:

```bash
mkdir -p ~/physical_ai_ws/src
cd ~/physical_ai_ws
```

### 1.2 Setting up the Package Structure

Create the basic package structure based on our specifications:

```bash
cd ~/physical_ai_ws/src
# Create ROS 2 packages
ros2 pkg create --build-type ament_python humanoid_description
ros2 pkg create --build-type ament_python humanoid_controllers
ros2 pkg create --build-type ament_python humanoid_msgs
ros2 pkg create --build-type ament_python ai_agents
ros2 pkg create --build-type ament_python simulation_bridge

# Create launch directory for each package
mkdir -p humanoid_description/launch
mkdir -p humanoid_controllers/launch
mkdir -p ai_agents/launch
mkdir -p simulation_bridge/launch
```

### 1.3 Environment Setup Script

Create an environment setup script:

```bash
#!/bin/bash
# setup_environment.sh

echo "Setting up Physical AI & Humanoid Robotics Environment"

# Update package lists
sudo apt update

# Install ROS 2 Humble dependencies
sudo apt install -y python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential

# Install Gazebo
sudo apt install -y gazebo libgazebo-dev

# Install additional dependencies
sudo apt install -y python3-colcon-common-extensions python3-vcstool

# Install Python dependencies
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install opencv-python numpy scipy

echo "Environment setup complete. Don't forget to source your ROS 2 installation:"
echo "source /opt/ros/humble/setup.bash"
```

## Step 2: Implementing Module 1 - ROS 2 Infrastructure

### 2.1 URDF Model for Humanoid Robot

In the `humanoid_description` package, create a URDF model. Let's create a simplified humanoid model:

`~/physical_ai_ws/src/humanoid_description/urdf/humanoid.urdf`

```xml
<?xml version="1.0"?>
<robot name="humanoid_robot">
  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.2 0.15 0.4"/>
      </geometry>
      <material name="gray">
        <color rgba="0.5 0.5 0.5 1.0"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.2 0.15 0.4"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5.0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Head -->
  <joint name="neck_joint" type="revolute">
    <parent link="base_link"/>
    <child link="head"/>
    <origin xyz="0 0 0.25" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.785" upper="0.785" effort="10.0" velocity="1.0"/>
  </joint>

  <link name="head">
    <visual>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
      <material name="skin">
        <color rgba="0.8 0.6 0.4 1.0"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.002" ixy="0.0" ixz="0.0" iyy="0.002" iyz="0.0" izz="0.002"/>
    </inertial>
  </link>

  <!-- Left Arm -->
  <joint name="left_shoulder_joint" type="revolute">
    <parent link="base_link"/>
    <child link="left_upper_arm"/>
    <origin xyz="0.12 0.15 0.1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="10.0" velocity="1.0"/>
  </joint>

  <link name="left_upper_arm">
    <visual>
      <geometry>
        <cylinder length="0.3" radius="0.03"/>
      </geometry>
      <material name="gray"/>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.3" radius="0.03"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.8"/>
      <inertia ixx="0.002" ixy="0.0" ixz="0.0" iyy="0.002" iyz="0.0" izz="0.002"/>
    </inertial>
  </link>

  <joint name="left_elbow_joint" type="revolute">
    <parent link="left_upper_arm"/>
    <child link="left_lower_arm"/>
    <origin xyz="0 0 -0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="2.35" effort="10.0" velocity="1.0"/>
  </joint>

  <link name="left_lower_arm">
    <visual>
      <geometry>
        <cylinder length="0.25" radius="0.025"/>
      </geometry>
      <material name="gray"/>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.25" radius="0.025"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <!-- Right Arm (similar to left, mirrored) -->
  <joint name="right_shoulder_joint" type="revolute">
    <parent link="base_link"/>
    <child link="right_upper_arm"/>
    <origin xyz="0.12 -0.15 0.1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="10.0" velocity="1.0"/>
  </joint>

  <link name="right_upper_arm">
    <visual>
      <geometry>
        <cylinder length="0.3" radius="0.03"/>
      </geometry>
      <material name="gray"/>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.3" radius="0.03"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.8"/>
      <inertia ixx="0.002" ixy="0.0" ixz="0.0" iyy="0.002" iyz="0.0" izz="0.002"/>
    </inertial>
  </link>

  <joint name="right_elbow_joint" type="revolute">
    <parent link="right_upper_arm"/>
    <child link="right_lower_arm"/>
    <origin xyz="0 0 -0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="2.35" effort="10.0" velocity="1.0"/>
  </joint>

  <link name="right_lower_arm">
    <visual>
      <geometry>
        <cylinder length="0.25" radius="0.025"/>
      </geometry>
      <material name="gray"/>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.25" radius="0.025"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <!-- Left Leg -->
  <joint name="left_hip_joint" type="revolute">
    <parent link="base_link"/>
    <child link="left_thigh"/>
    <origin xyz="-0.1 0.07 -0.2" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-0.78" upper="0.78" effort="20.0" velocity="1.0"/>
  </joint>

  <link name="left_thigh">
    <visual>
      <geometry>
        <cylinder length="0.4" radius="0.04"/>
      </geometry>
      <material name="gray"/>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.4" radius="0.04"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.5"/>
      <inertia ixx="0.005" ixy="0.0" ixz="0.0" iyy="0.005" iyz="0.0" izz="0.005"/>
    </inertial>
  </link>

  <joint name="left_knee_joint" type="revolute">
    <parent link="left_thigh"/>
    <child link="left_shin"/>
    <origin xyz="0 0 -0.4" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="2.35" effort="20.0" velocity="1.0"/>
  </joint>

  <link name="left_shin">
    <visual>
      <geometry>
        <cylinder length="0.35" radius="0.035"/>
      </geometry>
      <material name="gray"/>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.35" radius="0.035"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.2"/>
      <inertia ixx="0.003" ixy="0.0" ixz="0.0" iyy="0.003" iyz="0.0" izz="0.003"/>
    </inertial>
  </link>

  <joint name="left_ankle_joint" type="revolute">
    <parent link="left_shin"/>
    <child link="left_foot"/>
    <origin xyz="0 0 -0.35" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-0.78" upper="0.78" effort="10.0" velocity="1.0"/>
  </joint>

  <link name="left_foot">
    <visual>
      <geometry>
        <box size="0.15 0.08 0.04"/>
      </geometry>
      <material name="gray"/>
    </visual>
    <collision>
      <geometry>
        <box size="0.15 0.08 0.04"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.3"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <!-- Right Leg (similar to left, mirrored) -->
  <joint name="right_hip_joint" type="revolute">
    <parent link="base_link"/>
    <child link="right_thigh"/>
    <origin xyz="-0.1 -0.07 -0.2" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-0.78" upper="0.78" effort="20.0" velocity="1.0"/>
  </joint>

  <link name="right_thigh">
    <visual>
      <geometry>
        <cylinder length="0.4" radius="0.04"/>
      </geometry>
      <material name="gray"/>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.4" radius="0.04"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.5"/>
      <inertia ixx="0.005" ixy="0.0" ixz="0.0" iyy="0.005" iyz="0.0" izz="0.005"/>
    </inertial>
  </link>

  <joint name="right_knee_joint" type="revolute">
    <parent link="right_thigh"/>
    <child link="right_shin"/>
    <origin xyz="0 0 -0.4" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="2.35" effort="20.0" velocity="1.0"/>
  </joint>

  <link name="right_shin">
    <visual>
      <geometry>
        <cylinder length="0.35" radius="0.035"/>
      </geometry>
      <material name="gray"/>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.35" radius="0.035"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.2"/>
      <inertia ixx="0.003" ixy="0.0" ixz="0.0" iyy="0.003" iyz="0.0" izz="0.003"/>
    </inertial>
  </link>

  <joint name="right_ankle_joint" type="revolute">
    <parent link="right_shin"/>
    <child link="right_foot"/>
    <origin xyz="0 0 -0.35" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-0.78" upper="0.78" effort="10.0" velocity="1.0"/>
  </joint>

  <link name="right_foot">
    <visual>
      <geometry>
        <box size="0.15 0.08 0.04"/>
      </geometry>
      <material name="gray"/>
    </visual>
    <collision>
      <geometry>
        <box size="0.15 0.08 0.04"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.3"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>
</robot>
```

### 2.2 Python Nodes for Control

In the `humanoid_controllers` package, create a basic controller:

`~/physical_ai_ws/src/humanoid_controllers/humanoid_controllers/basic_controller.py`

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from builtin_interfaces.msg import Duration
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import JointTrajectoryControllerState

class BasicController(Node):
    def __init__(self):
        super().__init__('basic_controller')
        
        # Publisher for joint trajectory commands
        self.joint_trajectory_pub = self.create_publisher(
            JointTrajectory, '/joint_trajectory_controller/joint_trajectory', 10)
        
        # Subscriber for joint states
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)
        
        # Timer for control loop
        self.timer = self.create_timer(0.1, self.control_loop)
        
        # Store current joint states
        self.current_joint_positions = {}
        self.joint_names = [
            'neck_joint', 'left_shoulder_joint', 'left_elbow_joint',
            'right_shoulder_joint', 'right_elbow_joint', 'left_hip_joint',
            'left_knee_joint', 'left_ankle_joint', 'right_hip_joint',
            'right_knee_joint', 'right_ankle_joint'
        ]
        
        self.get_logger().info('Basic Controller initialized')

    def joint_state_callback(self, msg):
        """Update current joint positions"""
        for name, position in zip(msg.name, msg.position):
            self.current_joint_positions[name] = position

    def control_loop(self):
        """Main control loop function"""
        # Create a trajectory message
        trajectory_msg = JointTrajectory()
        trajectory_msg.joint_names = self.joint_names
        
        # Create a trajectory point
        point = JointTrajectoryPoint()
        
        # Set target positions (example: move to neutral position)
        target_positions = [0.0] * len(self.joint_names)
        point.positions = target_positions
        
        # Set velocities (optional)
        point.velocities = [0.0] * len(self.joint_names)
        
        # Set acceleration (optional)
        point.accelerations = [0.0] * len(self.joint_names)
        
        # Set time from start (1 second to reach target)
        point.time_from_start = Duration(sec=1)
        
        trajectory_msg.points = [point]
        
        # Publish the trajectory
        self.joint_trajectory_pub.publish(trajectory_msg)

def main(args=None):
    rclpy.init(args=args)
    controller = BasicController()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 2.3 Package Configuration

Update the `setup.py` for the `humanoid_controllers` package:

`~/physical_ai_ws/src/humanoid_controllers/setup.py`

```python
from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'humanoid_controllers'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include URDF files
        (os.path.join('share', package_name, 'urdf'), 
         glob('urdf/*')),
        # Include launch files
        (os.path.join('share', package_name, 'launch'), 
         glob('launch/*launch.[pxy][yma]*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='Controllers for humanoid robot',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'basic_controller = humanoid_controllers.basic_controller:main',
        ],
    },
)
```

## Step 3: Implementing Module 2 - Simulation Environment

### 3.1 Gazebo World File

Create a basic Gazebo world file:

`~/physical_ai_ws/src/humanoid_description/worlds/basic_world.sdf`

```xml
<?xml version="1.0"?>
<sdf version="1.6">
  <world name="basic_world">
    <!-- Physics parameters -->
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
      <gravity>0 0 -9.8</gravity>
    </physics>
    
    <!-- Include a ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>
    
    <!-- Include sun for lighting -->
    <include>
      <uri>model://sun</uri>
    </include>
    
    <!-- Add some obstacles -->
    <model name="table">
      <pose>-1 0 0.4 0 0 0</pose>
      <link name="table_base">
        <collision name="collision">
          <geometry>
            <box>
              <size>1.5 0.8 0.8</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1.5 0.8 0.8</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.6 0.2 1</ambient>
            <diffuse>0.8 0.6 0.2 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>10</mass>
          <inertia>
            <ixx>1</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>1</iyy>
            <iyz>0</iyz>
            <izz>1</izz>
          </inertia>
        </inertial>
      </link>
    </model>
    
    <!-- Add the humanoid model -->
    <include>
      <uri>model://humanoid_description</uri>
      <name>humanoid_robot</name>
      <pose>0 0 0.8 0 0 0</pose>
    </include>
  </world>
</sdf>
```

### 3.2 Simulation Launch File

Create a launch file to start the simulation:

`~/physical_ai_ws/src/humanoid_description/launch/simulation_launch.py`

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get package directories
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    pkg_humanoid_description = get_package_share_directory('humanoid_description')
    
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    
    # Gazebo launch
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={
            'world': os.path.join(pkg_humanoid_description, 'worlds', 'basic_world.sdf'),
            'verbose': 'true',
        }.items()
    )
    
    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'robot_description': open(os.path.join(pkg_humanoid_description, 'urdf', 'humanoid.urdf')).read()
        }]
    )
    
    # Spawn entity
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'humanoid_robot',
            '-x', '0', '-y', '0', '-z', '0.8'
        ],
        output='screen'
    )
    
    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation (Gazebo) clock if true'
        ),
        gazebo,
        robot_state_publisher,
        spawn_entity,
    ])
```

## Step 4: Implementing AI Agent Integration

### 4.1 Basic AI Agent

Create a basic AI agent in the `ai_agents` package:

`~/physical_ai_ws/src/ai_agents/ai_agents/simple_balancer.py`

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Vector3
from std_msgs.msg import Float64MultiArray
from builtin_interfaces.msg import Duration
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import numpy as np
from collections import deque

class SimpleBalancer(Node):
    def __init__(self):
        super().__init__('simple_balancer')
        
        # Publisher for joint trajectory commands
        self.joint_trajectory_pub = self.create_publisher(
            JointTrajectory, '/joint_trajectory_controller/joint_trajectory', 10)
        
        # Subscriber for joint states
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)
        
        # Subscriber for IMU data
        self.imu_sub = self.create_subscription(
            Imu, '/imu', self.imu_callback, 10)
        
        # Timer for control loop
        self.timer = self.create_timer(0.05, self.control_loop)
        
        # Store current joint states
        self.current_joint_positions = {}
        self.current_joint_velocities = {}
        self.imu_data = None
        
        # Joint names for humanoid
        self.joint_names = [
            'neck_joint', 'left_shoulder_joint', 'left_elbow_joint',
            'right_shoulder_joint', 'right_elbow_joint', 'left_hip_joint',
            'left_knee_joint', 'left_ankle_joint', 'right_hip_joint',
            'right_knee_joint', 'right_ankle_joint'
        ]
        
        # Target positions (neutral standing position)
        self.target_positions = np.array([0.0] * len(self.joint_names))
        
        # Initialize with a slight hip adjustment to stand up
        for i, name in enumerate(self.joint_names):
            if 'hip' in name:
                self.target_positions[i] = 0.1  # Slight forward lean
            elif 'knee' in name:
                self.target_positions[i] = -0.3  # Slight bend
            elif 'ankle' in name:
                self.target_positions[i] = 0.1  # Slight adjustment
        
        self.get_logger().info('Simple Balancer initialized')

    def joint_state_callback(self, msg):
        """Update current joint positions and velocities"""
        for name, pos, vel in zip(msg.name, msg.position, msg.velocity):
            if name in self.joint_names:
                idx = self.joint_names.index(name)
                self.current_joint_positions[name] = pos
                self.current_joint_velocities[name] = vel

    def imu_callback(self, msg):
        """Update IMU data"""
        self.imu_data = {
            'orientation': msg.orientation,
            'angular_velocity': msg.angular_velocity,
            'linear_acceleration': msg.linear_acceleration
        }

    def control_loop(self):
        """Main control loop with balancing logic"""
        if self.imu_data is not None:
            # Simple balancing algorithm based on IMU data
            linear_acc = self.imu_data['linear_acceleration']
            
            # Adjust target positions based on tilt
            # If the robot is tilting forward, lean back
            if linear_acc.x > 0.5:  # Forward tilt
                self.adjust_for_balance('forward')
            elif linear_acc.x < -0.5:  # Backward tilt
                self.adjust_for_balance('backward')
            
            # If tilting sideways
            if abs(linear_acc.y) > 0.5:
                self.adjust_for_balance('sideways')

        # Create trajectory message
        trajectory_msg = JointTrajectory()
        trajectory_msg.joint_names = self.joint_names
        
        # Create trajectory point
        point = JointTrajectoryPoint()
        point.positions = self.target_positions.tolist()
        point.velocities = [0.0] * len(self.joint_names)
        point.accelerations = [0.0] * len(self.joint_names)
        point.time_from_start = Duration(sec=0, nanosec=50000000)  # 50ms to reach target
        
        trajectory_msg.points = [point]
        
        # Publish trajectory
        self.joint_trajectory_pub.publish(trajectory_msg)

    def adjust_for_balance(self, direction):
        """Adjust joint positions to maintain balance"""
        if direction == 'forward':
            # Lean back by adjusting hip and ankle joints
            for i, name in enumerate(self.joint_names):
                if 'hip' in name:
                    self.target_positions[i] = min(0.2, self.target_positions[i] + 0.02)
                elif 'ankle' in name:
                    self.target_positions[i] = max(-0.1, self.target_positions[i] - 0.02)
        elif direction == 'backward':
            # Lean forward by adjusting hip and ankle joints
            for i, name in enumerate(self.joint_names):
                if 'hip' in name:
                    self.target_positions[i] = max(-0.1, self.target_positions[i] - 0.02)
                elif 'ankle' in name:
                    self.target_positions[i] = min(0.2, self.target_positions[i] + 0.02)
        elif direction == 'sideways':
            # Adjust hip joints to counter sideways tilt
            for i, name in enumerate(self.joint_names):
                if 'hip' in name:
                    if 'left' in name:
                        self.target_positions[i] = max(-0.2, self.target_positions[i] - 0.01)
                    elif 'right' in name:
                        self.target_positions[i] = min(0.2, self.target_positions[i] + 0.01)

def main(args=None):
    rclpy.init(args=args)
    balancer = SimpleBalancer()
    
    try:
        rclpy.spin(balancer)
    except KeyboardInterrupt:
        pass
    finally:
        balancer.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Step 5: Building and Running the System

### 5.1 Build the Workspace

```bash
cd ~/physical_ai_ws
colcon build --packages-select humanoid_description humanoid_controllers ai_agents
source install/setup.bash
```

### 5.2 Run the Simulation

To run the simulation with the humanoid robot:

```bash
# Terminal 1: Start Gazebo simulation
cd ~/physical_ai_ws
source install/setup.bash
ros2 launch humanoid_description simulation_launch.py

# Terminal 2: Start the AI balancer
cd ~/physical_ai_ws
source install/setup.bash
ros2 run ai_agents simple_balancer
```

## Troubleshooting Common Issues

### 1. Gazebo Not Starting

If Gazebo fails to start, ensure you have the appropriate graphics drivers installed:

```bash
# Check if gazebo runs
gazebo --version

# If using a virtual machine, you may need to enable 3D acceleration
```

### 2. Robot Falls Through Floor

This usually indicates issues with collision detection or physics parameters:

- Ensure all links have proper collision elements in the URDF
- Check that inertial parameters are properly set
- Increase physics update rate in the world file

### 3. Joint Limits Exceeded

If joints are exceeding their limits, check the controller and ensure it respects the URDF joint limits.

## Conclusion

This implementation guide has walked you through creating a basic but functional Physical AI system. The system includes:

1. A ROS 2-based communication infrastructure
2. A humanoid robot model with appropriate URDF
3. A simulation environment in Gazebo
4. A basic AI agent for balance control

The system can be extended with more sophisticated controllers, additional sensors, and complex behaviors. The modular design allows for easy integration of additional functionality as needed for specific applications.