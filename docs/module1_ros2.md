# Module 1: The Robotic Nervous System (ROS 2)

## Overview

In this module, we'll establish the foundational communication backbone for our humanoid robot using ROS 2 (Robot Operating System 2). ROS 2 serves as the "nervous system" of our robot, coordinating communication between sensors, actuators, AI agents, and simulation environments.

This module will cover the essential components of ROS 2 architecture, the development of Python-based agents using rclpy, and the creation of robot models using URDF (Unified Robot Description Format). By the end of this module, you'll have a functional ROS 2 workspace with a humanoid robot model ready for control.

## Learning Objectives

By the end of this module, you will be able to:
- Understand and implement ROS 2 architecture concepts (Nodes, Topics, Services)
- Develop Python-based ROS 2 nodes using rclpy
- Create and validate URDF models for humanoid robots
- Establish communication patterns between AI logic and robot actuators
- Implement motor control and joint actuation systems

## 1. ROS 2 Architecture Fundamentals

### Nodes, Topics, and Services

ROS 2 uses a distributed architecture where different components of a robot system run as separate processes called nodes. These nodes communicate with each other using:

- **Topics**: Asynchronous, one-way communication channels for continuous data streams
- **Services**: Synchronous, request-response communication for specific tasks
- **Actions**: Asynchronous, goal-oriented communication for long-running tasks

```
┌─────────────┐    publish    ┌─────────────┐
│  Publisher  │ ─────────────▶ │   Topic     │
│    Node     │               │   /joint_   │
└─────────────┘               │   states    │
                              └─────────────┘
                              ┌─────────────┐
                              │  Subscriber │
                              │    Node     │
                              └─────────────┘
```

### Implementation Example

Let's look at a basic publisher-subscriber pattern in Python:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

class JointStatePublisher(Node):
    def __init__(self):
        super().__init__('joint_state_publisher')
        self.publisher = self.create_publisher(JointState, '/joint_states', 10)
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        msg = JointState()
        msg.name = ['hip_joint', 'knee_joint', 'ankle_joint']
        msg.position = [0.0, 0.0, 0.0]  # radians for revolute joints
        self.publisher.publish(msg)
        self.get_logger().info('Published joint states')

def main(args=None):
    rclpy.init(args=args)
    joint_state_publisher = JointStatePublisher()
    rclpy.spin(joint_state_publisher)
    joint_state_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 2. rclpy-based Python Agents

### Creating Python ROS 2 Nodes

Python agents in ROS 2 are implemented as nodes using the rclpy library, which provides Python bindings for the ROS 2 client library. These agents can process sensor data, make decisions, and send commands to actuators.

### Example: AI Decision Node

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

class AIDecisionNode(Node):
    def __init__(self):
        super().__init__('ai_decision_node')
        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.listener_callback,
            10)
        self.publisher = self.create_publisher(Float64MultiArray, '/target_joints', 10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        # Simple decision-making based on current joint states
        current_positions = list(msg.position)
        
        # Example AI logic: maintain upright position
        target_positions = self.calculate_balance_targets(current_positions)
        
        # Publish target positions
        target_msg = Float64MultiArray()
        target_msg.data = target_positions
        self.publisher.publish(target_msg)

    def calculate_balance_targets(self, current_positions):
        # Simplified balance algorithm
        # In practice, this would use more sophisticated control methods
        return [pos * 0.95 for pos in current_positions]  # Gentle correction

def main(args=None):
    rclpy.init(args=args)
    ai_decision_node = AIDecisionNode()
    rclpy.spin(ai_decision_node)
    ai_decision_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 3. Communication Between AI Logic and Robot Actuators

Effective communication between AI decision-making components and robot actuators is crucial for successful Physical AI systems. This communication involves:

- **Sensor Data Processing**: Converting raw sensor readings into meaningful state information
- **Decision Making**: Using AI algorithms to determine appropriate actions
- **Actuator Commands**: Translating decisions into specific hardware commands
- **Feedback Integration**: Incorporating sensor feedback to adjust future decisions

### Communication Pattern Example

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from control_msgs.msg import JointTrajectoryControllerState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')
        
        # Subscriber for sensor feedback
        self.feedback_sub = self.create_subscription(
            JointTrajectoryControllerState,
            '/controller_state',
            self.feedback_callback,
            10)
            
        # Publisher for actuator commands
        self.command_pub = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory',
            10)
    
    def feedback_callback(self, msg):
        # Process feedback from actuators
        current_positions = msg.actual.positions
        current_velocities = msg.actual.velocities
        
        # Make decisions based on feedback
        target_trajectory = self.calculate_trajectory(
            current_positions, current_velocities)
            
        # Send commands to actuators
        self.command_pub.publish(target_trajectory)

    def calculate_trajectory(self, current_positions, current_velocities):
        # Implement trajectory planning logic
        trajectory = JointTrajectory()
        trajectory.joint_names = ['joint1', 'joint2', 'joint3']  # Example joint names
        
        point = JointTrajectoryPoint()
        # Example: move to neutral position
        point.positions = [0.0, 0.0, 0.0]
        point.velocities = [0.0, 0.0, 0.0]
        point.time_from_start.sec = 1  # Reach position in 1 second
        
        trajectory.points.append(point)
        return trajectory
```

## 4. URDF Design for Humanoid Robots

URDF (Unified Robot Description Format) is an XML format used to describe robot models in ROS. For humanoid robots, URDF files define the physical structure, including links (rigid parts), joints (connections), and visual/collision properties.

### Basic Humanoid URDF Structure

```xml
<?xml version="1.0"?>
<robot name="simple_humanoid">
  
  <!-- Base Link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.2 0.1 0.1"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.2 0.1 0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Hip joint and leg link -->
  <joint name="hip_joint" type="revolute">
    <parent link="base_link"/>
    <child link="left_leg"/>
    <origin xyz="0 -0.05 -0.15"/>
    <axis xyz="1 0 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  </joint>

  <link name="left_leg">
    <visual>
      <geometry>
        <box size="0.05 0.05 0.3"/>
      </geometry>
      <material name="white">
        <color rgba="1 1 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.05 0.05 0.3"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.005" ixy="0.0" ixz="0.0" iyy="0.005" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <!-- More joints and links would follow for a complete humanoid -->
  
</robot>
```

## 5. Motor Control and Joint Actuation

Motor control systems translate high-level commands into precise joint movements. This involves:
- Joint trajectory planning
- PID (Proportional-Integral-Derivative) control
- Safety and constraint management
- Feedback integration

### Joint Control Node Example

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
import numpy as np

class JointController(Node):
    def __init__(self):
        super().__init__('joint_controller')
        
        # PID controller parameters
        self.kp = 10.0  # Proportional gain
        self.ki = 0.5   # Integral gain
        self.kd = 0.1   # Derivative gain
        
        # Initialize errors for PID
        self.prev_error = 0.0
        self.integral_error = 0.0
        
        # Subscribers and publishers
        self.target_sub = self.create_subscription(
            Float64MultiArray,
            '/target_joints',
            self.target_callback,
            10)
        
        self.state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.state_callback,
            10)
        
        self.command_pub = self.create_publisher(
            Float64MultiArray,
            '/joint_commands',
            10)
    
    def target_callback(self, msg):
        # Store target joint positions
        self.target_positions = list(msg.data)
    
    def state_callback(self, msg):
        # Get current joint positions
        current_positions = list(msg.position)
        
        if hasattr(self, 'target_positions'):
            # Calculate control commands using PID
            commands = Float64MultiArray()
            control_outputs = []
            
            for i in range(len(current_positions)):
                if i < len(self.target_positions):
                    error = self.target_positions[i] - current_positions[i]
                    
                    # PID calculation
                    self.integral_error += error
                    derivative = error - self.prev_error
                    
                    control_output = (
                        self.kp * error + 
                        self.ki * self.integral_error + 
                        self.kd * derivative
                    )
                    
                    control_outputs.append(control_output)
                    self.prev_error = error
                else:
                    control_outputs.append(0.0)  # No target for this joint
            
            commands.data = control_outputs
            self.command_pub.publish(commands)

def main(args=None):
    rclpy.init(args=args)
    joint_controller = JointController()
    rclpy.spin(joint_controller)
    joint_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Summary

In this module, we've established the foundational components of our Physical AI system using ROS 2 as the robotic nervous system. We've covered:

- ROS 2 architecture concepts including nodes, topics, and services
- Implementation of Python-based agents using rclpy
- Communication patterns between AI logic and robot actuators
- URDF design for humanoid robots
- Motor control and joint actuation systems

These components will serve as the backbone for more advanced functionality in subsequent modules, where we'll integrate simulation environments and AI agents to create a complete embodied AI system.

## Exercises

1. Create a simple ROS 2 workspace with a publisher and subscriber node.
2. Implement a basic URDF model for a simple humanoid robot with at least 6 joints.
3. Develop a PID controller for a single joint and test its response to different target positions.
4. Create a simple AI agent that makes decisions based on joint position feedback to maintain an upright posture.