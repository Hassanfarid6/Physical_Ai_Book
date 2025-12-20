# Advanced Topics in Physical AI

## Overview

This chapter explores advanced concepts and techniques in Physical AI that build upon the foundational modules covered earlier. We'll examine cutting-edge approaches to sensorimotor learning, advanced control strategies, and implementation of more sophisticated AI behaviors for humanoid robots.

## Learning Objectives

After completing this chapter, you will be able to:
- Implement reinforcement learning algorithms for humanoid control
- Design advanced perception systems for real-world interaction
- Apply neural networks for sensorimotor learning
- Implement adaptive control mechanisms for changing environments
- Evaluate the simulation-to-reality transfer effectiveness

## 1. Reinforcement Learning for Humanoid Control

Reinforcement learning (RL) is particularly well-suited for Physical AI systems as it provides a framework for learning behaviors through environmental interaction. In this section, we'll implement RL algorithms specifically designed for humanoid robot control.

### 1.1 Deep Deterministic Policy Gradient (DDPG) for Continuous Control

DDPG is suitable for humanoid control because it handles continuous action spaces, which is essential for controlling joint positions, velocities, and forces.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import random
from collections import deque
import os

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        
        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        
        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.max_action = max_action
        self.replay_buffer = ReplayBuffer(max_size=int(1e6))
        
        self.discount = 0.99
        self.tau = 0.005
        self.policy_noise = 0.2 * max_action
        self.noise_clip = 0.5 * max_action
        self.policy_freq = 2

        self.total_it = 0

    def select_action(self, state, add_noise=True):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()
        
        if add_noise:
            noise = np.random.normal(0, scale=0.1, size=action.shape)
            action = action + noise
            action = np.clip(action, -self.max_action, self.max_action)
        
        return action

    def train(self, batch_size=100):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = self.replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        not_done = torch.FloatTensor(not_done).to(self.device)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            
            next_action = (
                self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor loss
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)

class ReplayBuffer:
    def __init__(self, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, 17))  # Adjust based on your state dimension
        self.action = np.zeros((max_size, 11))  # Adjust based on your action dimension
        self.next_state = np.zeros((max_size, 17))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )
```

### 1.2 Implementing the RL Agent in ROS 2

Now, let's create a ROS 2 node that integrates this DDPG agent:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Vector3
from std_msgs.msg import Float64MultiArray
from builtin_interfaces.msg import Duration
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import numpy as np
import torch
import os
import time

class RLBalanceController(Node):
    def __init__(self):
        super().__init__('rl_balance_controller')
        
        # Publisher for joint trajectory commands
        self.joint_trajectory_pub = self.create_publisher(
            JointTrajectory, '/joint_trajectory_controller/joint_trajectory', 10)
        
        # Subscribers for state estimation
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, '/imu', self.imu_callback, 10)
        
        # Timer for control loop
        self.timer = self.create_timer(0.05, self.control_loop)  # 20Hz control
        
        # Initialize RL agent
        self.state_dim = 17  # Joint positions, velocities, and IMU data
        self.action_dim = 11  # 11 joints for the humanoid
        self.max_action = 1.0  # Define maximum action value
        
        self.rl_agent = DDPGAgent(self.state_dim, self.action_dim, self.max_action)
        
        # Store current state
        self.current_joint_positions = {}
        self.current_joint_velocities = {}
        self.imu_data = None
        self.prev_action = np.zeros(self.action_dim)
        
        # Joint names
        self.joint_names = [
            'neck_joint', 'left_shoulder_joint', 'left_elbow_joint',
            'right_shoulder_joint', 'right_elbow_joint', 'left_hip_joint',
            'left_knee_joint', 'left_ankle_joint', 'right_hip_joint',
            'right_knee_joint', 'right_ankle_joint'
        ]
        
        # Normalization parameters (these would be computed from data)
        self.state_mean = np.zeros(self.state_dim)
        self.state_std = np.ones(self.state_dim)
        # Set std to a small value to avoid division by zero
        self.state_std[6:9] = 10.0  # For IMU readings which might be larger
        
        self.get_logger().info('RL Balance Controller initialized')

    def joint_state_callback(self, msg):
        """Update current joint positions and velocities"""
        for name, pos, vel in zip(msg.name, msg.position, msg.velocity):
            if name in self.joint_names:
                self.current_joint_positions[name] = pos
                self.current_joint_velocities[name] = vel

    def imu_callback(self, msg):
        """Update IMU data"""
        self.imu_data = {
            'orientation': msg.orientation,
            'angular_velocity': msg.angular_velocity,
            'linear_acceleration': msg.linear_acceleration
        }

    def get_state(self):
        """Construct state vector from sensor data"""
        if not self.current_joint_positions or self.imu_data is None:
            return None
        
        state = []
        
        # Add joint positions (first 11 values)
        for joint_name in self.joint_names:
            if joint_name in self.current_joint_positions:
                state.append(self.current_joint_positions[joint_name])
            else:
                state.append(0.0)  # Default if joint not found
        
        # Add joint velocities (next 11 values)
        for joint_name in self.joint_names:
            if joint_name in self.current_joint_velocities:
                state.append(self.current_joint_velocities[joint_name])
            else:
                state.append(0.0)  # Default if velocity not found
        
        # Add IMU data (orientation, angular velocity, linear acceleration)
        # Assuming we only use linear acceleration for balance
        state.append(self.imu_data['linear_acceleration'].x)
        state.append(self.imu_data['linear_acceleration'].y)
        state.append(self.imu_data['linear_acceleration'].z)
        
        return np.array(state[:self.state_dim])  # Ensure correct dimension

    def compute_reward(self, state, action):
        """Compute reward based on current state and action"""
        if state is None:
            return 0.0
        
        # Reward components:
        # 1. Stay upright (based on IMU linear acceleration)
        imu_z_acc = state[-1]  # Last value is linear acceleration in z
        upright_reward = np.clip(imu_z_acc / 9.81, 0.0, 1.0)  # Normalize by gravity
        
        # 2. Minimal joint velocities (smooth movement)
        joint_velocities = state[11:22]  # Assuming positions are first 11, velocities next 11
        smoothness_penalty = -0.01 * np.sum(np.abs(joint_velocities))
        
        # 3. Stay near neutral position
        target_positions = np.array([0.02, 0.0, 0.0, 0.0, 0.0, 0.1, -0.3, 0.1, 0.1, -0.3, 0.1])
        current_positions = state[:11]
        position_penalty = -0.1 * np.sum(np.abs(current_positions - target_positions))
        
        # 4. Penalty for excessive action magnitude
        action_penalty = -0.05 * np.sum(np.abs(action - self.prev_action))
        
        total_reward = upright_reward + smoothness_penalty + position_penalty + action_penalty
        self.prev_action = action.copy()
        
        return total_reward

    def control_loop(self):
        """Main control loop with RL"""
        state = self.get_state()
        
        if state is not None:
            # Normalize state
            normalized_state = (state - self.state_mean) / self.state_std
            
            # Get action from RL agent
            action = self.rl_agent.select_action(normalized_state)
            
            # Compute reward (in a real system, this would be computed after applying the action)
            reward = self.compute_reward(state, action)
            
            # For training, we would need to store (state, action, reward, next_state) in replay buffer
            # This is simplified for the demonstration
            
            # Convert action to joint positions
            # In a real system, action would represent desired changes or velocities
            target_positions = action  # This is a simplification
            
            # Create and publish trajectory
            trajectory_msg = JointTrajectory()
            trajectory_msg.joint_names = self.joint_names
            
            point = JointTrajectoryPoint()
            point.positions = target_positions.tolist()
            point.velocities = [0.0] * len(self.joint_names)
            point.time_from_start = Duration(sec=0, nanosec=50000000)  # 50ms to reach target
            
            trajectory_msg.points = [point]
            self.joint_trajectory_pub.publish(trajectory_msg)

def main(args=None):
    rclpy.init(args=args)
    controller = RLBalanceController()
    
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

## 2. Advanced Perception Systems

### 2.1 Sensor Fusion for State Estimation

For robust humanoid control, we need accurate state estimation combining multiple sensors. Here's an implementation of a sensor fusion system using a Kalman filter:

```python
import numpy as np
from scipy.linalg import block_diag

class HumanoidKalmanFilter:
    def __init__(self, dt=0.05):
        """
        Initialize Kalman filter for humanoid state estimation
        dt: Time step for prediction
        """
        self.dt = dt
        
        # State: [x, y, z, vx, vy, vz, roll, pitch, yaw, p, q, r]
        # where [x, y, z] is position, [vx, vy, vz] is velocity,
        # [roll, pitch, yaw] is orientation, [p, q, r] is angular velocity
        self.state_dim = 12
        self.observation_dim = 10  # Some combination of joint angles, IMU, etc.
        
        # State vector [position, velocity, orientation, angular_velocity]
        self.x = np.zeros(self.state_dim)
        
        # State transition matrix (simplified linear model)
        self.F = np.eye(self.state_dim)
        # Position updates from velocity
        self.F[0:3, 3:6] = np.eye(3) * dt
        # Orientation updates from angular velocity
        self.F[6:9, 9:12] = np.eye(3) * dt
        
        # Control input matrix (set to zero as we don't have direct control inputs)
        self.B = np.zeros((self.state_dim, 1))
        
        # Observation matrix (maps state to what we can observe)
        # Simplified: we observe position, orientation, and some joint angles
        self.H = np.zeros((self.observation_dim, self.state_dim))
        # Observe position
        self.H[0:3, 0:3] = np.eye(3)
        # Observe orientation
        self.H[3:6, 6:9] = np.eye(3)
        # Observe angular velocity
        self.H[6:9, 9:12] = np.eye(3)
        # Additional joint observations
        self.H[9, 0] = 1.0  # Example: observe first joint
        self.H[9, 1] = 1.0  # Example: observe second joint
        
        # Process noise covariance
        self.Q = np.eye(self.state_dim) * 0.1
        
        # Observation noise covariance
        self.R = np.eye(self.observation_dim) * 0.5
        
        # Error covariance matrix
        self.P = np.eye(self.state_dim) * 1.0
        
    def predict(self, u=None):
        """
        Predict next state based on current state and control input
        u: Control input (optional)
        """
        if u is not None:
            self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        else:
            self.x = np.dot(self.F, self.x)
            
        # Update error covariance
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        
    def update(self, z):
        """
        Update state estimate based on observation
        z: Observation vector
        """
        # Compute Kalman gain
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        
        # Update state estimate
        y = z - np.dot(self.H, self.x)  # Innovation
        self.x = self.x + np.dot(K, y)
        
        # Update error covariance
        I = np.eye(self.state_dim)
        self.P = np.dot((I - np.dot(K, self.H)), self.P)

# Integration with ROS 2
class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion_node')
        
        # Subscribers for different sensors
        self.imu_sub = self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        
        # Publisher for fused state
        self.state_pub = self.create_publisher(JointState, '/fused_state', 10)
        
        # Initialize Kalman filter
        self.kf = HumanoidKalmanFilter(dt=0.05)
        
        # Store sensor data
        self.imu_data = None
        self.joint_data = None
        
        # Timer for fusion loop
        self.timer = self.create_timer(0.05, self.fusion_loop)
        
    def imu_callback(self, msg):
        self.imu_data = msg
        
    def joint_state_callback(self, msg):
        self.joint_data = msg
        
    def fusion_loop(self):
        if self.imu_data is not None and self.joint_data is not None:
            # Create observation vector
            z = np.zeros(10)  # Example observation vector
            
            # Fill with relevant data
            # This is a simplified example - in practice, you would need to properly
            # map between the sensor data and the observation vector
            z[0:3] = [self.imu_data.linear_acceleration.x, 
                     self.imu_data.linear_acceleration.y, 
                     self.imu_data.linear_acceleration.z]
            
            # Predict step
            self.kf.predict()
            
            # Update step
            self.kf.update(z)
            
            # Publish fused state
            state_msg = JointState()
            state_msg.name = ['estimated_' + name for name in self.joint_data.name]
            state_msg.position = self.kf.x[0:len(state_msg.name)].tolist()  # Simplified mapping
            state_msg.header.stamp = self.get_clock().now().to_msg()
            
            self.state_pub.publish(state_msg)
```

## 3. Neural Networks for Sensorimotor Learning

### 3.1 Convolutional Networks for Visual Processing

For humanoid robots with vision capabilities, convolutional neural networks (CNNs) play a crucial role in processing visual input:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VisualPerceptionNet(nn.Module):
    def __init__(self, num_classes=10, input_channels=3):
        super(VisualPerceptionNet, self).__init__()
        
        # Feature extraction layers
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Fourth conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Calculate the size of the flattened features after conv layers
        # Assuming input is 224x224, after 4 pooling layers: 224/(2^4) = 14
        self.flattened_size = 256 * 14 * 14
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.flattened_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

# Integration with ROS 2 for vision processing
class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')
        
        # Subscriber for camera images
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        
        # Publisher for processed results
        self.result_pub = self.create_publisher(
            String, '/vision_result', 10)
        
        # Initialize the neural network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = VisualPerceptionNet(num_classes=10).to(self.device)
        self.net.eval()  # Set to evaluation mode
        
        # Load pre-trained weights if available
        # self.net.load_state_dict(torch.load('path_to_pretrained_model.pth'))
        
    def image_callback(self, msg):
        # Convert ROS image to PyTorch tensor
        # This requires cv_bridge to convert ROS Image to OpenCV format
        try:
            import cv2
            from cv_bridge import CvBridge
            
            bridge = CvBridge()
            cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Preprocess image
            image_tensor = self.preprocess_image(cv_image)
            
            # Run inference
            with torch.no_grad():
                output = self.net(image_tensor)
                predicted_class = torch.argmax(output, dim=1).item()
                
            # Publish result
            result_msg = String()
            result_msg.data = f"Detected class: {predicted_class}"
            self.result_pub.publish(result_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')
    
    def preprocess_image(self, image):
        # Resize image to model input size (example: 224x224)
        import cv2
        import numpy as np
        
        image_resized = cv2.resize(image, (224, 224))
        image_normalized = image_resized / 255.0  # Normalize to [0, 1]
        image_tensor = torch.FloatTensor(image_normalized).permute(2, 0, 1).unsqueeze(0)  # CHW format, add batch dim
        image_tensor = image_tensor.to(self.device)
        
        return image_tensor
```

## 4. Adaptive Control for Changing Environments

Humanoid robots must adapt to changing environments and conditions. Model-reference adaptive control (MRAC) is a technique for achieving this:

```python
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

class AdaptiveController:
    def __init__(self, reference_model_params, initial_adaptive_params, learning_rate=0.01):
        """
        Initialize adaptive controller
        reference_model_params: Parameters for the desired reference model
        initial_adaptive_params: Initial parameters for the adaptive controller
        learning_rate: Learning rate for parameter adaptation
        """
        self.ref_model_params = reference_model_params
        self.theta = initial_adaptive_params  # Adaptive parameters
        self.gamma = learning_rate  # Learning rate
        
        # For parameter projection (to keep parameters bounded)
        self.theta_min = -10 * np.ones_like(initial_adaptive_params)
        self.theta_max = 10 * np.ones_like(initial_adaptive_params)
        
    def control(self, state, reference_state, error):
        """
        Compute control action based on current state, reference, and error
        """
        # Compute control using adaptive parameters
        u = np.dot(self.theta, state)
        
        # Compute reference model response
        ref_dot = -self.ref_model_params * reference_state
        
        # Compute parameter update (gradient descent on tracking error)
        # This is a simplified version - in practice, you'd use more sophisticated adaptation laws
        phi = state  # Regressor vector
        error_dot = ref_dot - state  # Approximation of error derivative
        adaptation_law = -self.gamma * np.outer(error_dot, phi) * error
        
        # Update adaptive parameters
        self.theta += adaptation_law
        
        # Project parameters to stay within bounds
        self.theta = np.clip(self.theta, self.theta_min, self.theta_max)
        
        return u

# Example application to humanoid joint control
class AdaptiveJointController:
    def __init__(self, joint_name, learning_rate=0.01):
        self.joint_name = joint_name
        
        # Reference model: desired closed-loop dynamics
        # For a second-order system: s^2 + 2*zeta*wn*s + wn^2
        self.wn = 5.0  # Natural frequency
        self.zeta = 0.7  # Damping ratio
        self.ref_model_params = np.array([2*self.zeta*self.wn, self.wn**2])
        
        # Initial adaptive parameters (for controller)
        self.theta = np.array([1.0, 0.5])  # Initial proportional and derivative gains
        
        self.gamma = learning_rate
        self.error_history = []
        
    def compute_control(self, current_pos, current_vel, desired_pos, desired_vel, dt=0.01):
        """
        Compute adaptive control for a single joint
        """
        # State vector [position error, velocity error]
        pos_error = desired_pos - current_pos
        vel_error = desired_vel - current_vel
        
        state = np.array([pos_error, vel_error])
        
        # Reference model response
        ref_accel = -self.ref_model_params[0]*vel_error - self.ref_model_params[1]*pos_error
        
        # Compute control action
        u = np.dot(self.theta, state)
        
        # Compute adaptation law
        # Simplified gradient update
        dtheta = -self.gamma * np.outer(state, state) @ np.array([vel_error, ref_accel])
        
        # Update adaptive parameters
        self.theta += dtheta * dt
        
        # Store error history for monitoring
        self.error_history.append(abs(pos_error))
        
        return u
```

## 5. Simulation-to-Reality Transfer Techniques

Transferring control policies from simulation to reality is challenging due to the "reality gap". Domain randomization is one technique to address this:

```python
class DomainRandomization:
    def __init__(self, base_params):
        """
        Initialize domain randomization with base parameters
        base_params: Dictionary of base simulation parameters
        """
        self.base_params = base_params
        self.randomized_params = base_params.copy()
        
    def randomize_dynamics(self):
        """Randomize physical parameters like mass, friction, etc."""
        # Randomize mass of each link (±20% variation)
        for link_name in ['torso', 'head', 'upper_arm', 'lower_arm', 'thigh', 'shin', 'foot']:
            if f'{link_name}_mass' in self.randomized_params:
                base_mass = self.base_params[f'{link_name}_mass']
                variation = np.random.uniform(0.8, 1.2)  # ±20% variation
                self.randomized_params[f'{link_name}_mass'] = base_mass * variation
        
        # Randomize friction coefficients (±50% variation)
        for param_name, base_value in self.base_params.items():
            if 'friction' in param_name:
                variation = np.random.uniform(0.5, 1.5)  # ±50% variation
                self.randomized_params[param_name] = base_value * variation
    
    def randomize_sensors(self):
        """Randomize sensor parameters like noise, delay"""
        # Randomize IMU noise parameters
        self.randomized_params['imu_noise_linear_acceleration'] = \
            np.random.uniform(0.01, 0.1)  # Scale noise parameter
        self.randomized_params['imu_noise_angular_velocity'] = \
            np.random.uniform(0.001, 0.01)
    
    def randomize_actuators(self):
        """Randomize actuator parameters"""
        # Randomize motor torque limits
        for joint_name in ['hip', 'knee', 'ankle', 'shoulder', 'elbow']:
            if f'{joint_name}_torque_limit' in self.randomized_params:
                base_limit = self.base_params[f'{joint_name}_torque_limit']
                variation = np.random.uniform(0.8, 1.2)
                self.randomized_params[f'{joint_name}_torque_limit'] = base_limit * variation

class Sim2RealTransfer:
    def __init__(self, base_params):
        self.domain_rand = DomainRandomization(base_params)
        self.episode_count = 0
        
    def setup_randomized_episode(self):
        """Setup simulation with randomized parameters"""
        # Randomize parameters for this episode
        self.domain_rand.randomize_dynamics()
        self.domain_rand.randomize_sensors()
        self.domain_rand.randomize_actuators()
        
        # Apply randomized parameters to simulation (implementation depends on your sim)
        # self.apply_params_to_simulation(self.domain_rand.randomized_params)
        
        self.episode_count += 1
        return self.domain_rand.randomized_params
```

## 6. Evaluation Metrics for Physical AI Systems

Evaluating Physical AI systems requires metrics that capture both performance and robustness:

```python
import numpy as np

class PhysicalAIEvaluator:
    def __init__(self):
        self.metrics = {}
        
    def evaluate_balance(self, imu_data, duration=10.0):
        """
        Evaluate the robot's ability to maintain balance
        """
        # Calculate stability metrics from IMU data over time
        roll_list = []
        pitch_list = []
        vertical_accel_list = []
        
        # In a real implementation, you'd collect IMU data over the duration
        # For now, we'll use simulated data
        for i in range(int(duration / 0.01)):  # 100Hz sampling for 10s
            # Simulated IMU data collection would go here
            roll_list.append(np.random.normal(0, 0.05))  # Small variations
            pitch_list.append(np.random.normal(0, 0.05))
            vertical_accel_list.append(np.random.normal(9.81, 0.2))  # Around gravity
        
        # Calculate metrics
        avg_roll = np.mean(np.abs(roll_list))
        avg_pitch = np.mean(np.abs(pitch_list))
        stability_score = 1.0 - (avg_roll + avg_pitch) / 2.0  # Higher is more stable
        
        std_vertical_accel = np.std(vertical_accel_list)
        balance_consistency = 1.0 / (1.0 + std_vertical_accel)  # Lower std = more consistent
        
        self.metrics['balance_stability'] = stability_score
        self.metrics['balance_consistency'] = balance_consistency
        
        return stability_score, balance_consistency
    
    def evaluate_movement_efficiency(self, joint_positions, joint_velocities, time_taken):
        """
        Evaluate the efficiency of movements
        """
        # Calculate joint movement smoothness (lower jerk)
        total_energy = 0
        for velocities in joint_velocities:
            # Simplified energy calculation based on joint velocities
            total_energy += np.sum(np.abs(velocities))
        
        # Efficiency = task completion / energy expenditure
        movement_efficiency = time_taken / (total_energy + 1e-6)  # Add small value to prevent division by zero
        
        self.metrics['movement_efficiency'] = movement_efficiency
        return movement_efficiency
    
    def evaluate_adaptability(self, disturbance_response_times):
        """
        Evaluate how well the system adapts to disturbances
        """
        # Calculate mean and std of response times to disturbances
        mean_response = np.mean(disturbance_response_times)
        std_response = np.std(disturbance_response_times)
        
        # Adaptability score: faster and more consistent responses get higher scores
        adaptability_score = 1.0 / (mean_response + std_response + 1e-6)
        
        self.metrics['adaptability_score'] = adaptability_score
        return adaptability_score
    
    def generate_report(self):
        """
        Generate a comprehensive evaluation report
        """
        report = "Physical AI System Evaluation Report\n"
        report += "=" * 40 + "\n"
        
        for metric_name, value in self.metrics.items():
            report += f"{metric_name}: {value:.4f}\n"
        
        return report
```

## Summary

This chapter covered advanced topics in Physical AI, including:

1. Reinforcement learning techniques for humanoid control, specifically DDPG for continuous action spaces
2. Advanced perception systems with sensor fusion using Kalman filters
3. Neural network implementations for visual processing
4. Adaptive control strategies for changing environments
5. Simulation-to-reality transfer techniques like domain randomization
6. Evaluation metrics for Physical AI systems

These advanced techniques build upon the foundational modules to create more sophisticated, robust, and adaptive Physical AI systems. The implementation examples provide practical starting points for developing embodied AI systems that can effectively interact with the physical world.

## Exercises

1. Enhance the DDPG implementation to include more sophisticated reward shaping for complex humanoid behaviors.
2. Implement a particle filter for state estimation as an alternative to the Kalman filter.
3. Add depth camera processing to the perception system using CNNs for 3D object detection.
4. Develop a more advanced domain randomization strategy that includes visual appearance variations.
5. Create a comprehensive evaluation framework that tests the humanoid robot on a variety of tasks and environments.