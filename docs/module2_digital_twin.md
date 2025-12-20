# Module 2: The Digital Twin (Gazebo & Unity) - Physics Simulation and Environment Modeling

## Overview

This module focuses on creating high-fidelity simulation environments using Gazebo and Unity as digital twins for our humanoid robot. A digital twin represents a virtual counterpart of the physical system that enables comprehensive testing, validation, and training before deployment to real hardware. We'll implement sophisticated physics simulation including gravity, collisions, and friction, create Gazebo world environments for humanoid simulation, utilize Unity for high-fidelity rendering and human-robot interaction, and develop comprehensive sensor simulation for LiDAR, depth cameras, and IMUs with proper synchronization to the ROS 2 framework.

## Learning Objectives

By the end of this module, you will be able to:
- Implement physics simulation with gravity, collisions, and friction
- Design environments in Gazebo for robot testing
- Create human-robot interaction interfaces using Unity
- Simulate various sensors (LiDAR, depth cameras, IMUs)
- Integrate simulation with ROS 2 for seamless operation

## 1. Physics Simulation in Gazebo

Gazebo provides realistic physics simulation that includes gravity, collisions, friction, and other physical properties. This simulation allows us to test robot control algorithms in a safe, cost-effective environment before deploying to real hardware.

### Physics Engine Configuration

Gazebo uses the ODE (Open Dynamics Engine) physics engine by default, though it also supports Bullet and Simbody. The physics properties are configured in the world file:

```xml
<sdf version="1.6">
  <world name="default">
    <!-- Physics parameters -->
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
      <gravity>0 0 -9.8</gravity>
    </physics>
    
    <!-- Your robot and environment models go here -->
    
  </world>
</sdf>
```

### Collision and Friction

For realistic simulation, each link in our robot model needs appropriate collision and friction properties:

```xml
<link name="foot">
  <collision name="collision">
    <geometry>
      <box>
        <size>0.1 0.08 0.02</size>
      </box>
    </geometry>
    <surface>
      <friction>
        <ode>
          <mu>0.8</mu>
          <mu2>0.8</mu2>
        </ode>
      </friction>
      <contact>
        <ode>
          <kp>1e+6</kp>
          <kd>10</kd>
          <max_vel>100</max_vel>
          <min_depth>0.001</min_depth>
        </ode>
      </contact>
    </surface>
  </collision>
  <!-- Additional visual and inertial elements -->
</link>
```

## 2. Environment Design in Gazebo

Creating realistic environments is crucial for training embodied AI systems. Gazebo allows us to design complex environments with various objects, textures, and lighting conditions.

### Creating a Room Environment

```xml
<!-- Room with walls -->
<model name="room">
  <pose>0 0 2.5 0 0 0</pose>
  <link name="room_wall_north">
    <pose>0 2.5 2.5 0 0 0</pose>
    <collision name="collision">
      <geometry>
        <box>
          <size>5 0.2 5</size>
        </box>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <box>
          <size>5 0.2 5</size>
        </box>
      </geometry>
      <material>
        <ambient>0.7 0.7 0.7 1</ambient>
        <diffuse>0.7 0.7 0.7 1</diffuse>
      </material>
    </visual>
    <inertial>
      <mass>100</mass>
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
  <!-- Additional walls, floor, ceiling would follow -->
</model>

<!-- Add objects for interaction -->
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
  </link>
</model>
```

### Adding Physics Properties to Objects

```xml
<!-- Adding a ball that can be manipulated -->
<model name="ball">
  <pose>0.5 0 1.0 0 0 0</pose>
  <link name="ball_link">
    <collision name="collision">
      <geometry>
        <sphere>
          <radius>0.1</radius>
        </sphere>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <sphere>
          <radius>0.1</radius>
        </sphere>
      </geometry>
      <material>
        <ambient>1 0 0 1</ambient>
        <diffuse>1 0 0 1</diffuse>
      </material>
    </visual>
    <inertial>
      <mass>0.5</mass>
      <inertia>
        <ixx>0.005</ixx>
        <ixy>0</ixy>
        <ixz>0</ixz>
        <iyy>0.005</iyy>
        <iyz>0</iyz>
        <izz>0.005</izz>
      </inertia>
    </inertial>
  </link>
</model>
```

## 3. Human-Robot Interaction Using Unity

Unity provides a high-fidelity visualization platform that can be integrated with ROS 2 for human-robot interaction. We'll use the ROS-TCP-Connector to enable communication between ROS 2 and Unity.

### Setting up Unity for ROS Integration

1. Install the ROS-TCP-Connector Unity package
2. Configure the Unity scene with appropriate lighting and environment
3. Implement ROS communication scripts

### Unity C# Script for ROS Communication

```csharp
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Ros2UnityEx;

public class RobotController : MonoBehaviour
{
    private ROS2UnityConnection ros2UConn;
    private string robotTopic = "/unity_robot_pose";
    
    // Robot joint transforms
    public Transform head;
    public Transform leftArm;
    public Transform rightArm;
    public Transform leftLeg;
    public Transform rightLeg;
    
    void Start()
    {
        ros2UConn = ROS2UnityConnection.instance;
        ros2UConn.InitWithDefaultSettings();
        
        // Subscribe to robot state topic
        ros2UConn.ros2Service.Subscribe<sensor_msgs.JointState>(
            robotTopic, JointStateCallback);
    }
    
    void JointStateCallback(sensor_msgs.JointState msg)
    {
        // Update Unity model based on joint states from ROS
        for (int i = 0; i < msg.name.Count; i++)
        {
            string jointName = msg.name[i];
            float jointPosition = (float)msg.position[i];
            
            switch (jointName)
            {
                case "head_joint":
                    if (head != null) head.localRotation = 
                        Quaternion.Euler(0, 0, jointPosition * Mathf.Rad2Deg);
                    break;
                    
                case "left_arm_joint":
                    if (leftArm != null) leftArm.localRotation = 
                        Quaternion.Euler(0, jointPosition * Mathf.Rad2Deg, 0);
                    break;
                    
                // Additional joint mappings...
            }
        }
    }
    
    void OnDestroy()
    {
        if (ros2UConn != null)
        {
            ros2UConn.ros2Service.Unsubscribe<sensor_msgs.JointState>(robotTopic);
        }
    }
}
```

### Implementing Human Interaction

```csharp
using UnityEngine;
using Ros2UnityEx;

public class HumanInteraction : MonoBehaviour
{
    private ROS2UnityConnection ros2UConn;
    private string interactionTopic = "/human_interaction";
    
    void Start()
    {
        ros2UConn = ROS2UnityConnection.instance;
        ros2UConn.InitWithDefaultSettings();
    }
    
    void Update()
    {
        // Detect if user clicks on an object
        if (Input.GetMouseButtonDown(0))
        {
            Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);
            RaycastHit hit;
            
            if (Physics.Raycast(ray, out hit))
            {
                // Send interaction message to ROS
                std_msgs.String interactionMsg = new std_msgs.String();
                interactionMsg.data = "User clicked on: " + hit.collider.name;
                
                ros2UConn.ros2Service.Publish<std_msgs.String>(
                    interactionTopic, interactionMsg);
            }
        }
    }
}
```

## 4. Sensor Simulation

Accurate sensor simulation is crucial for the simulation-to-reality transfer. We'll implement simulation for LiDAR, depth cameras, and IMUs.

### LiDAR Sensor Simulation in Gazebo

```xml
<link name="lidar_link">
  <collision>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <cylinder radius="0.02" length="0.04"/>
    </geometry>
  </collision>
  <visual>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <cylinder radius="0.02" length="0.04"/>
    </geometry>
  </visual>
  <inertial>
    <mass value="0.1"/>
    <inertia ixx="1e-6" ixy="0" ixz="0" iyy="1e-6" iyz="0" izz="1e-6"/>
  </inertial>
  
  <!-- LiDAR sensor plugin -->
  <sensor name="lidar" type="ray">
    <pose>0 0 0 0 0 0</pose>
    <visualize>true</visualize>
    <update_rate>10</update_rate>
    <ray>
      <scan>
        <horizontal>
          <samples>720</samples>
          <resolution>1</resolution>
          <min_angle>-1.570796</min_angle>  <!-- -90 degrees -->
          <max_angle>1.570796</max_angle>   <!-- 90 degrees -->
        </horizontal>
      </scan>
      <range>
        <min>0.1</min>
        <max>10.0</max>
        <resolution>0.01</resolution>
      </range>
    </ray>
    <plugin name="lidar_controller" filename="libgazebo_ros_ray_sensor.so">
      <ros>
        <argument>~/out:=scan</argument>
      </ros>
      <output_type>sensor_msgs/LaserScan</output_type>
    </plugin>
  </sensor>
</link>
```

### Depth Camera Simulation in Gazebo

```xml
<link name="camera_link">
  <collision>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <box size="0.05 0.05 0.05"/>
    </geometry>
  </collision>
  <visual>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <box size="0.05 0.05 0.05"/>
    </geometry>
  </visual>
  <inertial>
    <mass value="0.01"/>
    <inertia ixx="1e-6" ixy="0" ixz="0" iyy="1e-6" iyz="0" izz="1e-6"/>
  </inertial>
  
  <!-- Depth camera sensor -->
  <sensor name="depth_camera" type="depth">
    <visualize>true</visualize>
    <update_rate>30</update_rate>
    <camera name="head">
      <horizontal_fov>1.047</horizontal_fov> <!-- 60 degrees -->
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>10</far>
      </clip>
    </camera>
    <plugin name="camera_controller" filename="libgazebo_ros_openni_kinect.so">
      <alwaysOn>true</alwaysOn>
      <updateRate>30.0</updateRate>
      <cameraName>camera</cameraName>
      <imageTopicName>rgb/image_raw</imageTopicName>
      <depthImageTopicName>depth/image_raw</depthImageTopicName>
      <pointCloudTopicName>depth/points</pointCloudTopicName>
      <cameraInfoTopicName>rgb/camera_info</cameraInfoTopicName>
      <depthImageCameraInfoTopicName>depth/camera_info</depthImageCameraInfoTopicName>
      <frameName>camera_depth_optical_frame</frameName>
      <baseline>0.1</baseline>
      <distortion_k1>0.0</distortion_k1>
      <distortion_k2>0.0</distortion_k2>
      <distortion_k3>0.0</distortion_k3>
      <distortion_t1>0.0</distortion_t1>
      <distortion_t2>0.0</distortion_t2>
      <pointCloudCutoff>0.5</pointCloudCutoff>
      <pointCloudCutoffMax>3.0</pointCloudCutoffMax>
      <CxPrime>0.0</CxPrime>
      <Cx>0.0</Cx>
      <Cy>0.0</Cy>
      <focalLength>0.0</focalLength>
      <hackBaseline>0.07</hackBaseline>
    </plugin>
  </sensor>
</link>
```

### IMU Sensor Simulation in Gazebo

```xml
<link name="imu_link">
  <collision>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <box size="0.01 0.01 0.01"/>
    </geometry>
  </collision>
  <visual>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <box size="0.01 0.01 0.01"/>
    </geometry>
  </visual>
  <inertial>
    <mass value="0.001"/>
    <inertia ixx="1e-9" ixy="0" ixz="0" iyy="1e-9" iyz="0" izz="1e-9"/>
  </inertial>
  
  <!-- IMU sensor -->
  <sensor name="imu_sensor" type="imu">
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <visualize>false</visualize>
    <imu>
      <angular_velocity>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
          </noise>
        </z>
      </angular_velocity>
      <linear_acceleration>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
          </noise>
        </z>
      </linear_acceleration>
    </imu>
    <plugin name="imu_plugin" filename="libgazebo_ros_imu_sensor.so">
      <ros>
        <argument>~/out:=imu</argument>
      </ros>
      <update_rate>100</update_rate>
      <topic>~/imu</topic>
      <body_name>imu_link</body_name>
      <frame_name>imu_link</frame_name>
      <initial_orientation_as_reference>false</initial_orientation_as_reference>
    </plugin>
  </sensor>
</link>
```

## 5. ROS 2 Integration with Simulation

To make the simulation part of the complete system, we need to properly integrate it with ROS 2:

### Launch File for Simulation

```xml
<launch>
  <!-- Start Gazebo with the world file -->
  <include file="$(find-pkg-share gazebo_ros)/launch/gzserver.launch.py">
    <arg name="world" value="$(find-pkg-share my_robot_description)/worlds/my_world.sdf"/>
  </include>
  
  <include file="$(find-pkg-share gazebo_ros)/launch/gzclient.launch.py"/>

  <!-- Spawn the robot model -->
  <node pkg="robot_state_publisher" exec="robot_state_publisher" name="robot_state_publisher">
    <param name="robot_description" value="$(var robot_description)"/>
  </node>

  <!-- Publish static transforms -->
  <node pkg="tf2_ros" exec="static_transform_publisher" name="odom_to_base_footprint">
    <arg>0 0 0 0 0 0 odom base_footprint</arg>
  </node>
  
  <!-- Start the controller manager -->
  <node pkg="controller_manager" exec="ros2_control_node" name="ros2_control_node">
    <param name="robot_description" value="$(var robot_description)"/>
  </node>
  
  <!-- Load and start controllers -->
  <node pkg="controller_manager" exec="spawner" name="joint_state_broadcaster_spawner" args="joint_state_broadcaster"/>
  <node pkg="controller_manager" exec="spawner" name="position_controller_spawner" args="position_controller"/>
  
  <!-- Start our custom nodes -->
  <node pkg="my_robot_controller" exec="ai_controller" name="ai_controller"/>
</launch>
```

## Summary

In this module, we've created high-fidelity simulation environments using Gazebo and Unity as digital twins for our humanoid robot. We covered:

- Physics simulation with gravity, collisions, and friction
- Environment design in Gazebo with various objects
- Human-robot interaction using Unity
- Simulation of various sensors (LiDAR, depth cameras, IMUs)
- Integration of simulation with ROS 2

The simulation environment serves as a crucial component for developing and testing our Physical AI system before deployment to real hardware, allowing for faster iteration and safer development.

## Exercises

1. Create a Gazebo world with a humanoid robot and various objects for interaction.
2. Implement a simple navigation task in the simulated environment.
3. Add sensor simulation to your robot model and verify sensor data publication.
4. Create a Unity scene that visualizes the robot's state in real-time through ROS communication.
5. Design a complex environment with multiple obstacles and interaction opportunities.