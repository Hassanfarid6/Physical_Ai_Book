# Appendices

## Appendix A: ROS 2 Commands Reference

### Common ROS 2 Commands

**Workspace Management:**
```bash
# Create a new workspace
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws

# Build the workspace
colcon build
colcon build --packages-select <package_name>  # Build specific package

# Source the workspace
source install/setup.bash
```

**Package Management:**
```bash
# Create a new package
ros2 pkg create --build-type ament_python <package_name>

# List all packages
ros2 pkg list

# Show package information
ros2 pkg xml <package_name> -t
```

**Node Management:**
```bash
# List all active nodes
ros2 node list

# Show information about a specific node
ros2 node info <node_name>

# Run a node
ros2 run <package_name> <executable_name>
```

**Topic Management:**
```bash
# List all topics
ros2 topic list

# Echo a topic (view messages)
ros2 topic echo /topic_name

# Show topic information
ros2 topic info /topic_name

# Publish to a topic (for testing)
ros2 topic pub /topic_name <msg_type> '{key: value}'
```

**Service Management:**
```bash
# List all services
ros2 service list

# Call a service
ros2 service call /service_name <service_type> '{request_fields}'

# Show service information
ros2 service info /service_name
```

**Launch Files:**
```bash
# Run a launch file
ros2 launch <package_name> <launch_file.py>

# Run with arguments
ros2 launch <package_name> <launch_file.py> arg_name:=arg_value
```

## Appendix B: URDF Reference

### URDF Elements

**Robot Root:**
```xml
<robot name="robot_name">
  <!-- Robot definition goes here -->
</robot>
```

**Links:**
```xml
<link name="link_name">
  <!-- Visual properties -->
  <visual>
    <origin xyz="0 0 0" rpy="0 0 0" />
    <geometry>
      <box size="1 1 1" />
      <!-- or <cylinder radius="0.5" length="1" /> -->
      <!-- or <sphere radius="0.5" /> -->
      <!-- or <mesh filename="package://package_name/meshes/link_name.stl" /> -->
    </geometry>
    <material name="color">
      <color rgba="0.8 0.2 0.2 1.0" />
    </material>
  </visual>
  
  <!-- Collision properties -->
  <collision>
    <origin xyz="0 0 0" rpy="0 0 0" />
    <geometry>
      <box size="1 1 1" />
    </geometry>
  </collision>
  
  <!-- Inertial properties -->
  <inertial>
    <origin xyz="0 0 0" rpy="0 0 0" />
    <mass value="1.0" />
    <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1" />
  </inertial>
</link>
```

**Joints:**
```xml
<joint name="joint_name" type="revolute">
  <parent link="parent_link" />
  <child link="child_link" />
  <origin xyz="0 0 0" rpy="0 0 0" />
  <axis xyz="0 0 1" />
  <limit lower="-1.57" upper="1.57" effort="10.0" velocity="1.0" />
  <dynamics damping="0.1" friction="0.0" />
</joint>
```

**Available Joint Types:**
- `revolute`: Rotational joint with limits
- `continuous`: Rotational joint without limits
- `prismatic`: Linear sliding joint with limits
- `fixed`: No movement
- `floating`: 6 DOF
- `planar`: Movement in a plane

## Appendix C: Gazebo Model Configuration

### Gazebo-Specific Extensions

**Transmission for ROS Control:**
```xml
<!-- Include transmission to connect joints to ROS control -->
<transmission name="tran1">
  <type>transmission_interface/SimpleTransmission</type>
  <joint name="joint_name">
    <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
  </joint>
  <actuator name="motor1">
    <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
    <mechanicalReduction>1</mechanicalReduction>
  </actuator>
</transmission>
```

**Gazebo-Specific Properties:**
```xml
<gazebo reference="link_name">
  <material>Gazebo/Red</material>
  <mu1>0.9</mu1>
  <mu2>0.9</mu2>
  <kp>1000000.0</kp>
  <kd>100.0</kd>
  <minDepth>0.001</minDepth>
  <maxVel>1.0</maxVel>
</gazebo>
```

**Sensor Plugins:**
```xml
<link name="camera_link">
  <!-- Camera sensor -->
  <sensor name="camera" type="camera">
    <camera name="head">
      <horizontal_fov>1.047</horizontal_fov>
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>100</far>
      </clip>
    </camera>
    <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
      <frame_name>camera_link_optical</frame_name>
    </plugin>
  </sensor>
</link>
```

## Appendix D: Troubleshooting Guide

### Common ROS 2 Issues

**1. Nodes not communicating across different terminals:**
- Ensure ROS_DOMAIN_ID is the same across terminals: `echo $ROS_DOMAIN_ID`
- If not set, all terminals should default to 0
- Set explicitly: `export ROS_DOMAIN_ID=0`

**2. "Command 'ros2' not found":**
- Check if ROS 2 is installed: `echo $ROS_DISTRO`
- Source ROS 2 setup: `source /opt/ros/humble/setup.bash`
- Add to ~/.bashrc: `echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc`

**3. "Package not found" errors:**
- Ensure package is built: `colcon build`
- Source the workspace: `source install/setup.bash`
- Check package.xml for proper dependencies

### Common Gazebo Issues

**1. Gazebo fails to start:**
- Check graphics drivers: `nvidia-smi` (for NVIDIA)
- Run with software rendering: `gazebo --verbose --rendering=ogre`
- Verify X11 forwarding if using SSH

**2. Robot falls through the ground or doesn't stay still:**
- Check URDF for proper inertial properties
- Verify joint limits and dynamics
- Increase physics update rate in world file

**3. Simulation runs too slowly:**
- Check real_time_factor in physics settings
- Reduce complexity of collision meshes
- Limit the number of active sensors

### Common Python Issues

**1. Import errors in ROS 2 Python nodes:**
- Ensure packages are built: `colcon build`
- Check Python path: `echo $PYTHONPATH`
- Verify package.xml for proper dependencies

**2. "Could not import" errors for custom messages:**
- Ensure message packages are built first: `colcon build --packages-select my_message_package`
- Verify message definitions are in msg/ directory
- Check build dependencies in package.xml

## Appendix E: Performance Optimization

### Optimizing Simulation Performance

**1. Reduce Physics Complexity:**
```xml
<!-- In world file -->
<physics type="ode">
  <max_step_size>0.001</max_step_size>  <!-- Smaller = more accurate but slower -->
  <real_time_update_rate>1000</real_time_update_rate>  <!-- Higher = more accurate but slower -->
  <real_time_factor>1</real_time_factor>  <!-- 1.0 = real-time -->
</physics>
```

**2. Optimize Collision Models:**
- Use simple geometric shapes (boxes, cylinders, spheres) instead of complex meshes
- Create separate collision meshes that are simpler than visual meshes
- Use `plane` elements for flat surfaces instead of thin boxes

**3. Sensor Optimization:**
- Reduce sensor update rates when possible (e.g., from 100Hz to 30Hz)
- Use smaller image resolutions for vision sensors
- Limit the range and resolution of range sensors (LiDAR, sonar)

### Code Optimization Tips

**1. Efficient ROS 2 Communication:**
```python
# Use appropriate QoS profiles
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

# For sensors, use best effort to avoid blocking
qos_sensor = QoSProfile(
    depth=10,
    durability=QoSDurabilityPolicy.VOLATILE,
    reliability=QoSReliabilityPolicy.BEST_EFFORT
)

# For critical commands, use reliable
qos_command = QoSProfile(
    depth=10,
    durability=QoSDurabilityPolicy.VOLATILE,
    reliability=QoSReliabilityPolicy.RELIABLE
)
```

**2. Efficient Data Processing:**
```python
import numpy as np

# Use NumPy for mathematical operations
def efficient_computation(data):
    # Instead of Python loops
    # result = [x * 2 for x in data]
    
    # Use NumPy
    result = np.array(data) * 2
    return result
```

## Appendix F: Safety Considerations

### Simulation Safety

**1. Virtual Environment Limits:**
- Implement software limits in joints to prevent self-collision
- Use safety controllers that prevent dangerous configurations
- Include emergency stops in simulation

**2. Sensor Range Checks:**
- Validate sensor readings to detect sensor failure in simulation
- Implement plausibility checks on sensor values

### Physical Robot Safety

**1. Hardware Safety:**
- Implement joint limits in hardware and software
- Use safety-rated controllers when available
- Include emergency stop functionality

**2. Operational Safety:**
- Start with simulation before physical deployment
- Use safety cages during initial physical testing
- Implement force/torque limits to prevent damage

**3. Control Safety:**
```python
def safety_check(joint_positions, joint_velocities):
    """Check if robot state is within safe limits"""
    # Check joint position limits
    for pos, (min_limit, max_limit) in zip(joint_positions, joint_limits):
        if pos < min_limit or pos > max_limit:
            return False, "Joint position limit exceeded"
    
    # Check joint velocity limits
    for vel, max_vel in zip(joint_velocities, max_velocities):
        if abs(vel) > max_vel:
            return False, "Joint velocity limit exceeded"
    
    return True, "Safe"
```

## Appendix G: Recommended Resources

### Learning Resources

**ROS 2:**
- Official ROS 2 Documentation: https://docs.ros.org/en/humble/
- ROS 2 Tutorials: https://docs.ros.org/en/humble/Tutorials.html
- Robot Ignite Academy: https://www.robotigniteacademy.com/

**Gazebo:**
- Gazebo Classic Documentation: http://gazebosim.org/
- Gazebo Garden Documentation: https://gazebosim.org/docs/garden/
- Gazebosim Tutorials: https://gazebosim.org/tutorials

**Python Robotics:**
- Python Robotics Library: https://github.com/AtsushiSakai/PythonRobotics
- Probabilistic Robotics by Thrun, Burgard, and Fox
- Programming Robots with ROS by Quigley and Gerkey

**Humanoid Robotics:**
- Kajita et al. "Humanoid Robotics: A Reference" 
- Humanoid Robotics Lab at Waseda University publications
- IEEE-RAS Humanoids Conference proceedings

### Tools and Libraries

**Simulation:**
- NVIDIA Isaac Sim for advanced simulation
- Webots: Open-source robotics simulator
- PyBullet: Python-based physics simulation

**AI/ML:**
- Stable-Baselines3 for reinforcement learning
- PyTorch Geometric for graph neural networks
- OpenAI Gym for reinforcement learning environments

**Development:**
- VS Code with ROS extension
- PyCharm with ROS plugin
- Docker for consistent development environments