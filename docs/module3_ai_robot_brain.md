# Module 3: The AI-Robot Brain (NVIDIA Isaac™) - Advanced Perception, Training, and Navigation

## Overview

This module focuses on implementing advanced perception, training, and navigation capabilities using NVIDIA Isaac technologies. The NVIDIA Isaac platform provides a comprehensive solution for developing intelligent robotic applications with hardware-accelerated perception, photorealistic simulation, and sophisticated navigation capabilities. This module covers the integration of Isaac Sim for synthetic data generation and domain randomization, Isaac ROS for accelerated perception pipelines, and Nav2 for autonomous navigation specifically tailored for bipedal humanoid movement.

## Learning Objectives

By the end of this module, you will be able to:
- Implement photorealistic simulation using NVIDIA Isaac Sim
- Generate synthetic training data for perception systems
- Apply domain randomization techniques to improve sim-to-real transfer
- Develop hardware-accelerated perception pipelines using Isaac ROS
- Implement Visual SLAM (VSLAM) for humanoid navigation
- Perform sensor fusion for enhanced environmental understanding
- Configure and deploy Nav2 for bipedal humanoid path planning and navigation
- Implement obstacle avoidance and localization for humanoid robots

## 1. NVIDIA Isaac Sim for Advanced Simulation

NVIDIA Isaac Sim is a next-generation simulation application and rendering engine based on NVIDIA Omniverse. It provides photorealistic virtual environments for training and testing robotic systems, offering capabilities that significantly enhance the development of Physical AI systems.

### Photorealistic Simulation

Isaac Sim leverages NVIDIA's RTX technology to create photorealistic environments that closely match real-world lighting, textures, and physics. This advanced simulation capability is crucial for developing robotic systems that can operate effectively in diverse real-world conditions.

Key features of Isaac Sim include:

- **PhysX 4.0 Physics Engine**: Provides accurate and stable physics simulation for complex interactions
- **RTX Denoising**: Real-time denoising of ray-traced lighting effects
- **Material Definition Language (MDL)**: Industry-standard material descriptions
- **GPU-accelerated Simulation**: Leverages CUDA and Tensor Cores for high-performance simulation
- **USD-based Scene Description**: Universal Scene Description for complex scene management

### Example: Setting up Isaac Sim Environment

```python
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import find_nucleus_server
from omni.isaac.core.utils.carb import set_carb_setting
import numpy as np

# Initialize Isaac Sim environment
class IsaacSimEnvironment:
    def __init__(self):
        self.world = World(stage_units_in_meters=1.0)
        self.setup_physics()
        
    def setup_physics(self):
        # Configure physics properties for humanoid simulation
        self.world.physics_scene.set_gravity(9.81)
        
        # Add ground plane
        self.world.scene.add_default_ground_plane()
        
        # Add humanoid robot
        self.add_humanoid_robot()
        
    def add_humanoid_robot(self):
        # Add humanoid model to the scene
        assets_root_path = find_nucleus_server()
        if assets_root_path is None:
            print("Could not find Isaac Sim Assets. Please enable Isaac Sim Nucleus.")
            return
            
        # Load humanoid robot model
        add_reference_to_stage(
            usd_path=f"{assets_root_path}/Isaac/Robots/Humanoid/humanoid.usd",
            prim_path="/World/Humanoid"
        )
        
    def run_simulation(self):
        # Reset the simulation
        self.world.reset()
        
        # Main simulation loop
        for i in range(1000):  # Run for 1000 steps
            # Step the simulation
            self.world.step(render=True)
            
            # Get robot state
            robot_position, robot_orientation = self.get_robot_state()
            
            # Perform AI decision making
            action = self.ai_decision(robot_position, robot_orientation)
            
            # Apply action to robot
            self.apply_action(action)
            
    def get_robot_state(self):
        # Implementation to get current robot state
        # This would interface with the simulation to retrieve pose, joint angles, etc.
        pass
    
    def ai_decision(self, position, orientation):
        # AI logic for decision making
        # This would implement perception, planning, and control algorithms
        pass
    
    def apply_action(self, action):
        # Apply action to robot joints
        # This would send commands to the simulated robot actuators
        pass

# Example usage
if __name__ == "__main__":
    env = IsaacSimEnvironment()
    env.run_simulation()
```

## 2. Synthetic Data Generation

One of the powerful capabilities of Isaac Sim is its ability to generate synthetic training data for machine learning models. This data can include RGB images, depth maps, segmentation masks, and other sensor modalities.

### Example: Synthetic Data Generation Pipeline

```python
import omni
from omni.isaac.core import World
from omni.isaac.sensor import Camera
from omni.isaac.synthetic_utils import SyntheticDataHelper
import numpy as np
import cv2
import os

class SyntheticDataGenerator:
    def __init__(self, output_dir="synthetic_data"):
        self.world = World(stage_units_in_meters=1.0)
        self.output_dir = output_dir
        self.cameras = []
        self.setup_environment()
        
        # Create output directories
        os.makedirs(f"{output_dir}/rgb", exist_ok=True)
        os.makedirs(f"{output_dir}/depth", exist_ok=True)
        os.makedirs(f"{output_dir}/seg", exist_ok=True)
        
    def setup_environment(self):
        # Add ground plane
        self.world.scene.add_default_ground_plane()
        
        # Add objects for diverse training data
        self.add_objects_for_training()
        
        # Setup cameras for data collection
        self.setup_cameras()
        
    def add_objects_for_training(self):
        # Add various objects with different materials and lighting conditions
        # This would typically include objects relevant to the humanoid's environment
        pass
    
    def setup_cameras(self):
        # Add RGB-D camera to the humanoid robot
        camera = Camera(
            prim_path="/World/Humanoid/Camera",
            position=np.array([0.0, 0.0, 0.5]),
            frequency=30,
            resolution=(640, 480)
        )
        self.cameras.append(camera)
        
    def generate_dataset(self, num_samples=1000):
        # Reset the environment
        self.world.reset()
        
        for i in range(num_samples):
            # Step simulation to a new configuration
            self.randomize_scene()
            self.world.step(render=True)
            
            # Capture synthetic data from all cameras
            for cam_idx, camera in enumerate(self.cameras):
                # Get RGB image
                rgb = camera.get_rgb()
                cv2.imwrite(f"{self.output_dir}/rgb/sample_{i:05d}_cam_{cam_idx}.png", 
                           cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                
                # Get depth image
                depth = camera.get_depth()
                cv2.imwrite(f"{self.output_dir}/depth/sample_{i:05d}_cam_{cam_idx}.png", 
                           (depth * 1000).astype(np.uint16))  # Convert to mm for 16-bit storage
                
                # Get segmentation mask
                seg = camera.get_semantic_segmentation()
                cv2.imwrite(f"{self.output_dir}/seg/sample_{i:05d}_cam_{cam_idx}.png", 
                           seg.astype(np.uint16))
            
            # Log progress
            if i % 100 == 0:
                print(f"Generated {i}/{num_samples} synthetic samples")
    
    def randomize_scene(self):
        # Randomize lighting, object positions, textures, etc.
        # This is essential for creating diverse training data
        pass

# Example usage
if __name__ == "__main__":
    generator = SyntheticDataGenerator()
    generator.generate_dataset(num_samples=10000)
    print("Synthetic dataset generation completed!")
```

## 3. Domain Randomization

Domain randomization is a technique to improve the sim-to-real transfer of robotic systems by randomizing the simulation environment during training. This includes randomizing texture, lighting, geometry, and dynamics.

### Implementation of Domain Randomization

```python
import random
import numpy as np
from omni.isaac.core import World
from pxr import Gf, UsdGeom, Sdf

class DomainRandomizer:
    def __init__(self, world: World):
        self.world = world
        self.randomization_params = {}
        self.setup_randomization_ranges()
        
    def setup_randomization_ranges(self):
        # Define ranges for different randomization parameters
        self.randomization_params = {
            'lighting': {
                'intensity_range': (300, 1500),
                'temperature_range': (3000, 8000),  # Kelvin
                'direction_range': (-0.5, 0.5)  # Randomize sun direction
            },
            'materials': {
                'roughness_range': (0.1, 0.9),
                'metallic_range': (0.0, 0.2),
                'albedo_range': (0.1, 1.0)
            },
            'geometry': {
                'object_size_range': (0.5, 1.5),  # Scale objects
                'position_jitter': (0.1, 0.1, 0.05)  # X, Y, Z jitter in meters
            },
            'dynamics': {
                'friction_range': (0.1, 0.9),
                'restitution_range': (0.0, 0.2),  # Bounciness
                'mass_range': (0.8, 1.2)  # Multiplier for mass
            }
        }
    
    def randomize_lighting(self):
        """Randomize lighting conditions in the scene"""
        # Randomize directional light (sun)
        sun_light = self.world.scene._lights.get("DistantLight", None)
        if sun_light:
            intensity = random.uniform(
                self.randomization_params['lighting']['intensity_range'][0],
                self.randomization_params['lighting']['intensity_range'][1]
            )
            temperature = random.uniform(
                self.randomization_params['lighting']['temperature_range'][0],
                self.randomization_params['lighting']['temperature_range'][1]
            )
            
            # Update light properties
            sun_light.intensity = intensity
            sun_light.color = self.temperature_to_rgb(temperature)
            
            # Randomize light direction
            direction_offset = [random.uniform(*self.randomization_params['lighting']['direction_range']) for _ in range(3)]
            # Ensure light still points roughly downward
            sun_light.direction = Gf.Vec3f(
                direction_offset[0],
                direction_offset[1],
                -abs(1 + direction_offset[2])  # Ensure it points downward
            )
    
    def temperature_to_rgb(self, temperature):
        """Convert color temperature in Kelvin to RGB values"""
        # Simplified approximation
        temp = temperature / 100
        if temp <= 66:
            red = 255
            green = temp
            green = 99.4708025861 * np.log(green) - 161.1195681661
        else:
            red = temp - 60
            red = 329.698727446 * (red ** -0.1332047592)
            green = temp - 60
            green = 288.1221695283 * (green ** -0.0755148492)
            
        blue = temp - 10
        blue = 138.5177312231 * np.log(blue) - 305.0447927307
        return Gf.Vec3f(
            max(0, min(255, red)) / 255.0,
            max(0, min(255, green)) / 255.0,
            max(0, min(255, blue)) / 255.0
        )
    
    def randomize_materials(self):
        """Randomize material properties in the scene"""
        # Get all materials in the scene
        stage = self.world.scene.stage
        material_prims = [prim for prim in stage.TraverseAll() if prim.IsA(UsdGeom.Material)]
        
        for material_prim in material_prims:
            material = UsdGeom.Material(material_prim)
            
            # Randomize material properties
            roughness = random.uniform(
                self.randomization_params['materials']['roughness_range'][0],
                self.randomization_params['materials']['roughness_range'][1]
            )
            metallic = random.uniform(
                self.randomization_params['materials']['metallic_range'][0],
                self.randomization_params['materials']['metallic_range'][1]
            )
            albedo = random.uniform(
                self.randomization_params['materials']['albedo_range'][0],
                self.randomization_params['materials']['albedo_range'][1]
            )
            
            # Update material properties in USD stage
            # This would involve modifying the material's shader inputs
            pass
    
    def randomize_geometry(self):
        """Randomize geometric properties of objects"""
        # Get all rigid bodies in the scene
        rigid_bodies = self.world.scene._physics_actors
        
        for name, rigid_body in rigid_bodies.items():
            # Randomize scale
            scale_factor = random.uniform(
                self.randomization_params['geometry']['object_size_range'][0],
                self.randomization_params['geometry']['object_size_range'][1]
            )
            
            # Randomize position with jitter
            current_pos = rigid_body.get_world_pose()[0]  # Get current position
            jitter = [
                random.uniform(-self.randomization_params['geometry']['position_jitter'][0],
                              self.randomization_params['geometry']['position_jitter'][0]),
                random.uniform(-self.randomization_params['geometry']['position_jitter'][1],
                              self.randomization_params['geometry']['position_jitter'][1]),
                random.uniform(-self.randomization_params['geometry']['position_jitter'][2],
                              self.randomization_params['geometry']['position_jitter'][2])
            ]
            new_pos = [current_pos[i] + jitter[i] for i in range(3)]
            
            # Apply new position and scale
            rigid_body.set_world_pose(position=new_pos)
    
    def randomize_dynamics(self):
        """Randomize dynamic properties of objects"""
        # Get all rigid bodies in the scene
        rigid_bodies = self.world.scene._physics_actors
        
        for name, rigid_body in rigid_bodies.items():
            friction = random.uniform(
                self.randomization_params['dynamics']['friction_range'][0],
                self.randomization_params['dynamics']['friction_range'][1]
            )
            
            restitution = random.uniform(
                self.randomization_params['dynamics']['restitution_range'][0],
                self.randomization_params['dynamics']['restitution_range'][1]
            )
            
            mass_multiplier = random.uniform(
                self.randomization_params['dynamics']['mass_range'][0],
                self.randomization_params['dynamics']['mass_range'][1]
            )
            
            # Apply dynamic properties (simplified - actual implementation depends on Isaac Sim API)
            # rigid_body.set_friction(friction)
            # rigid_body.set_restitution(restitution)
            # rigid_body.set_mass(rigid_body.get_mass() * mass_multiplier)
    
    def apply_randomization(self):
        """Apply all randomization techniques"""
        self.randomize_lighting()
        self.randomize_materials()
        self.randomize_geometry()
        self.randomize_dynamics()
        
        # Step the world to apply changes
        self.world.step(render=True)
```

## 4. Isaac ROS: Hardware-Accelerated Perception Pipelines

Isaac ROS provides hardware-accelerated perception algorithms optimized for NVIDIA GPUs. These packages enable efficient processing of sensor data for robotics applications.

### Visual SLAM (VSLAM) with Isaac ROS

Visual SLAM (Simultaneous Localization and Mapping) is critical for humanoid navigation. Isaac ROS provides optimized implementations that leverage NVIDIA GPUs for real-time performance.

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import cv2
import numpy as np

class IsaacVSLAMNode(Node):
    def __init__(self):
        super().__init__('isaac_vsalm_node')
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Publishers and subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/rgb/image_raw',
            self.image_callback,
            10
        )
        
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/rgb/camera_info',
            self.camera_info_callback,
            10
        )
        
        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )
        
        self.odom_pub = self.create_publisher(
            Odometry,
            '/visual_odom',
            10
        )
        
        self.pose_pub = self.create_publisher(
            PoseStamped,
            '/visual_pose',
            10
        )
        
        # VSLAM state variables
        self.prev_frame = None
        self.prev_desc = None
        self.prev_kp = None
        self.camera_matrix = None
        self.dist_coeffs = None
        self.position = np.array([0.0, 0.0, 0.0])
        self.orientation = np.array([0.0, 0.0, 0.0, 1.0])  # Quaternion
        
        # Feature detector (using ORB which is efficient for real-time)
        self.detector = cv2.ORB_create(nfeatures=1000)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        self.get_logger().info('Isaac VSLAM node initialized')
    
    def camera_info_callback(self, msg):
        """Process camera calibration information"""
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d)
    
    def imu_callback(self, msg):
        """Process IMU data to improve pose estimation"""
        # In a real implementation, this would be integrated with visual data
        # for more robust pose estimation
        pass
    
    def image_callback(self, msg):
        """Process incoming camera image for VSLAM"""
        if self.camera_matrix is None:
            return  # Wait for camera info
        
        # Convert ROS image to OpenCV
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # Undistort image
        h, w = cv_image.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h)
        )
        cv_image = cv2.undistort(cv_image, self.camera_matrix, self.dist_coeffs, None, new_camera_matrix)
        
        # Get current frame features
        kp, desc = self.detect_features(cv_image)
        
        if self.prev_frame is not None and len(kp) > 10 and len(self.prev_kp) > 10:
            # Match features between frames
            matches = self.match_features(self.prev_desc, desc)
            
            if len(matches) > 10:  # Minimum for reliable pose estimation
                # Estimate motion between frames
                motion = self.estimate_motion(matches, self.prev_kp, kp)
                
                if motion is not None:
                    # Update position and orientation
                    self.update_pose(motion)
                    
                    # Publish estimated pose
                    self.publish_pose()
        
        # Update previous frame
        self.prev_frame = cv_image.copy()
        self.prev_desc = desc
        self.prev_kp = kp
    
    def detect_features(self, image):
        """Detect and extract features from image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kp = self.detector.detect(gray, None)
        kp, desc = self.detector.compute(gray, kp)
        return kp, desc
    
    def match_features(self, desc1, desc2):
        """Match features between two sets of descriptors"""
        if desc1 is None or desc2 is None:
            return []
        
        matches = self.bf_matcher.match(desc1, desc2)
        # Sort by distance (lower is better)
        matches = sorted(matches, key=lambda x: x.distance)
        return matches
    
    def estimate_motion(self, matches, kp1, kp2):
        """Estimate camera motion between two frames"""
        if len(matches) < 10:
            return None
        
        # Get matching points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Find fundamental matrix
        F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.RANSAC, 4, 0.999)
        
        # Use camera matrix to compute essential matrix
        E = self.camera_matrix.T @ F @ self.camera_matrix
        
        # Decompose essential matrix to get rotation and translation
        _, R, t, _ = cv2.recoverPose(E, src_pts, dst_pts, self.camera_matrix)
        
        # Convert to transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = R
        transform[:3, 3] = t.flatten()
        
        return transform
    
    def update_pose(self, motion):
        """Update current pose based on motion estimate"""
        # Convert motion to position and orientation changes
        dx, dy, dz = motion[:3, 3]
        
        # Update position
        self.position += np.array([dx, dy, dz])
        
        # Extract rotation and update orientation
        R = motion[:3, :3]
        # Convert rotation matrix to quaternion
        qw = np.sqrt(1 + R[0,0] + R[1,1] + R[2,2]) / 2
        qx = (R[2,1] - R[1,2]) / (4 * qw)
        qy = (R[0,2] - R[2,0]) / (4 * qw)
        qz = (R[1,0] - R[0,1]) / (4 * qw)
        
        delta_q = np.array([qx, qy, qz, qw])
        delta_q = delta_q / np.linalg.norm(delta_q)  # Normalize
        
        # Update orientation quaternion
        # Multiply current orientation by delta orientation
        self.orientation = self.quaternion_multiply(self.orientation, delta_q)
    
    def quaternion_multiply(self, q1, q2):
        """Multiply two quaternions"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        
        return np.array([x, y, z, w])
    
    def publish_pose(self):
        """Publish estimated pose"""
        # Publish Odometry message
        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = 'map'
        odom_msg.child_frame_id = 'camera_frame'
        
        # Set position
        odom_msg.pose.pose.position.x = float(self.position[0])
        odom_msg.pose.pose.position.y = float(self.position[1])
        odom_msg.pose.pose.position.z = float(self.position[2])
        
        # Set orientation
        odom_msg.pose.pose.orientation.x = float(self.orientation[0])
        odom_msg.pose.pose.orientation.y = float(self.orientation[1])
        odom_msg.pose.pose.orientation.z = float(self.orientation[2])
        odom_msg.pose.pose.orientation.w = float(self.orientation[3])
        
        self.odom_pub.publish(odom_msg)
        
        # Publish PoseStamped message
        pose_msg = PoseStamped()
        pose_msg.header = odom_msg.header
        pose_msg.pose = odom_msg.pose.pose
        
        self.pose_pub.publish(pose_msg)

def main(args=None):
    rclpy.init(args=args)
    vsalm_node = IsaacVSLAMNode()
    
    try:
        rclpy.spin(vsalm_node)
    except KeyboardInterrupt:
        pass
    finally:
        vsalm_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Sensor Fusion with Isaac ROS

Isaac ROS provides sophisticated sensor fusion capabilities that combine data from multiple sensors to create a more accurate understanding of the environment.

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, Imu, PointCloud2, LaserScan
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float32MultiArray
from tf2_ros import TransformListener, Buffer
from tf2_geometry_msgs import do_transform_point
import numpy as np

class IsaacSensorFusionNode(Node):
    def __init__(self):
        super().__init__('isaac_sensor_fusion_node')
        
        # Initialize TF2
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Publishers and subscribers
        self.camera_sub = self.create_subscription(
            Image, '/camera/rgb/image_raw', self.camera_callback, 10)
        
        self.depth_sub = self.create_subscription(
            Image, '/camera/depth/image_raw', self.depth_callback, 10)
        
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10)
        
        self.lidar_sub = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, 10)
        
        self.fused_pub = self.create_publisher(
            PointCloud2, '/fused_pointcloud', 10)
        
        self.environment_model_pub = self.create_publisher(
            Float32MultiArray, '/environment_model', 10)
        
        # Data storage
        self.latest_camera = None
        self.latest_depth = None
        self.latest_imu = None
        self.latest_lidar = None
        
        # Sensor fusion timers
        self.fusion_timer = self.create_timer(0.1, self.perform_sensor_fusion)
        
        self.get_logger().info('Isaac Sensor Fusion node initialized')
    
    def camera_callback(self, msg):
        """Store latest camera data"""
        self.latest_camera = msg
    
    def depth_callback(self, msg):
        """Store latest depth data"""
        self.latest_depth = msg
    
    def imu_callback(self, msg):
        """Store latest IMU data"""
        self.latest_imu = msg
    
    def lidar_callback(self, msg):
        """Store latest LIDAR data"""
        self.latest_lidar = msg
    
    def perform_sensor_fusion(self):
        """Perform sensor fusion to create comprehensive environmental model"""
        if not all([self.latest_camera, self.latest_depth, self.latest_imu, self.latest_lidar]):
            return  # Wait for all sensors to have data
        
        # Process camera and depth data to create partial point cloud
        camera_cloud = self.camera_depth_to_pointcloud(
            self.latest_camera, self.latest_depth)
        
        # Process LIDAR data to point cloud
        lidar_cloud = self.lidar_to_pointcloud(self.latest_lidar)
        
        # Transform LIDAR points to camera frame
        lidar_cloud_transformed = self.transform_pointcloud(lidar_cloud, 'lidar_frame', 'camera_frame')
        
        # Fuse camera and LIDAR point clouds
        fused_cloud = self.fuse_pointclouds(camera_cloud, lidar_cloud_transformed)
        
        # Apply IMU data to refine pose estimation
        refined_cloud = self.refine_with_imu(fused_cloud, self.latest_imu)
        
        # Publish fused point cloud
        self.fused_pub.publish(refined_cloud)
        
        # Create and publish environmental model
        env_model = self.create_environmental_model(refined_cloud)
        self.environment_model_pub.publish(env_model)
    
    def camera_depth_to_pointcloud(self, camera_msg, depth_msg):
        """Convert camera and depth images to point cloud"""
        # This is a simplified implementation
        # In practice, Isaac ROS provides optimized implementations
        import cv2
        from cv_bridge import CvBridge
        
        bridge = CvBridge()
        depth_image = bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        camera_image = bridge.imgmsg_to_cv2(camera_msg, desired_encoding='bgr8')
        
        # Get camera parameters
        camera_info = self.get_camera_info()  # Would subscribe to camera info in practice
        
        # Create point cloud from depth image
        height, width = depth_image.shape
        points = []
        
        for v in range(0, height, 4):  # Downsample for performance
            for u in range(0, width, 4):
                z = depth_image[v, u]
                if z > 0 and z < 10.0:  # Valid depth range
                    x = (u - camera_info[0, 2]) * z / camera_info[0, 0]
                    y = (v - camera_info[1, 2]) * z / camera_info[1, 1]
                    points.append([x, y, z])
        
        # Convert to PointCloud2 message
        return self.create_pointcloud2_message(np.array(points))
    
    def lidar_to_pointcloud(self, lidar_msg):
        """Convert LIDAR scan to point cloud"""
        # Convert LIDAR scan to point cloud
        ranges = np.array(lidar_msg.ranges)
        angles = np.linspace(lidar_msg.angle_min, lidar_msg.angle_max, len(ranges))
        
        # Filter out invalid ranges
        valid_indices = np.isfinite(ranges) & (ranges > lidar_msg.range_min) & (ranges < lidar_msg.range_max)
        valid_ranges = ranges[valid_indices]
        valid_angles = angles[valid_indices]
        
        # Calculate x, y coordinates
        x = valid_ranges * np.cos(valid_angles)
        y = valid_ranges * np.sin(valid_angles)
        z = np.zeros_like(x)
        
        points = np.column_stack((x, y, z))
        
        # Convert to PointCloud2 message
        return self.create_pointcloud2_message(points)
    
    def transform_pointcloud(self, pointcloud, from_frame, to_frame):
        """Transform point cloud from one frame to another"""
        # In a real implementation, this would transform each point
        # using the TF2 transformation between frames
        # This is a simplified placeholder
        try:
            transform = self.tf_buffer.lookup_transform(
                to_frame, from_frame, rclpy.time.Time())
            # Apply transformation to each point
            # This is a simplified placeholder
            return pointcloud
        except Exception as e:
            self.get_logger().warn(f'Transform lookup failed: {e}')
            return pointcloud
    
    def fuse_pointclouds(self, cloud1, cloud2):
        """Fuse two point clouds into a single representation"""
        # In a real implementation, this would perform more sophisticated fusion
        # like ICP (Iterative Closest Point) or other registration algorithms
        fused_points = np.vstack((cloud1, cloud2))
        return self.create_pointcloud2_message(fused_points)
    
    def refine_with_imu(self, pointcloud, imu_msg):
        """Refine point cloud using IMU data"""
        # Use IMU data to correct for robot motion during scanning
        # This is a simplified implementation
        return pointcloud
    
    def create_pointcloud2_message(self, points):
        """Create a PointCloud2 message from numpy array of points"""
        # This is a simplified implementation for demonstration
        # Actual implementation would create proper PointCloud2 message
        from sensor_msgs.msg import PointCloud2, PointField
        import struct
        
        # Convert to PointCloud2 format
        # This is a simplified placeholder
        msg = PointCloud2()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        
        return msg
    
    def get_camera_info(self):
        """Get camera intrinsic parameters"""
        # This is a placeholder for camera intrinsics
        # In practice, this would be retrieved from camera_info topic
        return np.array([[554.256, 0.0, 320.5], 
                         [0.0, 554.256, 240.5], 
                         [0.0, 0.0, 1.0]])
    
    def create_environmental_model(self, pointcloud):
        """Create an environmental model from fused sensor data"""
        # Create a simplified environmental model
        # This could be occupancy grid, mesh, or other representation
        
        # For this example, we'll create a simple representation
        # of obstacles in different directions
        obstacles = np.zeros(8)  # 8 directions around the robot
        
        # Analyze point cloud for obstacles in different directions
        points = np.frombuffer(pointcloud.data, dtype=np.float32).reshape(-1, 3)
        
        for point in points:
            x, y, z = point
            if 0.1 < z < 2.0 and np.sqrt(x**2 + y**2) < 3.0:  # Relevant height and distance
                angle = int(np.arctan2(y, x) * 4 / np.pi) % 8  # Quantize to 8 directions
                obstacles[angle] += 1
        
        # Normalize obstacle representation
        if np.sum(obstacles) > 0:
            obstacles = obstacles / np.sum(obstacles)
        
        # Create message
        model_msg = Float32MultiArray()
        model_msg.data = obstacles.tolist()
        
        return model_msg

def main(args=None):
    rclpy.init(args=args)
    fusion_node = IsaacSensorFusionNode()
    
    try:
        rclpy.spin(fusion_node)
    except KeyboardInterrupt:
        pass
    finally:
        fusion_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 5. Nav2 for Autonomous Navigation

Navigation2 (Nav2) is a flexible, extensible, and performant navigation stack designed to work with ROS 2. For humanoid robots, Nav2 enables sophisticated path planning, obstacle avoidance, and localization.

### Setting up Nav2 for Humanoid Navigation

```python
import rclpy
from rclpy.node import Node
from nav2_simple_commander.robot_navigator import BasicNavigator
from geometry_msgs.msg import PoseStamped, Point
from nav2_msgs.action import NavigateToPose, NavigateThroughPoses
from std_msgs.msg import String
import math

class HumanoidNav2Node(Node):
    def __init__(self):
        super().__init__('humanoid_nav2_node')
        
        # Initialize the navigator
        self.navigator = BasicNavigator()
        
        # Initial pose setup
        self.initial_pose = PoseStamped()
        self.initial_pose.header.frame_id = 'map'
        self.initial_pose.header.stamp = self.get_clock().now().to_msg()
        self.initial_pose.pose.position.x = 0.0
        self.initial_pose.pose.position.y = 0.0
        self.initial_pose.pose.position.z = 0.0
        self.initial_pose.pose.orientation.x = 0.0
        self.initial_pose.pose.orientation.y = 0.0
        self.initial_pose.pose.orientation.z = 0.0
        self.initial_pose.pose.orientation.w = 1.0
        
        self.navigator.setInitialPose(self.initial_pose)
        
        # Wait for navigation to be active
        self.navigator.waitUntilNav2Active()
        
        # Command subscriber
        self.nav_command_sub = self.create_subscription(
            String, '/navigation_command', self.navigation_command_callback, 10)
        
        self.get_logger().info('Humanoid Nav2 node initialized')
    
    def navigation_command_callback(self, msg):
        """Process navigation commands"""
        command = msg.data
        if command == "init":
            self.initialize_navigation()
        elif command.startswith("move_to:"):
            # Parse coordinates: move_to:x,y
            try:
                coords = command.split(":")[1].split(",")
                x, y = float(coords[0]), float(coords[1])
                self.navigate_to_position(x, y)
            except ValueError:
                self.get_logger().error(f"Invalid navigation command format: {command}")
        elif command.startswith("waypoints:"):
            # Parse multiple waypoints: waypoints:x1,y1;x2,y2;x3,y3
            try:
                waypoints_str = command.split(":")[1]
                waypoints = []
                for wp in waypoints_str.split(";"):
                    x, y = map(float, wp.split(","))
                    waypoints.append((x, y))
                self.navigate_through_waypoints(waypoints)
            except ValueError:
                self.get_logger().error(f"Invalid waypoints format: {command}")
    
    def initialize_navigation(self):
        """Initialize the navigation system"""
        self.navigator.setInitialPose(self.initial_pose)
        self.navigator.waitUntilNav2Active()
        self.get_logger().info('Navigation system initialized')
    
    def navigate_to_position(self, x, y):
        """Navigate to a specific position"""
        goal_pose = self.create_pose_stamped(x, y)
        
        self.get_logger().info(f'Navigating to position: ({x}, {y})')
        
        # Go to pose
        result = self.navigator.goToPose(goal_pose)
        
        if not result:
            self.get_logger().warn('Navigation failed!')
        else:
            self.get_logger().info('Navigation succeeded!')
    
    def navigate_through_waypoints(self, waypoints):
        """Navigate through a series of waypoints"""
        goal_poses = []
        for x, y in waypoints:
            goal_poses.append(self.create_pose_stamped(x, y))
        
        self.get_logger().info(f'Navigating through {len(waypoints)} waypoints')
        
        # Go through poses
        result = self.navigator.goThroughPoses(goal_poses)
        
        if not result:
            self.get_logger().warn('Waypoint navigation failed!')
        else:
            self.get_logger().info('Waypoint navigation succeeded!')
    
    def create_pose_stamped(self, x, y, z=0.0, theta=0.0):
        """Create a pose stamped message"""
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = float(z)
        
        # Convert Euler angle (theta) to quaternion
        quat_z = math.sin(theta / 2.0)
        quat_w = math.cos(theta / 2.0)
        pose.pose.orientation.z = quat_z
        pose.pose.orientation.w = quat_w
        
        return pose

def main(args=None):
    rclpy.init(args=args)
    nav_node = HumanoidNav2Node()
    
    try:
        rclpy.spin(nav_node)
    except KeyboardInterrupt:
        pass
    finally:
        nav_node.navigator.lifecycle.shutdown()
        nav_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Custom Path Planner for Bipedal Humanoid Movement

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Path, OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from builtin_interfaces.msg import Duration
import numpy as np
from scipy.spatial import KDTree
import math

class BipedalPathPlannerNode(Node):
    def __init__(self):
        super().__init__('bipedal_path_planner_node')
        
        # Publishers
        self.global_plan_pub = self.create_publisher(Path, '/global_plan', 10)
        self.local_plan_pub = self.create_publisher(Path, '/local_plan', 10)
        self.path_viz_pub = self.create_publisher(MarkerArray, '/path_visualization', 10)
        
        # Subscribers
        self.goal_sub = self.create_subscription(
            PoseStamped, '/move_base_simple/goal', self.goal_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10)
        
        # Robot state
        self.current_pose = None
        self.current_twist = None
        self.costmap = None
        self.map_resolution = 0.05
        self.map_origin = [0.0, 0.0]
        self.map_width = 0
        self.map_height = 0
        
        # Parameters for bipedal navigation
        self.robot_radius = 0.3  # Humanoid robot radius
        self.step_height = 0.15  # Maximum step height for humanoid
        self.step_length = 0.6   # Typical step length
        self.max_angular_vel = 0.5  # Limited turning for bipedal stability
        self.min_turning_radius = 0.8
        
        # Path planning parameters
        self.min_distance_to_obstacle = 0.5
        self.max_planning_time = 5.0
        
        # Initialize visualization markers
        self.marker_id = 0
        
        self.get_logger().info('Bipedal Path Planner node initialized')
    
    def goal_callback(self, msg):
        """Handle new navigation goal"""
        if self.costmap is None or self.current_pose is None:
            self.get_logger().warn('Waiting for map and current pose...')
            return
        
        self.get_logger().info(f'New goal received: ({msg.pose.position.x}, {msg.pose.position.y})')
        
        # Plan path to goal
        path = self.plan_path(self.current_pose, msg.pose)
        
        if path:
            # Publish global plan
            path_msg = self.create_path_message(path)
            self.global_plan_pub.publish(path_msg)
            
            # Publish visualization
            self.publish_path_visualization(path)
        else:
            self.get_logger().warn('Failed to find valid path to goal')
    
    def odom_callback(self, msg):
        """Update current robot pose and velocity"""
        self.current_pose = msg.pose.pose
        self.current_twist = msg.twist.twist
    
    def scan_callback(self, msg):
        """Update local obstacle information from laser scan"""
        # Process laser scan to update local costmap
        # In a real implementation, this would update a local costmap
        pass
    
    def map_callback(self, msg):
        """Process map data for path planning"""
        self.map_resolution = msg.info.resolution
        self.map_origin = [msg.info.origin.position.x, msg.info.origin.position.y]
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        
        # Convert map data to costmap
        self.costmap = np.array(msg.data).reshape(self.map_height, self.map_width)
        
        self.get_logger().info(f'Map received: {self.map_width}x{self.map_height} with resolution {self.map_resolution}')
    
    def plan_path(self, start_pose, goal_pose):
        """Plan path for bipedal humanoid considering physical constraints"""
        if self.costmap is None:
            return None
        
        # Convert poses to grid coordinates
        start_grid = self.pose_to_grid_coords(start_pose)
        goal_grid = self.pose_to_grid_coords(goal_pose)
        
        # Check if start and goal are valid positions
        if not self.is_valid_position(start_grid[0], start_grid[1]) or \
           not self.is_valid_position(goal_grid[0], goal_grid[1]):
            self.get_logger().warn('Start or goal position is not valid')
            return None
        
        # Use A* algorithm with bipedal constraints
        path = self.bipedal_astar(start_grid, goal_grid)
        
        if path:
            # Convert grid path back to world coordinates
            world_path = []
            for grid_x, grid_y in path:
                world_x, world_y = self.grid_to_world_coords(grid_x, grid_y)
                world_path.append((world_x, world_y))
            return world_path
        else:
            return None
    
    def bipedal_astar(self, start, goal):
        """A* pathfinding algorithm with bipedal humanoid constraints"""
        # Check if start and goal are the same
        if start == goal:
            return [start]
        
        # Initialize open and closed sets
        open_set = [(start, 0 + self.heuristic(start, goal))]  # (position, f_score)
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        
        # Convert open_set to list for easier manipulation
        open_list = [start]
        
        while open_list:
            # Find node with lowest f_score
            current = min(open_list, key=lambda pos: f_score.get(pos, float('inf')))
            
            # Check if we've reached the goal
            if current == goal:
                return self.reconstruct_path(came_from, current)
            
            # Remove current from open set
            open_list.remove(current)
            
            # Check 8 neighbors (with bipedal movement constraints)
            neighbors = self.get_bipedal_neighbors(current)
            
            for neighbor in neighbors:
                # Skip if neighbor is not valid
                if not self.is_valid_position(neighbor[0], neighbor[1]):
                    continue
                
                # Calculate tentative g_score
                tentative_g_score = g_score[current] + self.distance(current, neighbor)
                
                # Check if this path to neighbor is better
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    
                    if neighbor not in open_list:
                        open_list.append(neighbor)
        
        # No path found
        return None
    
    def get_bipedal_neighbors(self, pos):
        """Get valid neighbors for bipedal humanoid movement"""
        x, y = pos
        neighbors = []
        
        # Generate 8-connected neighbors
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue  # Skip current position
                
                nx, ny = x + dx, y + dy
                neighbors.append((nx, ny))
        
        # Filter neighbors based on bipedal constraints
        valid_neighbors = []
        for nx, ny in neighbors:
            if self.is_bipedally_traversable(pos, (nx, ny)):
                valid_neighbors.append((nx, ny))
        
        return valid_neighbors
    
    def is_bipedally_traversable(self, pos1, pos2):
        """Check if transition between two cells is valid for bipedal movement"""
        x1, y1 = pos1
        x2, y2 = pos2
        
        # Check cost of target cell
        if self.get_cost(x2, y2) > 50:  # Cost threshold for traversability
            return False
        
        # Check for steep slopes based on height differences
        # In a real implementation, this would check elevation data
        height_diff = abs(self.get_elevation(x1, y1) - self.get_elevation(x2, y2))
        if height_diff > self.step_height:
            return False
        
        # Additional constraints for bipedal robots
        # For example, avoid diagonal movement that might be difficult for bipedal robots
        # This implementation is simplified
        
        return True
    
    def get_cost(self, x, y):
        """Get cost value at grid position"""
        if 0 <= x < self.map_width and 0 <= y < self.map_height:
            return self.costmap[y, x]
        else:
            return 100  # High cost for out-of-bounds
    
    def get_elevation(self, x, y):
        """Get elevation at grid position (simplified)"""
        # In a real implementation, this would access elevation data
        return 0.0
    
    def heuristic(self, pos1, pos2):
        """Calculate heuristic distance between two positions"""
        x1, y1 = pos1
        x2, y2 = pos2
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    
    def distance(self, pos1, pos2):
        """Calculate distance between two positions"""
        return self.heuristic(pos1, pos2)
    
    def is_valid_position(self, x, y):
        """Check if position is within bounds and not occupied"""
        if not (0 <= x < self.map_width and 0 <= y < self.map_height):
            return False
        
        # Check if cell is free (cost < 50 means traversable)
        if self.get_cost(x, y) >= 50:
            return False
        
        # Check if robot would collide with obstacles
        # Consider robot radius
        for dx in range(-int(self.robot_radius / self.map_resolution), 
                        int(self.robot_radius / self.map_resolution) + 1):
            for dy in range(-int(self.robot_radius / self.map_resolution), 
                            int(self.robot_radius / self.map_resolution) + 1):
                check_x, check_y = x + dx, y + dy
                if 0 <= check_x < self.map_width and 0 <= check_y < self.map_height:
                    if self.get_cost(check_x, check_y) >= 50:
                        return False
        
        return True
    
    def pose_to_grid_coords(self, pose):
        """Convert world pose to grid coordinates"""
        x = int((pose.position.x - self.map_origin[0]) / self.map_resolution)
        y = int((pose.position.y - self.map_origin[1]) / self.map_resolution)
        return (x, y)
    
    def grid_to_world_coords(self, grid_x, grid_y):
        """Convert grid coordinates to world coordinates"""
        x = grid_x * self.map_resolution + self.map_origin[0]
        y = grid_y * self.map_resolution + self.map_origin[1]
        return (x, y)
    
    def reconstruct_path(self, came_from, current):
        """Reconstruct path from A* result"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        
        path.reverse()
        return path
    
    def create_path_message(self, waypoints):
        """Create Path message from list of waypoints"""
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'
        
        for x, y in waypoints:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            
            path_msg.poses.append(pose)
        
        return path_msg
    
    def publish_path_visualization(self, path):
        """Publish visualization markers for the path"""
        marker_array = MarkerArray()
        
        # Clear old markers
        clear_marker = Marker()
        clear_marker.header.frame_id = 'map'
        clear_marker.header.stamp = self.get_clock().now().to_msg()
        clear_marker.ns = 'path'
        clear_marker.id = 0
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)
        
        # Draw path as connected line
        line_marker = Marker()
        line_marker.header.frame_id = 'map'
        line_marker.header.stamp = self.get_clock().now().to_msg()
        line_marker.ns = 'path'
        line_marker.id = self.marker_id
        self.marker_id += 1
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD
        line_marker.scale.x = 0.05  # Line width
        
        # Set color (green)
        line_marker.color.r = 0.0
        line_marker.color.g = 1.0
        line_marker.color.b = 0.0
        line_marker.color.a = 1.0
        
        # Add points to the line
        for x, y in path:
            point = Point()
            point.x = float(x)
            point.y = float(y)
            point.z = 0.05  # Slightly above ground
            line_marker.points.append(point)
        
        marker_array.markers.append(line_marker)
        
        # Draw path waypoints
        for i, (x, y) in enumerate(path):
            waypoint_marker = Marker()
            waypoint_marker.header.frame_id = 'map'
            waypoint_marker.header.stamp = self.get_clock().now().to_msg()
            waypoint_marker.ns = 'waypoints'
            waypoint_marker.id = self.marker_id
            self.marker_id += 1
            waypoint_marker.type = Marker.SPHERE
            waypoint_marker.action = Marker.ADD
            waypoint_marker.scale.x = 0.1
            waypoint_marker.scale.y = 0.1
            waypoint_marker.scale.z = 0.1
            
            # Set color (blue for waypoints)
            waypoint_marker.color.b = 1.0
            waypoint_marker.color.a = 1.0
            
            waypoint_marker.pose.position.x = float(x)
            waypoint_marker.pose.position.y = float(y)
            waypoint_marker.pose.position.z = 0.05
            
            marker_array.markers.append(waypoint_marker)
        
        self.path_viz_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    planner_node = BipedalPathPlannerNode()
    
    try:
        rclpy.spin(planner_node)
    except KeyboardInterrupt:
        pass
    finally:
        planner_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Summary

This module has covered the implementation of advanced perception, training, and navigation capabilities using NVIDIA Isaac technologies. We've explored:

1. NVIDIA Isaac Sim for photorealistic simulation, synthetic data generation, and domain randomization
2. Isaac ROS for hardware-accelerated perception pipelines, including Visual SLAM and sensor fusion
3. Nav2 for autonomous navigation, with specific adaptations for bipedal humanoid movement

The implementation provides a comprehensive foundation for creating intelligent humanoid robots capable of perceiving, understanding, and navigating complex environments using state-of-the-art tools and techniques from the NVIDIA Isaac ecosystem.

## Exercises

1. Implement a more sophisticated domain randomization technique that changes material properties during training.
2. Enhance the VSLAM implementation with loop closure detection for better long-term mapping.
3. Develop a custom controller for Nav2 that specifically accounts for humanoid robot dynamics.
4. Integrate object detection into the perception pipeline using Isaac ROS accelerated processing.
5. Implement a behavior tree for high-level humanoid robot planning and control.