# Module 4: Vision-Language-Action (VLA) - Convergence of LLMs and Robotics

## Overview

This module focuses on the convergence of Large Language Models (LLMs) and robotics through Vision-Language-Action (VLA) frameworks. This cutting-edge approach enables humanoid robots to understand natural language commands, perceive their environment visually, and execute complex tasks by bridging high-level cognitive planning with low-level motor control. The module covers voice command processing, vision-language understanding, cognitive planning with LLMs, and the mapping of high-level plans to robot behaviors.

## Learning Objectives

By the end of this module, you will be able to:
- Implement voice-to-action systems using OpenAI Whisper for voice command input
- Develop vision-language understanding systems for object detection and scene analysis
- Design cognitive planning architectures using LLMs to convert natural language into ROS 2 action sequences
- Map high-level plans to low-level robot behaviors using ROS 2 controllers
- Integrate all components into a complete Vision-Language-Action loop
- Create a capstone project featuring an autonomous humanoid robot executing complex tasks

## 1. Voice-to-Action: Voice Command Input Using OpenAI Whisper

Voice-to-action systems enable humanoid robots to understand and execute spoken commands. OpenAI Whisper is a state-of-the-art speech recognition model that can convert spoken commands into text, which can then be processed by AI systems.

### Implementing Voice Command Processing

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import AudioData
import openai
from openai import OpenAI
import numpy as np
import pyaudio
import wave
import threading
import queue
import json
import time

class VoiceToActionNode(Node):
    def __init__(self):
        super().__init__('voice_to_action_node')
        
        # Initialize OpenAI client
        # Note: In a real implementation, you would use your OpenAI API key
        # self.openai_client = OpenAI(api_key='your-openai-api-key')
        
        # Publishers
        self.command_pub = self.create_publisher(String, '/user_command', 10)
        self.action_pub = self.create_publisher(String, '/robot_action', 10)
        
        # Audio parameters
        self.chunk = 1024  # Buffer size
        self.format = pyaudio.paInt16  # 16-bit resolution
        self.channels = 1  # Mono
        self.rate = 44100  # 44.1kHz sampling rate
        
        # Voice processing variables
        self.audio_queue = queue.Queue()
        self.is_listening = False
        self.recording = False
        self.audio_data = []
        
        # Start audio listening thread
        self.audio_thread = threading.Thread(target=self.audio_capture_loop)
        self.audio_thread.daemon = True
        self.audio_thread.start()
        
        # Timer for continuous voice detection
        self.voice_timer = self.create_timer(0.1, self.check_for_voice)
        
        # Initialize audio interface
        self.p = pyaudio.PyAudio()
        self.stream = None
        
        self.get_logger().info('Voice-to-Action node initialized')
    
    def check_for_voice(self):
        """Check for voice activity and start recording if detected"""
        if not self.recording:
            # Simulate voice activity detection
            # In a real implementation, you would use VAD (Voice Activity Detection)
            # or continuously analyze audio for voice activity
            if self.is_voice_active():
                self.start_recording()
    
    def is_voice_active(self):
        """Check if voice activity is detected"""
        # This is a simplified check; in practice, you'd implement VAD
        return True  # For demo purposes
    
    def start_recording(self):
        """Start capturing audio"""
        self.get_logger().info('Starting audio recording...')
        
        # Open audio stream
        self.stream = self.p.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )
        
        self.recording = True
        self.audio_data = []
        
        # Schedule recording timeout
        self.create_timer(5.0, self.stop_recording)  # Record for 5 seconds max
    
    def stop_recording(self):
        """Stop capturing audio and process it"""
        if self.recording and self.stream:
            self.get_logger().info('Stopping audio recording...')
            
            # Stop and close the stream
            self.stream.stop_stream()
            self.stream.close()
            
            # Process the recorded audio
            audio_bytes = self.get_audio_bytes()
            if audio_bytes:
                # Transcribe audio using OpenAI Whisper (in a real implementation)
                # transcript = self.transcribe_audio(audio_bytes)
                # For this example, we'll simulate a transcript
                transcript = self.simulate_transcription()
                
                self.process_command(transcript)
        
        self.recording = False
    
    def get_audio_bytes(self):
        """Get recorded audio as bytes"""
        # In a real implementation, you would return the actual recorded audio data
        # For simulation purposes:
        return b'simulated_audio_data'
    
    def simulate_transcription(self):
        """Simulate Whisper transcription"""
        # In a real implementation, this would call the Whisper API
        # For demo purposes, we'll return a simulated command
        commands = [
            "Go to the kitchen",
            "Pick up the red bottle",
            "Clean the table",
            "Turn off the light",
            "Find the book"
        ]
        import random
        return random.choice(commands)
    
    def transcribe_audio(self, audio_bytes):
        """Transcribe audio to text using OpenAI Whisper"""
        try:
            # In a real implementation:
            # transcript = self.openai_client.audio.transcriptions.create(
            #     model="whisper-1",
            #     file=audio_bytes
            # )
            # return transcript.text
            pass
        except Exception as e:
            self.get_logger().error(f'Error transcribing audio: {e}')
            return ""
    
    def process_command(self, command):
        """Process the transcribed command"""
        if not command.strip():
            return
        
        self.get_logger().info(f'Processing command: "{command}"')
        
        # Publish the command
        cmd_msg = String()
        cmd_msg.data = command
        self.command_pub.publish(cmd_msg)
        
        # Convert natural language to actions using LLM (covered in next section)
        actions = self.natural_language_to_actions(command)
        
        # Publish the planned actions
        for action in actions:
            action_msg = String()
            action_msg.data = action
            self.action_pub.publish(action_msg)
    
    def audio_capture_loop(self):
        """Continuous audio capture loop"""
        # This is handled by the timer-based recording for simplicity
        # In a real implementation, you might use a continuous capture
        pass

def main(args=None):
    rclpy.init(args=args)
    voice_node = VoiceToActionNode()
    
    try:
        rclpy.spin(voice_node)
    except KeyboardInterrupt:
        pass
    finally:
        voice_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Advanced Voice Command Processing with Context

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
import numpy as np
import threading
import re
import requests
import json

class AdvancedVoiceToActionNode(Node):
    def __init__(self):
        super().__init__('advanced_voice_to_action_node')
        
        # Publishers
        self.action_pub = self.create_publisher(String, '/high_level_action', 10)
        self.command_pub = self.create_publisher(String, '/processed_command', 10)
        
        # Subscribers
        self.voice_cmd_sub = self.create_subscription(
            String, '/user_command', self.voice_command_callback, 10)
        
        # Context management
        self.room_context = "kitchen"  # Current room context
        self.object_context = {}  # Objects in current environment
        self.task_queue = []  # Queue of tasks to execute
        
        # Command patterns
        self.command_patterns = {
            'navigation': {
                'go to kitchen': ['go_to_kitchen', 'navigate_to_kitchen'],
                'go to bedroom': ['go_to_bedroom', 'navigate_to_bedroom'],
                'go to living room': ['go_to_living_room', 'navigate_to_living_room'],
                'go to bathroom': ['go_to_bathroom', 'navigate_to_bathroom'],
            },
            'manipulation': {
                'pick up (.*)': ['pick_up_object'],
                'grab (.*)': ['pick_up_object'],
                'take (.*)': ['pick_up_object'],
                'put (.*) down': ['put_down_object'],
                'place (.*)': ['place_object'],
            },
            'cleaning': {
                'clean (.*)': ['clean_object_or_area'],
                'clean up': ['clean_area'],
                'tidy up': ['clean_area'],
            },
        }
        
        self.get_logger().info('Advanced Voice-to-Action node initialized')
    
    def voice_command_callback(self, msg):
        """Process incoming voice command"""
        command = msg.data.lower().strip()
        self.get_logger().info(f'Received voice command: "{command}"')
        
        # Publish the original command
        original_cmd = String()
        original_cmd.data = f"ORIGINAL: {command}"
        self.command_pub.publish(original_cmd)
        
        # Parse and convert to actions
        actions = self.parse_command_to_actions(command)
        
        # Add actions to queue for execution
        self.task_queue.extend(actions)
        
        # Execute first action
        if self.task_queue:
            next_action = self.task_queue.pop(0)
            self.execute_action(next_action)
    
    def parse_command_to_actions(self, command):
        """Parse natural language command into executable actions"""
        actions = []
        
        # Check each command pattern
        for category, patterns in self.command_patterns.items():
            for pattern, action_list in patterns.items():
                # Use regex for pattern matching
                if re.search(pattern, command):
                    for action in action_list:
                        # Extract object name from command if needed
                        match = re.search(pattern, command)
                        if match:
                            # Extract the object name (first capture group)
                            if match.groups():
                                object_name = match.group(1)
                                actions.append(f"{action}:{object_name}")
                            else:
                                actions.append(action)
        
        # Handle specific cases that need more complex processing
        actions.extend(self.handle_special_cases(command))
        
        return actions
    
    def handle_special_cases(self, command):
        """Handle more complex command patterns"""
        actions = []
        
        # Complex commands that require multiple actions
        if "from the kitchen" in command and "to the bedroom" in command:
            # Example: "Pick up the bottle from the kitchen and bring it to the bedroom"
            actions.append("navigate_to_kitchen")
            actions.append("find_object:bottle")
            actions.append("grasp_object:bottle")
            actions.append("navigate_to_bedroom")
            actions.append("place_object:bottle")
        
        elif "and" in command:
            # Handle compound commands like "turn on the light and clean the table"
            sub_commands = command.split(" and ")
            for sub_cmd in sub_commands:
                actions.extend(self.parse_command_to_actions(sub_cmd.strip()))
        
        return actions
    
    def execute_action(self, action):
        """Execute a specific action"""
        self.get_logger().info(f'Executing action: "{action}"')
        
        # Publish action for execution by other nodes
        action_msg = String()
        action_msg.data = action
        self.action_pub.publish(action_msg)
    
    def get_current_context(self):
        """Get current environmental context"""
        context = {
            'room': self.room_context,
            'objects': self.object_context,
            'time': self.get_clock().now().seconds_nanoseconds()
        }
        return json.dumps(context)

def main(args=None):
    rclpy.init(args=args)
    advanced_voice_node = AdvancedVoiceToActionNode()
    
    try:
        rclpy.spin(advanced_voice_node)
    except KeyboardInterrupt:
        pass
    finally:
        advanced_voice_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 2. Vision-Language Understanding: Object Detection and Scene Understanding

Vision-language understanding combines visual perception with natural language processing to enable robots to understand their environment and connect it with human commands. This section covers object detection and scene understanding using state-of-the-art computer vision techniques.

### Object Detection with ROS 2 Integration

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from vision_msgs.msg import Detection2DArray, ObjectHypothesisWithPose
from cv_bridge import CvBridge
from std_msgs.msg import Header
import cv2
import numpy as np
import threading
import json

class VisionLanguageUnderstandingNode(Node):
    def __init__(self):
        super().__init__('vision_language_understanding_node')
        
        # Initialize computer vision
        self.bridge = CvBridge()
        
        # Publishers
        self.object_detections_pub = self.create_publisher(Detection2DArray, '/object_detections', 10)
        self.scene_description_pub = self.create_publisher(String, '/scene_description', 10)
        self.object_context_pub = self.create_publisher(String, '/object_context', 10)
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/rgb/image_raw', self.image_callback, 10)
        
        # Object detection parameters
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.4
        
        # COCO class names for YOLO
        self.coco_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 
            'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 
            'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 
            'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 
            'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 
            'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        # Detected objects context
        self.current_objects = {}
        
        self.get_logger().info('Vision-Language Understanding node initialized')
    
    def image_callback(self, msg):
        """Process incoming camera image for object detection"""
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Perform object detection
            detections = self.detect_objects(cv_image)
            
            # Create and publish detection array
            detection_array_msg = self.create_detection_array_message(detections, msg.header)
            self.object_detections_pub.publish(detection_array_msg)
            
            # Update context with detected objects
            self.current_objects = self.extract_object_context(detections)
            
            # Create scene description
            scene_description = self.generate_scene_description(detections)
            
            # Publish scene description
            scene_msg = String()
            scene_msg.data = scene_description
            self.scene_description_pub.publish(scene_msg)
            
            # Publish object context
            context_msg = String()
            context_msg.data = json.dumps(self.current_objects)
            self.object_context_pub.publish(context_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')
    
    def detect_objects(self, image):
        """Detect objects in the image using a pre-trained model"""
        # In a real implementation, this would use a model like YOLO or Detectron2
        # For this example, we'll simulate object detection
        
        # Simulated detections - in real implementation, this would come from a trained model
        height, width = image.shape[:2]
        
        simulated_detections = [
            {'label': 'person', 'confidence': 0.95, 
             'bbox': [width*0.1, height*0.3, width*0.3, height*0.6]},
            {'label': 'couch', 'confidence': 0.88,
             'bbox': [width*0.4, height*0.5, width*0.8, height*0.9]},
            {'label': 'bottle', 'confidence': 0.76,
             'bbox': [width*0.6, height*0.7, width*0.65, height*0.85]},
            {'label': 'dining table', 'confidence': 0.92,
             'bbox': [width*0.2, height*0.7, width*0.9, height*0.95]}
        ]
        
        # Filter by confidence threshold
        detections = [det for det in simulated_detections if det['confidence'] >= self.confidence_threshold]
        
        return detections
    
    def create_detection_array_message(self, detections, header):
        """Create Detection2DArray message from detection results"""
        detection_array = Detection2DArray()
        detection_array.header = header
        
        for det in detections:
            detection = Detection2D()
            
            # Set header
            detection.header = header
            
            # Set ID (optional)
            detection.id = det['label']
            
            # Set bounding box (converted from x1,y1,x2,y2 to x,y,w,h)
            x1, y1, x2, y2 = det['bbox']
            width = x2 - x1
            height = y2 - y1
            detection.bbox.center.x = x1 + width / 2
            detection.bbox.center.y = y1 + height / 2
            detection.bbox.size_x = width
            detection.bbox.size_y = height
            
            # Set scores
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = det['label']
            hypothesis.hypothesis.score = det['confidence']
            detection.results.append(hypothesis)
            
            detection_array.detections.append(detection)
        
        return detection_array
    
    def extract_object_context(self, detections):
        """Extract object context for scene understanding"""
        object_context = {}
        
        for det in detections:
            label = det['label']
            confidence = det['confidence']
            bbox = det['bbox']
            
            if label not in object_context:
                object_context[label] = []
            
            # Add object details
            obj_details = {
                'bbox': bbox,
                'confidence': confidence,
                'centroid': ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            }
            
            object_context[label].append(obj_details)
        
        return object_context
    
    def generate_scene_description(self, detections):
        """Generate natural language description of the scene"""
        if not detections:
            return "The scene is empty or no objects were detected."
        
        # Count objects
        object_counts = {}
        for det in detections:
            label = det['label']
            if label in object_counts:
                object_counts[label] += 1
            else:
                object_counts[label] = 1
        
        # Create description
        description = "Scene contains: "
        items = []
        
        for label, count in object_counts.items():
            if count == 1:
                items.append(f"a {label}")
            else:
                items.append(f"{count} {label}s")
        
        description += ", ".join(items) + "."
        
        return description

def main(args=None):
    rclpy.init(args=args)
    vision_node = VisionLanguageUnderstandingNode()
    
    try:
        rclpy.spin(vision_node)
    except KeyboardInterrupt:
        pass
    finally:
        vision_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Scene Understanding with Spatial Relationships

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import Point
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge
import numpy as np
import json

class AdvancedSceneUnderstandingNode(Node):
    def __init__(self):
        super().__init__('advanced_scene_understanding_node')
        
        # Initialize computer vision
        self.bridge = CvBridge()
        
        # Publishers
        self.spatial_context_pub = self.create_publisher(String, '/spatial_context', 10)
        self.scene_graph_pub = self.create_publisher(String, '/scene_graph', 10)
        self.semantic_map_pub = self.create_publisher(String, '/semantic_map', 10)
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/rgb/image_raw', self.image_callback, 10)
        self.detections_sub = self.create_subscription(
            Detection2DArray, '/object_detections', self.detections_callback, 10)
        
        # Scene understanding parameters
        self.spatial_threshold = 0.3  # Threshold for spatial relationships (normalized coordinates)
        
        # Store detections for scene analysis
        self.current_detections = []
        self.spatial_context = {}
        
        self.get_logger().info('Advanced Scene Understanding node initialized')
    
    def detections_callback(self, msg):
        """Process detection messages from the object detection node"""
        # Store detection data for scene analysis
        self.current_detections = []
        
        for detection in msg.detections:
            if detection.results:  # If there are results
                # Get the best hypothesis
                best_result = max(detection.results, 
                                key=lambda x: x.hypothesis.score)
                
                obj_info = {
                    'label': best_result.hypothesis.class_id,
                    'confidence': best_result.hypothesis.score,
                    'center': (detection.bbox.center.x, detection.bbox.center.y),
                    'size': (detection.bbox.size_x, detection.bbox.size_y)
                }
                
                self.current_detections.append(obj_info)
    
    def image_callback(self, msg):
        """Process image for advanced scene understanding"""
        if not self.current_detections:
            return  # Wait for detections
        
        # Analyze spatial relationships
        relationships = self.analyze_spatial_relationships()
        
        # Create spatial context
        self.spatial_context = self.create_spatial_context(relationships)
        
        # Publish spatial context
        context_msg = String()
        context_msg.data = json.dumps(self.spatial_context, indent=2)
        self.spatial_context_pub.publish(context_msg)
        
        # Create scene graph
        scene_graph = self.create_scene_graph(relationships)
        
        # Publish scene graph
        graph_msg = String()
        graph_msg.data = json.dumps(scene_graph, indent=2)
        self.scene_graph_pub.publish(graph_msg)
        
        # Create semantic map
        semantic_map = self.create_semantic_map()
        
        # Publish semantic map
        map_msg = String()
        map_msg.data = json.dumps(semantic_map, indent=2)
        self.semantic_map_pub.publish(map_msg)
    
    def analyze_spatial_relationships(self):
        """Analyze spatial relationships between detected objects"""
        relationships = []
        
        for i, obj1 in enumerate(self.current_detections):
            for j, obj2 in enumerate(self.current_detections):
                if i != j:  # Don't compare object with itself
                    # Calculate spatial relationship
                    rel = self.calculate_relationship(obj1, obj2)
                    if rel:
                        relationships.append(rel)
        
        return relationships
    
    def calculate_relationship(self, obj1, obj2):
        """Calculate spatial relationship between two objects"""
        # Calculate distance between object centers
        dx = obj2['center'][0] - obj1['center'][0]
        dy = obj2['center'][1] - obj1['center'][1]
        distance = np.sqrt(dx**2 + dy**2)
        
        # Determine spatial relationship based on position
        if abs(dx) > abs(dy):
            # Horizontal relationship
            if dx > 0:
                relationship = "to the right of"
            else:
                relationship = "to the left of"
        else:
            # Vertical relationship
            if dy > 0:
                relationship = "below"
            else:
                relationship = "above"
        
        # Only include relationships that are meaningful (not too far)
        # This threshold would be image-size dependent in practice
        if distance < 300:  # pixels in this example
            return {
                'subject': obj1['label'],
                'relationship': relationship,
                'object': obj2['label'],
                'distance': distance
            }
        
        return None
    
    def create_spatial_context(self, relationships):
        """Create spatial context from relationships"""
        context = {
            'objects': self.current_detections,
            'relationships': relationships,
            'layout_description': self.describe_layout(relationships)
        }
        
        return context
    
    def describe_layout(self, relationships):
        """Generate natural language description of the layout"""
        if not relationships:
            return "No clear spatial relationships identified in the scene."
        
        # Create a summary of the layout
        summary = "Layout summary: "
        
        # Group relationships by main objects
        obj_rels = {}
        for rel in relationships:
            subj = rel['subject']
            if subj not in obj_rels:
                obj_rels[subj] = []
            obj_rels[subj].append(rel)
        
        # Describe each object's position
        descriptions = []
        for obj, rels in obj_rels.items():
            if len(rels) > 0:
                # Take first few relationships to describe the object's position
                desc = f"{obj} is"
                rel_descs = []
                for rel in rels[:2]:  # Limit to first 2 relationships
                    rel_descs.append(f"{rel['relationship']} {rel['object']}")
                desc += " " + " and ".join(rel_descs)
                descriptions.append(desc)
        
        return summary + "; ".join(descriptions) + "."
    
    def create_scene_graph(self, relationships):
        """Create a scene graph representation"""
        nodes = []
        edges = []
        
        # Create nodes for each object
        for obj in self.current_detections:
            node = {
                'id': obj['label'],
                'type': 'object',
                'confidence': obj['confidence'],
                'position': obj['center']
            }
            nodes.append(node)
        
        # Create edges for spatial relationships
        for rel in relationships:
            edge = {
                'source': rel['subject'],
                'target': rel['object'],
                'relationship': rel['relationship'],
                'distance': rel['distance']
            }
            edges.append(edge)
        
        return {
            'nodes': nodes,
            'edges': edges
        }
    
    def create_semantic_map(self):
        """Create a semantic map of the environment"""
        # Semantic map includes objects and their spatial context
        semantic_map = {
            'objects': [
                {
                    'name': obj['label'],
                    'confidence': obj['confidence'],
                    'position': obj['center'],
                    'size': obj['size']
                }
                for obj in self.current_detections
            ],
            'spatial_context': self.spatial_context,
            'timestamp': self.get_clock().now().seconds_nanoseconds()
        }
        
        return semantic_map

def main(args=None):
    rclpy.init(args=args)
    scene_node = AdvancedSceneUnderstandingNode()
    
    try:
        rclpy.spin(scene_node)
    except KeyboardInterrupt:
        pass
    finally:
        scene_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 3. Cognitive Planning: Using LLMs to Convert Natural Language Commands

Cognitive planning bridges the gap between high-level natural language commands and executable robot actions. Large Language Models (LLMs) play a crucial role in parsing natural language, understanding intent, and generating structured action sequences.

### LLM-based Cognitive Planner

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose
import json
import re
import time
from typing import List, Dict, Any

class CognitivePlanningNode(Node):
    def __init__(self):
        super().__init__('cognitive_planning_node')
        
        # Publishers
        self.plan_pub = self.create_publisher(String, '/action_plan', 10)
        self.intent_pub = self.create_publisher(String, '/parsed_intent', 10)
        
        # Subscribers
        self.command_sub = self.create_subscription(
            String, '/processed_command', self.command_callback, 10)
        self.scene_sub = self.create_subscription(
            String, '/scene_description', self.scene_callback, 10)
        self.object_context_sub = self.create_subscription(
            String, '/object_context', self.object_context_callback, 10)
        
        # Context storage
        self.current_scene_description = ""
        self.current_object_context = {}
        
        # Action mapping from natural language to ROS actions
        self.action_map = {
            'navigate': ['go to', 'move to', 'travel to', 'walk to', 'head to'],
            'grasp': ['pick up', 'grab', 'take', 'collect', 'lift'],
            'place': ['put down', 'place', 'set down', 'release'],
            'manipulate': ['move', 'push', 'pull', 'turn', 'open', 'close'],
            'inspect': ['look at', 'examine', 'check', 'see'],
            'clean': ['clean', 'tidy', 'organize', 'clear'],
            'wait': ['wait', 'stop', 'pause'],
            'communicate': ['say', 'speak', 'tell', 'notify']
        }
        
        # Object aliases for better recognition
        self.object_aliases = {
            'bottle': ['water bottle', 'soda bottle', 'plastic bottle'],
            'cup': ['glass', 'mug', 'coffee cup', 'tea cup'],
            'book': ['novel', 'textbook', 'magazine'],
            'chair': ['seat', 'stool', 'armchair'],
            'table': ['desk', 'coffee table', 'dining table'],
            'light': ['lamp', 'bulb', 'chandelier']
        }
        
        self.get_logger().info('Cognitive Planning node initialized')
    
    def command_callback(self, msg):
        """Process incoming natural language command"""
        # Check if it's an original command or processed command
        if msg.data.startswith("ORIGINAL:"):
            command = msg.data[9:].strip()  # Remove "ORIGINAL: " prefix
        else:
            # This is a processed command from voice to action
            command = msg.data
        
        self.get_logger().info(f'Processing command: "{command}"')
        
        # Parse the command using LLM techniques
        parsed_command = self.parse_command(command)
        
        # Generate action plan based on command and context
        plan = self.generate_action_plan(parsed_command)
        
        # Publish the parsed intent
        intent_msg = String()
        intent_msg.data = json.dumps(parsed_command, indent=2)
        self.intent_pub.publish(intent_msg)
        
        # Publish the action plan
        plan_msg = String()
        plan_msg.data = json.dumps(plan, indent=2)
        self.plan_pub.publish(plan_msg)
    
    def scene_callback(self, msg):
        """Update current scene description"""
        self.current_scene_description = msg.data
    
    def object_context_callback(self, msg):
        """Update current object context"""
        try:
            self.current_object_context = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().error('Failed to parse object context JSON')
    
    def parse_command(self, command: str) -> Dict[str, Any]:
        """Parse natural language command into structured format"""
        command_lower = command.lower()
        
        # Extract action type
        action_type = self.identify_action_type(command_lower)
        
        # Extract object(s)
        objects = self.extract_objects(command_lower)
        
        # Extract location if present
        location = self.extract_location(command_lower)
        
        # Extract quantity if present
        quantity = self.extract_quantity(command_lower)
        
        # Extract additional details
        details = self.extract_details(command_lower)
        
        parsed_command = {
            'raw_command': command,
            'action_type': action_type,
            'objects': objects,
            'location': location,
            'quantity': quantity,
            'details': details,
            'timestamp': time.time()
        }
        
        return parsed_command
    
    def identify_action_type(self, command: str) -> str:
        """Identify the main action type from the command"""
        for action_type, keywords in self.action_map.items():
            for keyword in keywords:
                if keyword in command:
                    return action_type
        
        return 'unknown'
    
    def extract_objects(self, command: str) -> List[str]:
        """Extract object names from the command"""
        objects = []
        
        # Check for exact object names
        for obj_name in self.action_map.get('grasp', []):  # Using grasp as reference
            if obj_name in command:
                objects.append(obj_name)
        
        # Check for object aliases
        for canonical_name, aliases in self.object_aliases.items():
            for alias in aliases:
                if alias in command:
                    objects.append(canonical_name)
        
        # Use regex to extract potential objects
        # Simple approach: look for nouns after action words
        action_patterns = [r'pick up (\w+)', r'grab (\w+)', r'go to (\w+)', r'clean (\w+)']
        for pattern in action_patterns:
            matches = re.findall(pattern, command)
            for match in matches:
                objects.append(match)
        
        # Remove duplicates while preserving order
        unique_objects = []
        for obj in objects:
            if obj not in unique_objects:
                unique_objects.append(obj)
        
        return unique_objects
    
    def extract_location(self, command: str) -> str:
        """Extract location from the command"""
        locations = ['kitchen', 'bedroom', 'living room', 'bathroom', 'office', 'dining room', 
                     'hallway', 'garage', 'garden', 'patio', 'front door']
        
        for location in locations:
            if location in command:
                return location
        
        return 'unknown'
    
    def extract_quantity(self, command: str) -> int:
        """Extract quantity from the command"""
        # Simple quantity extraction
        quantity_words = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            '1': 1, '2': 2, '3': 3, '4': 4, '5': 5
        }
        
        for word, num in quantity_words.items():
            if word in command:
                return num
        
        return 1  # Default to 1
    
    def extract_details(self, command: str) -> Dict[str, str]:
        """Extract additional details from the command"""
        details = {}
        
        # Look for color adjectives
        colors = ['red', 'blue', 'green', 'yellow', 'white', 'black', 'orange', 'purple', 'pink']
        for color in colors:
            if color in command:
                details['color'] = color
                break
        
        # Look for size adjectives
        sizes = ['big', 'small', 'large', 'tiny', 'medium', 'huge']
        for size in sizes:
            if size in command:
                details['size'] = size
                break
        
        return details
    
    def generate_action_plan(self, parsed_command: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate executable action plan from parsed command"""
        plan = []
        
        action_type = parsed_command['action_type']
        objects = parsed_command['objects']
        location = parsed_command['location']
        
        if action_type == 'navigate':
            # Plan to navigate to location
            plan.append({
                'action': 'navigate_to_location',
                'location': location,
                'description': f'Navigate to {location}'
            })
        
        elif action_type == 'grasp':
            # Plan to find and grasp object
            for obj in objects:
                plan.append({
                    'action': 'find_object',
                    'object': obj,
                    'description': f'Find {obj}'
                })
                
                plan.append({
                    'action': 'approach_object',
                    'object': obj,
                    'description': f'Approach {obj}'
                })
                
                plan.append({
                    'action': 'grasp_object',
                    'object': obj,
                    'description': f'Grasp {obj}'
                })
        
        elif action_type == 'place':
            # Plan to place object at location
            for obj in objects:
                if location != 'unknown':
                    plan.append({
                        'action': 'navigate_to_location',
                        'location': location,
                        'description': f'Navigate to {location}'
                    })
                
                plan.append({
                    'action': 'place_object',
                    'object': obj,
                    'location': location,
                    'description': f'Place {obj} at {location}'
                })
        
        elif action_type == 'clean':
            # Plan to clean an area or object
            if objects:
                for obj in objects:
                    plan.append({
                        'action': 'clean_object',
                        'object': obj,
                        'description': f'Clean {obj}'
                    })
            else:
                plan.append({
                    'action': 'clean_area',
                    'location': location,
                    'description': f'Clean {location}'
                })
        
        elif action_type == 'manipulate':
            # Plan for general manipulation
            for obj in objects:
                plan.append({
                    'action': 'manipulate_object',
                    'object': obj,
                    'description': f'Manipulate {obj}'
                })
        
        elif action_type == 'inspect':
            # Plan to inspect an object
            for obj in objects:
                plan.append({
                    'action': 'navigate_to_object',
                    'object': obj,
                    'description': f'Navigate to {obj}'
                })
                
                plan.append({
                    'action': 'inspect_object',
                    'object': obj,
                    'description': f'Inspect {obj}'
                })
        
        # Add a completion step to any plan
        if plan:
            plan.append({
                'action': 'task_complete',
                'description': 'Task completed successfully'
            })
        
        return plan

def main(args=None):
    rclpy.init(args=args)
    planning_node = CognitivePlanningNode()
    
    try:
        rclpy.spin(planning_node)
    except KeyboardInterrupt:
        pass
    finally:
        planning_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Enhanced Cognitive Planning with LLM Integration

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from action_msgs.msg import GoalStatus
from lifecycle_msgs.msg import Transition
import json
import openai
from openai import OpenAI
import asyncio
import threading
from queue import Queue

class EnhancedCognitivePlanningNode(Node):
    def __init__(self):
        super().__init__('enhanced_cognitive_planning_node')
        
        # Publishers
        self.enhanced_plan_pub = self.create_publisher(String, '/enhanced_action_plan', 10)
        self.llm_response_pub = self.create_publisher(String, '/llm_response', 10)
        
        # Subscribers
        self.command_sub = self.create_subscription(
            String, '/processed_command', self.command_callback, 10)
        self.scene_sub = self.create_subscription(
            String, '/scene_description', self.scene_callback, 10)
        self.spatial_context_sub = self.create_subscription(
            String, '/spatial_context', self.spatial_context_callback, 10)
        
        # Initialize OpenAI client
        # self.openai_client = OpenAI(api_key='your-openai-api-key')
        
        # Context storage
        self.current_scene_description = ""
        self.spatial_context = {}
        
        # LLM prompt templates
        self.task_decomposition_prompt = """You are a helpful assistant that helps decompose high-level tasks into specific robotic actions. Given the following information:

Current Scene: {scene}
Spatial Relationships: {spatial_context}

Decompose this high-level command: "{command}"

Into a sequence of specific robot actions. Each action should be:
- Concrete and executable
- Consider the current scene and spatial relationships
- Take into account the robot's capabilities

Return the actions as a JSON list of action objects with 'action', 'target', 'location', and 'description' fields."""

        self.intent_classification_prompt = """Classify this command based on the intent:
Command: "{command}"

Available intents: navigation, manipulation, inspection, communication, cleaning, waiting

Return only the intent as a single word."""

        self.get_logger().info('Enhanced Cognitive Planning node initialized')
    
    def command_callback(self, msg):
        """Process incoming command with LLM enhancement"""
        if msg.data.startswith("ORIGINAL:"):
            command = msg.data[9:].strip()
        else:
            command = msg.data
        
        self.get_logger().info(f'Processing command with LLM: "{command}"')
        
        # Use threading to avoid blocking the main ROS loop
        thread = threading.Thread(
            target=self.process_command_with_llm,
            args=(command, self.current_scene_description, json.dumps(self.spatial_context))
        )
        thread.daemon = True
        thread.start()
    
    def scene_callback(self, msg):
        """Update scene description"""
        self.current_scene_description = msg.data
    
    def spatial_context_callback(self, msg):
        """Update spatial context"""
        try:
            self.spatial_context = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().error('Failed to parse spatial context JSON')
    
    def process_command_with_llm(self, command, scene_description, spatial_context):
        """Process command using LLM in a separate thread"""
        try:
            # Classify intent using LLM
            intent = self.classify_intent_with_llm(command)
            
            # Generate action plan using LLM
            action_plan = self.generate_plan_with_llm(command, scene_description, spatial_context)
            
            # Publish LLM response
            llm_response = {
                'command': command,
                'intent': intent,
                'plan': action_plan,
                'timestamp': self.get_clock().now().seconds_nanoseconds()
            }
            
            llm_msg = String()
            llm_msg.data = json.dumps(llm_response, indent=2)
            self.llm_response_pub.publish(llm_msg)
            
            # Publish enhanced action plan
            plan_msg = String()
            plan_msg.data = json.dumps(action_plan, indent=2)
            self.enhanced_plan_pub.publish(plan_msg)
            
            self.get_logger().info(f'LLM processing complete for command: "{command}"')
            
        except Exception as e:
            self.get_logger().error(f'Error in LLM processing: {e}')
    
    def classify_intent_with_llm(self, command):
        """Classify intent using LLM"""
        # In a real implementation:
        # prompt = self.intent_classification_prompt.format(command=command)
        # 
        # try:
        #     response = self.openai_client.chat.completions.create(
        #         model="gpt-3.5-turbo",
        #         messages=[{"role": "user", "content": prompt}],
        #         max_tokens=10,
        #         temperature=0.1
        #     )
        #     intent = response.choices[0].message.content.strip().lower()
        #     return intent
        # except Exception as e:
        #     self.get_logger().error(f'LLM intent classification failed: {e}')
        #     return "unknown"
        
        # For simulation, we'll use a simple rule-based approach
        command_lower = command.lower()
        
        if any(word in command_lower for word in ['go', 'move', 'travel', 'head']):
            return 'navigation'
        elif any(word in command_lower for word in ['pick', 'grab', 'take', 'lift', 'place', 'put']):
            return 'manipulation'
        elif any(word in command_lower for word in ['look', 'examine', 'check', 'see']):
            return 'inspection'
        elif any(word in command_lower for word in ['clean', 'tidy', 'organize', 'clear']):
            return 'cleaning'
        else:
            return 'unknown'
    
    def generate_plan_with_llm(self, command, scene_description, spatial_context):
        """Generate action plan using LLM"""
        # In a real implementation:
        # prompt = self.task_decomposition_prompt.format(
        #     command=command,
        #     scene=scene_description,
        #     spatial_context=spatial_context
        # )
        # 
        # try:
        #     response = self.openai_client.chat.completions.create(
        #         model="gpt-4",
        #         messages=[{"role": "user", "content": prompt}],
        #         max_tokens=500,
        #         temperature=0.3
        #     )
        #     
        #     # Parse the JSON response
        #     plan_text = response.choices[0].message.content.strip()
        #     
        #     # Clean up potential markdown formatting
        #     if plan_text.startswith('```json'):
        #         plan_text = plan_text[7:]  # Remove ```json
        #     if plan_text.endswith('```'):
        #         plan_text = plan_text[:-3]  # Remove ```
        #     
        #     plan = json.loads(plan_text)
        #     return plan
        # except Exception as e:
        #     self.get_logger().error(f'LLM plan generation failed: {e}')
        #     return [{"action": "task_failed", "description": f"Failed to generate plan: {e}"}]
        
        # For simulation, we'll return a sample plan based on the command
        return self.generate_sample_plan(command)
    
    def generate_sample_plan(self, command):
        """Generate a sample plan for demonstration"""
        command_lower = command.lower()
        
        if 'clean' in command_lower:
            return [
                {"action": "navigate_to_location", "location": "kitchen", "description": "Navigate to kitchen"},
                {"action": "find_broom", "description": "Locate cleaning tool"},
                {"action": "clean_area", "area": "counter", "description": "Clean the counter"},
                {"action": "put_away_tool", "description": "Return cleaning tool to storage"},
                {"action": "task_complete", "description": "Task completed successfully"}
            ]
        elif 'pick up' in command_lower or 'grab' in command_lower:
            # Extract object name
            obj_name = "object"
            for word in command_lower.split():
                if word in ['bottle', 'cup', 'book', 'box']:
                    obj_name = word
                    break
            
            return [
                {"action": "find_object", "object": obj_name, "description": f"Locate {obj_name}"},
                {"action": "approach_object", "object": obj_name, "description": f"Approach {obj_name}"},
                {"action": "verify_grasp", "object": obj_name, "description": f"Verify grasp capability for {obj_name}"},
                {"action": "grasp_object", "object": obj_name, "description": f"Grasp {obj_name}"},
                {"action": "lift_object", "object": obj_name, "description": f"Lift {obj_name}"},
                {"action": "task_complete", "description": "Object successfully picked up"}
            ]
        elif 'go to' in command_lower or 'move to' in command_lower:
            # Extract location
            location = "destination"
            for word in ['kitchen', 'bedroom', 'living room', 'bathroom']:
                if word in command_lower:
                    location = word
                    break
            
            return [
                {"action": "localize_robot", "description": "Determine current location"},
                {"action": "plan_path", "destination": location, "description": f"Plan path to {location}"},
                {"action": "navigate_to_location", "location": location, "description": f"Navigate to {location}"},
                {"action": "verify_reached", "location": location, "description": f"Verify reached {location}"},
                {"action": "task_complete", "description": f"Successfully reached {location}"}
            ]
        else:
            return [
                {"action": "parse_failed", "command": command, "description": f"Could not parse command: {command}"},
                {"action": "request_clarification", "description": "Request clarification from user"}
            ]

def main(args=None):
    rclpy.init(args=args)
    enhanced_planning_node = EnhancedCognitivePlanningNode()
    
    try:
        rclpy.spin(enhanced_planning_node)
    except KeyboardInterrupt:
        pass
    finally:
        enhanced_planning_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 4. Action Execution: Mapping High-Level Plans to Low-Level Robot Behaviors

Action execution is the final step in the VLA pipeline, where high-level plans generated by cognitive planners are translated into low-level robot behaviors. This involves converting abstract actions into specific ROS 2 commands that control robot joints, navigation systems, and manipulation capabilities.

### Action Execution Framework

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose, Point, Quaternion
from nav_msgs.msg import Path
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from moveit_msgs.action import MoveGroup
from rclpy.action import ActionClient
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from control_msgs.action import FollowJointTrajectory
import time
import json
import threading

class ActionExecutionNode(Node):
    def __init__(self):
        super().__init__('action_execution_node')
        
        # Publishers
        self.status_pub = self.create_publisher(String, '/action_status', 10)
        self.joint_trajectory_pub = self.create_publisher(JointTrajectory, '/joint_trajectory', 10)
        
        # Subscribers
        self.plan_sub = self.create_subscription(
            String, '/enhanced_action_plan', self.plan_callback, 10)
        
        # Action clients
        self.nav_client = ActionClient(self, MoveGroup, '/move_group')
        self.joint_traj_client = ActionClient(
            self, FollowJointTrajectory, '/joint_trajectory_controller/follow_joint_trajectory'
        )
        
        # Current execution state
        self.current_plan = []
        self.current_action_index = 0
        self.is_executing = False
        self.execution_lock = threading.Lock()
        
        # Robot joint names (for humanoid)
        self.joint_names = [
            'hip_joint', 'knee_joint', 'ankle_joint',    # Leg joints
            'shoulder_joint', 'elbow_joint', 'wrist_joint',  # Arm joints
            'neck_joint'  # Head joint
        ]
        
        # Initialize execution thread
        self.executor_thread = None
        
        self.get_logger().info('Action Execution node initialized')
    
    def plan_callback(self, msg):
        """Process incoming action plan"""
        try:
            plan_data = json.loads(msg.data)
            plan = plan_data if isinstance(plan_data, list) else plan_data.get('plan', [])
            
            self.get_logger().info(f'Received plan with {len(plan)} actions')
            
            # Stop current execution if running
            if self.is_executing:
                self.stop_execution()
            
            # Store new plan
            with self.execution_lock:
                self.current_plan = plan
                self.current_action_index = 0
                self.is_executing = True
            
            # Start execution thread
            self.executor_thread = threading.Thread(target=self.execute_plan)
            self.executor_thread.daemon = True
            self.executor_thread.start()
            
        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in action plan message')
        except Exception as e:
            self.get_logger().error(f'Error processing plan: {e}')
    
    def execute_plan(self):
        """Execute the action plan in a separate thread"""
        while self.is_executing and self.current_action_index < len(self.current_plan):
            with self.execution_lock:
                if self.current_action_index >= len(self.current_plan):
                    break
                
                action = self.current_plan[self.current_action_index]
            
            self.get_logger().info(f'Executing action {self.current_action_index + 1}: {action["action"]}')
            
            # Execute the action
            success = self.execute_single_action(action)
            
            if success:
                # Update action status
                status_msg = String()
                status_msg.data = f"ACTION_COMPLETED: {json.dumps(action)}"
                self.status_pub.publish(status_msg)
                
                # Move to next action
                with self.execution_lock:
                    self.current_action_index += 1
            else:
                # Execution failed
                status_msg = String()
                status_msg.data = f"ACTION_FAILED: {json.dumps(action)}"
                self.status_pub.publish(status_msg)
                
                # Stop execution on failure
                self.is_executing = False
                break
        
        # Plan completed or stopped
        with self.execution_lock:
            self.is_executing = False
        
        status_msg = String()
        status_msg.data = "PLAN_EXECUTION_COMPLETED" if not self.is_executing else "PLAN_EXECUTION_STOPPED"
        self.status_pub.publish(status_msg)
    
    def execute_single_action(self, action):
        """Execute a single action"""
        action_type = action.get('action', '')
        
        try:
            if action_type == 'navigate_to_location':
                return self.execute_navigate_action(action)
            elif action_type == 'find_object':
                return self.execute_find_object_action(action)
            elif action_type == 'grasp_object':
                return self.execute_grasp_action(action)
            elif action_type == 'place_object':
                return self.execute_place_action(action)
            elif action_type == 'clean_area':
                return self.execute_clean_action(action)
            elif action_type == 'manipulate_object':
                return self.execute_manipulate_action(action)
            elif action_type == 'task_complete':
                return self.execute_task_complete(action)
            else:
                return self.execute_generic_action(action)
        except Exception as e:
            self.get_logger().error(f'Error executing action {action_type}: {e}')
            return False
    
    def execute_navigate_action(self, action):
        """Execute navigation action"""
        location = action.get('location', 'unknown')
        
        # In a real implementation, this would call the navigation system
        # For simulation, we'll just log and wait
        self.get_logger().info(f'Navigating to {location}')
        
        # Publish navigation command (in a real implementation)
        # This would involve sending goals to Nav2
        time.sleep(2)  # Simulate navigation time
        
        return True
    
    def execute_find_object_action(self, action):
        """Execute object finding action"""
        obj_name = action.get('object', 'unknown')
        
        self.get_logger().info(f'Finding {obj_name}')
        
        # This would involve perception systems to locate the object
        time.sleep(1)  # Simulate perception time
        
        return True
    
    def execute_grasp_action(self, action):
        """Execute object grasping action"""
        obj_name = action.get('object', 'unknown')
        
        self.get_logger().info(f'Grasping {obj_name}')
        
        # This would involve manipulation planning and execution
        # Move arms to object position, close gripper, verify grasp
        
        # Simulate arm movement
        self.move_arm_for_grasp(obj_name)
        
        # Simulate gripper action
        self.close_gripper()
        
        time.sleep(1)  # Simulate action time
        
        return True
    
    def execute_place_action(self, action):
        """Execute object placement action"""
        obj_name = action.get('object', 'unknown')
        location = action.get('location', 'default')
        
        self.get_logger().info(f'Placing {obj_name} at {location}')
        
        # Move to placement location
        # Orient gripper appropriately
        # Release object
        
        # Simulate movement
        self.move_to_placement_location(location)
        
        # Simulate gripper release
        self.open_gripper()
        
        time.sleep(1)  # Simulate action time
        
        return True
    
    def execute_clean_action(self, action):
        """Execute cleaning action"""
        area = action.get('area', 'unknown')
        
        self.get_logger().info(f'Cleaning {area}')
        
        # This would involve path planning for cleaning motion
        # and executing those motions with appropriate tools
        
        time.sleep(2)  # Simulate cleaning time
        
        return True
    
    def execute_manipulate_action(self, action):
        """Execute general manipulation action"""
        obj_name = action.get('object', 'unknown')
        
        self.get_logger().info(f'Manipulating {obj_name}')
        
        # General manipulation could be push, pull, turn, etc.
        # Based on context in the action
        
        time.sleep(1)  # Simulate manipulation time
        
        return True
    
    def execute_task_complete(self, action):
        """Handle task completion"""
        self.get_logger().info('Task completed successfully')
        return True
    
    def execute_generic_action(self, action):
        """Execute generic action"""
        action_type = action.get('action', 'unknown')
        self.get_logger().info(f'Executing generic action: {action_type}')
        time.sleep(0.5)  # Simulate generic action time
        return True
    
    def move_arm_for_grasp(self, object_name):
        """Move arm to grasp position"""
        # In a real implementation, this would calculate inverse kinematics
        # and execute joint trajectories to reach the object
        
        # Create a sample joint trajectory for grasping
        trajectory = JointTrajectory()
        trajectory.joint_names = self.joint_names
        
        # Create trajectory point (simplified)
        point = JointTrajectoryPoint()
        point.positions = [0.1, -0.5, 0.3, 0.8, -0.4, 0.0, 0.0]  # Sample positions
        point.velocities = [0.0] * len(point.positions)
        point.time_from_start = Duration(sec=2)  # 2 seconds to reach position
        
        trajectory.points = [point]
        
        self.joint_trajectory_pub.publish(trajectory)
    
    def close_gripper(self):
        """Close the robot's gripper"""
        # This would send commands to gripper joints
        pass
    
    def move_to_placement_location(self, location):
        """Move to placement location"""
        # This would involve navigation to the placement area
        pass
    
    def open_gripper(self):
        """Open the robot's gripper"""
        # This would send commands to gripper joints
        pass
    
    def stop_execution(self):
        """Stop current plan execution"""
        with self.execution_lock:
            self.is_executing = False
    
    def get_execution_status(self):
        """Get current execution status"""
        return {
            'is_executing': self.is_executing,
            'current_action': self.current_action_index,
            'total_actions': len(self.current_plan) if self.current_plan else 0,
            'plan': self.current_plan
        }

def main(args=None):
    rclpy.init(args=args)
    execution_node = ActionExecutionNode()
    
    # Use MultiThreadedExecutor to handle callbacks and execution thread
    executor = MultiThreadedExecutor()
    executor.add_node(execution_node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        execution_node.stop_execution()
        execution_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Integration Node: Complete VLA System

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
import json
import time
from threading import Lock

class VLASystemNode(Node):
    def __init__(self):
        super().__init__('vla_system_node')
        
        # Publishers
        self.system_status_pub = self.create_publisher(String, '/vla_system_status', 10)
        self.completed_task_pub = self.create_publisher(String, '/completed_task', 10)
        
        # Subscribers for the entire VLA pipeline
        self.voice_cmd_sub = self.create_subscription(
            String, '/user_command', self.voice_command_callback, 10)
        self.scene_desc_sub = self.create_subscription(
            String, '/scene_description', self.scene_callback, 10)
        self.action_plan_sub = self.create_subscription(
            String, '/action_plan', self.action_plan_callback, 10)
        self.execution_status_sub = self.create_subscription(
            String, '/action_status', self.execution_status_callback, 10)
        self.llm_response_sub = self.create_subscription(
            String, '/llm_response', self.llm_response_callback, 10)
        
        # System state
        self.system_state = {
            'voice_command': '',
            'scene_description': '',
            'action_plan': [],
            'execution_status': 'idle',
            'execution_progress': 0,
            'current_action': 0,
            'total_actions': 0,
            'timestamp': time.time()
        }
        
        # State lock
        self.state_lock = Lock()
        
        # Task tracking
        self.active_task_id = None
        self.task_start_time = None
        
        self.get_logger().info('VLA System node initialized')
    
    def voice_command_callback(self, msg):
        """Handle voice command input"""
        with self.state_lock:
            self.system_state['voice_command'] = msg.data
            self.system_state['execution_status'] = 'processing_command'
        
        status_msg = String()
        status_msg.data = json.dumps({
            'status': 'received_command',
            'command': msg.data,
            'timestamp': time.time()
        })
        self.system_status_pub.publish(status_msg)
        
        self.get_logger().info(f'Received voice command: {msg.data}')
    
    def scene_callback(self, msg):
        """Handle scene description updates"""
        with self.state_lock:
            self.system_state['scene_description'] = msg.data
    
    def action_plan_callback(self, msg):
        """Handle action plan updates"""
        try:
            plan_data = json.loads(msg.data)
            plan = plan_data if isinstance(plan_data, list) else plan_data.get('plan', [])
            
            with self.state_lock:
                self.system_state['action_plan'] = plan
                self.system_state['total_actions'] = len(plan)
                self.system_state['current_action'] = 0
                self.system_state['execution_status'] = 'executing_plan'
            
            status_msg = String()
            status_msg.data = json.dumps({
                'status': 'plan_received',
                'action_count': len(plan),
                'timestamp': time.time()
            })
            self.system_status_pub.publish(status_msg)
            
        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in action plan')
    
    def execution_status_callback(self, msg):
        """Handle execution status updates"""
        status_text = msg.data
        
        # Parse status message
        is_action_complete = status_text.startswith('ACTION_COMPLETED')
        is_action_failed = status_text.startswith('ACTION_FAILED')
        is_plan_complete = status_text == 'PLAN_EXECUTION_COMPLETED'
        
        with self.state_lock:
            if is_action_complete:
                self.system_state['current_action'] += 1
                progress = (
                    (self.system_state['current_action'] / self.system_state['total_actions']) 
                    if self.system_state['total_actions'] > 0 else 0
                )
                self.system_state['execution_progress'] = progress
                self.system_state['execution_status'] = 'executing'
                
            elif is_action_failed:
                self.system_state['execution_status'] = 'error'
                
            elif is_plan_complete:
                self.system_state['execution_status'] = 'completed'
                self.finalize_task()
        
        # Publish system status
        self.publish_system_status()
    
    def llm_response_callback(self, msg):
        """Handle LLM response updates"""
        try:
            llm_data = json.loads(msg.data)
            command = llm_data.get('command', '')
            intent = llm_data.get('intent', 'unknown')
            
            self.get_logger().info(f'LLM parsed command: "{command}" as intent: {intent}')
            
        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in LLM response')
    
    def publish_system_status(self):
        """Publish current system status"""
        with self.state_lock:
            status_copy = self.system_state.copy()
        
        status_msg = String()
        status_msg.data = json.dumps(status_copy, indent=2)
        self.system_status_pub.publish(status_msg)
    
    def finalize_task(self):
        """Finalize completed task"""
        if self.active_task_id:
            completion_msg = String()
            completion_msg.data = json.dumps({
                'task_id': self.active_task_id,
                'start_time': self.task_start_time,
                'end_time': time.time(),
                'duration': time.time() - self.task_start_time,
                'status': 'completed'
            })
            self.completed_task_pub.publish(completion_msg)
            
            self.active_task_id = None
            self.task_start_time = None
    
    def get_system_status(self):
        """Get current system status"""
        with self.state_lock:
            return self.system_state.copy()

def main(args=None):
    rclpy.init(args=args)
    vla_node = VLASystemNode()
    
    try:
        rclpy.spin(vla_node)
    except KeyboardInterrupt:
        pass
    finally:
        vla_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Capstone Project: The Autonomous Humanoid

This capstone project demonstrates the complete Vision-Language-Action loop by implementing an autonomous humanoid robot that can receive voice commands, understand intent using LLMs, plan navigation paths, avoid obstacles, identify objects, and manipulate them in the environment.

### Autonomous Humanoid System Integration

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import Header
import json
import time
import threading
from queue import Queue, Empty

class AutonomousHumanoidNode(Node):
    def __init__(self):
        super().__init__('autonomous_humanoid_node')
        
        # Publishers
        self.voice_cmd_pub = self.create_publisher(String, '/user_command', 10)
        self.navigation_goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.velocity_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.system_status_pub = self.create_publisher(String, '/humanoid_status', 10)
        self.task_completed_pub = self.create_publisher(Bool, '/task_completed', 10)
        
        # Subscribers
        self.system_status_sub = self.create_subscription(
            String, '/vla_system_status', self.system_status_callback, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        self.image_sub = self.create_subscription(
            Image, '/camera/rgb/image_raw', self.image_callback, 10)
        
        # System components
        self.vla_system_active = False
        self.current_task = None
        self.task_queue = Queue()
        self.obstacle_detected = False
        self.obstacle_distance = float('inf')
        
        # Autonomous capabilities
        self.capabilities = [
            'voice_recognition',
            'vision_processing', 
            'language_understanding',
            'navigation',
            'manipulation',
            'decision_making'
        ]
        
        # Navigation parameters
        self.safety_distance = 0.5  # meters
        self.navigation_active = False
        
        # Task execution thread
        self.task_thread = None
        self.task_thread_running = False
        
        # Initialize the humanoid
        self.initialize_humanoid()
        
        self.get_logger().info('Autonomous Humanoid node initialized')
        
        # Start task monitoring
        self.task_timer = self.create_timer(1.0, self.monitor_tasks)
    
    def initialize_humanoid(self):
        """Initialize the autonomous humanoid system"""
        # Start VLA system components (these would be separate nodes in a real system)
        self.vla_system_active = True
        self.get_logger().info('Autonomous humanoid system initialized')
        
        # Update system status
        status_msg = String()
        status_msg.data = json.dumps({
            'status': 'initialized',
            'capabilities': self.capabilities,
            'timestamp': time.time()
        })
        self.system_status_pub.publish(status_msg)
    
    def monitor_tasks(self):
        """Monitor task queue and execute tasks"""
        try:
            # Check if there's a task to execute
            try:
                task = self.task_queue.get_nowait()
                self.execute_task(task)
            except Empty:
                # No task in queue
                pass
        except Exception as e:
            self.get_logger().error(f'Error monitoring tasks: {e}')
    
    def system_status_callback(self, msg):
        """Handle system status updates from VLA components"""
        try:
            status_data = json.loads(msg.data)
            
            # Monitor execution progress
            execution_status = status_data.get('execution_status', 'unknown')
            progress = status_data.get('execution_progress', 0)
            current_action = status_data.get('current_action', 0)
            total_actions = status_data.get('total_actions', 0)
            
            if execution_status == 'executing':
                self.get_logger().info(f'Execution progress: {progress:.2%} '
                                     f'({current_action}/{total_actions})')
                
            elif execution_status == 'completed':
                self.get_logger().info('Task execution completed successfully')
                
                # Notify that task is completed
                completed_msg = Bool()
                completed_msg.data = True
                self.task_completed_pub.publish(completed_msg)
                
        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in system status')
    
    def scan_callback(self, msg):
        """Handle laser scan data for obstacle detection"""
        # Find minimum distance in scan range
        if len(msg.ranges) > 0:
            valid_ranges = [r for r in msg.ranges if 0 < r < float('inf')]
            if valid_ranges:
                min_distance = min(valid_ranges)
                self.obstacle_distance = min_distance
                self.obstacle_detected = min_distance < self.safety_distance
            else:
                self.obstacle_distance = float('inf')
                self.obstacle_detected = False
        else:
            self.obstacle_distance = float('inf')
            self.obstacle_detected = False
    
    def image_callback(self, msg):
        """Handle camera images for visual processing"""
        # In a real implementation, this would trigger visual processing
        # For now, we'll just log that we received an image
        self.get_logger().debug('Received camera image for processing')
    
    def receive_voice_command(self, command):
        """Receive and process a voice command"""
        self.get_logger().info(f'Received voice command: {command}')
        
        # Publish command to VLA system
        cmd_msg = String()
        cmd_msg.data = command
        self.voice_cmd_pub.publish(cmd_msg)
        
        # Add task to queue
        task = {
            'command': command,
            'timestamp': time.time(),
            'processed': False
        }
        self.task_queue.put(task)
    
    def execute_task(self, task):
        """Execute a high-level task"""
        if self.task_thread_running:
            self.get_logger().warn('Task already executing, waiting...')
            return
        
        # Start execution in a separate thread
        self.task_thread_running = True
        self.task_thread = threading.Thread(target=self.run_task_execution, args=(task,))
        self.task_thread.daemon = True
        self.task_thread.start()
    
    def run_task_execution(self, task):
        """Run task execution in a separate thread"""
        try:
            command = task['command']
            self.get_logger().info(f'Starting execution of task: {command}')
            
            # Update status
            status_msg = String()
            status_msg.data = json.dumps({
                'status': 'executing_task',
                'command': command,
                'timestamp': time.time()
            })
            self.system_status_pub.publish(status_msg)
            
            # In a real implementation, this would:
            # 1. Parse the command using LLM (already handled by VLA system)
            # 2. Plan actions based on command and environment (handled by VLA)
            # 3. Execute actions sequentially (handled by VLA)
            
            # Simulate task execution (in reality, this is handled by the VLA system)
            time.sleep(2)  # Simulate processing time
            
            # Verify execution completed successfully
            self.get_logger().info(f'Task completed: {command}')
            
            # Update final status
            status_msg = String()
            status_msg.data = json.dumps({
                'status': 'task_completed',
                'command': command,
                'timestamp': time.time()
            })
            self.system_status_pub.publish(status_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error executing task: {e}')
            status_msg = String()
            status_msg.data = json.dumps({
                'status': 'task_error',
                'command': task['command'],
                'error': str(e),
                'timestamp': time.time()
            })
            self.system_status_pub.publish(status_msg)
        
        finally:
            self.task_thread_running = False
    
    def autonomous_navigation(self):
        """Handle autonomous navigation with obstacle avoidance"""
        if not self.navigation_active:
            return
        
        # Create twist message for movement
        twist = Twist()
        
        if self.obstacle_detected:
            # Stop or avoid obstacle
            twist.linear.x = 0.0
            twist.angular.z = 0.5  # Turn to avoid
            self.get_logger().warn(f'Obstacle detected at {self.obstacle_distance:.2f}m, avoiding')
        else:
            # Continue navigation
            twist.linear.x = 0.5  # Move forward
            twist.angular.z = 0.0  # No turning
        
        # Publish velocity command
        self.velocity_pub.publish(twist)
    
    def demonstrate_capstone(self):
        """Demonstrate the complete capstone functionality"""
        self.get_logger().info('Starting capstone demonstration...')
        
        # Example commands that demonstrate the full VLA loop
        demo_commands = [
            "Go to the kitchen", 
            "Pick up the red bottle", 
            "Take it to the bedroom",
            "Place it on the table"
        ]
        
        # Execute each command
        for cmd in demo_commands:
            self.get_logger().info(f'Executing demo command: {cmd}')
            self.receive_voice_command(cmd)
            
            # Wait before next command
            time.sleep(3)
    
    def get_system_status(self):
        """Get current system status"""
        return {
            'capabilities': self.capabilities,
            'vla_system_active': self.vla_system_active,
            'obstacle_detected': self.obstacle_detected,
            'obstacle_distance': self.obstacle_distance,
            'task_queue_size': self.task_queue.qsize(),
            'navigation_active': self.navigation_active,
            'timestamp': time.time()
        }

def main(args=None):
    rclpy.init(args=args)
    humanoid_node = AutonomousHumanoidNode()
    
    # For demonstration, start the capstone demo after 2 seconds
    def start_demo():
        time.sleep(2)
        humanoid_node.demonstrate_capstone()
    
    demo_thread = threading.Thread(target=start_demo)
    demo_thread.daemon = True
    demo_thread.start()
    
    try:
        rclpy.spin(humanoid_node)
    except KeyboardInterrupt:
        pass
    finally:
        humanoid_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Summary

This module has provided a comprehensive implementation of Vision-Language-Action (VLA) systems for humanoid robotics. We've covered:

1. **Voice-to-Action systems** using OpenAI Whisper for voice command processing
2. **Vision-Language understanding** with object detection and scene analysis
3. **Cognitive planning** using LLMs to convert natural language to executable action sequences
4. **Action execution** mapping high-level plans to low-level robot behaviors

The capstone project demonstrates an autonomous humanoid robot that implements a complete Vision-Language-Action loop, receiving voice commands, understanding intent through LLMs, planning navigation paths, avoiding obstacles, identifying objects, and manipulating them in the environment.

## Exercises

1. Integrate a real speech recognition model like OpenAI Whisper into the voice-to-action system.
2. Enhance the object detection system with a pre-trained model like YOLOv8 or Detectron2.
3. Implement a more sophisticated LLM-based planning system that considers environmental constraints.
4. Create a more robust action execution framework that handles failures and recovery.
5. Develop a complete simulation environment where the autonomous humanoid can execute complex tasks.

The VLA system represents the cutting edge of robotics, combining the latest advances in natural language processing, computer vision, and robotics to create truly intelligent and interactive humanoid robots.