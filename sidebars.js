// @ts-check

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.

 @type {import('@docusaurus/plugin-content-docs').SidebarsConfig}
 */
const sidebars = {
  // Professional sidebar for the Physical AI & Humanoid Robotics Documentation
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Introduction to Physical AI & Humanoid Robotics',
      items: ['intro_to_physical_ai'],
    },
    {
      type: 'category',
      label: 'Module 1: The Robotic Nervous System (ROS 2) - Middleware for Humanoid Control',
      items: [
        'module1_ros2',
      ],
    },
    {
      type: 'category',
      label: 'Module 2: The Digital Twin (Gazebo & Unity) - Physics Simulation and Environment Modeling',
      items: [
        'module2_digital_twin',
      ],
    },
    {
      type: 'category',
      label: 'Module 3: The AI-Robot Brain (NVIDIA Isaac™) - Advanced Perception, Training, and Navigation',
      items: [
        'module3_ai_robot_brain',
      ],
    },
    {
      type: 'category',
      label: 'Module 4: Vision-Language-Action (VLA) - Convergence of LLMs and Robotics',
      items: [
        'module4_vla',
      ],
    },
    {
      type: 'category',
      label: 'Implementation Guide',
      items: [
        'project_implementation',
      ],
    },
    {
      type: 'category',
      label: 'Advanced Techniques',
      items: [
        'advanced_topics',
      ],
    },
    {
      type: 'category',
      label: 'Appendices & References',
      items: [
        'appendices',
      ],
    },
  ],
};

export default sidebars;
