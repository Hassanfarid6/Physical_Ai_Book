# Contributing to Physical AI Book

We're excited that you're interested in contributing to the Physical AI Book! This document outlines the process for contributing content, improvements, and fixes to our documentation.

## Table of Contents
- [Content Structure](#content-structure)
- [How to Contribute](#how-to-contribute)
- [Style Guide](#style-guide)
- [Technical Requirements](#technical-requirements)
- [Testing](#testing)

## Content Structure

The Physical AI Book follows a specific structure:
```
docs/
├── chapter-1/
│   ├── lesson-1-getting-started-with-physical-ai.md
│   ├── lesson-2-understanding-robot-control-systems.md
│   └── lesson-3-exploring-simulation-and-modeling.md
├── intro.md
└── ...
```

Each lesson should follow this format:
```markdown
---
id: unique-identifier
title: Lesson Title
sidebar_label: Sidebar Display Text
description: Brief description of the lesson
---

# Lesson Title

## Overview
Brief introduction to the topic...

## Main Content
Detailed explanation with examples...

## Practical Exercise
Hands-on exercise for readers to try...

## Summary
Recap of key concepts covered...
```

## How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-chapter`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some amazing content'`)
5. Push to the branch (`git push origin feature/amazing-chapter`)
6. Open a Pull Request

### Adding a New Chapter
1. Create a new directory in `docs/` (e.g., `docs/chapter-2/`)
2. Add lesson files following the naming convention `lesson-{number}-{title}.md`
3. Update `sidebars.js` to include the new chapter and lessons
4. Ensure proper frontmatter is included in all lesson files

### Adding a New Lesson
1. Create a new markdown file in the appropriate chapter directory
2. Follow the naming convention: `lesson-{number}-{title}.md`
3. Include proper frontmatter with id, title, sidebar_label, and description
4. Follow the lesson format described above

## Style Guide

### Writing Style
- Use clear, concise language appropriate for beginners
- Include practical examples with code snippets where relevant
- Focus on hands-on learning approach
- Connect concepts to real-world Physical AI applications

### Code Examples
- Use Python for code examples unless otherwise specified
- Include comments in code to explain important concepts
- Make examples runnable and testable
- Follow Python PEP 8 style guidelines

### Markdown Formatting
- Use sentence case for headings
- Include alt text for all images
- Use relative links for internal navigation
- Use absolute URLs for external references

## Technical Requirements

### Prerequisites
- Node.js (version 18 or higher)
- npm or yarn package manager

### Local Development
1. Install dependencies: `npm install`
2. Start development server: `npm start`
3. Visit `http://localhost:3000` to view your changes

### File Naming Convention
- Use kebab-case for file names: `my-lesson-title.md`
- Include chapter and lesson numbers: `lesson-2-advanced-concepts.md`
- Use descriptive names that reflect the content

## Testing

### Content Validation
- Ensure all links are functional
- Verify code examples run correctly
- Confirm exercises have working solutions
- Test navigation between lessons

### Building
- Run `npm run build` to test the build process
- Check for any build errors or warnings
- Verify the built site displays correctly

## Need Help?

If you have questions about contributing:
- Check the existing documentation in the repository
- Open an issue for technical questions
- Review existing pull requests for examples

Thank you for contributing to the Physical AI Book!