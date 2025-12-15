# Content Guidelines: Physical AI Book

## Purpose
This document defines the content guidelines for maintaining consistent lesson structure and quality throughout the Physical AI Book.

## Document Structure

Each lesson must follow this standardized format:

### Frontmatter Requirements
```yaml
---
id: unique-identifier
title: Lesson Title
sidebar_label: Lesson Display Name
description: Brief description of the lesson content
---
```

### Content Sections (in order)
1. **Title** (H1) - Same as frontmatter title
2. **Overview** - Brief introduction to the topic (2-3 sentences)
3. **Main Content** - Detailed explanation with examples
4. **Code Examples** - Where applicable, with explanations
5. **Practical Exercise** - Hands-on activity for the reader
6. **Summary** - Key takeaways from the lesson

## Writing Style

### Language & Tone
- Use clear, concise language appropriate for beginners
- Write in an instructive yet conversational tone
- Avoid jargon unless clearly defined
- Use active voice where possible

### Terminology
- Establish new terms in italics on first use: *actuator*, *sensor*, *feedback loop*
- Use consistent terminology throughout the book
- When introducing Physical AI concepts, contrast with traditional AI where relevant

## Content Requirements

### Learning Objectives
Each lesson must include 2-3 clear learning objectives in a bullet format after the overview.

### Examples & Exercises
- Include at least one practical example in each lesson
- Create exercises that reinforce the key concepts
- Provide solutions for exercises (can be hidden initially for student discovery)

### Code Examples
- Use Python as the default language for examples
- Include comments explaining important concepts
- Make examples self-contained where possible
- Use consistent variable naming conventions
- Include output examples where relevant

## Formatting Standards

### Headings
- H1: Lesson title (automatically generated from frontmatter)
- H2: Major sections (Overview, Main Content, etc.)
- H3: Subsections within main content
- H4: Further subdivisions if needed

### Lists & Tables
- Use numbered lists for procedures
- Use bulleted lists for non-sequential information
- Left-align table content unless it's numerical
- Include descriptive table headers

### Images & Diagrams
- Place images in `static/img/` directory
- Use descriptive alt text for accessibility
- Include captions explaining the importance of visual elements

## Quality Standards

### Completeness
- Each lesson must be self-contained
- Prerequisites must be clearly stated
- All exercises must have valid solutions

### Accuracy
- Verify code examples function as described
- Ensure all links and references are valid
- Cross-check technical information with authoritative sources

### Consistency
- Maintain consistent formatting across all lessons
- Use the same structural organization for each lesson
- Apply consistent terminology throughout the book

## Lesson Progression

### Prerequisite Management
- Clearly state prerequisite knowledge at the beginning of each lesson
- Reference previous lessons when building on concepts
- Ensure each lesson can stand alone while connecting to the larger narrative

### Conceptual Flow
- Start with fundamental concepts before moving to advanced topics
- Relate new concepts back to previous lessons where appropriate
- Preview how concepts will be used in future lessons

## Review Checklist

Before finalizing a lesson, verify:
- [ ] Frontmatter is complete and correct
- [ ] Content follows required structure
- [ ] At least one code example is included
- [ ] At least one practical exercise is included
- [ ] Learning objectives are clearly stated
- [ ] All links and references are valid
- [ ] Code examples have been tested
- [ ] Prerequisites are clearly defined
- [ ] Writing style is consistent with guidelines
- [ ] Appropriate images/diagrams are included where needed