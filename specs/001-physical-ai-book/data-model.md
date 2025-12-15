# Data Model: Physical AI Book

## Entities

### Lesson
- **Fields**:
  - id: string (unique identifier, e.g., "chapter-1-lesson-1")
  - title: string (lesson title)
  - description: string (brief description of the lesson)
  - content: string (main content in Markdown format)
  - prerequisites: array of string (IDs of required lessons)
  - exercises: array of object (practical tasks for users)
  - objectives: array of string (learning goals)
  - duration: number (estimated time to complete in minutes)

- **Validation**:
  - id must follow pattern: "chapter-{number}-lesson-{number}"
  - title and content must not be empty
  - duration must be a positive number

- **Relationships**:
  - Belongs to one Chapter
  - May link to other Lessons as prerequisites

### Chapter
- **Fields**:
  - id: string (unique identifier, e.g., "chapter-1")
  - title: string (chapter title)
  - description: string (brief description of the chapter)
  - lessons: array of Lesson objects
  - prerequisites: array of string (IDs of required chapters)
  - objectives: array of string (learning goals for the chapter)

- **Validation**:
  - id must follow pattern: "chapter-{number}"
  - title must not be empty
  - lessons array must contain at least one lesson

- **Relationships**:
  - Contains multiple Lessons
  - May depend on other Chapters as prerequisites

### Exercise
- **Fields**:
  - id: string (unique identifier)
  - title: string (exercise title)
  - description: string (detailed instructions)
  - type: string (e.g., "coding", "simulation", "analysis", "research")
  - difficulty: string (e.g., "beginner", "intermediate", "advanced")
  - solution: string (optional - for reference only)

- **Validation**:
  - id must be unique within the lesson
  - title and description must not be empty
  - type must be one of the allowed values

- **Relationships**:
  - Belongs to one Lesson
  - May reference external resources

### Navigation Path
- **Fields**:
  - id: string (unique identifier)
  - title: string (path title)
  - lessons: array of string (ordered list of lesson IDs)
  - description: string (purpose of this path)

- **Validation**:
  - lessons array must contain valid lesson IDs
  - lessons must follow prerequisite requirements

- **Relationships**:
  - References multiple Lessons
  - May span across multiple Chapters

## State Transitions

### Lesson Completion
- **States**: `not-started` → `in-progress` → `completed`
- **Transitions**:
  - When user starts reading: `not-started` → `in-progress`
  - When user completes exercises: `in-progress` → `completed`

### Chapter Completion
- **States**: `not-started` → `in-progress` → `completed`
- **Transitions**:
  - When user starts first lesson: `not-started` → `in-progress`
  - When all lessons are completed: `in-progress` → `completed`