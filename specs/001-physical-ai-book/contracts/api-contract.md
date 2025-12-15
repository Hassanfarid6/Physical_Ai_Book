# API Contract: Physical AI Book Backend Services

## Purpose
This document defines the API contracts for backend services that support the Physical AI Book, including user progress tracking, feedback collection, and community features.

## Base URL
```
https://api.physicalai-book.com/v1
```

## Authentication
All endpoints require authentication via Bearer token, except for public content access.

## Endpoints

### User Progress Tracking

#### POST /users/{userId}/progress
Track user progress through lessons and chapters

**Request Body**:
```json
{
  "lessonId": "chapter-1-lesson-1",
  "status": "completed",
  "timestamp": "2025-01-01T10:00:00Z",
  "exerciseResults": [
    {
      "exerciseId": "exercise-1",
      "status": "passed",
      "score": 100
    }
  ]
}
```

**Response**:
```json
{
  "success": true,
  "message": "Progress saved successfully"
}
```

#### GET /users/{userId}/progress
Get user progress summary

**Response**:
```json
{
  "userId": "user-123",
  "completedLessons": ["chapter-1-lesson-1"],
  "currentChapter": "chapter-1",
  "currentLesson": "chapter-1-lesson-2",
  "totalChapters": 5,
  "completedChapters": 0,
  "lastAccessed": "2025-01-01T10:00:00Z"
}
```

### Feedback Collection

#### POST /feedback
Submit feedback on content quality

**Request Body**:
```json
{
  "userId": "user-123",
  "contentId": "chapter-1-lesson-1",
  "type": "correction|suggestion|compliment",
  "message": "This section was unclear about...",
  "rating": 3
}
```

**Response**:
```json
{
  "feedbackId": "feedback-456",
  "success": true,
  "message": "Feedback submitted successfully"
}
```

### Content Search

#### GET /search
Search across all content

**Query Parameters**:
- q: Search query string
- limit: Max results to return (default: 10)
- offset: Results offset (default: 0)

**Response**:
```json
{
  "query": "robot control",
  "totalResults": 5,
  "results": [
    {
      "id": "chapter-2-lesson-1",
      "title": "Understanding Robot Control Systems",
      "contentPreview": "This lesson covers how robots make decisions and control their movements...",
      "url": "/docs/chapter-2/lesson-1"
    }
  ]
}
```

### Community Contributions

#### GET /contributions
Get list of community contributions

**Response**:
```json
{
  "contributions": [
    {
      "id": "contribution-789",
      "title": "Improved explanation of PID controllers",
      "author": "contributor-101",
      "status": "approved|pending|rejected",
      "date": "2024-12-15T10:00:00Z",
      "contentId": "chapter-2-lesson-1"
    }
  ]
}
```

## Error Responses
All error responses follow this format:
```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Descriptive error message",
    "details": "Additional error details if applicable"
  }
}
```