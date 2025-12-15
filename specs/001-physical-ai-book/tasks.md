# Tasks: Physical AI Book

**Input**: Design documents from `/specs/[001-physical-ai-book]/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: The examples below include test tasks. Tests are OPTIONAL - only include them if explicitly requested in the feature specification.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Docusaurus project**: Content in `docs/` directory following Docusaurus conventions
- **Configuration**: Main config files at project root: `docusaurus.config.js`, `sidebars.js`
- **Content**: Documentation files in `docs/` with chapter/lesson structure
- **Assets**: Images and static files in `static/img/` and `docs/assets/`

<!--

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Docusaurus project initialization and basic structure

- [X] T001 Install Node.js v18+ and npm on development machine
- [X] T002 Initialize Docusaurus project in project root with classic template
- [X] T003 [P] Update package.json with Physical AI Book metadata and author information
- [X] T004 Configure docusaurus.config.js with Physical AI Book settings
- [X] T005 Set up initial sidebar navigation in sidebars.js for 1 chapter with 3 lessons

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core documentation infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T006 Create docs/chapter-1/ directory structure for the first chapter
- [X] T007 [P] Set up basic CSS styling in src/css/custom.css for Physical AI Book theme
- [X] T008 [P] Configure Docusaurus plugins for search, sitemap, and code blocks
- [X] T009 Create initial intro.md file with Physical AI Book introduction
- [X] T010 Define content guidelines document for consistent lesson structure

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Getting Started with Physical AI (Priority: P1) 🎯 MVP

**Goal**: Create the first lesson that introduces users to Physical AI concepts, explaining the differences between traditional AI and Physical AI systems

**Independent Test**: Users can explain the core concepts of Physical AI and identify basic components of physical systems after completing this lesson

### Implementation for User Story 1

- [X] T011 [US1] Create lesson-1-getting-started-with-physical-ai.md file in docs/chapter-1/
- [X] T012 [US1] Write content for lesson 1 covering core Physical AI concepts and differences from traditional AI
- [X] T013 [P] [US1] Add practical examples to lesson 1 demonstrating Physical AI vs traditional AI
- [X] T014 [P] [US1] Create exercises for lesson 1 that help users identify key differences between AI types
- [X] T015 [US1] Update sidebar navigation to include lesson 1 in chapter 1
- [X] T016 [US1] Add prerequisite information to lesson 1

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 - Understanding Robot Control Systems (Priority: P2)

**Goal**: Create the second lesson that teaches users about robot control systems, explaining how robots make decisions and control their movements

**Independent Test**: Users can analyze a simple control loop and identify its components after completing this lesson

### Implementation for User Story 2

- [X] T017 [US2] Create lesson-2-understanding-robot-control-systems.md file in docs/chapter-1/
- [X] T018 [US2] Write content for lesson 2 covering control systems, sensors, actuators, and processing units
- [ ] T019 [P] [US2] Add diagrams and visual aids to illustrate control system components (to be placed in static/img/)
- [ ] T020 [P] [US2] Create practical examples to lesson 2 showing simple control system diagrams
- [ ] T021 [US2] Develop exercises for lesson 2 where users identify sensor, processing, and actuator components
- [X] T022 [US2] Update sidebar navigation to include lesson 2 in chapter 1
- [X] T023 [US2] Add prerequisite information linking to lesson 1 concepts

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently

---

## Phase 5: User Story 3 - Exploring Simulation and Modeling (Priority: P3)

**Goal**: Create the third lesson that explains how to model physical systems and test AI algorithms in simulated environments

**Independent Test**: Users can describe the advantages and limitations of simulation vs. real-world testing after completing this lesson

### Implementation for User Story 3

- [X] T024 [US3] Create lesson-3-exploring-simulation-and-modeling.md file in docs/chapter-1/
- [X] T025 [US3] Write content for lesson 3 covering simulation benefits, limitations, and use cases
- [ ] T026 [P] [US3] Add practical examples of simulation platforms/tools to lesson 3
- [ ] T027 [P] [US3] Create exercises for lesson 3 where users compare simulation vs. real-world testing
- [ ] T028 [US3] Develop interactive elements (if possible) to demonstrate simulation concepts
- [X] T029 [US3] Update sidebar navigation to include lesson 3 in chapter 1
- [X] T030 [US3] Add prerequisite information linking to previous lessons

**Checkpoint**: All user stories should now be independently functional

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [X] T031 [P] Update site-wide navigation and header with Physical AI Book branding
- [X] T032 Add consistent metadata and frontmatter to all lesson files
- [X] T033 [P] Add images and diagrams to enhance learning experience across all lessons
- [X] T034 Review and proofread all content for clarity and accuracy
- [X] T035 Test site locally to ensure all links and navigation work properly
- [X] T036 Validate that all three lessons meet hands-on learning requirements with practical exercises
- [X] T037 Document content creation process for future chapters in CONTRIBUTING.md

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2 → P3)
- **Polish (Final Phase)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P2)**: Can start after Foundational (Phase 2) - Builds on US1 concepts but should be independently testable
- **User Story 3 (P3)**: Can start after Foundational (Phase 2) - Builds on US1/US2 concepts but should be independently testable

### Within Each User Story

- Core content before exercises
- Exercises before integration with navigation
- Story complete before moving to next priority

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational tasks marked [P] can run in parallel (within Phase 2)
- Once Foundational phase completes, all user stories can start in parallel (if team capacity allows)
- Content writing and asset preparation for different lessons can run in parallel
- Different user stories can be worked on in parallel by different team members

---

## Parallel Example: User Story 2

```bash
# Launch all tasks for User Story 2 together:
Task: "Create lesson-2-understanding-robot-control-systems.md file in docs/chapter-1/"
Task: "Write content for lesson 2 covering control systems, sensors, actuators, and processing units"
Task: "Add diagrams and visual aids to illustrate control system components (to be placed in static/img/)"
Task: "Create practical examples to lesson 2 showing simple control system diagrams"
Task: "Develop exercises for lesson 2 where users identify sensor, processing, and actuator components"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Test User Story 1 independently
5. Deploy/demo if ready

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → Deploy/Demo (MVP!)
3. Add User Story 2 → Test independently → Deploy/Demo
4. Add User Story 3 → Test independently → Deploy/Demo
5. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1
   - Developer B: User Story 2
   - Developer C: User Story 3
3. Stories complete and integrate independently

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence

-->