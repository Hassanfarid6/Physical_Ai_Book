# Quickstart Guide: Physical AI Book

## Setup

1. **Install Node.js and npm**
   - Download and install Node.js (v18 or higher) from [nodejs.org](https://nodejs.org/)
   - npm is included with Node.js installation

2. **Install Docusaurus**
   ```bash
   npm init docusaurus@latest docs classic
   ```

3. **Navigate to the project directory**
   ```bash
   cd docs
   ```

## Project Structure

After setup, your project will have this structure:

```
docs/
├── blog/              # Blog posts (optional)
├── docs/              # Documentation files
│   ├── intro.md       # Introduction page
│   └── ...            # More documentation files
├── src/
│   ├── components/    # Custom React components
│   ├── css/           # Custom styles
│   └── pages/         # Custom pages
├── static/            # Static assets
├── docusaurus.config.js # Configuration file
├── package.json       # Dependencies and scripts
├── sidebars.js        # Sidebar configuration
└── yarn.lock          # Dependency lock file
```

## Adding Content

1. **Create a new lesson**
   - Add a new Markdown file in `docs/chapter-1/`
   - Example: `lesson-1-getting-started-with-physical-ai.md`

2. **Format your lesson**
   ```markdown
   ---
   id: lesson-1-getting-started-with-physical-ai
   title: Getting Started with Physical AI
   sidebar_label: Lesson 1
   ---

   # Getting Started with Physical AI

   This lesson introduces the basic concepts of Physical AI...
   ```

3. **Update sidebar configuration**
   - Edit `sidebars.js` to include your new lesson in the navigation

4. **Add exercises and examples**
   - Use Docusaurus code blocks for examples
   - Use custom components for interactive exercises

## Development Server

Run the development server:

```bash
cd docs
npm start
```

This will start a local development server at `http://localhost:3000` and open it in your browser. Most changes are reflected live without having to restart the server.

## Build

To build the static site for production:

```bash
npm run build
```

This creates a `build/` directory with the static files that can be deployed to any static hosting service.

## Deployment

To deploy your site:

1. **Prepare for GitHub Pages** (if applicable):
   ```bash
   npm run deploy
   ```

2. **Deploy to other static hosting**:
   - Use the contents of the `build/` directory
   - Follow the specific instructions for your hosting service

## Key Configuration

### docusaurus.config.js

Important settings for your Physical AI book:

```javascript
module.exports = {
  title: 'Physical AI Book',
  tagline: 'A hands-on approach to learning Physical AI',
  url: 'https://your-domain.com',
  baseUrl: '/',
  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',
  // Additional configuration for search, themes, etc.
};
```

### sidebars.js

Define the navigation structure for your book:

```javascript
module.exports = {
  docs: [
    {
      type: 'category',
      label: 'Chapter 1: Getting Started',
      items: [
        'chapter-1/lesson-1-getting-started-with-physical-ai',
        'chapter-1/lesson-2-understanding-robot-control-systems',
        'chapter-1/lesson-3-exploring-simulation-and-modeling'
      ],
    },
  ],
};
```