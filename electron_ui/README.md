# Privacy Proxy UI

A React-based user interface for the Privacy Proxy Service that demonstrates PII masking, AI processing, and demasking.

## Features

- **Interactive Demo**: Test the privacy proxy flow with your own data
- **Multiple Views**: Flow view, side-by-side comparison, and diff view
- **PII Detection**: Visual highlighting of detected entities (emails, phones, SSNs, names)
- **Confidence Scores**: Shows detection confidence levels
- **Real-time Processing**: Simulates the complete A → A' → B' → B pipeline

## Development

### Prerequisites

- Node.js 18+
- npm or yarn

### Local Development

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Start production server
npm start
```

### Docker Development

```bash
# Build UI image
make docker-build-ui

# Start UI service
make docker-up-ui

# View logs
make docker-logs-ui
```

## Usage

1. **Input Data**: Enter text containing PII (emails, phone numbers, SSNs, names)
2. **Process**: Click "Process Data" to run through the masking pipeline
3. **View Results**: Switch between different views to see the transformation
4. **Copy Data**: Use copy buttons to copy data from any stage

## Views

### Flow View
Shows the complete pipeline: Original → Masked → AI Response → Final

### Side-by-Side View
Compares original vs masked input and masked vs final output

### Diff View
Highlights exactly what changed during masking and demasking

## API Integration

This UI is currently a demo with simulated processing. To connect to the actual Privacy Proxy API:

1. Update the `handleSubmit` function to make real API calls
2. Replace simulation functions with actual HTTP requests
3. Configure the proxy endpoint URL

## Architecture

- **React 18**: Modern React with hooks
- **Tailwind CSS**: Utility-first CSS framework
- **Lucide React**: Beautiful icons
- **Webpack**: Module bundler
- **Express**: Production server
