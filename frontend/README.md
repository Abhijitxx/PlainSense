# PlainSense Frontend

A modern React.js frontend for the PlainSense document simplification system.

## Features

- **Two Interface Tabs**: Legal Documents and Medical Reports
- **Drag & Drop Upload**: Easy file upload with drag-and-drop support
- **Text Paste**: Direct text input option
- **Multi-language Support**: Plain English, Hindi (Plain & Colloquial), Tamil (Plain & Colloquial)
- **Risk Analysis**: Visual risk indicators for legal clauses
- **Medical Insights**: Causes and effects explanations for abnormal test results
- **Domain Validation**: Automatic detection to prevent processing wrong document types
- **Responsive Design**: Works on desktop and mobile devices

## Prerequisites

- Node.js 16+ and npm
- Python 3.8+ (for backend)
- PlainSense backend server running on port 5000

## Installation

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install
```

## Running the Application

### Development Mode

1. **Start the Backend Server** (from project root):
   ```bash
   python server.py
   ```

2. **Start the Frontend** (from frontend directory):
   ```bash
   npm start
   ```

The frontend will be available at http://localhost:3000 and will proxy API requests to the backend at http://localhost:5000.

### Production Build

```bash
# Build the production bundle
npm run build

# The build output will be in the 'build' folder
# The Flask server will automatically serve these files
```

## Project Structure

```
frontend/
├── public/
│   ├── index.html
│   ├── manifest.json
│   └── robots.txt
├── src/
│   ├── components/
│   │   ├── Header.tsx          # App header with logo
│   │   ├── LegalInterface.tsx  # Legal document processing
│   │   ├── MedicalInterface.tsx # Medical report processing
│   │   ├── ClauseCard.tsx      # Legal clause display card
│   │   └── MedicalResultCard.tsx # Medical test result card
│   ├── services/
│   │   └── api.ts              # API service layer
│   ├── App.tsx                 # Main app component
│   ├── App.css                 # Main styles
│   └── index.tsx               # Entry point
├── package.json
└── README.md
```

## API Endpoints

The frontend communicates with these backend endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/process/legal` | POST | Process legal documents |
| `/api/process/medical` | POST | Process medical documents |
| `/api/process/text` | POST | Process raw text |
| `/api/validate/domain` | POST | Validate document domain |
| `/api/translate` | POST | Translate text |
| `/api/health` | GET | Health check |

## Language Options

The app supports the following language outputs:
- **Plain English** (default)
- **Plain Hindi** (formal)
- **Colloquial Hindi** (conversational)
- **Plain Tamil** (formal)
- **Colloquial Tamil** (conversational)

## Design System

The UI follows a minimalist design with:
- Clean, modern aesthetics
- Consistent color palette
- Smooth animations
- Responsive layouts
- Accessible components

## Scripts

| Command | Description |
|---------|-------------|
| `npm start` | Start development server |
| `npm run build` | Create production build |
| `npm test` | Run tests |

## License

MIT License

### `npm run eject`

**Note: this is a one-way operation. Once you `eject`, you can’t go back!**

If you aren’t satisfied with the build tool and configuration choices, you can `eject` at any time. This command will remove the single build dependency from your project.

Instead, it will copy all the configuration files and the transitive dependencies (webpack, Babel, ESLint, etc) right into your project so you have full control over them. All of the commands except `eject` will still work, but they will point to the copied scripts so you can tweak them. At this point you’re on your own.

You don’t have to ever use `eject`. The curated feature set is suitable for small and middle deployments, and you shouldn’t feel obligated to use this feature. However we understand that this tool wouldn’t be useful if you couldn’t customize it when you are ready for it.

## Learn More

You can learn more in the [Create React App documentation](https://facebook.github.io/create-react-app/docs/getting-started).

To learn React, check out the [React documentation](https://reactjs.org/).
