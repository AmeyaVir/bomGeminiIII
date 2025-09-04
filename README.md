# BOM Platform Enhanced v3.0

A comprehensive Bill of Materials (BOM) processing platform with autonomous document processing, knowledge base management, and human intervention workflows. This version is fully integrated with the Gemini API.

## New Features in v3.0

### 🚀 Key Enhancements
- **Gemini API Integration**: Replaces mock agent logic with live API calls to Google's Gemini models for document processing.
- **FastAPI Backend**: Reconstructs the entire backend using a modern, asynchronous FastAPI framework for improved performance.
- **Enhanced Data Orchestration**: Frontend now communicates with the real FastAPI endpoints, orchestrating data flow seamlessly.

### 🔧 Technical Improvements
- **Live API Calls**: Frontend fetches real data from the FastAPI backend, enabling actual document processing.
- **Improved Service Layer**: Backend services are updated to handle asynchronous operations and integrate with the new Gemini agent.
- **Simplified Frontend Logic**: Removed mock data, making the frontend a pure client for the API.

## Architecture

### Backend
- **FastAPI** for a high-performance, asynchronous API
- **Gemini API** for intelligent document processing and classification
- **SQLite Database** with comprehensive schema
- **Service Layer**: `WorkflowService`, `KnowledgeBaseService`, and the new `GeminiAgentService`
- **File Management**: Upload handling and results storage

### Frontend  
- **React 18** with modern hooks
- **React Router** for navigation
- **Tailwind CSS** for styling
- **Lucide React** for icons
- **React Hot Toast** for notifications

### Database Schema
- `workflows` - Workflow tracking and status
- `knowledge_base` - Historical processed items
- `pending_approvals` - Items awaiting human review
- `workflow_results` - Processed results storage

## Installation

### Prerequisites
- Python 3.8+
- Node.js 16+
- **A Gemini API Key** (set as `GEMINI_API_KEY` in a `.env` file in the `backend` directory)

### Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python -c "from models import init_db; init_db()"
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend Setup
```bash
cd frontend
npm install
npm start
```

## Usage

### 1. Configure API Key
Create a `.env` file inside the `backend` directory and add your Gemini API key:
```dotenv
GEMINI_API_KEY=YOUR_GEMINI_API_KEY_HERE
```

### 2. Run the Application
You will need to run the backend and frontend separately in two terminals.
```bash
# Terminal 1 (Backend)
./start_backend.sh

# Terminal 2 (Frontend)
./start_frontend.sh
```

### 3. Use the Platform
- Access the dashboard at `http://localhost:3000`.
- Upload documents and monitor their processing in real time.
- View detailed results and approve/reject items for the knowledge base.

## API Endpoints

### Workflows
- `GET /api/workflows` - List all workflows
- `GET /api/autonomous/workflow/{id}/status` - Get workflow status
- `GET /api/autonomous/workflow/{id}/results` - Get workflow results
- `POST /api/autonomous/upload` - Upload documents and start processing

### Knowledge Base
- `GET /api/knowledge-base` - Get knowledge base items with search
- `GET /api/knowledge-base/pending` - Get pending approval items
- `POST /api/knowledge-base/approve` - Approve items for knowledge base
- `POST /api/knowledge-base/reject` - Reject items for knowledge base

---
## Deployment

This version is designed for a full-stack deployment. You can use a process manager like Gunicorn for the backend and a static file server for the frontend.

## Troubleshooting

### Common Issues
1.  **API Key Error**: Ensure `GEMINI_API_KEY` is correctly set in the `.env` file.
2.  **`Failed to fetch`**: Verify both the backend and frontend servers are running, and check for CORS issues.
3.  **Database not found**: Run `python -c "from models import init_db; init_db()"` in the `backend` directory.

---
## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
- Check the troubleshooting section
- Review application logs
- Open an issue on GitHub
