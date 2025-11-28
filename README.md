# Plant Leaf Disease Detector

An AI-powered system that identifies plant diseases from leaf images, provides detailed information about causes and symptoms, and recommends comprehensive treatments. Built with Streamlit frontend and FastAPI backend.

## Features

- **AI-Powered Analysis**: Hybrid approach using Grok API + specialized plant disease models
- **Comprehensive Coverage**: Supports all major crops with top 20 most damaging diseases
- **Detailed Treatment Recommendations**: Organic, chemical, and prevention measures with priority ranking
- **Visual Disease Reference**: Disease symptom photos for comparison and verification
- **User-Friendly Interface**: Simple upload-and-analyze workflow
- **Mobile Responsive**: Works on all devices

## Quick Start

```bash
# Clone and setup
git clone <repository-url>
cd samad
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env
# Edit .env with your API keys

# Start services
cd backend && uvicorn main:app --reload --host 0.0.0.0 --port 8000
# In separate terminal:
cd frontend && streamlit run app.py --server.port 8501
```

## Project Structure

```
samad/
├── frontend/           # Streamlit application
├── backend/            # FastAPI backend
├── database/           # Disease and treatment data
├── tests/              # Test suites
├── docker-compose.yml  # Container orchestration
├── Dockerfile          # Container configuration
└── requirements.txt    # Python dependencies
```

## API Endpoints

- `POST /api/v1/analyze` - Analyze plant leaf image
- `GET /api/v1/diseases/{disease_id}` - Get disease information
- `GET /api/v1/images/reference/{disease_id}` - Get reference images
- `GET /api/v1/health` - Service health check

## Configuration

Environment variables required:
- `GROK_API_KEY` - Grok API key for image analysis
- `PLANTNET_API_KEY` - PlantNet API key for specialized plant identification

## License

[Your License Here]