# Multi-Agent AI System

A sophisticated multi-agent system that processes and classifies different types of input (PDF, JSON, Email) and routes them to specialized agents for intelligent processing.

## System Architecture

### 1. Classifier Agent
- **Input**: Raw files (PDF/JSON/Email)
- **Responsibilities**:
  - Format Classification (PDF/JSON/Email)
  - Intent Classification (Invoice, RFQ, Complaint, Regulation, etc.)
  - Intelligent routing to specialized agents
  - Logging format and intent in shared memory

### 2. JSON Agent
- **Input**: Structured JSON payloads
- **Responsibilities**:
  - Schema validation and reformatting
  - Field extraction and normalization
  - Anomaly detection
  - Missing field identification

### 3. Email Agent
- **Input**: Email content
- **Responsibilities**:
  - Sender information extraction
  - Intent classification
  - Urgency assessment
  - CRM-compatible formatting

### Shared Memory Module
- **Storage Types**:
  - Redis (distributed)
  - SQLite (local)
  - In-memory (development)
- **Stored Information**:
  - Source metadata
  - Document type
  - Timestamps
  - Extracted values
  - Thread/conversation IDs
  - Processing history

## Example Flow
1. User uploads email
2. Classifier Agent detects "Email + RFQ"
3. Email Agent processes content
4. Information is extracted and formatted
5. Results are logged in shared memory

## Tech Stack
- **Language**: Python
- **LLM Integration**: 
  - OpenAI API
  - Open-source alternatives
- **Memory Storage**:
  - Redis
  - SQLite
  - JSON store
- **Web Framework**: Streamlit

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file with:
```
OPENROUTER_API_KEY=your_api_key_here
REDIS_URL=redis://localhost:6379
```

4. Run the application:
```bash
streamlit run app.py
```

## Project Structure
```
.
├── app.py
│----shared_memory.py
│── document-loader.py
├── langchain_agent.py
├── requirements.txt
└── README.md
```

## Features
- Multi-format document processing
- Intelligent classification and routing
- Specialized agent processing
- Persistent memory storage
- Real-time processing status
- Processing history tracking
- Error handling and validation
- Beautiful and responsive UI

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
MIT License - See LICENSE file for details

## API Endpoints

- POST `/process`: Accepts input files and processes them through the agent system
- GET `/status/{task_id}`: Check the status of a processing task
- GET `/history`: View processing history

## Memory System

The system uses Redis for shared memory storage, maintaining:
- Source information
- Processing timestamps
- Extracted values
- Conversation/thread IDs
- Processing history 
