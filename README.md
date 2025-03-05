# Financial News Chat Agent

A sophisticated application that combines RAG (Retrieval Augmented Generation) with chat capabilities to provide intelligent, contextual responses to queries about financial news, market trends, and company information.

## Overview

This system consists of a FastAPI backend that implements advanced RAG techniques and a React frontend that provides a user-friendly chat interface. The application allows users to:

- Query financial news and information about specific companies
- Maintain conversational context across multiple exchanges
- Receive relevant article links and supporting quotes
- Start new conversations or clear existing history

## Architecture

The application follows a client-server architecture:

### Backend (Python/FastAPI)

```
app/
├── __init__.py
├── api/
│   ├── __init__.py
│   └── app.py              # FastAPI application entrypoint
└── core/
    ├── __init__.py
    ├── config.py           # Configuration settings
    ├── memory.py           # Conversation history management
    └── rag.py              # Advanced RAG implementation
```

### Frontend (React/TypeScript)

```
frontend/
├── public/
│   └── index.html
├── src/
│   ├── App.tsx             # Main React component
│   ├── index.css
│   └── index.tsx
├── package.json
├── tailwind.config.js
└── tsconfig.json
```

## Key Components

### AdvancedRAG System

The core of the application is the `AdvancedRAG` class in `rag.py`, which:

1. Loads financial news data from a JSON file
2. Splits articles into smaller chunks for effective retrieval
3. Generates embeddings for all chunks using OpenAI's embedding model
4. Performs semantic search based on user queries
5. Processes results and generates contextual responses

### Conversation Management

The `ConversationManager` class in `memory.py` handles:

- Storing conversation history with a configurable number of turns
- Formatting messages for LLM context
- Managing conversation state across multiple user sessions

### API Layer

The FastAPI application in `app.py` provides:

- `/chat` endpoint for processing user queries
- `/new-conversation` endpoint for starting fresh conversations
- Cookie-based conversation tracking
- Error handling and logging

### User Interface

The React frontend provides:

- A clean, intuitive chat interface
- Display of chat messages with useful links and supporting quotes
- Options to clear history or start new conversations
- Loading indicators and error handling

## Test Script

The application includes a test script that:

- Validates the RAG system's accuracy against a predefined set of test cases
- Tests financial data retrieval across various companies and metrics
- Performs automated evaluation of answers against expected results
- Calculates and reports the system's overall accuracy

The test script provides 25 diverse test cases covering company statistics, market data, executive information, and partnership details. It extracts answers from the RAG system and compares them to expected values, supporting both text matching and numerical comparisons.

To run the tests:

```bash
poetry run python test.py
```

Test results are saved to `qa_test_results.json` with detailed information about each test case, including questions, expected answers, actual system responses, and pass/fail status.

## How It Works

1. **Data Processing**: Financial news articles are loaded, chunked, and embedded
2. **User Query**: The user submits a question about financial news/companies
3. **Semantic Search**: The system finds relevant article chunks based on semantic similarity
4. **Context Integration**: Previous conversation context is considered for follow-up questions
5. **Response Generation**: An LLM generates a comprehensive response with relevant sources
6. **Results Display**: The response, along with useful links and quotes, is displayed to the user

## Setup & Installation

### Prerequisites

- Python 3.9+
- Node.js 14+
- OpenAI API key
- Make sure you add the `stock_news.json` file in a data folder in the root directory

### Backend Setup

1. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

2. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

3. Add the database file

Put it under `/data/stock_news.json`

4. Run the FastAPI server from the app folder:
   ```bash
   (cd app)
   poetry run uvicorn api.app:app --reload
   ```

The API will be available at `http://localhost:8000`.

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm start
   ```

The frontend will be available at `http://localhost:3000`.

## Data Format

The system expects financial news data in a JSON file with the following structure (make sure to include that file under `/data/stock_news.json`):

```json
{
  "AAPL": [
    {
      "title": "Article title",
      "full_text": "Full article content...",
      "date": "2023-01-01",
      "link": "https://example.com/article"
    }
  ],
  "MSFT": [
    {
      "title": "Another article title",
      "full_text": "Article content...",
      "date": "2023-01-02",
      "link": "https://example.com/another-article"
    }
  ]
}
```

## Key Features

### Semantic Search Enhancement

The system implements several enhancements to basic semantic search:

- **Query Type Detection**: Automatically detects if a query is about a specific ticker/company
- **Title Boost**: Boosts relevance of titles for certain query types
- **Follow-up Handling**: Maintains context for follow-up questions

### Advanced LLM Integration

- Uses function calling to generate structured responses
- Supports extraction of useful links and supporting quotes
- Handles conversation context for coherent multi-turn exchanges

### Error Handling & Robustness

- Graceful fallback mechanisms when search returns no results
- Structured error handling in both frontend and backend
- Comprehensive logging for debugging

## Future Enhancements

Potential areas for improvement:

1. **Real-time data updates**: Integrate with financial APIs for current market data
2. **Multi-modal support**: Add chart generation and visualization capabilities
3. **User authentication**: Add user accounts and personalized experiences
4. **Advanced filtering**: Allow filtering by date range, source, and sentiment
5. **Performance optimization**: Implement caching and query optimization
