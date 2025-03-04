# Advanced RAG with LangGraph

A sophisticated Retrieval-Augmented Generation (RAG) application that enhances LLM responses with contextual document retrieval and multi-stage verification.

## Getting Started

This guide will help you set up and run the Advanced RAG application on your local machine.

### Prerequisites

Before you begin, make sure you have the following installed:
- Python 3.12 or higher
- Git

### Installation

1. **Clone the Repository**

   Open your terminal and run:
   ```bash
   git clone https://github.com/segave/Advanced-RAG---LangGraph
   cd Advanced-RAG-LangGraph
   ```

2. **Install Poetry**

   ```bash
   # Install Poetry using pip
   pip install -r requirements.txt
   ```

3. **Install Project Dependencies**

   ```bash
   # Install all project dependencies
   poetry install
   
   # Activate the virtual environment
   poetry shell
   ```

4. **Run the Application**

   Once you have activated the virtual environment with `poetry shell`, you can run the application:
   ```bash
   # Make sure you are in the project directory
   streamlit run app.py
   ```

   The application will start and automatically open in your default web browser. If it doesn't open automatically, you can access it at:
   - Local URL: http://localhost:8501
   - Network URL: http://192.168.X.X:8501 (for accessing from other devices on your network)

## Environment Configuration

1. **Create Environment File**
   
   Copy the example environment file and add your API keys:
   ```bash
   cp .env.example .env
   ```

   Then edit the `.env` file and add your API keys:
   ```env
   # Required
   OPENAI_API_KEY=your_openai_api_key
   TAVILY_API_KEY=your_tavily_api_key
   ```

2. **Optional Environment Variables**
   
   You can customize these settings in your `.env` file:
   ```env
    LANGCHAIN_API_KEY=your_langchain_api_key_here
    LANGCHAIN_PROJECT=your_project_name
    LANGCHAIN_TRACING_V2=true
    PYTHONPATH=your_project_path 
   ```

## Features and Functionality

### 1. Document Processing
- Upload and process multiple document formats:
  * PDF documents (.pdf)
  * Word documents (.docx)
  * Text files (.txt)
- Automatic text chunking and vectorization
- Persistent vector storage using ChromaDB

### 2. Advanced RAG Pipeline
- Multi-stage document retrieval and verification:
  * Entry classification for query routing
  * Contextual document retrieval
  * Relevance grading of retrieved documents
  * Response generation with context
  * Hallucination detection
  * Answer quality assessment
  * Web search if needed

### 3. LLM Model Selection
- Choose between different OpenAI models:
  * GPT-4o
  * GPT-4o-mini
- Dynamic model switching without application restart

### 4. Document Management
- Upload interface for multiple documents
- Document database cleanup functionality
- Vector store persistence across sessions
- Efficient document chunking and retrieval

### 5. Chat Interface
- Interactive chat with context-aware responses
- Chat history maintenance
- Clear conversation option
- Real-time response generation

### 6. Quality Control
- Automatic verification of responses against sources
- Detection of potential hallucinations
- Relevance scoring of retrieved documents
- Answer quality assessment

## Docker Installation

If you prefer to use Docker, you can run the application using our Docker image 
([available on Docker Hub](https://hub.docker.com/r/sergiogaliana/advanced-rag-langgraph)):

### Option 1: Using Docker Run

1. **Pull the Docker image**:
   ```bash
   docker pull sergiogaliana/advanced-rag-langgraph:latest
   ```

2. **Run the container with environment variables**:
   ```bash
   docker run -p 8501:8501 \
     -e OPENAI_API_KEY=your_key \
     -e TAVILY_API_KEY=your_key \
     sergiogaliana/advanced-rag-langgraph:latest
   ```

### Option 2: Using Docker Compose (Recommended)

1. **Create a docker-compose.yml file**:
   ```bash
   cp docker-compose.yml.example docker-compose.yml
   ```

2. **Add your environment variables**:
   Either edit the docker-compose.yml directly or create a .env file with your variables:
   ```env
   OPENAI_API_KEY=your_key
   TAVILY_API_KEY=your_key
   ```

3. **Run the container**:
   ```bash
   docker-compose up
   ```

The application will be available at:
- Local URL: http://localhost:8501

More sections about configuration and usage will be added soon...
