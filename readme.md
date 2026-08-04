# GitHub Repository Assistant

An AI-powered GitHub Repository Assistant built with Streamlit, ChromaDB, Mistral AI, and RAG (Retrieval-Augmented Generation).

The application allows users to provide any GitHub repository URL and interact with the repository through an intelligent chat interface. It analyzes source code, generates repository summaries, detects technologies, visualizes dependencies, reviews code quality, and answers repository-specific questions.

---

# Features

## Repository Processing

- Clone any public GitHub repository
- Read Python source files automatically
- Create intelligent code chunks
- Generate embeddings
- Store vectors in ChromaDB
- Build repository knowledge base

---

## AI Repository Chat

Ask questions such as:

- What does this repository do?
- Explain the project architecture.
- How do I run this project?
- What are the important files?
- Explain a specific function.
- List all classes.
- What technologies are used?

The assistant retrieves relevant code context before generating answers.

---

## Repository Summary

Automatically generates:

- Project overview
- Core functionality
- Important modules
- Architecture explanation

---

## Repository Statistics

Displays:

- Total Python files
- Total lines of code
- Largest file
- Largest file size

---

## Technology Stack Detection

Automatically identifies:

- Streamlit
- Flask
- FastAPI
- Django
- LangChain
- ChromaDB
- OpenAI
- Mistral
- Other frameworks

---

## Dependency Analysis

Builds a dependency graph showing:

- Module imports
- Internal dependencies
- File relationships

---

## AI Code Review

Generates recommendations for:

- Code quality
- Performance improvements
- Maintainability
- Best practices

---

## Repository Health Score

Calculates project health based on:

- File count
- Code size
- Project structure
- Repository complexity

---

## File Explorer

Browse repository files directly from the sidebar.

Features:

- File selection
- Source code viewer
- Expandable code display

---

## Chat History

Maintains:

- Conversation history
- Previous questions
- Previous answers

Stored in Streamlit session state.

---

## Quick Actions

One-click repository analysis:

- Repository Summary
- Architecture
- Main Classes
- Main Functions

---

# Project Architecture

```text
github1/
│
├── app.py
│
├── src/
│   ├── github_loader.py
│   ├── file_reader.py
│   ├── chunker.py
│   ├── mistral_embeddings.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── llm_service.py
│   ├── repository_summary.py
│   ├── repository_analyzer.py
│   ├── dependency_analyzer.py
│   ├── code_reviewer.py
│   ├── tech_stack.py
│   └── repository_health.py
│
├── chroma_db/
├── repos/
├── .env
├── requirements.txt
└── README.md
```

---

# Tech Stack

## Frontend

- Streamlit

## LLM

- Mistral AI
- Codestral

## Embeddings

- Mistral Embeddings

## Vector Database

- ChromaDB

## Repository Access

- GitPython

## Environment Management

- python-dotenv

---

# Installation

## Clone Project

```bash
git clone <your-project-url>
cd github1
```

---

## Create Virtual Environment

```bash
python -m venv .venv
```

Activate:

### Windows

```bash
.venv\Scripts\activate
```

### Linux/Mac

```bash
source .venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Environment Variables

Create a `.env` file:

```env
MISTRALAI_API_KEY=your_api_key

REPO_STORAGE_PATH=repos
```

---

# Run Application

```bash
streamlit run app.py
```

Application URL:

```text
http://localhost:8501
```

---

# Usage

## Step 1

Paste a GitHub repository URL.

Example:

```text
https://github.com/streamlit/streamlit
```

---

## Step 2

Click:

```text
Process Repository
```

---

## Step 3

Wait for:

- Repository cloning
- File reading
- Chunk creation
- Embedding generation
- ChromaDB indexing

---

## Step 4

Explore:

- Repository Summary
- Statistics
- Health Score
- Tech Stack
- Dependency Graph
- AI Code Review

---

## Step 5

Ask questions in Repository Chat.

Example:

```text
How does authentication work?
```

```text
Explain repository architecture.
```

```text
Which file contains the main application?
```

```text
How do I run this project?
```

---

# Future Enhancements

## Phase 6

- Multi-repository comparison
- Architecture visualization diagrams
- Repository flowcharts
- Auto-generated documentation
- README improvement suggestions
- Code smell detection

## Phase 7

- GitHub OAuth Login
- Private repository support
- Pull Request analysis
- Issue analysis
- Commit history insights

## Phase 8

- Multi-language support
- Java repositories
- JavaScript repositories
- C++ repositories
- TypeScript repositories

---

# Author

Developed by Suryanandan

GitHub Repository Assistant using:

- Streamlit
- Mistral AI
- ChromaDB
- Retrieval Augmented Generation (RAG)

---