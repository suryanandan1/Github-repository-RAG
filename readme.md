# GitHub Repository Assistant

An AI-powered Streamlit app that clones a GitHub repository, indexes its code with embeddings, and lets you chat with it — ask questions, get a summary, review dependencies, check code health, and more. Supports both pasting a public repo URL and logging in with GitHub to browse and load your own public, private, and organization repositories.

---

## Features

### Repository Indexing
- Clone any public repository by URL, or load one directly from your GitHub account
- Extracts and chunks Python source files
- Generates embeddings and stores them in ChromaDB
- Progress bar during indexing

### Repository Insights
- **Repository Summary** — LLM-generated overview of purpose, functionality, and architecture
- **Repository Statistics** — Python file count, total lines, largest file
- **Repository Health Score** — heuristic score out of 100
- **Dependency Graph** — imports per file
- **Tech Stack Detection** — frameworks/libraries in use
- **Code Review** — LLM-generated recommendations (architecture, performance, security, code quality, scalability)
- **Architecture Analysis** — LLM-generated structural/architecture overview
- **File Preview** — browse and view any indexed file's source

### Chat
- Ask natural-language questions about the repository (RAG-backed via ChromaDB retrieval)
- **Streamed, ChatGPT-style responses** — answers appear token-by-token with a typing cursor (▌)
- **Conversation memory** — last 10 messages are passed to the model, so follow-ups like *"who created it?"* resolve back to the repository being discussed
- **Automatic question routing** — repository questions use retrieval; general programming questions (e.g. *"What is OOP?"*) are answered directly without touching the repository
- **Quick Actions** — one-click buttons for common questions (Summarize Repository, Explain Architecture, List Main Classes/Functions, Show Dependencies, Show Tech Stack, Show Repository Health)
- **Clickable Chat History** — click any past question to resend it
- **New Chat** — clears the conversation only; the repository, its embeddings, and ChromaDB data are untouched

### GitHub OAuth & Repository Explorer
- **Login with GitHub** — OAuth flow with profile display (avatar, name, followers, public repo count)
- **Repository Explorer** — browse your repositories (owned, collaborator, and organization), with search
- **One-click load** — select a repo and load it directly, no manual URL needed
- **Private repository support** — authenticated cloning for private and org repos
- **Repository metadata** — stars, forks, language, description, last updated, public/private status
- **Manual mode preserved** — pasting a repository URL still works without logging in

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
|   ├── github_auth.py 
|   ├── github_api.py
|   ├── github_loader.py
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

## Setup

### 1. Clone and install

```bash
git clone <this-repo-url>
cd your-project
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

### 2. Environment variables

Create a `.env` file in the project root:

```env
# Mistral AI
MISTRALAI_API_KEY=your_mistral_api_key

# GitHub OAuth (see below for how to get these)
GITHUB_CLIENT_ID=your_github_client_id
GITHUB_CLIENT_SECRET=your_github_client_secret
GITHUB_REDIRECT_URI=http://localhost:8501

# Optional
REPO_STORAGE_PATH=repos
```

### 3. Create a GitHub OAuth App

1. Go to **GitHub → Settings → Developer settings → OAuth Apps → New OAuth App**
2. Fill in:
   - **Application name**: anything recognizable, e.g. `GitHub Repository Assistant`
   - **Homepage URL**: `http://localhost:8501`
   - **Authorization callback URL**: `http://localhost:8501` (must exactly match `GITHUB_REDIRECT_URI`)
3. Click **Register application**
4. Copy the **Client ID**, then click **Generate a new client secret** and copy it immediately (shown only once)
5. Paste both into your `.env`

> If you deploy this app elsewhere, update both the OAuth App's callback URL and `GITHUB_REDIRECT_URI` to match the deployed address. A GitHub OAuth App supports only one callback URL at a time.

### 4. Run the app

```bash
streamlit run app.py
```

---

## Usage

**Manual mode** — paste a public repository URL into the text field and click **Process Repository**.

**GitHub mode** — click **Login with GitHub** in the sidebar, authorize the app, then use the **Repository Explorer** to search/select one of your repositories and click **Load Repository**. This works for public, private, and organization repos.

Once a repository is loaded, use the sidebar to jump between Repository Actions (summary, dependencies, tech stack, code review, architecture, file preview), fire off Quick Actions, or just start chatting at the bottom of the page.

---

## Tech Stack

- **Streamlit** — UI
- **Mistral AI** (`codestral-latest`) — chat completions, streaming, question classification
- **ChromaDB** — vector storage
- **sentence-transformers** (`BAAI/bge-small-en-v1.5`) — embeddings
- **GitPython** — repository cloning
- **GitHub OAuth / REST API** — login and repository access

---

## Notes

- Only Python (`.py`) files are indexed.
- Each new repository load clears the previous ChromaDB collection before indexing the new one.
- The health score is a simple heuristic based on file count and total line count — not a full static-analysis tool.

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
