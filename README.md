# Agentic AI Studio

Agentic AI Studio is a unified platform for building, deploying, and interacting with multi-agent AI systems, enhanced by Retrieval-Augmented Generation (RAG) across Vector (Milvus), Graph (Neo4j), and SQL (SQLite) databases.

## Features

-   **Multi-Modal Retrieval**: Seamlessly query unstructured text (Vector), structured relationships (Graph), and tabular data (SQL).
-   **Work Item Context**: Organize data into specific "Work Items" (Milvus Collections) for targeted retrieval.
-   **Two-Step Ingestion**: robust data pipeline that dumps to a local DB (Milvus Lite) first for safety, then automatically migrates to the production Server.
-   **Dynamic Agent Loading**: Load CrewAI agents dynamically from the `src/studio/crews` directory.
-   **Unified UI**: specific Tabs for Data Ingestion, Agent Configuration, and Chat.
-   **Extensible Architecture**: Modular design for adding new database providers or agent types.

## Installation

1.  **Prerequisites**:
    -   Python 3.10+
    -   Milvus (running locally or remote)
    -   Neo4j (running locally or remote)
    -   CrewAI installed

2.  **Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    (Note: Ensure `pymilvus`, `neo4j`, `gradio`, `crewai`, `sentence-transformers` are installed)

## Configuration

Edit `src/studio/config/studio_config.yaml` to set your database credentials and model preferences:

```yaml
databases:
  milvus:
    uri: "http://localhost:19530"
  neo4j:
    uri: "bolt://localhost:7687"
  mysql:
    database: "studio_data.db"
```

## Usage

Run the studio:

```bash
python3 src/studio/main.py
```

The app will launch at `http://localhost:7860`.

### Workflow

1.  **Data Setup**:
    -   Go to the **"Data Setup"** tab.
    -   Enter a **Work Item ID** (e.g., `ticket_123`). This ID acts as the content group (Milvus Collection Name).
    -   Upload your text files.
    -   Click "Process & Upload". The system will ingest data into a local safekeeping DB, then migrate it to your Milvus Server.

2.  **Agent Config**:
    -   Go to **"Agent Configuration"**.
    -   Select a Crew (e.g., `test_crew`) and connect the desired databases.
    -   Create a session.

3.  **Chat**:
    -   Go to **"Chat"**.
    -   Enter the **Work Item ID** you want to query against (e.g., `ticket_123`).
    -   Enable/Disable RAG and start chatting. The agent will retrieve specific context from that Work Item.

## Project Structure

```
src/studio/
├── core/          # Core abstractions (DB Manager, Agent Factory)
├── data/          # Data pipeline (Ingestion, Chunking, Embedding, Utils)
├── services/      # Business logic (Agent, RAG, Chat services)
├── ui/            # Gradio components (Chat, Data Uploader, Agent Config)
├── config/        # Configuration files
└── main.py        # Entry point
```
