# Agentic AI Studio

Agentic AI Studio is a unified platform for building, deploying, and interacting with multi-agent AI systems, enhanced by Retrieval-Augmented Generation (RAG) across Vector (Milvus), Graph (Neo4j), and SQL (SQLite) databases.

## Features

- **Multi-Modal Retrieval**: Seamlessly query unstructured text (Vector), structured relationships (Graph), and tabular data (SQL).
- **Dynamic Agent Loading**: Load CrewAI agents dynamically from the `src/studio/crews` directory.
- **Unified UI**: specific Tabs for Data Ingestion, Agent Configuration, and Chat.
- **Extensible Architecture**: Modular design for adding new database providers or agent types.

## Installation

1.  **Prerequisites**:
    *   Python 3.10+
    *   Milvus (running locally or remote)
    *   Neo4j (running locally or remote)
    *   CrewAI installed

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

Open your browser at `http://localhost:7860`.

### Workflow

1.  **Data Setup**: Go to the "Data Setup" tab. Upload your text files (e.g., policy documents, rules). Select which databases to ingest into (Milvus for semantic search, Neo4j for graph rules).
2.  **Agent Config**: Go to "Agent Configuration". Select a Crew (e.g., `test_crew`) and connect the desired databases. Create a session.
3.  **Chat**: Go to "Chat". Enable/Disable RAG and start chatting with your agent.

## Project Structure

```
src/studio/
├── core/          # Core abstractions (DB Manager, Agent Factory)
├── data/          # Data pipeline (Ingestion, Chunking, Embedding)
├── services/      # Business logic (Agent, RAG, Chat services)
├── ui/            # Gradio components
├── config/        # Configuration files
└── main.py        # Entry point
```
