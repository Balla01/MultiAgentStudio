import os
import sys
import logging
import json
import sqlite3
import pandas as pd
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from threading import Lock

# Third-party imports
try:
    from pymilvus import MilvusClient, DataType, FieldSchema, CollectionSchema, connections, Collection
    from sentence_transformers import SentenceTransformer
except ImportError:
    logging.warning("Milvus dependencies not found. MilvusProvider will not work.")

try:
    from neo4j import GraphDatabase
except ImportError:
    logging.warning("Neo4j dependencies not found. Neo4jProvider will not work.")

try:
    from crewai import LLM
except ImportError:
    logging.warning("CrewAI dependencies not found. SQL provider LLM features may not work.")

# Configure Logger
logger = logging.getLogger(__name__)

class DatabaseProvider(ABC):
    """Abstract base class for all database providers."""
    
    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to the database."""
        pass
    
    @abstractmethod
    def disconnect(self):
        """Close connection."""
        pass

    @abstractmethod
    def dump_data(self, data: Any, metadata: Optional[Dict] = None) -> bool:
        """Insert data into the database."""
        pass
    
    @abstractmethod
    def query(self, query_params: Dict) -> Any:
        """Query the database."""
        pass
    
    @abstractmethod
    def health_check(self) -> bool:
        """Check if database is accessible."""
        pass

class MilvusProvider(DatabaseProvider):
    def __init__(self, config: Dict):
        self.uri = config.get("uri", "http://localhost:19530")
        self.host = config.get("host", "localhost")
        self.port = config.get("port", "19530")
        self.collection_name = config.get("collection_name", "itf_testing")
        self.dimension = config.get("dimension", 1024)
        self.client = None
        self.collection = None
        
        # We might need an embedding model for queries if not passed from outside
        # For now, we assume query params include the vector
        self.embedding_model_name = config.get("embedding_model", "Alibaba-NLP/gte-large-en-v1.5")
        self.embedding_model = None

    def connect(self) -> bool:
        try:
            logger.info(f"Connecting to Milvus at {self.uri}...")
            self.client = MilvusClient(uri=self.uri)
            
            # Also use connections for ORM-style usage if needed
            connections.connect(alias="default", host=self.host, port=self.port)
            
            if self.client.has_collection(self.collection_name):
                self.collection = Collection(self.collection_name)
                self.collection.load()
                logger.info(f"Milvus Collection '{self.collection_name}' loaded.")
            else:
                logger.warning(f"Milvus Collection '{self.collection_name}' does not exist.")
            
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            return False

    def disconnect(self):
        if self.client:
            self.client.close()
        connections.disconnect("default")

    def dump_data(self, data: List[Dict], metadata: Optional[Dict] = None) -> bool:
        """
        data: List of dicts with keys matching schema (id, vector, text, metadata)
        """
        if not self.client:
            if not self.connect():
                return False

        try:
            # Check if collection exists, if not create it
            if not self.client.has_collection(self.collection_name):
               self._create_collection()

            # Insert data
            res = self.client.insert(collection_name=self.collection_name, data=data)
            logger.info(f"Inserted {len(data)} entities into Milvus.")
            return True
        except Exception as e:
            logger.error(f"Error dumping data to Milvus: {e}")
            return False

    def _create_collection(self):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="metadata", dtype=DataType.JSON)
        ]
        schema = CollectionSchema(fields, description="Studio Collection")
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector", 
            index_type="HNSW", 
            metric_type="COSINE", 
            params={"M": 16, "efConstruction": 200}
        )
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params
        )

    def query(self, query_params: Dict) -> List[Dict]:
        """
        query_params: {
            "vector": List[float] (optional, if query_text provided),
            "query_text": str (optional),
            "top_k": int,
            "filter": str
        }
        """
        if not self.client:
            self.connect()
            
        top_k = query_params.get("top_k", 5)
        metadata_filter = query_params.get("filter", "")
        vector = query_params.get("vector")
        query_text = query_params.get("query_text")
        
        if vector is None and query_text is None:
            raise ValueError("Must provide either 'vector' or 'query_text' in query_params")

        if vector is None:
             # Lazy load embedding model
            if not self.embedding_model:
                try:
                    self.embedding_model = SentenceTransformer(self.embedding_model_name, trust_remote_code=True)
                except Exception as e:
                    logger.error("Failed to load embedding model: %s", e)
                    return []
            vector = self.embedding_model.encode(query_text).tolist()

        search_params = {"metric_type": "COSINE", "params": {"ef": 100}}
        
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                data=[vector],
                limit=top_k,
                filter=metadata_filter,
                output_fields=["text", "metadata"]
            )
            
            # Format results
            formatted_results = []
            for res in results[0]:
                formatted_results.append({
                    'score': res['distance'],
                    'text': res['entity'].get('text'),
                    'metadata': res['entity'].get('metadata'),
                    'id': res['id']
                })
            return formatted_results

        except Exception as e:
            logger.error(f"Milvus search failed: {e}")
            return []

    def health_check(self) -> bool:
        if not self.client:
            return False
        try:
            self.client.get_collection_stats(self.collection_name)
            return True
        except:
            return False


class Neo4jProvider(DatabaseProvider):
    def __init__(self, config: Dict):
        self.uri = config.get("uri", "bolt://localhost:7687")
        self.user = config.get("username", "neo4j")
        self.password = config.get("password", "password")
        self.driver = None

    def connect(self) -> bool:
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            self.driver.verify_connectivity()
            logger.info("Connected to Neo4j.")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            return False

    def disconnect(self):
        if self.driver:
            self.driver.close()

    def dump_data(self, data: List[Dict], metadata: Optional[Dict] = None) -> bool:
        """
        data: List of "rule" dicts as parsed by rules_graph_ingest.py
        """
        if not self.driver:
            if not self.connect(): return False
            
        try:
            with self.driver.session() as s:
                for rule in data:
                    s.execute_write(self._create_rule_tx, rule)
            return True
        except Exception as e:
            logger.error(f"Neo4j data dump failed: {e}")
            return False

    @staticmethod
    def _create_rule_tx(tx, rule):
        # Adapted from neo4j_with_vector/a.py
        # Expects rule dict to have keys like 'id', 'name', 'group', etc.
        tx.run("""
        MERGE (r:Rule {id: $id})
        SET r.name = $name,
            r.group = $group,
            r.description = $description,
            r.condition_json = $condition_json
        """, 
        id=rule.get('id', 'unknown'),
        name=rule.get('name'),
        group=rule.get('group'),
        description=rule.get('description'),
        condition_json=json.dumps(rule.get('condition')) if rule.get('condition') else None
        )
        
        # Simple linkage for now - can be expanded
        if rule.get('group'):
            tx.run("""
            MATCH (r:Rule {id:$id})
            MERGE (g:RuleGroup {name:$group})
            MERGE (r)-[:BELONGS_TO]->(g)
            """, id=rule.get('id'), group=rule.get('group'))

    def query(self, query_params: Dict) -> List[Dict]:
        """
        query_params: {
            "cypher": str (optional),
            "entities": List[str] (optional - for simple keyword lookup)
        }
        """
        if not self.driver: self.connect()
        
        cypher = query_params.get("cypher")
        entities = query_params.get("entities")
        
        results = []
        try:
            with self.driver.session() as s:
                if cypher:
                    res = s.run(cypher)
                    results = [r.data() for r in res]
                elif entities:
                    # Simple retrieval of Rules connected to these entities (documents/fields)
                    # This is a placeholder for more complex traversal logic
                    res = s.run("""
                    MATCH (r:Rule)
                    WHERE any(word IN split(toLower(r.description), ' ') WHERE word IN $entities)
                    RETURN r.id, r.name, r.description, r.group
                    LIMIT 10
                    """, entities=entities)
                    results = [r.data() for r in res]
                    
        except Exception as e:
            logger.error(f"Neo4j query failed: {e}")
            
        return results

    def health_check(self) -> bool:
        if not self.driver: return False
        try:
            self.driver.verify_connectivity()
            return True
        except:
            return False


class SQLProvider(DatabaseProvider):
    """
    Wraps SQLite logic but is named SQLProvider for future extensibility.
    Uses 'sql_retrieval/infer.py' logic for Text-to-SQL.
    """
    def __init__(self, config: Dict):
        self.db_path = config.get("database", "studio_data.db")
        self.conn = None
        self.llm_config = config.get("llm_config", {})
        self.llm = None
        
        # Initialize LLM for Text-to-SQL if API key present
        if api_key := self.llm_config.get("api_key"):
            self.llm = LLM(
                model=self.llm_config.get("model", "mistral/mistral-large-latest"),
                api_key=api_key
            )

    def connect(self) -> bool:
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            logger.info(f"Connected to SQLite: {self.db_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to SQLite: {e}")
            return False

    def disconnect(self):
        if self.conn:
            self.conn.close()

    def dump_data(self, data: Any, metadata: Optional[Dict] = None) -> bool:
        """
        data: pandas DataFrame
        metadata: {'table_name': str, 'if_exists': 'append'|'replace'}
        """
        if not self.conn: self.connect()
        
        try:
            if not isinstance(data, pd.DataFrame):
                logger.error("SQLProvider.dump_data expects a pandas DataFrame")
                return False
                
            table_name = metadata.get("table_name", "default_table")
            if_exists = metadata.get("if_exists", "append")
            
            data.to_sql(table_name, self.conn, if_exists=if_exists, index=False)
            logger.info(f"Dumped dataframe to table '{table_name}'")
            return True
        except Exception as e:
            logger.error(f"SQL dump failed: {e}")
            return False

    def query(self, query_params: Dict) -> Dict:
        """
        query_params: {
            "sql": str (optional - direct execution),
            "natural_query": str (optional - text-to-sql)
        }
        """
        if not self.conn: self.connect()
        
        sql = query_params.get("sql")
        natural_query = query_params.get("natural_query")
        
        if natural_query and self.llm:
            sql = self._generate_sql(natural_query)
        
        if not sql:
            return {"error": "No SQL query provided or generated"}
            
        try:
            cursor = self.conn.cursor()
            cursor.execute(sql)
            results = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            return {
                "columns": columns,
                "data": results,
                "sql_used": sql
            }
        except Exception as e:
            logger.error(f"SQL execution failed: {e}")
            return {"error": str(e)}

    def _generate_sql(self, text: str) -> str:
        # Simplified Text-to-SQL logic
        schema = self._get_schema()
        prompt = f"""Convert to SQLite SQL:
        Schema: {schema}
        Question: {text}
        SQL:"""
        try:
            response = self.llm.call(prompt)
            # Basic cleanup
            sql = response.strip().replace("```sql", "").replace("```", "")
            return sql
        except Exception as e:
            logger.error(f"Text-to-SQL failed: {e}")
            return ""

    def _get_schema(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        schema = []
        for t in tables:
            t_name = t[0]
            cursor.execute(f"PRAGMA table_info({t_name})")
            cols = [c[1] for c in cursor.fetchall()]
            schema.append(f"{t_name} ({', '.join(cols)})")
        return "\n".join(schema)

    def health_check(self) -> bool:
        if not self.conn: return False
        try:
            self.conn.execute("SELECT 1")
            return True
        except:
            return False


class DatabaseManager:
    def __init__(self):
        self.providers: Dict[str, DatabaseProvider] = {}
        self._lock = Lock()

    def register_database(self, db_name: str, provider: DatabaseProvider):
        with self._lock:
            self.providers[db_name] = provider
            logger.info(f"Registered database provider: {db_name}")

    def get_provider(self, db_name: str) -> Optional[DatabaseProvider]:
        return self.providers.get(db_name)
    
    def connect_all(self):
        for name, provider in self.providers.items():
            if provider.connect():
                logger.info(f"Connected to {name}")
            else:
                logger.error(f"Failed to connect to {name}")

    def disconnect_all(self):
        for provider in self.providers.values():
            provider.disconnect()
