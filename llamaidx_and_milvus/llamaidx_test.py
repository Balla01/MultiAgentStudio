from llama_cloud_services import LlamaExtract
from pydantic import BaseModel, Field

# bring in our LLAMA_CLOUD_API_KEY
from dotenv import load_dotenv
load_dotenv()

# Initialize client
extractor = LlamaExtract()

# Define schema using Pydantic
class Resume(BaseModel):
    name: str = Field(description="Full name of candidate")
    email: str = Field(description="Email address")
    skills: list[str] = Field(description="Technical skills and technologies")

# Create extraction agent
agent = extractor.create_agent(name="resume-parser", data_schema=Resume)

# Extract data from document
result = agent.extract("/home/ntlpt19/Downloads/Final_Delivery_Training_itter_4/BEFORE/CI_train_604/Images/Invoice(2012_08_21_13_51_21_4587)_629.TIF_0.png")
print(result.data)

'''

from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.vector_stores.types import VectorStoreQuery, VectorStoreQueryMode

vector_store = MilvusVectorStore(
    uri="http://localhost:19530",
    collection_name="your_collection",
    dim=768,
    overwrite=False
)
embed_model = {}#put your model here
query_text = "What is deep learning?"
query_embedding = embed_model.get_text_embedding(query_text)
query = VectorStoreQuery(
    query_embedding=query_embedding,  # your embedding vector
    similarity_top_k=5,
    mode=VectorStoreQueryMode.DEFAULT
)
result = vector_store.query(query)
'''