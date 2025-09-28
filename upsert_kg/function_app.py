import azure.functions as func
import os
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from json import loads
from llama_index.core import PropertyGraphIndex
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core import Document, Settings
from neo4j import GraphDatabase
from utils import *

load_dotenv()

NLTK_DATA = os.getenv("NLTK_DATA")
TIKTOKEN_CACHE_DIR = os.getenv("TIKTOKEN_CACHE_DIR")

from llama_index.llms.azure_openai import AzureOpenAI

NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_URI = os.getenv("NEO4J_URI")

AZURE_BLOB_CONN_STR = os.getenv("AZURE_BLOB_CONN_STR")
SEMANTIC_CONTAINER_NAME = os.getenv("SEMANTIC_CONTAINER_NAME")

AZURE_OPENAI_EMBEDDING_MODEL_NAME = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL_NAME")
AZURE_OPENAI_EMBEDDING_ENGINE = os.getenv("AZURE_OPENAI_EMBEDDING_ENGINE")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_MODEL_NAME = os.getenv("AZURE_OPENAI_MODEL_NAME")
AZURE_OPENAI_ENGINE = os.getenv("AZURE_OPENAI_ENGINE")
AZURE_OPENAI_TYPE = os.getenv("AZURE_OPENAI_TYPE")

llm = AzureOpenAI(
    model=AZURE_OPENAI_MODEL_NAME,
    engine=AZURE_OPENAI_ENGINE,
    api_key=AZURE_OPENAI_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_type=AZURE_OPENAI_TYPE,
    api_version="2024-03-01-preview",
    temperature=0.3,
)

embeddings = AzureOpenAIEmbedding(
    model=AZURE_OPENAI_EMBEDDING_MODEL_NAME,
    deployment_name=AZURE_OPENAI_EMBEDDING_ENGINE,
    api_key=AZURE_OPENAI_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version="2024-02-01",
)

Settings.llm = llm
Settings.embed_model = embeddings

driver = GraphDatabase.driver(auth=(NEO4J_USERNAME, NEO4J_PASSWORD),uri = NEO4J_URI)

#app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)
#@app.route(route="upsert_kg")
def main(req: func.HttpRequest) -> func.HttpResponse:

    req_json = req.get_json()
    database = req_json["database"]

    try:
        blob_name = f"{database}_semantic.json"
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_BLOB_CONN_STR)
        container_client = blob_service_client.get_container_client(SEMANTIC_CONTAINER_NAME)
        blob_client = container_client.get_blob_client(blob_name)
        blob_data = blob_client.download_blob().readall()
        json_data = loads(blob_data)

    except Exception as e:
         return func.HttpResponse(f"Error in loading from Blob Storage: {str(e)}", status_code=500)
    
    try:
        documents = extract_documents_from_semantic(json_data)
        nodes = create_nodes(documents)

        with driver.session() as session:
            session.run("""MATCH(n)\n DETACH DELETE n""")

        neo4jpg = Neo4jPropertyGraphStore(
            username = NEO4J_USERNAME,
            password = NEO4J_PASSWORD,
            url = NEO4J_URI
        )

        kg_extractor = MyGraphExtractor()

        index = PropertyGraphIndex(
            nodes = nodes,
            kg_extractors = [kg_extractor],
            show_progress = True,
            property_graph_store = neo4jpg
        )

    except Exception as e:
         return func.HttpResponse(f"Error in Creation of KG: {str(e)}", status_code = 500)

    return func.HttpResponse("Knowledge Graph updated to reflect the changes in the semantic layer.", status_code = 200)
