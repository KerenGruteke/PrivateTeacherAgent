from propcache import cached_property
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from qdrant_client import QdrantClient
from src.utils.constants import QDRANT_CLUSTER_URL, QDRANT_API_KEY, EMBEDDING_DIM
from src.utils.LLM_utils import get_embedding_object
from typing import List
from loguru import logger
from tqdm import tqdm
import uuid
import pandas as pd

def convert_to_uuid(id):
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, str(id)))

class DB:
    def __init__(self):
        self.qdrant_client = QdrantClient(
            url=QDRANT_CLUSTER_URL,
            api_key=QDRANT_API_KEY,
        )
        self.embeder_client = get_embedding_object()

    def create_collection(self, collection_name, dim=1):
        if not self.qdrant_client.collection_exists(collection_name):
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def insert_data(self, collection_name, data: List[tuple], id_col: str):
        """
        Data points as a list of tuples, where each tuple contains:
        - A vector (list of floats)
        - A dictionary with fields and values.
        
        Example of Format:
            [(
            [0.0],
            {
                "student_id": "S001",
                "name": "Alice",
                "status": [{
                    "math": [{"score": 80, "note": "needs more algebra", "date": "18-07-2025"},
                            {"score": 70, "note": "should practice multiplication", "date": "18-07-2025"}],
                    "history": [{"score": 90, "note": "Practice Renaissance", "date": "10-06-2024"}]
                }]
            }
        ),
        ]

        Args:
        collection_name (str): The name of the Qdrant collection.
        data (List[tuple]): List of tuples containing vectors and payloads.
        """
        data_points = []
        for vector, payload in tqdm(data, desc=f"Inserting data into {collection_name}"):
            data_points.append(
                PointStruct(
                    id=convert_to_uuid(payload[id_col]),
                    vector=vector,
                    payload=payload
                )
            )

        self.qdrant_client.upsert(collection_name=collection_name, points=data_points)

    def update_metadata(self, collection_name, item_id, new_metadata: dict):
        uuid_id = convert_to_uuid(item_id)

        self.qdrant_client.set_payload(
            collection_name=collection_name,
            payload=new_metadata,
            points=[uuid_id]
        )

    def get_items_data(self, collection_name, item_ids: List, id_col: str):
        uuid_ids = [convert_to_uuid(item_id) for item_id in item_ids]

        results = self.qdrant_client.retrieve(
            collection_name=collection_name,
            ids=uuid_ids,
            with_payload=True,
            with_vectors=False
        )

        if results:
            return {item.payload[id_col]: item.payload for item in results}

        return None

    def clean_collection(self, collection_name, vector_dim=384):
        """
        Clean the specified Qdrant collection by deleting all its vectors.
        :param collection_name: The name of the collection to clean.
        :param vector_dim: The dimensionality of the vectors in the collection.
        """
        self.qdrant_client.delete_collection(collection_name=collection_name)
        self.qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
        )
        logger.info(f"Collection '{collection_name}' has been cleaned and recreated.")


    def delete_all_collections(self):
        """
        Delete all Qdrant collections.
        """
        collections = self.qdrant_client.get_collections()
        for collection in collections.collections:
            self.qdrant_client.delete_collection(collection_name=collection.name)
            logger.info(f"Collection '{collection.name}' has been deleted.")
            
    def delete_collection(self, collection_name: str):
        """
        Delete a specific Qdrant collection.
        :param collection_name: The name of the collection to delete.
        """
        self.qdrant_client.delete_collection(collection_name=collection_name)
        logger.info(f"Collection '{collection_name}' has been deleted.")

    def print_collections(self):
        print(self.qdrant_client.get_collections())
        
    def print_example(self, collection_name: str, limit: int = 1):
        points = self.qdrant_client.scroll(collection_name=collection_name, limit=limit)
        print(f"Example from collection '{collection_name}': \n{points}")
        
    def print_collection_size(self, collection_name: str):
        """
        Print the size of a specific Qdrant collection.
        :param collection_name: The name of the collection to check.
        """
        size = self.qdrant_client.count(collection_name=collection_name)
        print(f"Collection '{collection_name}' size: {size}")

    def search_by_query_vec(self, collection_name: str, query: str, top_k: int = 5) -> list[str]:
        vecs, _ = self.embeder_client.embed([query])
        vec = vecs[0] # need only one vector for the query
        results = self.qdrant_client.search(collection_name=collection_name, query_vector=vec, limit=top_k)
        return [result.payload for result in results]


@cached_property
def get_db_object():
    return DB()


def index_df(df, index_by_col: str, need_to_embed_col: bool, id_col: str, collection_name: str):
    """
    Index the data in the DataFrame into a Qdrant collection.
    :param df: DataFrame containing the data to index.
    :param index_by_col: Column name to use for indexing.
    :param collection_name: Name of the Qdrant collection to create or use.
    :param dim: Dimensionality of the vectors in the collection.
    """
    # embed indexed col
    if need_to_embed_col:
        embeder_client = get_embedding_object()
        vectors, dim = embeder_client.embed(df[index_by_col].tolist())
    else:
        # vectors os just a list of zeros
        dim = 1
        vectors = [[0.0] * dim for _ in range(len(df))]
    
    db = DB()
    db.create_collection(collection_name, dim)
    db.insert_data(collection_name, list(zip(vectors, df.to_dict(orient='records'))), id_col=id_col)
    db.print_collection_size(collection_name)

def _get_studens_DB_for_test():
    students = [
        (
            [0.0],
            {
                "student_id": "S001",
                "name": "Alice",
                "status": [{
                    "math": [{"score": 80, "note": "needs more algebra", "date": "18-07-2025"},
                            {"score": 70, "note": "should practice multiplication", "date": "18-07-2025"}],
                    "history": [{"score": 90, "note": "Practice Renaissance", "date": "10-06-2024"}]
                }]
            }
        ),
        (
            [0.0],
            {
                "student_id": "S002",
                "name": "Bob",
                "status": [{
                    "math": [{"score": 60, "note": "needs more algebra", "date": "18-07-2025"},
                            {"score": 90, "note": "should practice multiplication", "date": "18-07-2025"}],
                    "history": [{"score": 95, "note": "Practice Renaissance", "date": "10-06-2024"}]
                }]
            }
        ),
    ]
    
def _test_index_df():
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "text": ["Hello", "World", "Test"],
        "feature1": [0.1, 0.2, 0.3],
        "feature2": [0.4, 0.5, 0.6],
        "feature3": [0.7, 0.8, 0.9]
    })
    
    index_df(df, index_by_col="text", need_to_embed_col=True, id_col="id", collection_name="test_collection")
    logger.debug("Indexing completed successfully.")

def _test_search_by_query_vec():    
    query = "Bye Bye"
    db = DB()
    results = db.search_by_query_vec(collection_name="test_collection", query=query, top_k=2)
    print(f"Search results for query '{query}': \n{results}")
    logger.debug("Search completed successfully.")

if __name__ == "__main__":
    # _test_index_df()
    _test_search_by_query_vec()


