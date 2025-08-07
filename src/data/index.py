from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from qdrant_client import QdrantClient
from src.utils.constants import QDRANT_CLUSTER_URL, QDRANT_API_KEY
import uuid

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

def student_id_to_uuid(student_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, student_id))

class DB:
    def __init__(self):
        self.qdrant_client = QdrantClient(
            url=QDRANT_CLUSTER_URL,
            api_key=QDRANT_API_KEY,
        )

    def create_collection(self, collection_name, dim=1):
        if not self.qdrant_client.collection_exists(collection_name):
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def insert_student(self, collection_name, student_data):
        student_points = []
        for vector, student_payload in student_data:
            student_points.append(
                PointStruct(
                    id=student_id_to_uuid(student_payload["student_id"]),
                    vector=vector,
                    payload=student_payload
                )
            )

        self.qdrant_client.upsert(collection_name=collection_name, points=student_points)

    def update_metadata(self, collection_name, student_id, new_metadata: dict):
        uuid_id = student_id_to_uuid(student_id)

        self.qdrant_client.set_payload(
            collection_name=collection_name,
            payload=new_metadata,
            points=[uuid_id]
        )

    def get_student_data(self, collection_name, student_id):
        uuid_id = student_id_to_uuid(student_id)

        results = self.qdrant_client.retrieve(
            collection_name=collection_name,
            ids=[uuid_id],
            with_payload=True,
            with_vectors=False
        )

        if results:
            student = results[0]
            return student.payload["student_id"],  student.payload

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
        print(f"Collection '{collection_name}' has been cleaned and recreated.")


    def delete_all_collections(self):
        """
        Delete all Qdrant collections.
        """
        collections = self.qdrant_client.get_collections()
        for collection in collections.collections:
            collection_name = collection.name
            self.qdrant_client.delete_collection(collection_name=collection.name)
            print(f"Collection '{collection_name}' has been deleted.")

    def print_collections(self):
        print(self.qdrant_client.get_collections())


db = DB()
db.delete_all_collections()
db.create_collection(
    collection_name="student_db",
    dim=1
)
db.insert_student("student_db", students)
_, student_payload = db.get_student_data("student_db", "S002")
print(student_payload)
student_payload["name"] = "Ali Mohamad"
db.update_metadata(
    collection_name="student_db",
    student_id=student_payload["student_id"],
    new_metadata=student_payload
)

student_id, student_payload = db.get_student_data("student_db", "S002")
print(student_payload)
1 == 1