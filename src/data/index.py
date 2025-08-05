import pandas as pd
# from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client import QdrantClient
from src.utils.constants import QDRANT_CLUSTER_URL, QDRANT_API_KEY

# 1. Load your CSV
# df = pd.read_csv("your_history_qa.csv")

# # 2. Combine question + answer into a document field (optional)
# df["document"] = df["question"] + " " + df["answer"]

# # 3. Initialize encoder
# encoder = SentenceTransformer("all-MiniLM-L6-v2")  # You can choose another model

# 4. Connect to Qdrant (e.g., cloud or local instance)
qdrant_client = QdrantClient(
    url=QDRANT_CLUSTER_URL, 
    api_key=QDRANT_API_KEY,
)

students = [
    {
        "student_id": "S001",
        "name": "Alice",
        "status": {
            "math": {"score": 80, "note": "needs more algebra"},
            "history": {"score": 90, "note": "good"}
        }
    },
    {
        "student_id": "S002",
        "name": "Bob",
        "status": {
            "math": {"score": 65, "note": "practice fractions"},
            "history": {"score": 70, "note": "needs to review WWI"}
        }
    },
]

# Create student collection (dummy 1-dim vector)
qdrant_client.create_collection(
    collection_name="student_db",
    vectors_config=VectorParams(size=1, distance=Distance.COSINE),
)

# Insert with dummy vector
student_points = []
for i, student in enumerate(students):
    student_points.append(
        PointStruct(
            id=i,
            vector=[0.0],  # dummy vector
            payload=student
        )
    )

qdrant_client.upsert(collection_name="student_db", points=student_points)

print(qdrant_client.get_collections())


# # 5. Create a collection
# collection_name = "history_qa"
# qdrant_client.recreate_collection(
#     collection_name=collection_name,
#     vectors_config=VectorParams(size=384, distance=Distance.COSINE),
# )

# # 6. Create embeddings and upload them
# points = []
# for idx, row in df.iterrows():
#     vector = encoder.encode(row["question"])  # or row["document"] if you want
#     points.append(
#         PointStruct(
#             id=idx,
#             vector=vector.tolist(),
#             payload={"question": row["question"], "answer": row["answer"]},
#         )
#     )

# qdrant_client.upsert(collection_name=collection_name, points=points)
