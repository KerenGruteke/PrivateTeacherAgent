import pandas as pd
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# 1. Load your CSV
df = pd.read_csv("your_history_qa.csv")

# 2. Combine question + answer into a document field (optional)
df["document"] = df["question"] + " " + df["answer"]

# 3. Initialize encoder
encoder = SentenceTransformer("all-MiniLM-L6-v2")  # You can choose another model

# 4. Connect to Qdrant (e.g., cloud or local instance)
client = QdrantClient(host="localhost", port=6333)  # change if cloud

# 5. Create a collection
collection_name = "history_qa"
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

# 6. Create embeddings and upload them
points = []
for idx, row in df.iterrows():
    vector = encoder.encode(row["question"])  # or row["document"] if you want
    points.append(
        PointStruct(
            id=idx,
            vector=vector.tolist(),
            payload={"question": row["question"], "answer": row["answer"]},
        )
    )

client.upsert(collection_name=collection_name, points=points)
