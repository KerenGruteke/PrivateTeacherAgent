query = "Who won the American Civil War?"
query_vector = encoder.encode(query)

search_result = client.search(
    collection_name=collection_name,
    query_vector=query_vector,
    limit=3,
)

for result in search_result:
    print("Score:", result.score)
    print("Q:", result.payload["question"])
    print("A:", result.payload["answer"])
    print()
