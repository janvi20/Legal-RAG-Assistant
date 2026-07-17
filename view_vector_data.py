import chromadb

VECTOR_DB_PATH = "db/chroma_db"
COLLECTION_NAME = "caselaw_rag"

# Connect to ChromaDB
client = chromadb.PersistentClient(
    path=VECTOR_DB_PATH
)

collection = client.get_collection(
    COLLECTION_NAME
)


def preview_vector_db(limit=5):

    print("\n========== VECTOR DB INFO ==========")

    print("Collection:", collection.name)
    print("Total Chunks:", collection.count())


    print("\n========== SAMPLE CHUNKS ==========")


    data = collection.get(
        limit=limit,
        include=[
            "documents",
            "metadatas"
        ]
    )


    for i in range(len(data["documents"])):

        print("\n------------------------------")

        print("ID:")
        print(data["ids"][i])

        print("\nMetadata:")
        print(data["metadatas"][i])

        print("\nText Preview:")
        print(data["documents"][i][:500])


if __name__ == "__main__":

    preview_vector_db(limit=5)