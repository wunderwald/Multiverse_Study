from pymongo import MongoClient

# Connect to the local MongoDB instance
client = MongoClient('mongodb://localhost:27017/')

# Create or switch to a database
db = client['mydatabase']

# Create or switch to a collection
collection = db['mycollection']

# Insert a document into the collection
document = {"name": "John Doe", "email": "john.doe@example.com", "age": 30}
insert_result = collection.insert_one(document)
print(f"Document inserted with _id: {insert_result.inserted_id}")

# Insert multiple documents at once
documents = [
    {"name": "Jane Doe", "email": "jane.doe@example.com", "age": 25},
    {"name": "Alice", "email": "alice@example.com", "age": 28}
]
collection.insert_many(documents)

# Querying documents
query_result = collection.find({"age": {"$gt": 25}})  # Find documents where age is greater than 25

# Print the query results
for doc in query_result:
    print(doc)

# Close the connection
client.close()
