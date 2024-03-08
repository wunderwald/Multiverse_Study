from pymongo import MongoClient

def get_db_entries():
    # Connect to the local MongoDB instance and open database
    client = MongoClient('mongodb://localhost:27017/')
    db = client['genetic_rsa']

    # initialize results
    results = []

    # Iterate over each collection in the database
    for collection_name in db.list_collection_names():
        if "fittest_individuals" not in collection_name:
            continue

        # Retrieve all documents in the current collection
        collection = db[collection_name]
        documents = collection.find()
        
        # make result entries for each individual in the document
        for document in documents:
            hyperparameters = document['hyperparameters']
            for individual in document['fittest_individuals']:
                results.append({
                    'hyperparameters': hyperparameters,
                    'dyad_parameters': individual
                })


    # Close the MongoDB connection
    client.close()

    return results
