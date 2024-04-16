from pymongo import MongoClient

def get_db_entries(database_id='genetic_rsa'):
    # Connect to the local MongoDB instance and open database
    client = MongoClient('mongodb://localhost:27017/')
    db = client[database_id]

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
            for d in document['fittest_individuals']:
                results.append({
                    'hyperparameters': hyperparameters,
                    'dyad_parameters': d['individual']
                })


    # Close the MongoDB connection
    client.close()

    return results

def get_db_entries_random_dyads(database_id='randomized_synchronous_dyads'):
    # Connect to the local MongoDB instance and open database
    client = MongoClient('mongodb://localhost:27017/')
    db = client[database_id]

    # initialize results
    results = []

    # Iterate over each collection in the database
    for collection_name in db.list_collection_names():

        # Retrieve all documents in the current collection
        collection = db[collection_name]
        documents = collection.find()
        
        # make result entries for each individual in the document
        for d in documents:
            results.append({
                'dyad_parameters': d['dyad']
            })


    # Close the MongoDB connection
    client.close()

    return results
