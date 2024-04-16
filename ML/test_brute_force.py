from genetic import initialize_individual, evaluate_fitness_individual
from pymongo import MongoClient

# connect to mongo db client
mongodb_client = MongoClient('mongodb://localhost:27017/')

# open or create database
db = mongodb_client['randomized_synchronous_dyads']

# open or create collection for current optimization batch
db_collection = db[f"dyads_080424"]

NUM_DYADS = 20000

for dyad_index in range(NUM_DYADS):

    dyad = initialize_individual(use_noise=False)
    fitness = evaluate_fitness_individual(individual=dyad, target_zlc=200, distance_metric='euclidian')
    if fitness > .01:
        print(f'# dyad {dyad_index} ... storing')
        # export data
        # make database record
        record = {
            'dyad': dyad
        }

        # write record to database
        db_collection.insert_one(record)

