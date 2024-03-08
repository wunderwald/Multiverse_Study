from dbaccess import get_db_entries

db_entries = get_db_entries()
print(f"Number of dyads in database: {len(db_entries)}")