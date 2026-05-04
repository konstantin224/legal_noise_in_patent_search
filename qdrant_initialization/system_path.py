BATCH_SIZE = 128
COLLECTION = "patents_collection"
COLUMNS = ['source_file', 'record_type', 'ip_office', 'publication_number',
       'kind_code', 'publication_date', 'title', 'description', 'claims',
       'abstract', 'citations', 'citations_all', 'title_all',
       'description_all', 'claims_all', 'abstract_all', 'analog_number'] 

CACHE_FOLDER_PATH = "/scratch/shared/soma_storage/llm/models"
PATH_TO_ALL_FILES = r"/scratch/shared/soma_storage/фипс/Тест-октябрь"
PATH_TO_EMBED_MODEL = "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5"