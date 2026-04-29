## running for index build  
./.venv/Scripts/python.exe -m utils.create_lab_vectorindex.ingest_pdfs --document_store ./utils/create_lab_vectorindex/document_store.json
./.venv/Scripts/python.exe -m utils.create_lab_vectorindex.ingest_pdfs --name PSA_SSA --document_store ./utils/create_lab_vectorindex/document_store.json 

## remove documents not listed:
./.venv/Scripts/python.exe -m utils.create_lab_vectorindex.prune_index --name DigiUng_lab --dry-run #fires
./.venv/Scripts/python.exe -m utils.create_lab_vectorindex.prune_index --name DigiUng_lab