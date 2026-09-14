#!/bin/bash
set -e

pip install -r requirements.txt
pip install gdown

# Download the SentenceTransformer model during the build so the running web
# service does not need a slow first-request download from Hugging Face.
python - <<'PY'
from sentence_transformers import SentenceTransformer
SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
print("Embedding model cached successfully")
PY

gdown https://drive.google.com/uc?id=1Xu0NXMCmCZJVUvQVeiHk0qtJJIM1RD2h -O chroma_db.zip

# Windows Compress-Archive stores path separators as backslashes. On Linux,
# unzip extracts these entries correctly but may return status 1 as a warning.
# Treat warning status 1 as non-fatal, but still fail on real extraction errors.
set +e
unzip -o chroma_db.zip -d .
UNZIP_STATUS=$?
set -e
if [ "$UNZIP_STATUS" -gt 1 ]; then
  echo "ChromaDB archive extraction failed with status $UNZIP_STATUS"
  exit "$UNZIP_STATUS"
fi

# Verify that the database was actually extracted before continuing.
test -f chroma_db/chroma.sqlite3
rm chroma_db.zip

echo "ChromaDB archive extracted successfully"
