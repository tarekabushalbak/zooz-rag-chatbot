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
unzip -o chroma_db.zip -d .
rm chroma_db.zip
