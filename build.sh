#!/bin/bash
pip install -r requirements.txt
pip install gdown
gdown https://drive.google.com/uc?id=1U1aQ9mlYxE0U1f6-fWcD7JUncXEwozn4 -O chroma_db.zip
unzip -o chroma_db.zip -d .
rm chroma_db.zip