#!/bin/bash
pip install -r requirements.txt
pip install gdown
gdown https://drive.google.com/uc?id=1At3L_TCZE0IY6XHKl1FGSyU1Hzetn3RQ -O chroma_db.zip
unzip -o chroma_db.zip -d .
rm chroma_db.zip