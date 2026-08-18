#!/bin/bash
pip install -r requirements.txt
pip install gdown
gdown https://drive.google.com/uc?id=1PFYRBHW41UJSNnz-fTkJsiPVDjOj4xQR -O chroma_db.zip
unzip -o chroma_db.zip -d .
rm chroma_db.zip