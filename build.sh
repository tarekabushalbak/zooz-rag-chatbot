#!/bin/bash
pip install -r requirements.txt
pip install gdown
gdown https://drive.google.com/uc?id=1emBSTIYR_FGeHFTCaXOROWjmVUN4NfYb -O chroma_db.zip
unzip -o chroma_db.zip -d .
rm chroma_db.zip