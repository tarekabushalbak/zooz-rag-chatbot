#!/bin/bash
pip install -r requirements.txt
pip install gdown
gdown https://drive.google.com/uc?id=1-J5T5_ruwcx87PVx7A124uVLddL7SI4w -O chroma_db.zip
unzip -o chroma_db.zip -d .
rm chroma_db.zip