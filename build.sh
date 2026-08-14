#!/bin/bash
pip install -r requirements.txt
pip install gdown
gdown https://drive.google.com/uc?id=13WJL9eSwx1wOpHK5A1xNMZv-4poIXjZo -O chroma_db.zip
unzip -o chroma_db.zip -d .
rm chroma_db.zip