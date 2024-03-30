# AuroraFlow
 A tool for RAG pipeline and easy processing of files. 
 The idea behind this is that one can feed in files in form of pdf such as scientific papers, books etc.
and the system gives you the most relevant answers it found in all these papers along with the page number and file it found it in. 

## Installation

1. Make a virtual environment:
````commandline
python -m venv .venv
````
2. Install requirements:
````commandline
pip install -r requirements.txt
````
3. Now run
````commandline
python -m apps.rag_app
````
The demo below shows how aurora flow helps one to extract relevant information for there search from the papers


![Flow Demo](auroraflowdemo.gif)

