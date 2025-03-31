# test_palm_embeddings.py
from langchain_community.embeddings import GooglePalmEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

embeddings = GooglePalmEmbeddings()
text = "This is a test sentence."
try:
    embedded = embeddings.embed_query(text)
    print("Embedding successful! First 5 values:", embedded[:5])
except Exception as e:
    print("Error:", str(e))