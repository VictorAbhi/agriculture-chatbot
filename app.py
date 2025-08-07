from flask import Flask, render_template, jsonify, request
from src.helper import download_embeddings, load_pdf, filter_to_minimal_docs, text_split
from src.prompt import system_prompt
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os

app = Flask(__name__)



@app.route('/')
def index():
    return render_template('index.html')



if __name__ == '__main__':
    load_dotenv()
    app.run(debug=True, host='0.0.0.0', port=5000)