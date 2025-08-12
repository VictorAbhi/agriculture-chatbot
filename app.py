from flask import Flask, render_template, request
import traceback
import os
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

try:
    from src.helper import download_embeddings
    from src.prompt import system_prompt as imported_system_prompt
except ImportError as e:
    print(f"Warning: Could not import from src: {e}")
    imported_system_prompt = None

try:
    from langchain_pinecone import PineconeVectorStore
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain_core.prompts import ChatPromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    print(f"Warning: LangChain imports failed: {e}")
    LANGCHAIN_AVAILABLE = False

app = Flask(__name__)

def download_embeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Safe embedding initialization"""
    try:
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        return embeddings
    except Exception as e:
        print(f"Error initializing embeddings: {e}")
        return None

def initialize_rag_system():
    """Initialize RAG system with proper error handling"""
    if not LANGCHAIN_AVAILABLE:
        return None, "LangChain dependencies not available"
    
    try:
        # Get API keys
        GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
        PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
        
        if not GOOGLE_API_KEY or not PINECONE_API_KEY:
            return None, "API keys not found in environment"
        
        # Initialize embeddings
        embeddings = download_embeddings()
        if embeddings is None:
            return None, "Failed to initialize embeddings"
        
        # Initialize vector store
        index_name = "agri-bot"
        try:
            vector_store = PineconeVectorStore.from_existing_index(
                embedding=embeddings,
                index_name=index_name
            )
        except Exception as e:
            return None, f"Failed to connect to Pinecone: {str(e)}"
        
        # Initialize the language model
        try:
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
        except Exception as e:
            return None, f"Failed to initialize Google GenAI: {str(e)}"
        
        # Create system prompt
        system_prompt_text = imported_system_prompt if imported_system_prompt else (
            "You are AgriBot, a helpful Agriculture assistant specializing in crop disease diagnosis and management. "
            "Provide accurate, practical advice based on the provided agricultural documents. "
            "Focus on symptoms, causes, prevention, and treatment of plant diseases. "
            "Keep responses concise and actionable for farmers."
            "\n\n"
            "{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt_text),
            ("human", "{input}"),
        ])
        
        # Create retriever and chains
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, answer_chain)
        
        return rag_chain, None
        
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"

# Initialize RAG system
rag_chain, init_error = initialize_rag_system()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get', methods=["POST"])
def get_bot_response():
    try:
        # Get user message
        userText = request.form['msg']
        
        if not userText or not userText.strip():
            return "Please provide a valid question about agriculture or crop diseases."
        
        print(f"User query: {userText}")
        
        # Check if RAG system is available
        if rag_chain is None:
            return "AgriBot is currently unavailable. I'm an AI assistant that helps with crop disease identification and agricultural advice. Please try again later or check with the system administrator."
        
        try:
            # Get response from RAG chain
            response = rag_chain.invoke({"input": userText.strip()})
            
            # Extract answer safely
            if isinstance(response, dict) and 'answer' in response:
                bot_response = response['answer']
            else:
                bot_response = str(response)
            
            # Ensure response is not empty
            if not bot_response or not bot_response.strip():
                bot_response = "I couldn't find specific information about that topic. Could you please rephrase your question or ask about a specific crop disease?"
            
            print(f"Bot response: {bot_response}")
            return bot_response
            
        except Exception as rag_error:
            print(f"RAG processing error: {rag_error}")
            return "I'm having trouble processing your question right now. Please try asking about specific crop diseases, symptoms, or management techniques."
    
    except Exception as e:
        print(f"Error in get_bot_response: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return "I encountered an error while processing your request. Please try again."

@app.route('/health')
def health_check():
    """Simple health check"""
    status = {
        'status': 'healthy' if rag_chain else 'degraded',
        'rag_available': rag_chain is not None
    }
    if init_error:
        status['error'] = init_error
    return status

if __name__ == '__main__':
    print("Starting AgriBot Application...")
    print(f"RAG System: {'✓ Ready' if rag_chain else '✗ Failed'}")
    if init_error:
        print(f"Error: {init_error}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)