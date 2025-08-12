import os
import traceback
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

class AgriBotSystem:
    def __init__(self):
        self.initialized = False
        self.error = None
        self.conversation_chain = None
        self.initialize()

    def initialize(self):
        """Initialize the RAG system with proper error handling"""
        try:
            from langchain.memory import ConversationBufferWindowMemory, ChatMessageHistory
            from langchain_pinecone import PineconeVectorStore
            from langchain_community.embeddings import HuggingFaceEmbeddings
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain.chains import ConversationalRetrievalChain
            from langchain_core.prompts import ChatPromptTemplate

            # Verify API keys
            GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
            PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
            
            if not GOOGLE_API_KEY or not PINECONE_API_KEY:
                raise ValueError("Missing required API keys (GOOGLE_API_KEY, PINECONE_API_KEY)")

            # Initialize embeddings
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )

            # Initialize vector store
            vector_store = PineconeVectorStore.from_existing_index(
                embedding=embeddings,
                index_name="agri-bot"
            )

            # Initialize LLM
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0.7,
                google_api_key=GOOGLE_API_KEY
            )

            # Configure memory
            message_history = ChatMessageHistory()
            memory = ConversationBufferWindowMemory(
                memory_key="chat_history",
                output_key="answer",
                chat_memory=message_history,
                return_messages=True,
                k=3
            )

            # Create prompt template
            system_prompt = (
                "You are AgriBot, an expert agricultural assistant specializing in crop disease diagnosis. "
                "Provide detailed, accurate information about symptoms, causes, prevention, and treatment. "
                "Always cite specific details from the provided context. If unsure, say you don't know.\n\n"
                "Context:\n{context}\n\n"
                "Question: {question}"
            )

            # Create conversation chain
            self.conversation_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
                memory=memory,
                verbose=True,
                chain_type="stuff",
                combine_docs_chain_kwargs={
                    "prompt": ChatPromptTemplate.from_template(system_prompt)
                }
            )

            self.initialized = True
            print("✓ AgriBot RAG system initialized successfully")

        except Exception as e:
            self.error = str(e)
            print(f"✗ Failed to initialize AgriBot: {self.error}")
            if self.conversation_chain is None:
                print("⚠️ Running in fallback mode without RAG capabilities")

    def get_response(self, query):
        """Get response from the RAG system with proper error handling"""
        if not query or not query.strip():
            return "Please provide a valid question about agriculture or crop diseases."

        try:
            if not self.initialized:
                return ("AgriBot is currently initializing. I can answer basic questions about crop diseases, "
                       "but advanced features are unavailable. Please try again later.")

            # Process the query
            response = self.conversation_chain.invoke({"question": query.strip()})
            
            if not response or 'answer' not in response:
                return "I couldn't generate a response. Please try asking about specific crop diseases or symptoms."
            
            return response['answer']

        except Exception as e:
            print(f"Error processing query: {str(e)}")
            return ("I encountered an error while processing your question. "
                   "Please try asking about specific crop diseases like tomato blight or wheat rust.")

# Initialize the AgriBot system
agri_bot = AgriBotSystem()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=["POST"])
def get_bot_response():
    """API endpoint for chat messages - accepts both JSON and form data"""
    try:
        # Handle both JSON and form data
        if request.is_json:
            data = request.get_json()
            user_query = data.get('message', '').strip()
        else:
            user_query = request.form.get('msg', '').strip()
        
        if not user_query:
            return jsonify({"error": "Please enter a question"}), 400
        
        response = agri_bot.get_response(user_query)
        return jsonify({"answer": response})

    except Exception as e:
        print(f"API Error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            "error": "An internal error occurred",
            "answer": "I'm having technical difficulties. Please try again later."
        }), 500

@app.route('/health')
def health_check():
    """Comprehensive health check endpoint"""
    status = {
        'status': 'healthy' if agri_bot.initialized else 'degraded',
        'rag_available': agri_bot.initialized,
        'initialization_error': agri_bot.error,
        'environment_loaded': bool(os.getenv('GOOGLE_API_KEY') and os.getenv('PINECONE_API_KEY'))
    }
    return jsonify(status)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)