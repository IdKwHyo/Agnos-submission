import os
import requests
from bs4 import BeautifulSoup
import datetime
import shutil
import pandas as pd
import chromadb
from langchain_huggingface import HuggingFaceEndpoint
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import logging
import threading
import time
import json
import traceback
from googlesearch import search  # For fallback web search

# Suppress tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize Flask app
app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set environment variables for HuggingFaceHub
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "#"  # Replace with your valid token

# Global variables
PERSIST_DIRECTORY = "db"
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
llm_model_name = "google/flan-t5-base"  # A smaller model for demonstration
scraping_url = "https://www.agnoshealth.com/forums"
CHAT_HISTORY_DIR = "chat_histories"
GOOGLE_API_KEY = ""  # Replace with your actual key
GOOGLE_CX = ""  # Replace with your Custom Search Engine ID

# Create necessary directories with proper permissions
os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)

# Ensure directories are writable
try:
    os.chmod(PERSIST_DIRECTORY, 0o777)
    os.chmod(CHAT_HISTORY_DIR, 0o777)
except Exception as e:
    logger.warning(f"Could not change permissions: {str(e)}")

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

def get_vector_db():
    try:
        os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
        os.chmod(PERSIST_DIRECTORY, 0o777)

        if not os.path.exists("forum_data.csv"):
            create_sample_data()

        if os.path.exists(PERSIST_DIRECTORY) and any(os.listdir(PERSIST_DIRECTORY)):
            logger.info("Loading existing vector database")
            try:
                return Chroma(persist_directory=PERSIST_DIRECTORY, 
                            embedding_function=embeddings)
            except Exception as e:
                logger.error(f"Failed to load existing database: {str(e)}")
                raise

        logger.info("Creating new vector database")
        documents = load_forum_documents()
        if not documents:
            logger.warning("No valid documents found, creating sample data.")
            create_sample_data()
            documents = load_forum_documents()  # Load again after creating sample data
            
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)
        
        if not texts:
            logger.error("No texts available for embedding.")
            raise Exception("No valid texts found for embedding.")
        
        db = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory=PERSIST_DIRECTORY
        )
        return db

    except Exception as e:
        logger.error(f"Error creating vector DB: {str(e)}")
        logger.info("Using in-memory vector database as fallback")
        documents = load_forum_documents()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)
        return Chroma.from_documents(documents=texts, embedding=embeddings)



def scrape_forum_data():
    logger.info(f"Starting web scraping from {scraping_url}")
    
    try:
        headers = {
            'User -Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(scraping_url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        forum_topics = []
        topics = soup.find_all('div', class_='forum-post') or soup.find_all('article')
        
        logger.info(f"Found {len(topics)} topics on the page.")
        
        for topic in topics:
            try:
                title = topic.find('h3', class_='post-title').text.strip() if topic.find('h3', class_='post-title') else "No title"
                content = topic.find('div', class_='post-body').text.strip() if topic.find('div', class_='post-body') else "No content available"
                author = topic.find('span', class_='post-author').text.strip() if topic.find('span', class_='post-author') else "Anonymous"
                date_elem = topic.find('time', class_='post-date')
                date = date_elem['datetime'] if date_elem and 'datetime' in date_elem.attrs else datetime.datetime.now().isoformat()
                
                forum_topics.append({
                    'title': title,
                    'content': content,
                    'author': author,
                    'date': date,
                    'url': scraping_url
                })
            except Exception as e:
                logger.warning(f"Error parsing a topic: {str(e)}")
                continue
        
        if forum_topics:
            df = pd.DataFrame(forum_topics)
            df.to_csv("forum_data.csv", index=False)
            logger.info(f"Scraped {len(forum_topics)} forum posts")
        else:
            logger.warning("No forum topics were scraped")
            create_sample_data()
            
    except Exception as e:
        logger.error(f"Error during web scraping: {str(e)}")
        create_sample_data()



def create_sample_data():
    logger.info("Creating sample forum data")
    sample_topics = [
        {
            'title': 'Nutrition Tips for Better Health',
            'content': 'Discussing the benefits of a balanced diet with fruits, vegetables, and lean proteins.',
            'author': 'HealthGuru',
            'date': datetime.datetime.now().isoformat(),
            'url': scraping_url
        },
        {
            'title': 'Mental Health Awareness',
            'content': 'Sharing strategies for managing stress and improving mental well-being.',
            'author': 'MindfulUser',
            'date': datetime.datetime.now().isoformat(),
            'url': scraping_url
        },
        {
            'title': 'Exercise Routines for Beginners',
            'content': 'Starting an exercise routine doesn\'t have to be complicated. Begin with 10-15 minutes of activity and increase over time. Choose activities you enjoy like walking, swimming, or cycling for 20-30 minutes, 3-5 times weekly. For strength, try bodyweight exercises like squats and push-ups 2-3 times weekly. Add daily stretching for flexibility. Remember to have rest days between workouts.',
            'author': 'FitCoach',
            'date': datetime.datetime.now().isoformat(),
            'url': scraping_url
        },
        {
            'title': 'Cardio vs. Strength Training',
            'content': 'Both cardio and strength training are important for overall fitness. Cardio improves heart health and burns calories, while strength training builds muscle and boosts metabolism. For beginners, start with 2-3 days of each per week, allowing for rest days between strength sessions.',
            'author': 'FitnessExpert',
            'date': datetime.datetime.now().isoformat(),
            'url': scraping_url
        },
        {
            'title': 'How to Start Exercising Safely',
            'content': 'Before starting any exercise program, consult with your healthcare provider if you have existing health conditions. Start slowly with low-impact activities like walking, swimming, or gentle cycling. Warm up before exercise and cool down afterward. Stay hydrated and listen to your body - pain is different from normal muscle fatigue.',
            'author': 'SafetyFirst',
            'date': datetime.datetime.now().isoformat(),
            'url': scraping_url
        }
    ]
    
    df = pd.DataFrame(sample_topics)
    df.to_csv("forum_data.csv", index=False)
    logger.info(f"Created {len(sample_topics)} sample forum posts")

# Function to convert scraped data to documents
def load_forum_documents():
    if not os.path.exists("forum_data.csv"):
        create_sample_data()
    
    documents = []
    try:
        df = pd.read_csv("forum_data.csv")
        # Filter out empty rows
        df = df[df['content'].notna() & (df['content'] != "No content available")]
        
        for _, row in df.iterrows():
            if not pd.isna(row['content']) and str(row['content']).strip():
                metadata = {
                    'title': row.get('title', 'No title'),
                    'author': row.get('author', 'Unknown'),
                    'date': row.get('date', datetime.datetime.now().isoformat()),
                    'url': row.get('url', '')
                }
                documents.append(Document(
                    page_content=f"Title: {metadata['title']}\nContent: {row['content']}",
                    metadata=metadata
                ))
        
        logger.info(f"Loaded {len(documents)} documents from CSV")
        return documents
    except Exception as e:
        logger.error(f"Error loading forum documents: {str(e)}")
        return []

# Initialize the conversational chain
def get_conversational_chain():
    try:
        vectordb = get_vector_db()
        
        # Validate collection exists
        if not hasattr(vectordb, '_collection') or not vectordb._collection.count():
            logger.error("Empty or invalid vector database")
            raise Exception("Invalid vector database")
            
        llm = HuggingFaceEndpoint(
            repo_id=llm_model_name,
            huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
            temperature=0.5,
            max_new_tokens=512
        )
        
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
            memory=memory,
            verbose=True
        )
        
        return chain
    except Exception as e:
        logger.error(f"Chain creation failed: {str(e)}")
        return FallbackResponseGenerator()

def verify_exercise_content():
    try:
        if os.path.exists("forum_data.csv"):
            df = pd.read_csv("forum_data.csv")
            
            exercise_keywords = ["exercise", "workout", "fitness", "training", "gym"]
            has_exercise_content = False
            
            for _, row in df.iterrows():
                content = str(row.get('content', '')).lower()
                title = str(row.get('title', '')).lower()
                
                if any(keyword in content or keyword in title for keyword in exercise_keywords):
                    has_exercise_content = True
                    break
            
            if not has_exercise_content:
                logger.info("Adding exercise content")
                exercise_data = [
                    {
                        'title': 'Best Beginner Exercise Routines',
                        'content': 'For beginners, start with 20-30 minutes of cardio 3 times per week. Include bodyweight exercises like squats, lunges, push-ups, and planks twice weekly.',
                        'author': 'FitTrainer',
                        'date': datetime.datetime.now().isoformat(),
                        'url': scraping_url
                    },
                    {
                        'title': 'Home Workout Tips',
                        'content': 'You can get a full workout at home with just a few resistance bands or dumbbells. Focus on compound movements that work multiple muscle groups at once.',
                        'author': 'HomeGymPro',
                        'date': datetime.datetime.now().isoformat(),
                        'url': scraping_url
                    }
                ]
                
                new_df = pd.DataFrame(exercise_data)
                df = pd.concat([df, new_df], ignore_index=True)
                df.to_csv("forum_data.csv", index=False)
    except Exception as e:
        logger.error(f"Error in verify_exercise_content: {str(e)}")

class FallbackResponseGenerator:
    def __call__(self, inputs):
        query = inputs.get("question", "").lower()
        
        exercise_keywords = ["exercise", "workout", "fitness", "training", "gym", "cardio", "strength"]
        if any(keyword in query for keyword in exercise_keywords):
            return {
                "answer": "To start exercising, begin with activities you enjoy like walking, swimming, or cycling for 20-30 minutes, 3-5 times weekly. Add bodyweight exercises like squats and push-ups 2-3 times weekly for strength. Always warm up before and cool down after exercise. Start slowly and gradually increase intensity as your fitness improves.",
                "source_documents": []
            }
            
        return {
            "answer": "I'm sorry, I'm having trouble connecting to my knowledge base. Let me search online for that information...",
            "source_documents": []
        }
        
    def invoke(self, inputs):
        return self.__call__(inputs)

# Enhanced web search function
# Enhanced web search function with better error handling
def perform_web_search(query, num_results=3):
    try:
        logger.info(f"Performing web search for: {query}")
        # First try Google Custom Search JSON API
        if GOOGLE_API_KEY and GOOGLE_CX:
            search_url = f"https://www.googleapis.com/customsearch/v1?key={GOOGLE_API_KEY}&cx={GOOGLE_CX}&q={query}&num={num_results}"
            response = requests.get(search_url, timeout=10)
            if response.status_code == 200:
                results = response.json().get('items', [])
                if results:
                    logger.info(f"Google API search found {len(results)} results")
                    return [{
                        'title': item.get('title'),
                        'link': item.get('link'),
                        'snippet': item.get('snippet')
                    } for item in results]
                else:
                    logger.warning("Google API returned no results")
        else:
            logger.warning("Google API keys not configured, falling back to googlesearch module")
        
        # Fallback to python-googlesearch
        fallback_results = list(search(query, num_results=num_results, stop=num_results, pause=2.0))
        logger.info(f"Fallback search found {len(fallback_results)} results")
        return fallback_results
    except Exception as e:
        logger.error(f"Web search failed: {str(e)}")
        return []

# Improved function to scrape health information from a webpage
def scrape_health_info(url):
    try:
        logger.info(f"Scraping content from {url}")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Try to get main content - these selectors work for many health sites
        content = ""
        
        # Try multiple common content containers
        for selector in ['article', 'main', '.article-content', '.post-content', '.entry-content', '#content', '.content']:
            container = soup.select_one(selector)
            if container:
                paragraphs = container.find_all('p')
                content = "\n".join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
                if content:
                    logger.info(f"Found content using selector: {selector}")
                    break
        
        if not content:
            # Fallback to body text
            body = soup.find('body')
            if body:
                paragraphs = body.find_all('p')
                content = "\n".join([p.get_text().strip() for p in paragraphs if p.get_text().strip()][:15])  # Limit to first 15 paragraphs
                logger.info("Using fallback body content")
        
        # Clean up the content
        content = ' '.join(content.split())  # Remove excessive whitespace
        content = content.replace('\n\n', '\n').replace('\n\n', '\n')
        
        return content[:2000]  # Limit to 2000 chars
    except Exception as e:
        logger.error(f"Error scraping {url}: {str(e)}")
        return ""

# Initialize background scraper
def start_background_scraper(interval_hours=24):
    def scrape_periodically():
        while True:
            try:
                logger.info("Starting periodic scrape...")
                scrape_forum_data()
                time.sleep(5)
                
                if os.path.exists(PERSIST_DIRECTORY):
                    try:
                        shutil.rmtree(PERSIST_DIRECTORY)
                        time.sleep(2)
                        os.makedirs(PERSIST_DIRECTORY, mode=0o777, exist_ok=True)
                    except Exception as e:
                        logger.error(f"Error removing old DB: {str(e)}")
                
                time.sleep(2)
                get_vector_db()
                
            except Exception as e:
                logger.error(f"Error in background scraper: {str(e)}")
            finally:
                time.sleep(interval_hours * 3600)
    
    thread = threading.Thread(target=scrape_periodically, daemon=True)
    thread.start()

# Flask routes
@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/api/save_chat', methods=['POST'])
def save_chat():
    try:
        data = request.json
        chat_history = data.get('chat_history', [])
        user_id = data.get('user_id', 'anonymous')
        
        if not chat_history:
            return jsonify({"error": "No chat history provided"}), 400
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{user_id}_{timestamp}.json"
        filepath = os.path.join(CHAT_HISTORY_DIR, filename)
        
        with open(filepath, 'w') as f:
            json.dump(chat_history, f)
        
        return jsonify({"message": "Chat history saved successfully", "filename": filename})
    except Exception as e:
        logger.error(f"Error saving chat history: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/load_chats', methods=['GET'])
def load_chats():
    try:
        user_id = request.args.get('user_id', 'anonymous')
        
        user_chats = []
        for filename in os.listdir(CHAT_HISTORY_DIR):
            if filename.startswith(user_id):
                filepath = os.path.join(CHAT_HISTORY_DIR, filename)
                with open(filepath, 'r') as f:
                    chat_data = json.load(f)
                user_chats.append({
                    "filename": filename,
                    "timestamp": filename.split('_')[1].split('.')[0],
                    "preview": chat_data[0] if chat_data else "Empty chat"
                })
        
        user_chats.sort(key=lambda x: x["timestamp"], reverse=True)
        return jsonify({"chats": user_chats})
    except Exception as e:
        logger.error(f"Error loading chat histories: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/load_chat', methods=['GET'])
def load_chat():
    try:
        filename = request.args.get('filename')
        if not filename:
            return jsonify({"error": "Filename not provided"}), 400
        
        filepath = os.path.join(CHAT_HISTORY_DIR, filename)
        if not os.path.exists(filepath):
            return jsonify({"error": "Chat not found"}), 404
        
        with open(filepath, 'r') as f:
            chat_history = json.load(f)
        
        return jsonify({"chat_history": chat_history})
    except Exception as e:
        logger.error(f"Error loading chat: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Common greetings and health keywords
GREETINGS = {"hi", "hello", "hey", "yo", "hola"}
HEALTH_KEYWORDS = {
    "health", "medical", "doctor", "symptom", "diagnosis", "treatment", 
    "medicine", "illness", "disease", "condition", "pain", "fever",
    "headache", "cough", "covid", "vaccine", "prescription"
}

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        query = data.get('query', '').strip().lower()
        chat_history = data.get('chat_history', [])

        # Handle greetings
        if query in GREETINGS:
            return jsonify({
                "answer": "Hello! I'm your health assistant. How can I help you today?",
                "source_documents": []
            })

        # Handle ChatGPT questions
        if "chat gpt" in query or "chatbot" in query:
            return jsonify({
                "answer": "ChatGPT is an AI language model that can answer questions and have conversations. I'm a similar AI assistant specialized in health and fitness topics.",
                "source_documents": []
            })

        # Handle exercise queries
        if "exercise" in query or "workout" in query or "push" in query or "fitness" in query:
            verify_exercise_content()

        # First try to get answer from knowledge base
        kb_response = None
        try:
            chain = get_conversational_chain()
            formatted_history = []
            for i in range(0, len(chat_history), 2):
                if i + 1 < len(chat_history):
                    formatted_history.append((chat_history[i], chat_history[i+1]))
            
            kb_response = chain.invoke({"question": query, "chat_history": formatted_history})
            
            # If we got a good answer, return it
            if (kb_response and 
                "answer" in kb_response and 
                kb_response["answer"] and 
                len(kb_response["answer"]) > 20 and
                kb_response["answer"].lower() not in ["i don't know", "i'm sorry, i don't know"]):
                
                return jsonify({
                    "answer": kb_response["answer"],
                    "source_documents": [doc.metadata for doc in kb_response.get("source_documents", [])]
                })
                
        except Exception as e:
            logger.error(f"Error in chain processing: {str(e)}")
            # Continue to fallback methods

        # If we're here, the knowledge base didn't have an answer
        # Define expanded keywords for different topics
        health_keywords = {
            "health", "medical", "doctor", "symptom", "diagnosis", "treatment", 
            "medicine", "illness", "disease", "condition", "pain", "fever",
            "headache", "cough", "covid", "vaccine", "prescription"
        }
        
        exercise_keywords = {
            "exercise", "workout", "fitness", "training", "gym", "cardio", "strength",
            "push", "pull", "squat", "run", "running", "jog", "walking", "lift", "muscle"
        }
        
        # Check if question relates to our supported topics
        is_health_question = any(keyword in query for keyword in health_keywords)
        is_exercise_question = any(keyword in query for keyword in exercise_keywords)
        
        # For specific exercise questions where we know we should have answers
        if "push up" in query or "push-up" in query:
            return jsonify({
                "answer": "To perform a basic push-up: Start in a plank position with hands shoulder-width apart, arms straight, and body forming a straight line from head to heels. Bend your elbows to lower your chest toward the floor while keeping your body straight. Push back up to the starting position. For beginners, you can do modified push-ups with knees on the floor. Focus on proper form rather than quantity.",
                "source_documents": []
            })
        
        # Try web search for relevant questions
        if is_health_question or is_exercise_question:
            try:
                # Perform web search
                search_results = perform_web_search(query)
                
                if search_results:
                    # Try to scrape the first result for more detailed info
                    if isinstance(search_results[0], dict):
                        first_result = search_results[0]
                        url = first_result.get('link', '')
                    else:  # Handle string URLs from googlesearch fallback
                        first_result = {"title": "Web resource", "link": search_results[0]}
                        url = search_results[0]
                    
                    detailed_content = scrape_health_info(url)
                    
                    if detailed_content:
                        answer = f"I found this information from {first_result.get('title', 'a reputable source')}:\n\n{detailed_content[:1000]}\n\nRead more: {url}"
                    else:
                        answer = "Here are some resources I found:\n"
                        for i, res in enumerate(search_results[:3]):
                            if isinstance(res, dict):
                                answer += f"- {res.get('title', 'Resource ' + str(i+1))}: {res.get('link', 'No URL')}\n" 
                            else:
                                answer += f"- Resource {i+1}: {res}\n"
                    
                    return jsonify({
                        "answer": answer,
                        "source_documents": []
                    })
            except Exception as e:
                logger.error(f"Web search failed: {str(e)}")

        # General exercise fallback
        if is_exercise_question:
            return jsonify({
                "answer": "For beginners starting an exercise routine, focus on activities you enjoy like walking, swimming, or cycling for 20-30 minutes, 3-5 times weekly. Add basic strength exercises like squats, lunges, push-ups, and planks 2-3 times weekly. Always warm up before and cool down after exercising. Start slowly and gradually increase intensity as your fitness improves. Remember to include rest days between strength workouts.",
                "source_documents": []
            })

        # Final fallback response
        return jsonify({
            "answer": "I couldn't find a specific answer to your question. You might want to consult a healthcare professional for medical advice or try rephrasing your question.",
            "source_documents": []
        })

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            "answer": "I'm sorry, I encountered a technical issue. Our team has been notified and is working to resolve it. Please try again later.",
            "source_documents": []
        }), 200

@app.route('/health')
def health():
    try:
        db = get_vector_db()
        return jsonify({
            "status": "healthy",
            "db_connected": True,
            "documents": db._collection.count() if hasattr(db, '_collection') else 0
        })
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

@app.route('/debug/db')
def debug_db():
    try:
        db = get_vector_db()
        count = db._collection.count() if hasattr(db, '_collection') else 0
        sample = db.similarity_search("exercise", k=1) if count > 0 else []
        return jsonify({
            "status": "ok",
            "document_count": count,
            "sample": sample[0].page_content if sample else None,
            "sample_metadata": sample[0].metadata if sample else None
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/api/scrape', methods=['POST'])
def trigger_scrape():
    try:
        thread = threading.Thread(target=scrape_forum_data)
        thread.start()
        return jsonify({"message": "Scraping started in background"})
    except Exception as e:
        logger.error(f"Error triggering scrape: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.errorhandler(Exception)
def handle_error(e):
    logger.error(f"Error: {str(e)}\n{traceback.format_exc()}")
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    try:
        logger.info("Starting initial setup...")
        try:
            scrape_forum_data()
        except Exception as e:
            logger.error(f"Initial scraping failed: {str(e)}")
            create_sample_data()
            
        get_vector_db()
        start_background_scraper()
        app.run(debug=True, port=5000, threaded=True)
    except Exception as e:
        logger.error(f"Application failed to start: {str(e)}")
        raise
