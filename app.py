from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
from deep_translator import GoogleTranslator
from langdetect import detect_langs
from src.config import KatzBotConfig
from src.data_load import DataManager
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.utils import secure_filename
from langchain_community.document_loaders.pdf import UnstructuredPDFLoader

import aiohttp
import asyncio
import os
import pickle
import pandas as pd
import requests

from flask_cors import CORS

load_dotenv()  # Load env variables from .env file

app = Flask(__name__, static_folder='build', static_url_path='/')
CORS(app, resources={r"/*": {"origins": "*"}})
# Serve the React app
@app.route('/')
def serve():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/manifest.json')
def manifest():
    return send_from_directory(app.static_folder, 'manifest.json')


config = KatzBotConfig()
manager = DataManager(config)

serpapi_key = os.getenv("SERPAPI_KEY")

# Set up the LLM model
model_choices = {
    "Mixtral-8x7b-32768": "Mixtral-8x7b-32768",
    "Gemma2-9b-It": "Gemma2-9b-It"
}
model_name = model_choices["Mixtral-8x7b-32768"]
llm = ChatGroq(groq_api_key=config.GROQ_API_KEY, model_name=model_name)

# Load embeddings and vector database
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load documents for FAISS vectors
with open(config.DATA_FILE, "rb") as f:
    docs = pickle.load(f)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
final_documents = text_splitter.split_documents(docs)

# Initialize FAISS vectors and retriever
try:
    vectors = FAISS.from_documents(final_documents, embeddings)
    print("Vectors db created")
except Exception as e:
    print(f"Error creating vector store: {e}")

prompt_template = ChatPromptTemplate.from_template(
    """
    You are Katzbot, the official chatbot for Yeshiva University. Your role is to assist users by providing accurate and helpful information about Yeshiva University. Please follow these guidelines when answering questions:

    1. Ensure your response is detailed and concise, relevant to the University, and at least two sentences long.
    2. Provide links to Yeshiva University webpages or resources only if you are certain they are correct. If you are not sure about the link, do not provide it.
    3. Maintain a polite, professional, and helpful tone.
    5. Focus on Yeshiva University-related topics such as admissions, academic programs, campus life, events, news, policies, and support services.
    6. When appropriate, include relevant news from the Yeshiva University news site (yu.edu/news) to provide the most up-to-date information. Ensure that all the hyperlinks are clickable.
    7. Whenever any links related to faculty is to be provided, give the link as https://www.yu.edu/katz/faculty.
    
    <context>
    {context}
    </context>
    Question: {input}
    """
)

document_chain = create_stuff_documents_chain(llm, prompt_template)
retriever = vectors.as_retriever()
# Create retrieval chain using the retriever and document chain
retrieval_chain = create_retrieval_chain(retriever, document_chain)

@app.route('/chat', methods=['POST'])
async def chat():
    try:
        data = request.json
        user_input = data.get("message")
        user_language = data.get("language", "en-US")
        
        # Detect language and translate if necessary
        translated_input, input_lang = detect_and_translate(user_input, user_language)
        
        # Asynchronous web search and ranking
        search_task = search_web(translated_input)
        chatbot_task = asyncio.to_thread(retrieval_chain.invoke, {'input': translated_input})

        # Run web search and chatbot inference in parallel
        search_results, chatbot_response = await asyncio.gather(search_task, chatbot_task)

        # Asynchronous ranking of results
        ranked_results = await rank_results(search_results, translated_input, embeddings)

        # Summarize the ranked results
        summary = summarize_with_llm(ranked_results)

        # Extract and format source information
        sources = format_sources(ranked_results)

        # Combine the LLM response with the summary from the web search
        enriched_response = f"{chatbot_response['answer']}\n\n**Additional Information from the Web:** {summary}**Sources:** {sources}"

        # Get response from the LLM chain
        # chatbot_response = retrieval_chain.invoke({'input': translated_input})['answer']
        # print(f"LLM Chain Response: {chatbot_response}")  # Print the entire response
        
        # Translate response back to original language (if necessary)
        if input_lang != 'en':
            print(f'Detected input lang: {input_lang}')
            if input_lang in {'zh-CN', 'zh-TW', 'ko', 'zh-cn', 'zh-tw'}:
                chatbot_response = GoogleTranslator(source='auto', target='zh-CN').translate(enriched_response)
            else:
                    chatbot_response = GoogleTranslator(source='auto', target='en').translate(enriched_response)
        elif user_language == 'zh-CN':
            chatbot_response = GoogleTranslator(
                source='auto',
                target='zh-CN').translate(enriched_response)

        # Create a conversation history record
        conversation_history = [
            {"sender": "user", "message": user_input, "language": user_language},
            {"sender": "bot", "message": enriched_response, "language": "en" if input_lang == "en" else input_lang}
        ]

        # Call save_history route
        save_chat_history(conversation_history)
    
        return jsonify({'response': enriched_response})
    except Exception as e:
        # Log the error message
        print(f"Error occurred: {e}")
        return jsonify({'error': 'Internal Server Error'}), 500

async def search_web(query):
    url = "https://serpapi.com/search"
    params = {
        "q": query + " Yeshiva University",
        "api_key": serpapi_key
    }
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as response:
            return await response.json()

async def rank_results(search_results, user_input, embeddings):
    # Extract the 'organic_results' list
    organic_results = search_results.get('organic_results', [])
    
    # Embed the user input for similarity calculation
    input_embedding = embeddings.embed_query(user_input)

    # Process and rank the organic results based on similarity to the user input
    result_embeddings = [embeddings.embed_query(res['snippet']) for res in organic_results]
    
    # Calculate cosine similarity
    similarities = [cosine_similarity([input_embedding], [emb])[0][0] for emb in result_embeddings]
    
    # Sort results by similarity
    ranked_results = sorted(zip(organic_results, similarities), key=lambda x: x[1], reverse=True)
    
    # Return the top 3 ranked results
    return [result[0] for result in ranked_results[:3]]

def summarize_with_llm(ranked_results):
    # Extract snippets from the ranked results
    snippets = [result['snippet'].replace("...", "") for result in ranked_results]
    
    # Combine snippets into a single text
    combined_text = " ".join(snippets)
    
    # Define a summarization prompt for the LLM
    summarization_prompt = f"""
    Summarize the following information concisely:
    {combined_text}
    """
    
    # Call the existing LLM for summarization
    summary = llm.invoke(summarization_prompt)
    
    return summary.content

def format_sources(ranked_results):
    # Format the top 3 sources
    sources = "\n".join(
        [f"[{result['title']}]({result['link']})" for result in ranked_results]
    )
    return sources

def detect_and_translate(user_input, user_language):
    detected_languages = detect_langs(user_input)
    probable_language = [lang.lang for lang in detected_languages if lang.lang in ['en', 'zh-cn', 'zh-tw', 'ko']]
    
    input_lang = probable_language[0] if probable_language else 'en'
    print(f'user language: {user_language}')
    print(f'input lang: {input_lang}')
    if input_lang in {'zh-CN', 'zh-TW', 'ko', 'zh-cn', 'zh-tw'}:
        translated_input = GoogleTranslator(source='auto', target='en').translate(user_input)
    else:
        translated_input = user_input  # Already in English
    
    if user_language in ['zh-CN', 'zh-TW', 'zh-cn', 'zh-tw']:
        translated_input = GoogleTranslator(source='en', target='zh-cn').translate(translated_input)

    return translated_input, input_lang

def save_chat_history(history):
    try:
        response = requests.post('http://127.0.0.1:5000/save_history', json={'history': history})
        print(f"Save history response: {response.json()}")
    except Exception as e:
        print(f"Error saving history: {e}")

@app.route('/save_history', methods=['POST'])
def save_history():
    data = request.json
    history_df = pd.DataFrame(data['history'])

    # Check if the file exists and is non-empty
    if os.path.exists(config.CHAT_HISTORY_FILE) and os.path.getsize(config.CHAT_HISTORY_FILE) > 0:
        try:
            existing_df = pd.read_csv(config.CHAT_HISTORY_FILE, encoding='utf-8')
        except pd.errors.EmptyDataError:
            # Handle the case where the file exists but is empty
            existing_df = pd.DataFrame()
    else:
        existing_df = pd.DataFrame()  # Initialize an empty DataFrame

    # Append to existing chat history and save
    updated_df = pd.concat([existing_df, history_df])
    updated_df.to_csv(config.CHAT_HISTORY_FILE, index=False, encoding='utf-8')

    return jsonify({"message": "History saved successfully"})

@app.route('/get_history', methods=['GET'])
def get_history():
    history_df = pd.read_csv(config.CHAT_HISTORY_FILE, encoding='utf-8')
    return history_df.to_dict(orient='records')

@app.route('/update_urls', methods=['POST'])
def update_urls():
    try:
        data = request.json
        additional_urls = data.get('urls', [])
        
        if not additional_urls:
            return jsonify({"error": "No URLs provided"}), 400

        # Call the update_existing_data method with the provided URLs
        manager.update_existing_data(additional_urls)
        return jsonify({"message": "Data updated successfully!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    try:
        if 'file' not in request.files:
            print("Log: No file part in the request")
            return jsonify({'error': 'No file part in the request'}), 400
        file = request.files['file']
        if file.filename == '':
            print("Log: No file selected for upload")
            return jsonify({'error': 'No selected file'}), 400

        # Save the uploaded file
        filename = secure_filename(file.filename)
        print(f"Log: File '{filename}' received for upload")
        upload_folder = os.path.join(os.getcwd(), 'data', 'pdfs')  # Path to 'data/pdfs' folder
        print(f"Log: Upload folder is '{upload_folder}'")

        file_path = os.path.join(upload_folder, filename)
        file.save(file_path)
        print(f"Log: File saved successfully at '{file_path}'")

        # Try loading the PDF content
        try:
            loader = UnstructuredPDFLoader(file_path)
            pdf_data = loader.load()
            print("Log: PDF loaded successfully")
        except Exception as e:
            print(f"Log: Error loading PDF - {e}")
            return jsonify({'error': f'Failed to load PDF: {str(e)}'}), 500
        
        # Load and update data from the pickle file
        try:
            with open(config.DATA_FILE, 'rb') as f:
                existing_data = pickle.load(f)
                print("Log: Pickle data loaded successfully")

            updated_data = existing_data + pdf_data
            with open(config.DATA_FILE, 'wb') as f:
                pickle.dump(updated_data, f)
                print("Log: Pickle data updated successfully")
        except Exception as e:
            print(f"Log: Error updating data file - {e}")
            return jsonify({'error': f'Failed to update data file: {str(e)}'}), 500

        return jsonify({'message': 'PDF uploaded and data updated successfully'}), 200

    except Exception as e:
        print(f"Log: Internal server error - {e}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500