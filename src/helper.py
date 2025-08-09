import os
import logging
from typing import Optional, List
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from google.api_core import exceptions as google_exceptions
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class GoogleAPIKeyManager:
    """Enhanced API key manager with quota awareness and retry logic"""
    def __init__(self):
        self.keys = [
            os.getenv("GOOGLE_API_KEY_1"),
            os.getenv("GOOGLE_API_KEY_2"),
            os.getenv("GOOGLE_API_KEY_3")
        ]
        self.current_key_index = 0
        self.failed_keys = set()
        self.quota_exceeded_keys = set()
        self.last_rotation_time = 0
        self.rotation_cooldown = 60  # seconds
    
    def get_current_key(self) -> Optional[str]:
        """Get current active API key"""
        if not self.keys:
            return None
        return self.keys[self.current_key_index]
    
    def rotate_key(self) -> bool:
        """Rotate to next available API key with cooldown"""
        now = time.time()
        if now - self.last_rotation_time < self.rotation_cooldown:
            logger.warning(f"Key rotation cooldown active. Waiting {self.rotation_cooldown} seconds")
            return False
            
        original_index = self.current_key_index
        while True:
            self.current_key_index = (self.current_key_index + 1) % len(self.keys)
            if self.current_key_index == original_index:
                logger.error("All API keys exhausted")
                return False
            if (self.keys[self.current_key_index] not in self.failed_keys and 
                self.keys[self.current_key_index] not in self.quota_exceeded_keys):
                logger.info(f"Rotated to key index {self.current_key_index}")
                self.last_rotation_time = time.time()
                return True
    
    def mark_failed(self, key: str, is_quota_error: bool = False):
        """Mark a key as failed or quota exceeded"""
        if is_quota_error:
            self.quota_exceeded_keys.add(key)
            logger.warning(f"Marked key as quota exceeded: {key[-6:]}")
        else:
            self.failed_keys.add(key)
            logger.warning(f"Marked key as failed: {key[-6]}")
        
        if key == self.get_current_key():
            self.rotate_key()

# Initialize key manager
key_manager = GoogleAPIKeyManager()

def get_pdf_text(pdf_docs) -> str:
    """Extract text from uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            if page.extract_text():
                text += page.extract_text()
    return text

def get_text_chunks(text: str) -> List[str]:
    """Split the text into smaller chunks for processing."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return text_splitter.split_text(text)

def get_embeddings():
    """Get embeddings instance with current API key"""
    current_key = key_manager.get_current_key()
    if not current_key:
        raise ValueError("No valid Google API keys available")
    
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        # Test with a small embedding to verify key works
        embeddings.embed_query("test")
        return embeddings
    except google_exceptions.ResourceExhausted as e:
        key_manager.mark_failed(current_key, is_quota_error=True)
        logger.error(f"Embedding quota exceeded for key: {current_key[-6:]}")
        return get_embeddings()
    except Exception as e:
        key_manager.mark_failed(current_key)
        logger.error(f"Embedding API call failed with key: {current_key[-6:]}. Error: {str(e)}")
        return get_embeddings()

def get_vector_store(text_chunks: List[str]):
    """Convert text chunks into a FAISS vector store."""
    embeddings = get_embeddings()
    return FAISS.from_texts(text_chunks, embedding=embeddings)

def get_llm(model: str = "gemini-1.5-flash", temperature: float = 0, max_retries: int = 3):
    """Get LLM instance with current API key and proper error handling"""
    current_key = key_manager.get_current_key()
    if not current_key:
        raise ValueError("No valid Google API keys available")
    
    try:
        llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=current_key,
            temperature=temperature
        )
        # Test with a small request to verify key works
        llm.invoke("test")
        return llm
    except google_exceptions.ResourceExhausted as e:
        key_manager.mark_failed(current_key, is_quota_error=True)
        logger.error(f"Quota exceeded for key: {current_key[-6:]}. Rotating...")
        if max_retries > 0:
            return get_llm(model, temperature, max_retries-1)
        raise
    except Exception as e:
        key_manager.mark_failed(current_key)
        logger.error(f"API call failed with key: {current_key[-6:]}. Error: {str(e)}")
        if max_retries > 0:
            return get_llm(model, temperature, max_retries-1)
        raise

def get_conversational_chain(vector_store, max_retries: int = 3):
    """Create a conversational retrieval chain with retry logic"""
    try:
        llm = get_llm(model="gemini-1.5-flash", temperature=0)
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        return ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(),
            memory=memory
        )
    except Exception as e:
        if max_retries > 0:
            logger.warning(f"Retrying conversation chain creation ({max_retries} left)")
            return get_conversational_chain(vector_store, max_retries-1)
        raise

def generate_quiz(text: str, max_retries: int = 3) -> str:
    """Generate quiz questions with retry logic"""
    try:
        llm = get_llm(model="gemini-1.5-flash", temperature=0)
        prompt = f"Create 5 multiple-choice quiz questions with 4 options each and the correct answer marked from the following text:\n\n{text}"
        return llm.invoke(prompt).content
    except Exception as e:
        if max_retries > 0:
            logger.warning(f"Retrying quiz generation ({max_retries} left)")
            return generate_quiz(text, max_retries-1)
        raise

__all__ = [
    'get_pdf_text',
    'get_text_chunks',
    'get_vector_store',
    'get_conversational_chain',
    'generate_quiz'
]