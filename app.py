import streamlit as st
import os
import tempfile
import subprocess
import shutil
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import TextLoader, DirectoryLoader

st.set_page_config(page_title="Chat with GitHub Repository", page_icon="💬", layout="wide")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "repo_loaded" not in st.session_state:
    st.session_state.repo_loaded = False
if "repo_name" not in st.session_state:
    st.session_state.repo_name = ""
if "llm" not in st.session_state:
    st.session_state.llm = None

# Function to clone GitHub repository
def clone_github_repo(repo_url):
    try:
        with st.spinner(f"Cloning repository from {repo_url}..."):
            # Create temporary directory
            temp_dir = tempfile.mkdtemp()
            
            # Clone the repository
            subprocess.run(["git", "clone", repo_url, temp_dir], 
                        check=True, capture_output=True, text=True)
            
            # Extract repo name from URL
            repo_name = repo_url.split("/")[-1]
            if repo_name.endswith(".git"):
                repo_name = repo_name[:-4]
                
            return temp_dir, repo_name
    except subprocess.CalledProcessError as e:
        st.error(f"Error cloning repository: {e.stderr}")
        return None, None
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None, None

# Function to process repository files
def process_repository(repo_path):
    try:
        with st.spinner("Processing repository files..."):
            # Define extensions to include 
            suffixes = [
                ".py", ".js", ".jsx", ".ts", ".tsx", ".html", ".css", ".java", 
                ".c", ".cpp", ".h", ".hpp", ".go", ".rs", ".php", ".rb", ".swift",
                ".kt", ".md", ".txt", ".json", ".yml", ".yaml", ".toml"
            ]
            
            # Use DirectoryLoader instead of GenericLoader with LanguageParser
            loader = DirectoryLoader(
                repo_path,
                glob="**/*.*",
                loader_cls=TextLoader,
                loader_kwargs={"autodetect_encoding": True},
                show_progress=True,
                use_multithreading=True,
                silent_errors=True
            )
            
            documents = loader.load()
            
            # Filter documents based on suffixes
            filtered_docs = []
            for doc in documents:
                file_path = doc.metadata.get("source", "")
                if any(file_path.endswith(suffix) for suffix in suffixes):
                    # Skip hidden files and directories
                    if not any(part.startswith('.') for part in file_path.split(os.sep)):
                        filtered_docs.append(doc)
            
            st.info(f"Loaded {len(filtered_docs)} documents from repository")
            
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(filtered_docs)
            st.info(f"Split into {len(splits)} chunks")
            
            # Create embeddings and vector store
            embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
            vectorstore = FAISS.from_documents(splits, embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
            
            return retriever
    except Exception as e:
        st.error(f"Error processing repository: {str(e)}")
        return None

# Function to generate response using direct LLM invocation
def generate_response(query, retriever, llm):
    try:
        # Retrieve relevant documents
        docs = retriever.get_relevant_documents(query)
        
        # Format context from documents
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Create prompt
        prompt = f"""You are an assistant specialized in explaining code from GitHub repositories.
        Use the following context about the repository to answer the question.
        
        Context: {context}
        
        Question: {query}
        
        Answer the question based on the context. If you don't know the answer or can't find it in the context, say so."""
        
        # Direct invocation of the LLM
        response = llm.invoke(prompt)
        
        return response
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Streamlit UI
st.title("💬 Chat with GitHub Repository")
st.subheader("Load a GitHub repository and ask questions about its code")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    repo_url = st.text_input("GitHub Repository URL", placeholder="https://github.com/username/repository")
    
    if st.button("Load Repository"):
        if repo_url:
            # Clear previous messages when loading a new repo
            st.session_state.messages = []
            st.session_state.repo_loaded = False
            
            # Clone repository
            repo_path, repo_name = clone_github_repo(repo_url)
            
            if repo_path:
                # Process repository
                retriever = process_repository(repo_path)
                
                if retriever:
                    # Initialize Ollama with Gemma model
                    llm = Ollama(model="goekdenizguelmez/JOSIEFIED-Qwen2.5:1.5b")
                    
                    st.session_state.retriever = retriever
                    st.session_state.repo_loaded = True
                    st.session_state.repo_name = repo_name
                    st.session_state.llm = llm
                    st.success(f"Successfully loaded repository: {repo_name}")
                else:
                    st.error("Failed to process repository")
                
                # Clean up temporary directory
                try:
                    shutil.rmtree(repo_path)
                except:
                    pass
        else:
            st.warning("Please enter a GitHub repository URL")
    
    if st.session_state.repo_loaded:
        st.info(f"Currently loaded: {st.session_state.repo_name}")
        
        # System info
        st.divider()
        st.caption("System Information")
        st.caption("- Model: Gemma3:1b (via Ollama)")
        st.caption("- Embeddings: Nomic (via Ollama)")
        st.caption("- Vector DB: FAISS (CPU)")
        st.caption("- LLM Invocation: Direct (Fast Mode)")

# Display chat messages
st.subheader("Chat")
if not st.session_state.repo_loaded:
    st.info("Please load a GitHub repository to start chatting")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if st.session_state.repo_loaded:
    if prompt := st.chat_input("Ask something about the repository"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Use direct LLM invocation for faster response
                    response = generate_response(
                        prompt, 
                        st.session_state.retriever,
                        st.session_state.llm
                    )
                    st.write(response)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Footer
st.divider()
st.caption("This is a local RAG application. All processing happens on your machine.")
st.caption("Requires Ollama running locally with the Gemma3:1b model and Nomic embeddings available.")