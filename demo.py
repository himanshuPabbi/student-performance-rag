import streamlit as st
import os
import pandas as pd
import csv
import time
import uuid
from datetime import datetime
from dotenv import load_dotenv

# LangChain & AI Modules
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# Load Environment Variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- RESEARCH CONFIGURATION ---
LOG_FILE = "research_metrics_log.csv"
DB_FAISS_PATH = "vectorstore/db_faiss"

def log_experiment(data):
    """Saves research data to CSV. Handles header creation only if file is new."""
    file_exists = os.path.isfile(LOG_FILE) and os.path.getsize(LOG_FILE) > 0
    with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

# --- SYSTEM INITIALIZATION ---
@st.cache_resource
def initialize_engine():
    if not os.path.exists("StudentPerformanceFactors.csv"):
        st.error("Dataset not found! Please upload StudentPerformanceFactors.csv.")
        st.stop()
        
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if os.path.exists(DB_FAISS_PATH):
        vector_db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        df = pd.read_csv("StudentPerformanceFactors.csv").fillna("Not Reported")
        docs = []
        for idx, row in df.iterrows():
            text_content = f"Student Record {idx}: " + ". ".join([f"{k.replace('_',' ')}: {v}" for k, v in row.items()])
            docs.append(Document(page_content=text_content, metadata={"row_id": idx}))
        vector_db = FAISS.from_documents(docs, embeddings)
        vector_db.save_local(DB_FAISS_PATH)

    retriever = vector_db.as_retriever(search_kwargs={"k": 5})
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    
    system_msg = (
        "You are an Advanced Educational Data Scientist. Analyze the provided student data "
        "and answer inquiries with academic rigor. Context: {context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("human", "{input}"),
    ])
    
    chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, chain)

# --- UI ARCHITECTURE ---
st.set_page_config(page_title="RAG Research Portal", layout="wide")
st.title("🔬 Student Performance Analysis: Llama-3.1-8b (RAG)")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found in .env file.")
    st.stop()

engine = initialize_engine()

tab1, tab2, tab3 = st.tabs(["💬 Interactive Lab", "📂 Batch Processing", "📊 Export Metrics"])

# --- TAB 1: INTERACTIVE CHAT ---
with tab1:
    st.subheader("Exploratory Analysis")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]): st.markdown(chat["content"])

    if query := st.chat_input("Ex: Analyze the correlation between study hours and scores."):
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"): st.markdown(query)

        with st.chat_message("assistant"):
            try:
                start_t = time.perf_counter()
                response = engine.invoke({"input": query})
                latency = round(time.perf_counter() - start_t, 4)
                
                st.markdown(response['answer'])
                
                log_experiment({
                    "Run_ID": str(uuid.uuid4())[:8], 
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Query": query, 
                    "Latency": latency, 
                    "Mode": "Interactive",
                    "Model": "Llama-3.1-8b"
                })
                st.session_state.chat_history.append({"role": "assistant", "content": response['answer']})
            except Exception as e:
                st.error(f"Inference Error: {e}")

# --- TAB 2: BATCH PROCESSING (WITH RATE-LIMIT PROTECTION) ---
with tab2:
    st.subheader("Systematic Batch Evaluation")
    st.info("Safety: 2.0s delay between queries + 60s auto-pause on Rate Limits.")
    batch_input = st.text_area("Paste Research Queries (One per line):", height=250)
    
    if st.button("🚀 Execute Batch Research"):
        queries = [q.strip() for q in batch_input.split('\n') if q.strip()]
        if not queries:
            st.warning("Please enter at least one query.")
        else:
            results_list = []
            progress_bar = st.progress(0)
            status_update = st.empty()
            
            for idx, q in enumerate(queries):
                success = False
                retries = 0
                
                while not success and retries < 2:
                    try:
                        status_update.text(f"Processing {idx+1}/{len(queries)}: {q[:50]}...")
                        t0 = time.perf_counter()
                        
                        res = engine.invoke({"input": q})
                        
                        dt = round(time.perf_counter() - t0, 4)
                        
                        entry = {
                            "Run_ID": f"BATCH-{uuid.uuid4().hex[:5]}", 
                            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Query": q, 
                            "Answer": res['answer'].replace('\n', ' '), 
                            "Latency": dt, 
                            "Mode": "Batch",
                            "Model": "Llama-3.1-8b"
                        }
                        log_experiment(entry)
                        results_list.append(entry)
                        
                        # 2-second cooldown to avoid RPM limits
                        time.sleep(2.0)
                        success = True
                        
                    except Exception as e:
                        err = str(e).lower()
                        if "429" in err or "rate_limit" in err:
                            st.warning(f"Rate limit reached at query {idx+1}. Pausing for 60 seconds...")
                            time.sleep(60)
                            retries += 1
                        else:
                            st.error(f"Critical Error on query {idx+1}: {e}")
                            break
                
                progress_bar.progress((idx + 1) / len(queries))
            
            st.success(f"Successfully processed {len(results_list)} queries.")
            if results_list:
                st.dataframe(pd.DataFrame(results_list))

# --- TAB 3: LOGS & ANALYTICS (FIXED EMPTY DATA ERROR) ---
with tab3:
    st.subheader("Data Export for Analysis")
    if os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE) > 0:
        try:
            logs_df = pd.read_csv(LOG_FILE)
            st.write(f"Total Successful Experiments logged: **{len(logs_df)}**")
            st.dataframe(logs_df)
            
            csv_data = logs_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Research Logs (CSV)",
                data=csv_data,
                file_name=f"research_metrics_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Could not read log file: {e}")
    else:
        st.warning("No data found in log file. Please run queries first.")