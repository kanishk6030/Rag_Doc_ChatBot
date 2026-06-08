# RAG Document Chatbot

This project is a Streamlit app that creates embeddings from PDF documents, stores them in a FAISS vector store, and answers user queries using a Groq LLM via LangChain-style chains.

This README documents every step required to set up, run, and troubleshoot the project.

## Prerequisites

- Windows (tested here, but should run on macOS/Linux with minor path adjustments)
- Python 3.10+ (virtual environment recommended)
- Git (optional)
- Internet connection for model and embeddings API access

## Files

- `app.py` - main Streamlit application.
- `requirements.txt` - Python dependencies.
- `files/` - directory to place PDF documents to index.

## 1. Clone or copy the repository

If using Git:

```bash
git clone <repo-url> RAG_document_chatbot
cd RAG_document_chatbot
```

Or copy the project files into a working directory.

## 2. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On cmd.exe:

```cmd
.\.venv\Scripts\activate.bat
```

On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

If installation fails, check the output for missing system libraries (e.g. for FAISS or PyTorch) and follow the package-specific instructions.

## 4. Environment variables

Create a `.env` file in the project root with the following variables (replace placeholders):

```
GROQ_API_KEY=your_groq_api_key_here
HUGGING_FACE=your_huggingface_api_key_here
LANGCHAIN_API_KEY=your_langchain_api_key_here
```

- `GROQ_API_KEY` is used by `ChatGroq` in `app.py`.
- `HUGGING_FACE` is used by `HuggingFaceEmbeddings` if required.
- `LANGCHAIN_API_KEY` is optional depending on used libraries.

Load the `.env` file by ensuring `python-dotenv` is installed (it's in `requirements.txt`) and the app calls `load_dotenv()` (already present in `app.py`).

## 5. Add PDF documents

Place any PDF files you want processed into the `files/` directory. The app uses `PyPDFDirectoryLoader` to read PDFs from this folder.

## 6. Run the app

```bash
streamlit run app.py
```

Open the URL printed by Streamlit in your browser (usually `http://localhost:8501`).

## 7. Usage

- Click `Document Embedding` to create embeddings and build the FAISS index from PDFs in `files/`.
- Enter a question in the text input and press Enter (or the UI will automatically run when a question is provided).
- The app will return an answer and show retrieved documents in an expander.

## 8. Troubleshooting

Common errors and fixes:

- ValueError: "Prompt must accept context as an input variable"
  - Cause: `create_stuff_documents_chain` expects the prompt template to include a document/context variable named `context` (default) or a custom variable name passed to `document_variable_name`.
  - Fix: Update the `ChatPromptTemplate` to use `{context}` instead of `{retrieved_docs}`, or construct the chain with the matching variable name. Example prompt:

    ```python
    prompt = ChatPromptTemplate.from_template(
        """
        Use only the following retrieved documents to answer the question.
        <documents>
        {context}
        <documents>
        Question: {question}
        Answer:
        """
    )
    ```

- No documents found / `files/` is empty
  - Ensure PDFs are present in `files/` and readable.

- FAISS installation errors
  - On Windows, prefer installing `faiss-cpu` via pip wheels or use conda:

    ```bash
    conda install -c conda-forge faiss-cpu
    ```

- HuggingFace embeddings failing due to missing API key
  - Ensure `HUGGING_FACE` is set in `.env` and referenced by `os.getenv("HUGGING_FACE")`.

## 9. Code notes and suggestions

- `create_vector_embeddings` stores objects in `st.session_state` to avoid rebuilding index on every interaction.
- The prompt template variable must match what `create_stuff_documents_chain` expects. Use `{context}` or pass `document_variable_name` when creating the chain.
- Add error handling around embedding/index creation to surface problems in the UI.

## 9. How it works (detailed pipeline)

This section describes step-by-step what happens from raw PDFs to a question-answer response.

- **1) Document loading**: `PyPDFDirectoryLoader('files')` scans the `files/` directory and produces a list of `Document` objects. Each `Document` contains `page_content` and optional metadata such as `source` and `page_number`.

- **2) Text splitting (chunking)**: The `RecursiveCharacterTextSplitter` is used to split each `Document` into smaller chunks to balance context length and retrieval precision. Configurable parameters in the app:
  - `chunk_size` (default in this repo: 1000 characters)
  - `chunk_overlap` (default: 200 characters)

  Why split? LLMs have context-length limits. Splitting makes it more likely that a relevant excerpt will fit in a single retrieved chunk and reduces noise during similarity search.

- **3) Embeddings creation**: For every chunk produced by the splitter, an embedding vector is computed using `HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")`. These are fixed-size numeric vectors (e.g., 384 dimensions for many MiniLM models) which capture semantic meaning.

  Notes:
  - The embedding model choice affects retrieval quality and speed. `all-MiniLM-L6-v2` is a popular, small, fast model suitable for many RAG setups.
  - The code caches the `HuggingFaceEmbeddings` object in `st.session_state` to avoid re-initializing it repeatedly.

- **4) Vector store (FAISS) indexing**: The embeddings for all chunks are added to a FAISS index via `FAISS.from_documents(...)`. FAISS stores vectors and supports fast approximate nearest-neighbor search.

  Internals:
  - FAISS indexes can be configured for speed/accuracy tradeoffs (IVF, HNSW, PQ, etc.). The default builder in `langchain_community` chooses a reasonable CPU-backed index.
  - The project keeps the FAISS index in `st.session_state['vectors']` to reuse it across Streamlit interactions.

- **5) Retriever**: `st.session_state.vectors.as_retriever()` produces a retriever object that accepts a text query, computes its embedding, and returns the top-N most similar document chunks.

- **6) Combine documents chain (stuffing)**: `create_stuff_documents_chain` expects a prompt template which accepts two variables: the user `question` and the retrieved document variable (default name: `context`). The chain fills the prompt with the retrieved chunks (in the `context` variable) and sends it to the LLM to generate an answer.

  Important: If the prompt uses a different placeholder name (for example `{retrieved_docs}`), the chain will raise the error seen earlier: `ValueError: Prompt must accept context as an input variable.` Fix by using `{context}` in the prompt or by passing the appropriate `document_variable_name` when constructing the chain.

- **7) LLM call (Groq via ChatGroq)**: The filled prompt is sent to `ChatGroq` which produces the final answer. The app prints timing for the invoke call to help measure latency.

- **8) UI output**: The app writes the answer to the Streamlit page and exposes the retrieved chunks in an expander for transparency.

## 10. Implementation details and tuning

- **Session caching**: `st.session_state` caches `embeddings`, `loader`, `docs`, `final_documents`, and `vectors`. This prevents repeated CPU and network work while using the app interactively.

- **Chunk size / overlap**: Increasing `chunk_size` reduces the number of vectors but may produce chunks that contain irrelevant info; increasing `chunk_overlap` helps preserve context at chunk boundaries but increases index size.

- **Embedding model**: Tradeoffs: smaller models are faster and cheaper but less precise. Try `all-mpnet-base-v2` or larger SBERT models for better quality.

- **FAISS persistence**: For faster startups, serialize the FAISS index to disk (e.g., `vectors.save_local(path)`) and load it on startup if present. This prevents re-embedding on every process start.

- **Error handling**: The app should surface clear messages when:
  - No PDFs are found in `files/`.
  - Embedding creation fails (missing API key or network error).
  - FAISS index creation fails (memory / installation issues).

## 11. Where to look in code

- Loader and splitting: [app.py](app.py#L1-L80)
- Prompt template: [app.py](app.py#L20-L36)
- Embedding and vector creation: [app.py](app.py#L38-L80)
- Retrieval and chain invocation: [app.py](app.py#L82-L130)

---

If you'd like, I can now modify `app.py` to switch the prompt placeholder to `{context}`, harden session state usage, and add FAISS persistence. Which change should I apply next?

## 12. Next steps (optional)

- Add an uploader in Streamlit to add files via the UI.
- Persist the FAISS index to disk for faster startup.
- Add unit tests for the document-loading and embedding pipeline.

---
