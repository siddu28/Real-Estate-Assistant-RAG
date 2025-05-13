# 🏙️ **RealEstate Research Tool**

This is a user-friendly news research tool designed for effortless information retrieval. Users can input article URLs and ask questions to receive relevant insights from the real-estate domain. (But it's features can be extended to any domain.)
![image](https://github.com/user-attachments/assets/8eedc948-2768-4138-9541-6f697b6e332d)


### Features

- Load URLs to fetch article content.
- Process article content through LangChain's UnstructuredURL Loader
- Construct an embedding vector using HuggingFace embeddings and leverage ChromaDB as the vectorstore, to enable swift and effective retrieval of relevant information.
- Interact with the LLM's (Llama3 via Groq) by inputting queries and receiving answers along with source URLs.

## 🔍 Process Behind It (Retrieval-Augmented Generation - RAG)
This project uses **Retrieval-Augmented Generation (RAG)** to provide accurate, context-aware responses by combining document retrieval with the power of a Large Language Model (LLM). Below is the step-by-step process:
1. **Document Loading**  
   * The relevant documents are collected from various sources.
2. **Chunking**  
   * These documents are split into smaller, manageable chunks to preserve context during retrieval.
3. **Embedding and Storage**  
   * Each chunk is converted into a vector (embedding) and stored in a **vector database**.  
   * In this project, **ChromaDB** is used to handle vector storage and similarity search efficiently.
4. **Query Embedding and Retrieval**  
   * When a user enters a query, it is also converted into a vector.  
   * The vector database finds the most relevant chunks by comparing the query vector with stored vectors based on **similarity scores**.
5. **Context + Prompt Construction**  
   * The retrieved chunks (context) are combined with a prompt (instruction) and passed to the LLM.
6. **Response Generation**  
   * The LLM uses this combined input to generate a coherent, human-readable response grounded in the retrieved information.

  ![image](https://github.com/user-attachments/assets/3df2e611-3487-4f82-9b6e-9608b0ec4932)

## Here is project Demo
https://youtu.be/zQ1Kw0SPB78



### Set-up

1. Run the following command to install all dependencies. 

    ```bash
    pip install -r requirements.txt
    ```

2. Create a .env file with your GROQ credentials as follows:
    ```text
    GROQ_MODEL=MODEL_NAME_HERE
    GROQ_API_KEY=GROQ_API_KEY_HERE
    ```

3. Run the streamlit app by running the following command.

    ```bash
    streamlit run main.py
    ```


### Usage/Examples

The web app will open in your browser after the set-up is complete.

- On the sidebar, you can input URLs directly.

- Initiate the data loading and processing by clicking "Process URLs."

- Observe the system as it performs text splitting, generates embedding vectors using HuggingFace's Embedding Model.

- The embeddings will be stored in ChromaDB.

- One can now ask a question and get the answer based on those news articles


</br>
