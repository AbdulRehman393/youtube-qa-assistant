from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
import config
import os
import shutil


def get_llm():
    return ChatOpenAI(
        openai_api_key=config.OPENROUTER_API_KEY,
        openai_api_base=config.OPENROUTER_BASE_URL,
        model_name=config.LLM_MODEL,
        temperature=0.3,
    )


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def split_transcript(transcript_text):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = text_splitter.split_text(transcript_text)
    documents = [
        Document(page_content=chunk, metadata={"chunk_index": i})
        for i, chunk in enumerate(chunks)
    ]
    return documents


def create_vectorstore(documents):
    embeddings = get_embeddings()
    if os.path.exists(config.CHROMA_PERSIST_DIR):
        shutil.rmtree(config.CHROMA_PERSIST_DIR)
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=config.CHROMA_PERSIST_DIR,
        collection_name=config.COLLECTION_NAME,
    )
    return vectorstore


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def get_conversational_chain(vectorstore):
    llm = get_llm()
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": config.RETRIEVER_K},
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that answers questions about a YouTube video 
based on its transcript. Use the following context from the transcript to answer 
the user's question. If you cannot find the answer in the context, say so honestly.

Context from transcript:
{context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])
    chain = (
        {
            "context": lambda x: format_docs(retriever.invoke(x["question"])),
            "chat_history": lambda x: x.get("chat_history", []),
            "question": lambda x: x["question"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever


def get_summary(transcript_text):
    llm = get_llm()
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=3000,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_text(transcript_text)

    if len(chunks) == 1:
        prompt = f"""You are a helpful assistant. Please provide a comprehensive summary 
of the following transcript. Include the main topics discussed, key points, 
and any important takeaways.

Transcript:
{chunks[0]}

Summary:"""
        response = llm.invoke(prompt)
        return response.content

    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        prompt = f"""Summarize this part ({i+1}/{len(chunks)}) of a transcript concisely:

{chunk}

Summary:"""
        response = llm.invoke(prompt)
        chunk_summaries.append(response.content)

    combined = "\n\n".join(chunk_summaries)
    final_prompt = f"""You are a helpful assistant. Combine these partial summaries into 
one comprehensive, well-structured summary. Include main topics, key points, 
and important takeaways.

Partial Summaries:
{combined}

Final Comprehensive Summary:"""
    final_response = llm.invoke(final_prompt)
    return final_response.content