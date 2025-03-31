import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
import streamlit as st

def get_pdf_text(pdf_docs):
    if not pdf_docs:
        st.warning("No PDFs uploaded.")
        return ""
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted
                else:
                    st.warning(f"Could not extract text from a page in {pdf.name}")
        except Exception as e:
            st.error(f"Error processing {pdf.name}: {str(e)}")
    if not text:
        st.warning("No text extracted from uploaded PDFs.")
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

# def get_conversational_chain(vector_store):
#     # Use google/flan-t5-large for better performance
#     model_name = "google/flan-t5-large"
#     tokenizer = T5Tokenizer.from_pretrained(model_name)
#     model = T5ForConditionalGeneration.from_pretrained(model_name)
    
#     # Set up pipeline
#     pipe = pipeline(
#         "text2text-generation",
#         model=model,
#         tokenizer=tokenizer,
#         max_new_tokens=512,  # Allow longer, more detailed responses
#         truncation=True
#     )
#     llm = HuggingFacePipeline(pipeline=pipe)
    
#     # Memory setup with explicit output_key
#     memory = ConversationBufferMemory(
#         memory_key="chat_history",
#         return_messages=True,
#         output_key="answer"  # Specify which key to store in memory
#     )
    
#     # Custom prompt to improve answer quality
#     prompt_template = """
#     Given the following context from a PDF: {context}
#     Answer this question: {question}
#     Provide a concise and accurate response based only on the context.
#     """
#     prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)
    
#     # Conversational chain with custom prompt and output_key
#     conversation_chain = ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=vector_store.as_retriever(search_kwargs={"k": 3}),  # Retrieve top 3 chunks
#         memory=memory,
#         combine_docs_chain_kwargs={"prompt": prompt},
#         return_source_documents=True,  # For debugging
#         output_key="answer"  # Explicitly set the output key for the chain
#     )
    
#     # Wrap the chain in a debug function
#     def debug_chain(input_dict):
#         result = conversation_chain(input_dict)  # Direct call to the chain
#         print("Retrieved docs:", [doc.page_content for doc in result.get("source_documents", [])])
#         print("Generated answer:", result["answer"])
#         return result
    
#     return debug_chain  # Return the wrapped function
