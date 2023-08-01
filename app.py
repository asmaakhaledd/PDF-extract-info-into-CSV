import os
import streamlit as st
from PyPDF2 import PdfReader
import PyPDF2
import re
import csv
import sqlite3
import pickle
import redis
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.vectorstores.redis import Redis
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback


os.environ["OPENAI_API_KEY"] = "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"


def extractFromPDF(text):
     name_pattern = r'[A-Z][a-z]+\s[A-Z][a-z]+'  
     names = re.findall(name_pattern, text, flags=re.MULTILINE)
        
     gpa_pattern = r'\d+\.\d+' 
     gpa = re.findall(gpa_pattern, text)[0]
        
     phone_pattern = r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b' 
     phones = re.findall(phone_pattern, text)
        
     address_pattern = r'\b[A-Z][a-z]+(?:\s*,\s*[A-Z][a-z]+)*,\s*[A-Z][a-z]+(?:\s*,\s*[A-Z][a-z]+)*\b'
     address = re.findall(address_pattern, text)

     if address:
        addresses = address[0]
     else:
        addresses = ""

     email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
     emails = re.findall(email_pattern, text, re.IGNORECASE)
     
     return {"Name": names[0] if names else None,
        "GPA": float(gpa) if gpa else None,
        "Phone": phones if phones else None,
        "Address": addresses if addresses else None,
        "Email": emails if emails else None}

def writeInCSV(info_dict,csv_filename):
      with open(csv_filename, mode="w", newline="", encoding='utf-8') as csvfile:
        fieldnames = ["Name", "GPA", "Phone", "Address", "Email"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(info_dict)
      
def main():
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF 💬")
    
    # upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    # extract the text
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
            
        # split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        
        # create embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        
        user_question = st.text_input("Ask a question about your PDF:")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            
            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cb:
                docs = knowledge_base.similarity_search(user_question)
                response = chain.run(input_documents=docs, question=user_question)
                print(cb)
            
            extractedInfo = extractFromPDF(text)
            if extractedInfo:
                writeInCSV(extractedInfo, "DataBase.csv")
                
            st.write(response)


if __name__ == '__main__':
    main()
   
       

