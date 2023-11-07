import json
import os
import numpy as np
import pandas as pd
import openai

import requests
from bs4 import BeautifulSoup

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# Load text data
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
os.environ['OPENAI_API_KEY'] = "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"

embeddings = OpenAIEmbeddings()

website_ls = ["https://www.bcg.com/publications/2023/generative-ai-in-marketing",
              "https://www.accenture.com/us-en/insights/generative-ai",
              "https://www.mckinsey.com/featured-insights",
              "https://blog.hubspot.com/sales/business-development",
              "https://www.redhat.com/en/blog/generative-ai-business-applications",
              "https://www.mckinsey.com/industries/travel-logistics-and-infrastructure/our-"
              "insights/what-ai-means-for-travel-now-and-in-the-future"]


def store_in_vector(text_file_name, vector_db_path, text_path=".tmp"):
    vector_path = os.path.join(vector_db_path, text_file_name)
    text_file_path = os.path.join(text_path, text_file_name)
    doc = TextLoader(text_file_path, encoding='utf-8').load()
    text_splitter = CharacterTextSplitter(chunk_size=256, chunk_overlap=32)
    documents = text_splitter.split_documents(doc)
    vectordb = Chroma.from_documents(documents=documents,
                                     embedding=embeddings,
                                     persist_directory=vector_path)
    vectordb.persist()


def get_text_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')
    text_content = ' '.join([paragraph.get_text() for paragraph in paragraphs])
    return text_content


def save_to_txt(site_name, text_content):
    file_name = f"{site_name.replace('://', '_').replace('/', '_')}.txt"
    # with open(r".tmp/" + file_name, 'w', encoding='utf-8') as file:
    #     file.write(text_content)
    return file_name


# os.makedirs(".tmp")

for url in website_ls:
    site_name = url.split('//')[1].split('/')[0]
    text_content = get_text_content(url)
    file_name = save_to_txt(site_name, text_content)
    store_in_vector(file_name, "vector_db")






