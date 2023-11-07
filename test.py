from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

# Load the text documents with explicit encoding (replace 'utf-8' with 'latin-1' if needed)
documents = TextLoader(r"D:\text_files\accenture.txt", encoding='utf-8').load()

splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

db = Chroma.from_documents(documents, embeddings)
