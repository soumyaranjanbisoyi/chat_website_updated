import os
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts.chat import *
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
import json


load_dotenv()

api_key = st.secrets.api_credential.openai_key
os.environ['OPENAI_API_KEY'] = api_key

class WebsiteChatApp:
    """Class representing a Streamlit web app for chatting with predefined websites."""

    def __init__(self):
        """Initialize the WebsiteChatApp instance."""
        self.OPENAI_API_KEY = api_key
        self.system_template = """SYSTEM: You are an intelligent assistant helping the users with their questions on the context from the 
        given website. Don't make up the answer. Strictly, never answer from your previous knowledge. If the answer to the 
        question is present on website, give the answer, otherwise respond as 'Not Available'
        Question: {question}
        Strictly Use ONLY the following pieces of context to answer the question at the end. Think step-by-step and then answer.
        Do not try to make up an answer:
        - If the answer to the question cannot be determined from the context alone, say "I cannot determine the answer to that."
        - If the context is empty, just say "I do not know the answer to that."
        =============
        {context}
        =============
        Question: {question}
        Helpful Answer:
        """

        self.messages = [
            SystemMessagePromptTemplate.from_template(self.system_template),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]
        self.prompt = ChatPromptTemplate.from_messages(self.messages)
        self.chain_type_kwargs = {"prompt": self.prompt}
        self.session_state = st.session_state
        self.load_predefined_websites()

    def load_predefined_websites(self):
        """Load predefined websites from the session file."""
        if "predefined_websites" not in self.session_state:
            self.session_state.predefined_websites = self.load_websites_from_session()

    def load_websites_from_session(self):
        """Load websites from the session file."""
        session_file_path = "websites_session.json"
        if os.path.exists(session_file_path):
            with open(session_file_path, "r") as f:
                return json.load(f)
        else:
            return [
                "https://hbsp.harvard.edu/home/",
                "https://hbr.org/",
                "https://www.accenture.com/us-en/insights/voices",
                "https://www.zs.com/insights",
                "https://www.mckinsey.com/featured-insights",
                "https://www.bcg.com/publications",
                "https://publishing.insead.edu/",
            ]

    def save_websites_to_session(self):
        """Save websites to the session file."""
        session_file_path = "websites_session.json"
        with open(session_file_path, "w") as f:
            json.dump(self.session_state.predefined_websites, f)


    def ask_website(self, webpage, question):
        QNA_PROMPT = PromptTemplate.from_template(self.system_template)
        loader = WebBaseLoader(webpage)
        data = loader.load()
        text_splitter = CharacterTextSplitter(separator='\n', chunk_size=200, chunk_overlap=40)
        docs = text_splitter.split_documents(data)
        openai_embeddings = OpenAIEmbeddings()
        vectordb = Chroma.from_documents(documents=docs,
                                         embedding=openai_embeddings,
                                         persist_directory=f".tmp_db/{webpage.replace(r'/', '').replace(':', '').replace('.', '')}")
        retriever = vectordb.as_retriever(search_kwargs={"k": 1})
        llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0, model_kwargs={"top_p": 0.9})
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",
                                         retriever=retriever,
                                         return_source_documents=True,
                                         verbose=True,
                                         chain_type_kwargs={"prompt": PromptTemplate(
                                             template=self.system_template,
                                             input_variables=["context", "question"])})
        response = qa(question)
        return response["result"]

    def get_website_resp(self, question):
        res = {}
        for website_url in self.session_state.predefined_websites:
            val = self.ask_website(website_url, question)
            res[website_url] = val
        return res


    def add_website(self):
        """Add more websites dynamically."""
        website_url = st.text_input("Enter Website URL")

        if st.button("Add Website", type="primary"):
            if website_url:
                self.session_state.predefined_websites.append(website_url)
                self.save_websites_to_session()
                st.success(f"Website {website_url} added successfully!")


# def main():
#     """Main function to run the Streamlit web app."""
#     st.markdown("<h1 style='text-align: center; color: yellow;'>WebAnswerHub</a></h1>",
#                 unsafe_allow_html=True)
#     st.markdown("<h3 style='text-align: center; color:red;'>Enter your question (query/prompt) below 👇</h3>",
#                 unsafe_allow_html=True)
#
#     website_chat_app = WebsiteChatApp()
#
#     st.subheader("Predefined Websites:")
#     for website_url in website_chat_app.session_state.predefined_websites:
#         st.write(f"- {website_url}")
#
#     st.subheader("Add More Websites:")
#     website_chat_app.add_website()
#
#     prompt = st.text_input("Ask a question (query/prompt)")
#
#     if st.button("Generate Individual Answers", type="primary"):
#         website_chat_app.get_website_resp(prompt)



# if __name__ == '__main__':
#     main()












