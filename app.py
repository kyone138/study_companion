import gradio as gr
import pandas as pd
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma, faiss
from langchain_community.llms import HuggingFaceEndpoint, HuggingFaceHub
from langchain.chains import LLMChain
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community import vectorstores
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
import panel as pn
import param
import re
import os

api_token = os.environ.get('HUGGINGFACEHUB_API_TOKEN')

model = HuggingFaceHub(
    huggingfacehub_api_token=api_token,
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="conversational",
    model_kwargs={"temperature": 0.8, "max_length": 1000},
)
template = """<s>[INST] You are a friendly study tutor chatbot that has access to a database of documents provided by the students. Use the chat history and your existing knowledge to answer the follow up question in a helpful and friendly way. Make sure your tone is that of a friendly study buddy. [/INST]
Chat History: {context}
Follow up question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

def load_db(file, k):
    # load documents
    loader = PyPDFLoader(file)
    documents = loader.load()
    # split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    # define embedding
    embeddings = HuggingFaceEmbeddings()
    # create vector database from data
    db = vectorstores.FAISS.from_documents(docs, embeddings)
    # define retriever
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    # create a chatbot chain. Memory is managed externally.
    question_generator_chain = LLMChain(llm=model, prompt=QA_CHAIN_PROMPT)

    qa = ConversationalRetrievalChain.from_llm(
        llm=model,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        return_generated_question=True,
    )

    return qa

chat_history = []  # initialize chat history

def greet(question, pdf_file):
    global chat_history
    print("chat_history: ", chat_history)
    a = load_db(pdf_file, 5)
    r = a.invoke({"question": question, "chat_history": chat_history})
    print(a.return_source_documents)
    match = re.search(r'Helpful Answer:(.*)', r['answer'])
    if match:
        helpful_answer = match.group(1).strip()
        # Extend chat history with the current question and answer
        chat_history.extend([(question, helpful_answer)])
        return helpful_answer
    else:
        return "No helpful answer found."

iface = gr.Interface(fn=greet, inputs=["text", "file"], outputs="text")
iface.launch(share=True)



# import gradio as gr
# import pandas as pd
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import Chroma, faiss
# from langchain_community.llms import HuggingFaceEndpoint, HuggingFaceHub
# from langchain.chains import LLMChain
# from langchain_community.document_loaders.csv_loader import CSVLoader
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain_community.document_loaders import TextLoader
# from langchain_community import vectorstores
# from langchain.prompts import PromptTemplate
# from langchain.chains import RetrievalQA
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
# from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
# from langchain.vectorstores import DocArrayInMemorySearch
# from langchain.document_loaders import TextLoader
# from langchain.chains import RetrievalQA, ConversationalRetrievalChain
# from langchain.memory import ConversationBufferMemory
# from langchain.chat_models import ChatOpenAI
# from langchain.document_loaders import TextLoader
# from langchain.document_loaders import PyPDFLoader
# import panel as pn
# import param
# import re
# import os

# api_token = os.environ.get('HUGGINGFACEHUB_API_TOKEN')

# model = HuggingFaceHub(
#     huggingfacehub_api_token=api_token,
#     repo_id="mistralai/Mistral-7B-Instruct-v0.2",
#     task="conversational",
#     model_kwargs={"temperature": 0.8, "max_length": 1000},
# )
# template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
# {context}
# Question: {question}
# Helpful Answer:"""
# QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# # Updated greet function to handle file upload
# def greet(message, pdf_file):
#     global chat_history
#     user_message = message
#     if pdf_file is not None:
#         # Save the uploaded PDF file
#         with open("uploaded_file.pdf", "wb") as f:
#             f.write(pdf_file.read())
#         a = load_db("uploaded_file.pdf", 3)
#     # else:
#     #     a = load_db("temp.pdf", 3)  # assuming you've uploaded the file and saved it as "temp.pdf"
    
#     r = a.invoke({"question": user_message, "chat_history": chat_history})
#     match = re.search(r'Helpful Answer:(.*)', r['answer'])
#     if match:
#         helpful_answer = match.group(1).strip()
#         # Extend chat history with the current question and answer
#         chat_history.extend([{"role": "user", "content": user_message}, {"role": "assistant", "content": helpful_answer}])
#         return [{"role": "assistant", "content": helpful_answer}]
#     else:
#         return [{"role": "assistant", "content": "No helpful answer found."}]

# # Gradio ChatInterface with file upload support
# iface = gr.ChatInterface(fn=greet, title="Your Chatbot Title", additional_inputs="file")
# iface.launch(share=True)