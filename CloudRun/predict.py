from pathlib import Path
import langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter,CharacterTextSplitter
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
import pickle
from langchain import OpenAI, LLMChain
from langchain.prompts import Prompt
import sys
import subprocess
import os
from langchain.text_splitter import Language
from google.cloud import aiplatform
from langchain_google_vertexai import VertexAI
from vertexai.language_models import CodeGenerationModel
from langchain_community.llms import VertexAI
from langchain_google_vertexai import VertexAIEmbeddings
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import RetrievalQA
import os
from langchain.schema.document import Document
from langchain.chains import ConversationChain, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.prompts import QA_PROMPT
from langchain.prompts.chat import SystemMessagePromptTemplate,HumanMessagePromptTemplate
from vertexai.preview.generative_models import GenerativeModel
from vertexai.generative_models._generative_models import HarmCategory, HarmBlockThreshold, ResponseBlockedError
import google.generativeai as genai
from google.cloud import secretmanager
import warnings
from google.cloud import aiplatform
import vertexai

with warnings.catch_warnings():
    warnings.simplefilter('ignore')

### CREATE A SECRET IN SECRETS MANAGER WITH THE NAME 'GOOGLE_APPLICATION_CREDENTIALS' 
    ## AND ADD THE CONTENT OF YOUR KEY.JSON FILE OBTAINED FROM THE COMPUTE SERVICE ACCOUNT

def access_secret_version(secret_version_id):
  client = secretmanager.SecretManagerServiceClient()
  response = client.access_secret_version(name=secret_version_id)
  return response.payload.data.decode('UTF-8')

## HERE IS THE PROJECT NUMBER, NOT NAME --  GET WITH: gcloud projects list
secret_version_id = f"projects/your-project/secrets/GOOGLE_APPLICATION_CREDENTIALS/versions/latest"

key=access_secret_version(secret_version_id)
os.getenv(key)

## HERE IS THE PROJECT NAME
vertexai.init(project='your-project', location='us-east1')


import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore')



## THE PLACE WHERE YOUR RAG DOCUMENT IS
trainingData = os.listdir("/home/training/facts/")


## MODEL USED TO EMBED THE RAG DOCUMENT
EMBEDDING_QPM = 100
EMBEDDING_NUM_BATCH = 5
embeddings = VertexAIEmbeddings(
    requests_per_minute=EMBEDDING_QPM,
    num_instances_per_batch=EMBEDDING_NUM_BATCH,
    model_name = "textembedding-gecko",max_output_tokens=512,
temperature=0.1,
top_p=0.8,
top_k=40
)

# CHUNK CREATION IN THE RAG
text_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,chunk_size=2000, chunk_overlap=500
)


## CREATE DOCUMENTS FROM CHUNKS
docs=[]
for training in trainingData:
    with open('/home/training/facts/'+training) as f:
        print(f"Add {f.name} to dataset")
        texts=text_splitter.create_documents([f.read()])
        docs+=texts


# CREATE AN INDEX FROM CHUNKED DOCUMENTS AND SAVE LOCALLY IN THE CONTAINER
store = FAISS.from_documents(docs, embeddings)

store.save_local("/home/faiss_index")
  

## FLASK APPLICATION

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask import session, app
import json
from collections import Counter
import numpy as np

app = Flask(__name__)

app.secret_key = '12345'  ## added
CORS(app)

history=[]

@app.route('/predict', methods= ['POST'])
def predict():
    # THIS GETS 'x' (PAYLOAD) FROM DIALOGFLOW
    if request.get_json():
        x=json.dumps(request.get_json())
        print(x)
        x=json.loads(x)
    else:
        x={}

    # HERE WE CREATE A STATEFUL SESSION SO THAT WE CAN USE LANCHAIN MEMORY IN DIALOGFLOW
    session['username'] = str(np.random.randint(10000000, size=1)[0]) ## GET FROM DATABASE
    app.config['SESSION_TYPE'] = 'filesystem' 

    data=x["text"]  # EXTRACT THE QUESTION (text) FROM DIALOGFLOW PAYLOAD


    # CREATE A CHATBOT WITH MEMORY POWERED BY LANGCHAIN AND GEMINI 1.0

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True, output_key='answer')

    code_llm = VertexAI(
    model_name="gemini-1.0-pro",
    max_output_tokens=256,
    temperature=0.1,
    verbose=False,
    )

    EMBEDDING_QPM = 100
    EMBEDDING_NUM_BATCH = 5
    embeddings = VertexAIEmbeddings(
        requests_per_minute=EMBEDDING_QPM,
        num_instances_per_batch=EMBEDDING_NUM_BATCH,
        model_name = "textembedding-gecko"
    )

    # LOAD LOCAL RAG INDEX PRE-TRAINED

    store = FAISS.load_local("/home/faiss_index", embeddings)

    ## RETRIEVE BEST K=2 RESULTS FROM EMBEDDING SIMILARITY

    retriever = store.as_retriever(
    search_type="similarity",  # Also test "similarity", "mmr"
    search_kwargs={"k": 2},)


    # MAKE THE LLM ANSWER USER'S QUESTIONS BASED ON CONTEXT (RAG) AND CHAT HISTORY (MEMORY)

    promptTemplate = """" Please answer the user question. You can use the chat history: {chat_history} and {context}
    to answer users' question: {question}.
    If you don't know, please answer considering your knowledge base. Please be polite and answer in english.
    """

    # MACHINE AND HUMAN INTERACT
    messages = [
        SystemMessagePromptTemplate.from_template(promptTemplate),
        HumanMessagePromptTemplate.from_template("{question}")
        ]
    qa_prompt = ChatPromptTemplate.from_messages(messages)

    # CREATE THE CONVERSATION CHAIN WITH MEMORY

    qa_chain = ConversationalRetrievalChain.from_llm(
        code_llm, retriever, memory=memory,get_chat_history=lambda h : h,combine_docs_chain_kwargs={"prompt": qa_prompt})

    answer = qa_chain({"question":data,"chat_history":history})
    history.append(("{question}",data, answer))

    response=answer["answer"]
  
    # JASONIFY RESPONSE TO BE ACCEPETED BY DIALOGFLOW

    response=jsonify({
        "fulfillment_response": {
            "messages": [{
                "text": {
                    "text": [
                        response
                    ]
                }
            }]
        }
    })

    # HERE IS FOR CORS - Cross-Origin Resource Sharing in the website

    response.headers['Access-Control-Allow-Origin'] = '*'
    return response 



if __name__ == "__main__":

    ## SAME PORT EXPOSED IN DOCKERFILE, ACCEPTING CONNECTIONS FROM ANY IP
    app.run(port=8080, host='0.0.0.0', debug=True)

