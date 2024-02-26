# Gemini-LangChain-RAG Powered Chatbot in Dialogflow

This project is part of the GDE's Gemini Sprint. The idea was to develop a chatbot that uses Gemini-1.0-Pro to answer questions, and has memory of past interactions by using LangChain. Also, it has its context enriched by RAG (Retrieval Augmented Generation). This memory obtained through allows the chatbot to remember past interactions independently of Dialogflow $session.params. RAG document is vectorized with gecko-embeddings, chunked and a FAISS index is created. Later, a TOP-K result of embeddings similarity is retrieved to answer the questions, along with the chat history. The application is served via Flask and deployed in Cloud Run. It can also be deployed in GKE (Google Kubernetes Engine). The Flask application is stateful for demonstration purposes. However, user session must be saved in a database (SQL or BigQuery)

<b>Papers:</b>

* AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation: https://arxiv.org/abs/2308.08155
* Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks: https://arxiv.org/abs/2005.11401

<b>Articles:</b>
* Cost-Efficient Multi-Agent Collaboration with LangGraph + Gemma for Code Generation: https://medium.com/@rubenszimbres/cost-efficient-multi-agent-collaboration-with-langgraph-gemma-for-code-generation-88d6cf87fc99
* Code Generation using Retrieval Augmented Generation + LangChain: https://medium.com/@rubenszimbres/code-generation-using-retrieval-augmented-generation-langchain-861e3c1a1a53

<img src=https://github.com/RubensZimbres/Gemini-RAG/blob/main/pics/RAG_.png>

In this repo, you have the steps to create a RAG (Retrieval Augmented Generation) application with Gemini and Langchain, build the image and deploy it in Cloud Run, add the Flask interface, and then deploy a Dialogflow chatbot to a website.


<b>Steps:</b>  
* Test locally
* Make deployment in Cloud Run
* Generate the flow and pages in Dialogflow + test webhook
* Add HTML code to website + create event called 'sayhi' in Default Welcome Page in Dialogflow so that the bot starts a conversation

<b>Deployment in Cloud Run:</b>  

```
gcloud builds submit --tag gcr.io/your-project/container-name . --timeout=85000
```

```
gcloud run deploy container-name --image gcr.io/your-project/container-name --min-instances 1 --max-instances 5 --cpu 1 --allow-unauthenticated --memory 512Mi --region us-east1 --concurrency 10
```
  
<b>Final Project Screenshot 1:</b>

<img scr=https://github.com/RubensZimbres/Gemini-RAG/blob/main/pics/dialog0.png>  
  
<b>Final Project Screenshot 2:</b>
  
<img src=https://github.com/RubensZimbres/Gemini-RAG/blob/main/pics/dialog1.png>
