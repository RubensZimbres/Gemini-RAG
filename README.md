# Gemini-RAG

In this code, you have the steps to create a RAG (Retrieval Augmented Generation) application with Gemini and Langchain, build the image and deploy it in Cloud Run, add the Flask interface, and then deploy a Dialogflow chatbot to a website.


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
