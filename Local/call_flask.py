import requests
import time

### BEFORE DEPLOYING TO CLOUD RUN, TEST LOCALLY !!!

## USE CLOUD RUN ENDPOINT IN THIS FORMAT: https://container-name-bli76422ut76.a.run.app/predict
url = 'http://127.0.0.1:8080/predict' 

r = requests.post(
    url, json={"text": "Who was Albert Einstein?"}) # send custom payload from dialogflow to webhook
print(r.json())
time.sleep(8)

r = requests.post(
    url, json={"text": "When was he born?"}) # send custom payload from dialogflow to webhook
print(r.json())
time.sleep(8)

r = requests.post(
    url, json={"text": "What was my last question?"}) # send custom payload from dialogflow to webhook
print(r.json())
