# from transformers import AutoTokenizer, AutoModelForSequenceClassification

# import numpy as np
# import torch
import math
import requests


def predict(model, tweet):
    API_URL = "https://api-inference.huggingface.co/models/dibsondivya/ernie-phmtweets-sutd"
    if model == 'distilbert':
        API_URL = "https://api-inference.huggingface.co/models/dibsondivya/distilbert-phmtweets-sutd"

    headers = {"Authorization": "Bearer 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"}

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    output = query({
        "inputs": tweet,
    })
    print(output)
    output = output[0]

    max_score = max(label['score'] for label in output)

    for result in output:
        l = result['label']
        s = result['score']
        if s == max_score:
            pred = l[-1]

    print(pred)
    return pred


# def predict1(tweet):
#     device = torch.device('cpu')

#     tokenizer = AutoTokenizer.from_pretrained(
#         'dibsondivya/distilbert-phmtweets-sutd')

#     model = AutoModelForSequenceClassification.from_pretrained(
#         'dibsondivya/distilbert-phmtweets-sutd')

#     # Tokenize inputs
#     inputs = tokenizer(tweet, padding=True, truncation=True, return_tensors="pt").to(
#         device)  # Move the tensor to the GPU

#     # Inference model and get logits
#     outputs = model(**inputs)

#     # get predictions for test data
#     with torch.no_grad():
#         predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
#         preds = predictions.detach().cpu().numpy()

#     print(preds)
#     # model's performance
#     preds = np.argmax(predictions)
#     # print(classification_report(test_y, preds))

#     # model's performance
#     preds = np.argmax(predictions)
#     # print(classification_report(test_y, preds))

#     return preds.item()

# pred = predict("Alzheimer's is the worst disease on the planet")
# print(pred)


def convert_label(label):
    label = int(label)

    if label == 0:
        return 'NON-HEALTH'
    elif label == 1:
        return 'AWARENESS'
    elif label == 2:
        return 'OTHER MENTION'
    elif label == 3:
        return 'SELF-MENTION'
