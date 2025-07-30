import boto3
import json
import streamlit as st

class BedrockEmbedding:
    def __init__(self):
        self.client = boto3.client(
            service_name="bedrock-runtime",
            region_name=st.secrets["AWS_REGION"],
            aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
        )
        self.model_id = "amazon.titan-embed-text-v1"

    def embed(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        embeddings = []
        for text in texts:
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps({"inputText": text}),
                contentType="application/json",
                accept="application/json",
            )
            result = json.loads(response['body'].read())
            embeddings.append(result['embedding'])

        return embeddings
