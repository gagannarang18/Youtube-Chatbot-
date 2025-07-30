import boto3
import json
import streamlit as st

class BedrockEmbedding:
    def __init__(self):
        region = st.secrets.get("AWS_DEFAULT_REGION")
        access_key = st.secrets.get("AWS_ACCESS_KEY_ID")
        secret_key = st.secrets.get("AWS_SECRET_ACCESS_KEY")
        self.client = boto3.client(
            service_name="bedrock-runtime",
            region_name="us-east-1" ,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )
        self.model_id = "amazon.titan-embed-text-v2:0"

    def embed_documents(self, texts):
        return self._embed(texts)

    def embed_query(self, text):
        # Bedrock expects list input, so wrap single query
        return self._embed([text])[0]

    def __call__(self, texts):
        """
        Make the embedder itself callable for compatibility with FAISS bindings.
        If input is a list, embed documents; otherwise embed a single query.
        """
        if isinstance(texts, list):
            return self.embed_documents(texts)
        return self.embed_query(texts)

    def _embed(self, texts):
        embeddings = []
        for text in texts:
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps({"inputText": text}),
                contentType="application/json",
                accept="application/json",
            )
            result = json.loads(response["body"].read())
            embeddings.append(result["embedding"])
        return embeddings
