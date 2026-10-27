import os

from dotenv import load_dotenv
from langchain_aws import BedrockEmbeddings, ChatBedrockConverse

load_dotenv()

agent_llm = ChatBedrockConverse(
    model_id=os.getenv("AGENT_LLM"),
    region_name="us-east-1",
    temperature=0.0,
)

embedder = BedrockEmbeddings(
    region_name="us-east-1", model_id="amazon.titan-embed-text-v2:0", normalize=True
)
