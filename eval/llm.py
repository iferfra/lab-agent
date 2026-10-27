import os

from dotenv import load_dotenv
from langchain_aws import ChatBedrockConverse

load_dotenv()


eval_llm = ChatBedrockConverse(
    model_id=os.getenv("EVALUATION_LLM"),
    region_name="us-east-1",
    temperature=0.0,
)
