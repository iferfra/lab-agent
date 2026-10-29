import json
import os

import boto3
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

# Optional: leave RERANK_MODEL empty to rank chunks by vector similarity only.
RERANK_MODEL = os.getenv("RERANK_MODEL")
bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")


def rerank(query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
    """(index, relevance) of the top_n documents for the query, best first.

    Called through bedrock-runtime because the Bedrock Rerank API used by
    langchain-aws does not accept Bedrock API keys.
    """
    response = bedrock.invoke_model(
        modelId=RERANK_MODEL,
        body=json.dumps(
            {"query": query, "documents": documents, "top_n": top_n, "api_version": 2}
        ),
    )
    results = json.loads(response["body"].read())["results"]
    return [(r["index"], r["relevance_score"]) for r in results]
