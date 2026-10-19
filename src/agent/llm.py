from langchain_aws import BedrockEmbeddings, ChatBedrockConverse

agent_llm = ChatBedrockConverse(
    model_id="us.amazon.nova-lite-v1:0",
    region_name="us-east-1",
    temperature=0.0,
)

embedder = BedrockEmbeddings(
    region_name="us-east-1", model_id="amazon.titan-embed-text-v2:0", normalize=True
)
