# app/utils/llm_factory.py
import os
from langchain_openai import ChatOpenAI
# 필요하면 HuggingFaceHub import

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME")

def get_llm_instance(temperature: float = 0.5):
    """OpenAI 또는 HF 모델을 반환."""
    if LLM_PROVIDER == "hf":
        # from langchain import HuggingFaceHub
        # return HuggingFaceHub(repo_id=LLM_MODEL_NAME, model_kwargs={"temperature": temperature})
        return ChatOpenAI(
            model_name=LLM_MODEL_NAME,
            temperature=temperature,
            max_tokens=1000,
            openai_api_base="http://localhost:12000/v1",
        )
    return ChatOpenAI(model_name=LLM_MODEL_NAME, temperature=temperature, max_tokens=1000)

