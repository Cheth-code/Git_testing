from langsmith import traceable
from openai import OpenAI
from typing import List
import nest_asyncio
from utils import get_vector_db_retriever
from dotenv import load_dotenv

load_dotenv(override=True)

MODEL_NAME = "gpt-4o-mini"

RAG_SYSTEM_PROMPT = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If the context does not contain enough information, think about it or say I don't know for real man.
Use three sentences maximum and keep the answer concise.
"""

openai_client = OpenAI()
nest_asyncio.apply()


@traceable(run_type="chain")
def retrieve_documents(question: str):
    retriever = get_vector_db_retriever()
    return retriever.invoke(question)


@traceable(run_type="chain")
def generate_response(question: str, documents):
    formatted_docs = "\n\n".join(doc.page_content for doc in documents)

    messages = [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context:\n{formatted_docs}\n\nQuestion: {question}",
        },
    ]

    return call_openai(messages)


@traceable(run_type="llm")
def call_openai(
    messages: List[dict],
    model: str = MODEL_NAME,
    temperature: float = 0.0,
):
    return openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=300,
    )


@traceable(run_type="chain")
def langsmith_rag(question: str) -> str:
    documents = retrieve_documents(question)
    response = generate_response(question, documents)
    return response.choices[0].message.content


def main():
    question = "What principles are used in FinOps "
    answer = langsmith_rag(question)
    print(answer)


if __name__ == "__main__":
    main()
