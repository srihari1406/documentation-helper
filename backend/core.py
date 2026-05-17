import os
from typing import Any, Dict
from dotenv import load_dotenv
from langchain.agents import create_tool_calling_agent, tool, AgentExecutor, create_react_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import ToolMessage, HumanMessage, SystemMessage
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = PineconeVectorStore(index_name=os.getenv("INDEX_NAME"), embedding=embeddings)

model = init_chat_model(f"openai:gpt-4o-mini", temperature=0)

_extracted_artifacts = []
@tool(response_format="content_and_artifact")
def retrieve_context(query):
    """Retrieve relevant documentation to help answer user queries about langchain."""
    retrieved_docs = vector_store.as_retriever().invoke(query, k=4)
    serialized = "\n\n".join([f"Source: {doc.metadata.get('source', 'unknown')}\n\nContent: {doc.page_content}" for doc in retrieved_docs])
    _extracted_artifacts.extend(retrieved_docs)
    return serialized, retrieved_docs

def run_llm(query):
    """
    Run the RAG pipeline to answer a query using retrived documentation.
    
    Args:
        query: The user query to answer.

    Returns:
        Dict: A dictionary containing the answer and context retrived from relevant artifacts.
    
    """
    prompt = ChatPromptTemplate.from_messages([
        ('system', (
            "You are a helpful assistant that answers questions about the Langchain documentation. "
            "You have access to the following tools to retrieve relevant documentation to help answer user queries about langchain. "
            "Use the tools to find relevant information to answer the user's question. "
            "Always cite the sources you use in your answer. "
            "If you cannot find the answer in the documentation, say you don't know."
        )),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    agent = create_tool_calling_agent(llm=model, tools=[retrieve_context], prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=[retrieve_context], verbose=True, return_intermediate_steps=True)

    response = agent_executor.invoke({"input": query})
    answer = response["output"]

    context_docs = list(_extracted_artifacts)

    return {
        "answer": answer,
        "context": context_docs
    }

if __name__ == "__main__":
    result = run_llm("what are deep agents?")
    print(result)