import os, asyncio
from openai import OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_core.caches import BaseCache
from utlis.config import (
    DB_PATH,
    DISTANCE_THRESHOLD,
    require_openai_api_key,
)
from jinja2 import Template
import logging

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def create_llm():
    return ChatOpenAI(
        temperature=0,
        model_name="gpt-3.5-turbo",
        max_tokens=500,
        openai_api_key=require_openai_api_key(),
        cache=None,
    )


def create_openai_client():
    return OpenAI(api_key=require_openai_api_key())

MAX_CONTEXT_DOCS = 3
MAX_CHARS_PER_DOC = 1500


def _truncate_documents(documents, max_docs=MAX_CONTEXT_DOCS, max_chars=MAX_CHARS_PER_DOC):
    """Keep context small by trimming the number and length of documents."""
    if not isinstance(documents, list):
        documents = [documents]

    trimmed = []
    for doc in documents[:max_docs]:
        text = doc if isinstance(doc, str) else str(doc)
        if max_chars and len(text) > max_chars:
            text = text[:max_chars].rstrip() + "..."
        trimmed.append(text)
    return trimmed

'''
Here, three different versions of the fetchGptResponse function are defined.
From the average of testing the time taken for each version, here are the average time:
fetchLangchainResponse: 2.840806246	
fetchGptResponseTwo: 4.176655531	
fetchGptResponse: 3.762374473
'''


async def fetchGptResponse(query, role, data=None):
    # Build a context for rendering templates
    context = {}
    if data is None:
        data = []

    llm = create_llm()

    if isinstance(data, dict):
        context.update(data)

    # ensure common keys are available
    context.setdefault('query', query)
    if 'student_submission' in context and 'submission' not in context:
        context['submission'] = context['student_submission']
    if 'retrieved_context' in context and 'rubric' not in context:
        context['rubric'] = context['retrieved_context']
    # handle alternate keys
    if 'relevant_messages' in context and 'relevant_documents' not in context:
        context['relevant_documents'] = context['relevant_messages']

    # Trim large collections before converting to strings
    for key in ('relevant_documents', 'relevant_messages', 'rubric'):
        if key in context and context[key]:
            context[key] = _truncate_documents(context[key])

    # Render the role prompt if it contains template placeholders.
    role_rendered = role
    try:
        if '{{' in role and '}}' in role:
            role_rendered = Template(role).render(**context)
        elif '{' in role and '}' in role:
            # safe formatting: missing keys become empty string
            class SafeDict(dict):
                def __missing__(self, key):
                    return ''
            role_rendered = role.format_map(SafeDict(context))
    except Exception as e:
        # fallback to original role on any render error
        logging.info(f'fetchGPT role error: {e}')
        role_rendered = role

    # If we rendered the role template, don't append raw data again (to avoid duplication).
    if role_rendered != role:
        system_content = role_rendered
    else:
        system_content = f"{role} Here are the relevant information {str(data)}."

    response = await asyncio.to_thread(
        llm.invoke,
        [
            ("system", system_content),
            ("user", query)
        ]
    )
    return response.content


async def fetchLangchainResponse(query, collection_name, top_k=10):
    llm = create_llm()
    embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002", api_key= require_openai_api_key,)
    # embedding_model = SentenceTransformerEmbeddings(model="all-MiniLM-L6-v2")

    # Initialize the ChromaDB client and retriever
    client = Chroma(
        embedding_function=embedding_model,
        persist_directory=DB_PATH, 
        collection_name=str(collection_name)
    )

    try:
        # Initialize the RetrievalQA chain
        chatbot_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=client.as_retriever(
                search_type="similarity_score_threshold", 
                search_kwargs={"k": top_k, "score_threshold": DISTANCE_THRESHOLD}
            ),
            verbose=True, 
            return_source_documents=True,
        )

        # Define the prompt template
        template = f"""
        Respond as clearly as possible with more than 100 words {query}?
        """

        prompt = PromptTemplate(
            input_variables=["query"],
            template=template,
        )

        # Run the query through the chatbot chain
        response = chatbot_chain.invoke(prompt.format(query=query))

        # Extract and print source documents
        source_documents = response.get('source_documents', [])
        sources = [doc.metadata['source'] for doc in source_documents]
        sources = list(set(sources))
        response["sources"] = sources

        return response

    except Exception as e:
        print(f"Error with fetching Langchain response: {e}")
        return "I'm sorry, I couldn't find an answer to that question."


async def fetchGptResponseTwo(query, role, data=[]):
    if data is None:
        data = []

    client = create_openai_client()
    
    messages = [
        {"role": "system", "content": role},
        {"role": "user", "content": f"Here are the relevant information: {str(data)}"},
        {"role": "user", "content": query},
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0,
            max_tokens=500,
        )

        print(f"Response from OpenAI: {response}")
        assistant_reply = response.choices[0].message.content
        return assistant_reply
    
    except Exception as e:
        print(f"Error with fetching GPT response: {e}")
        return "I'm sorry, I couldn't find an answer to that question."
