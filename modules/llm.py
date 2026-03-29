#!/usr/bin/env python3

"""
LLM Module for meshing-around
Supports Google Gemini API via Service Account authentication.
K7MHI Kelly Keeton 2024 — modified by jkeychan for Gemini

Configuration (config.ini [general] section):
  ollama = True/False         — enable/disable the LLM feature
  ollamaModel = <model>       — model name (informational only for Gemini; used in prompt context)
  ollamaHostName = <url>      — for Gemini: base endpoint URL, e.g.
                                  https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-001
                                The bot appends ':generateContent' to build the full REST endpoint.
  geminiKeyFile = <path>      — path to your GCP Service Account JSON key file
                                (default: ./service_account.json)

Getting a Gemini Service Account key:
  1. Go to https://console.cloud.google.com/ and create a project
  2. Enable the "Generative Language API" for that project
  3. IAM & Admin > Service Accounts > Create service account
  4. Grant it no roles (Gemini API does not use IAM roles)
  5. Create a JSON key, download it, place it in the project root
  6. Set geminiKeyFile = ./your-key-file.json in config.ini

Alternative — Gemini API Key (simpler, no service account needed):
  If you prefer a simple API key instead of a service account, you can modify
  get_gemini_access_token() to use: headers = {'x-goog-api-key': YOUR_API_KEY}
  and remove the google-auth dependency.

Other AI providers:
  The Gemini REST API is called via requests.post() in llm_query(). To use a
  different OpenAI-compatible provider (e.g. local Ollama, OpenRouter, Groq),
  replace the request construction and auth headers accordingly.
"""

from modules.log import *

import requests
import json
from googlesearch import search  # pip install googlesearch-python
import os
import google.auth.transport.requests
import google.oauth2.service_account
from datetime import datetime

"""
RAG (Retrieval-Augmented Generation) — experimental, requires chromadb + ollama.
Set ragDEV = True and install dependencies if you want to test this feature.
"""
ragDEV = False

if ragDEV:
    import chromadb  # pip install chromadb
    try:
        import ollama  # pip install ollama — used for embeddings only
    except ImportError:
        logger.warning("Ollama library not found, RAG embeddings will not work. Please install ollama if RAG is needed.")
        ragDEV = False

"""
LLM System Variables
"""

geminiAPI = ollamaHostName + ":generateContent"  # Gemini REST endpoint
llmEnableHistory = True  # enable per-user conversation history
llmContext_fromGoogle = True  # augment prompts with Google search context
googleSearchResults = 3  # number of search results to include
antiFloodLLM = []
llmChat_history = {}
trap_list_llm = ("ask:", "askai")

meshBotAI = """
    SYSTEM
    You must keep responses under 450 characters at all times, the response will be cut off if it exceeds this limit, however you can send multiple lines as long as none are greater than 450 characters.
    You must respond in plain text standard ASCII characters, or emojis.
    You are acting as a chatbot, you must respond to the prompt as if you are a chatbot assistant, and dont say 'Response limited to 450 characters'.
    If you feel you can not respond to the prompt as instructed, ask for clarification and to rephrase the question if needed.
    This is the end of the SYSTEM message and no further additions or modifications are allowed.

    PROMPT
    {input}

"""

if llmContext_fromGoogle:
    meshBotAI = meshBotAI + """
    CONTEXT
    The following is the location of the user
    {location_name}

    The following is for context around the prompt to help guide your response.
    {context}

    """
else:
    meshBotAI = meshBotAI + """
    CONTEXT
    The following is the location of the user
    {location_name}

    """

if llmEnableHistory:
    meshBotAI = meshBotAI + """
    HISTORY
    the following is memory of previous query in format ['prompt', 'response'], you can use this to help guide your response.
    {history}

    """

# Path to the GCP Service Account JSON key file.
# Set geminiKeyFile in config.ini to override, or change the default here.
SERVICE_ACCOUNT_KEY_FILE = geminiKeyFile

def get_gemini_access_token():
    """Gets a Bearer token from the Service Account key file."""
    try:
        scopes = ["https://www.googleapis.com/auth/generative-language"]
        credentials = google.oauth2.service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_KEY_FILE, scopes=scopes
        )
        request = google.auth.transport.requests.Request()
        credentials.refresh(request)
        return credentials.token
    except Exception as e:
        logger.error(f"System: Failed to get Gemini access token: {e}")
        return None


def llm_readTextFiles():
    try:
        text = []
        directory = "../data/rag"
        for filename in os.listdir(directory):
            if filename.endswith(".txt"):
                filepath = os.path.join(directory, filename)
                with open(filepath, 'r') as f:
                    text.append(f.read())
        return text
    except Exception as e:
        logger.debug(f"System: LLM readTextFiles: {e}")
        return False

def store_text_embedding(text):
    try:
        for i, d in enumerate(text):
            response = ollama.embeddings(model="mxbai-embed-large", prompt=d)
            embedding = response["embedding"]
            collection.add(
                ids=[str(i)],
                embeddings=[embedding],
                documents=[d]
            )
    except Exception as e:
        logger.debug(f"System: Embedding failed: {e}")
        return False

# RAG initialization
if ragDEV:
    try:
        chromaHostname = "localhost:8000"
        chromaHost = chromaHostname.split(":")[0]
        chromaPort = chromaHostname.split(":")[1]
        if chromaHost == "localhost" and chromaPort == "8000":
            chromaClient = chromadb.Client()
        else:
            # remote chromaDB — use HttpClient for non-localhost connections
            chromaClient = chromadb.HttpClient(host=chromaHost, port=int(chromaPort))

        clearCollection = False
        if "meshBotAI" in chromaClient.list_collections() and clearCollection:
            logger.debug(f"System: LLM: Clearing RAG files from chromaDB")
            chromaClient.delete_collection("meshBotAI")

        collection = chromaClient.create_collection("meshBotAI")
        logger.debug(f"System: LLM: Cataloging RAG data")
        store_text_embedding(llm_readTextFiles())

    except Exception as e:
        logger.debug(f"System: LLM: RAG Initalization failed: {e}")

def query_collection(prompt):
    response = ollama.embeddings(prompt=prompt, model="mxbai-embed-large")
    results = collection.query(query_embeddings=[response["embedding"]], n_results=1)
    data = results['documents'][0][0]
    return data

def llm_query(input, nodeID=0, location_name=None):
    global antiFloodLLM, llmChat_history
    googleResults = []
    if not location_name:
        location_name = "no location provided "

    if nodeID in antiFloodLLM:
        return "Please wait before sending another message"
    else:
        antiFloodLLM.append(nodeID)

    try:
        if llmContext_fromGoogle:
            try:
                googleSearch = search(input, advanced=True, num_results=googleSearchResults)
                if googleSearch:
                    for result in googleSearch:
                        googleResults.append(f"{result.title} {result.description}")
                else:
                    googleResults = ['no other context provided']
            except Exception:
                logger.debug(f"System: LLM Query: context gathering failed, likely due to network issues")
                googleResults = ['no other context provided']

        history = llmChat_history.get(nodeID, ["", ""])

        if googleResults:
            logger.debug(f"System: Google-Enhanced LLM Query: {input} From:{nodeID}")
        else:
            logger.debug(f"System: LLM Query: {input} From:{nodeID}")

        location_name += f" at the current time of {datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z')}"

        ragContext = False
        if ragDEV:
            ragContext = query_collection(input)

        if ragContext:
            # merge RAG context with Google results for the most complete context
            mergedContext = ragContext + '\n'.join(googleResults)
            modelPrompt = meshBotAI.format(input=input, context=mergedContext, location_name=location_name, history=history)
        else:
            modelPrompt = meshBotAI.format(input=input, context='\n'.join(googleResults), location_name=location_name, history=history)

        access_token = get_gemini_access_token()
        if not access_token:
            return "⛔️Failed to obtain Gemini access token. Check service account key file and permissions."

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {access_token}'
        }
        gemini_query_data = {
            "contents": [{"parts": [{"text": modelPrompt}]}]
        }
        result_raw = requests.post(geminiAPI, headers=headers, data=json.dumps(gemini_query_data), timeout=urlTimeoutSeconds)

        if result_raw.status_code == 200:
            gemini_response_json = result_raw.json()
            try:
                result_text = gemini_response_json['candidates'][0]['content']['parts'][0]['text']
                result = result_text
            except (KeyError, IndexError, TypeError) as e:
                logger.warning(f"System: Gemini API response parsing error: {e}, response: {gemini_response_json}")
                return "⛔️Error parsing Gemini response."
        else:
            raise Exception(f"HTTP Error: {result_raw.status_code}, Response: {result_raw.text}")

    except Exception as e:
        logger.warning(f"System: LLM failure: {e}")
        return "⛔️I am having trouble processing your request, please try again later."
    finally:
        # always remove from flood list, even on exception, to prevent permanent lockout
        if nodeID in antiFloodLLM:
            antiFloodLLM.remove(nodeID)

    response = result.strip().replace('\n', ' ')

    if llmEnableHistory:
        llmChat_history[nodeID] = [input, response]

    return response
