#!/usr/bin/env python3

"""
LLM Module for meshing-around (modified for Service Account Auth)
This module is used to interact with LLM API to generate responses to user input
K7MHI Kelly Keeton 2024

"""

from modules.log import *

"""
Google Gemini API
https://ai.google.dev/tutorials/python_quickstart
"""

import requests
import json
from googlesearch import search # pip install googlesearch-python
import os # Import os module for API key
import google.auth.transport.requests
import google.oauth2.service_account
from datetime import datetime

"""
This is my attempt at a simple RAG implementation it will require some setup
you will need to have the RAG data in a folder named rag in the data directory (../data/rag)
This is lighter weight and can be used in a standalone environment, needs chromadb
"chat with a file" is the use concept here, the file is the RAG data
is anyone using this please let me know if you are Dec62024 -kelly
"""

ragDEV = False

if ragDEV:
    import chromadb # pip install chromadb
    # from ollama import Client as OllamaClient # ollama import removed
    # ollamaClient = OllamaClient(host=ollamaHostName) # ollama client removed
    import chromadb # pip install chromadb
    try:
        import ollama # pip install ollama - keep ollama for embeddings for now for RAG
    except ImportError:
        logger.warning("Ollama library not found, RAG embeddings will not work. Please install ollama if RAG is needed.")
        ragDEV = False # disable RAG if ollama lib is missing


"""
LLM System Variables
"""

# ollamaAPI = ollamaHostName + "/api/generate" # old ollama API - not used
geminiAPI = ollamaHostName + ":generateContent" # Gemini API endpoint
openaiAPI = "https://api.openai.com/v1/completions" # not used, if you do push a enhancement!
llmEnableHistory = True # enable last message history for the LLM model
llmContext_fromGoogle = True # enable context from google search results adds to compute time but really helps with responses accuracy
googleSearchResults = 3 # number of google search results to include in the context more results = more compute time
antiFloodLLM = []
llmChat_history = {}
trap_list_llm = ("ask:", "askai")

meshBotAI = """
    FROM {llmModel}
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

# --- Service Account Authentication Setup ---
# IMPORTANT: Replace with the actual path to your downloaded Service Account key JSON file
SERVICE_ACCOUNT_KEY_FILE = "./meshbotai-2438a94d61ee.json"  # <-- REPLACE THIS PATH if needed

def get_gemini_access_token():
    """Gets an access token using the Service Account key file with explicit scope."""
    try:
        scopes = ["https://www.googleapis.com/auth/generative-language"] # Add the Gemini API scope
        credentials = google.oauth2.service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_KEY_FILE, scopes=scopes
        )
        request = google.auth.transport.requests.Request()
        credentials.refresh(request) # Refresh the token if needed
        return credentials.token
    except Exception as e:
        logger.error(f"System: Failed to get Gemini access token: {e}")
        return None


def llm_readTextFiles():
    # read .txt files in ../data/rag
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
        # store each document in a vector embedding database
        for i, d in enumerate(text):
            response = ollama.embeddings(model="mxbai-embed-large", prompt=d) # using ollama embeddings for now
            embedding = response["embedding"]
            collection.add(
                ids=[str(i)],
                embeddings=[embedding],
                documents=[d]
            )

    except Exception as e:
        logger.debug(f"System: Embedding failed: {e}")
        return False

# INITALIZATION of RAG

if ragDEV:
    try:
        chromaHostname = "localhost:8000"
        # connect to the chromaDB
        chromaHost = chromaHostname.split(":")[0]
        chromaPort = chromaHostname.split(":")[1]
        if chromaHost == "localhost" and chromaPort == "8000":
            # create a client using local python Client
            chromaClient = chromadb.Client()
        else:
            # create a client using the remote python Client
            # this isnt tested yet please test and report back
            chromaClient = chromadb.Client(host=chromaHost, port=chromaPort)

        clearCollection = False
        if "meshBotAI" in chromaClient.list_collections() and clearCollection:
            logger.debug(f"System: LLM: Clearing RAG files from chromaDB")
            chromaClient.delete_collection("meshBotAI")

        # create a new collection
        collection = chromaClient.create_collection("meshBotAI")

        logger.debug(f"System: LLM: Cataloging RAG data")
        store_text_embedding(llm_readTextFiles())

    except Exception as e:
        logger.debug(f"System: LLM: RAG Initalization failed: {e}")

def query_collection(prompt):
    # generate an embedding for the prompt and retrieve the most relevant doc
    response = ollama.embeddings(prompt=prompt, model="mxbai-embed-large") # using ollama embeddings for now
    results = collection.query(query_embeddings=[response["embedding"]], n_results=1)
    data = results['documents'][0][0]
    return data

from datetime import datetime # import datetime if not already imported

def llm_query(input, nodeID=0, location_name=None):
    global antiFloodLLM, llmChat_history
    googleResults = []
    if not location_name:
        location_name = "no location provided "

    # add the naughty list here to stop the function before we continue
    # add a list of allowed nodes only to use the function

    # anti flood protection
    if nodeID in antiFloodLLM:
        return "Please wait before sending another message"
    else:
        antiFloodLLM.append(nodeID)

    if llmContext_fromGoogle:
        # grab some context from the internet using google search hits (if available)
        # localization details at https://pypi.org/project/googlesearch-python/

        # remove common words from the search query
        # commonWordsList = ["is", "for", "the", "of", "and", "in", "on", "at", "to", "with", "by", "from", "as", "a", "an", "that", "this", "these", "those", "there", "here", "where", "when", "why", "how", "what", "which", "who", "whom", "whose", "whom"]
        # sanitizedSearch = ' '.join([word for word in input.split() if word.lower() not in commonWordsList])
        try:
            googleSearch = search(input, advanced=True, num_results=googleSearchResults)
            if googleSearch:
                for result in googleSearch:
                    # SearchResult object has url= title= description= just grab title and description
                    googleResults.append(f"{result.title} {result.description}")
            else:
                googleResults = ['no other context provided']
        except Exception as e:
            logger.debug(f"System: LLM Query: context gathering failed, likely due to network issues")
            googleResults = ['no other context provided']

    history = llmChat_history.get(nodeID, ["", ""])

    if googleResults:
        logger.debug(f"System: Google-Enhanced LLM Query: {input} From:{nodeID}")
    else:
        logger.debug(f"System: LLM Query: {input} From:{nodeID}")

    response = ""
    result = ""
    location_name += f" at the current time of {datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z')}"

    try:
        # RAG context inclusion testing
        ragContext = False
        if ragDEV:
            ragContext = query_collection(input)

        if ragContext:
            ragContextGooogle = ragContext + '\n'.join(googleResults)
            # Build the query from the template
            modelPrompt = meshBotAI.format(input=input, context=ragContext, location_name=location_name, llmModel=llmModel, history=history)
            # Query the model with RAG context - Gemini API Call


        else:
            # Build the query from the template
            modelPrompt = meshBotAI.format(input=input, context='\n'.join(googleResults), location_name=location_name, llmModel=llmModel, history=history)
            # Query the model via Gemini web API - Gemini API Call


        # --- Get Access Token using Service Account ---
        access_token = get_gemini_access_token()
        if not access_token:
            return "⛔️Failed to obtain Gemini access token. Check service account key file and permissions."

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {access_token}' # Use the access token in the header
        }
        gemini_query_data = {
            "contents": [{"parts": [{"text": modelPrompt}]}]
        }
        result_raw = requests.post(geminiAPI, headers=headers, data=json.dumps(gemini_query_data))

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


        #logger.debug(f"System: LLM Response: " + result.strip().replace('\n', ' '))
    except Exception as e:
        logger.warning(f"System: LLM failure: {e}")
        return "⛔️I am having trouble processing your request, please try again later."

    # cleanup for message output
    response = result.strip().replace('\n', ' ')
    # done with the query, remove the user from the anti flood list
    antiFloodLLM.remove(nodeID)

    if llmEnableHistory:
        llmChat_history[nodeID] = [input, response]

    return response
