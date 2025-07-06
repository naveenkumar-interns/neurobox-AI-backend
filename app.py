## added follow up question and increased speed of response
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import ast
from datetime import datetime
import pymongo
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from typing import List
from pymongo import MongoClient
from multiprocessing import Pool, cpu_count
from pymongo.operations import UpdateOne
import tqdm
import time
from pinecone import Pinecone
from pinecone import ServerlessSpec  # Import ServerlessSpec
import numpy as np  # Import NumPy for handling NaN
from sentence_transformers import SentenceTransformer


# hf_token = "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"
MONGO_URI = "mongodb+srv://jsckson_store:jsckson_store@cluster0.9a981.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(MONGO_URI)

# model = HuggingFaceInferenceAPIEmbeddings(api_key=hf_token, model_name="sentence-transformers/all-MiniLM-L6-v2")

model = SentenceTransformer('all-MiniLM-L6-v2')


Pinecone_index = None

app = Flask(__name__)
CORS(app)




# Move generate_embedding_batch to module level
def generate_embedding_batch(doc_batch, model):
    
    try:
        texts = []
        valid_docs = []
        for doc in doc_batch:
            desc = doc.get('expanded_description', '')
            if desc and isinstance(desc, str):
                texts.append(desc)
                valid_docs.append(doc)
            else:
                print(f"Skipping document {doc['_id']}: Invalid or missing expanded_description")
        if not texts:
            return []
        embeddings = model.encode(texts, batch_size=128).tolist()
        return [(doc['_id'], embedding) for doc, embedding in zip(valid_docs, embeddings)]
    except Exception as e:
        print(f"Error processing batch: {e}")
        return []

# Wrapper for multiprocessing
def process_batch(batch):
    global model
    return generate_embedding_batch(batch, model)

def build_embedding(database_name, collection_name):
    MONGO_URI = "mongodb+srv://sudhakaran:URvEVWjORGTkaeaq@cluster0.znyhl.mongodb.net/chatbot?retryWrites=true&w=majority&appName=Cluster0"
    global model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    client = MongoClient(MONGO_URI)
    db = client[database_name]
    collection = db[collection_name]

    # Process documents in chunks
    chunk_size = 10000
    total_docs = collection.count_documents({})
    embeddings = []

    # Initialize multiprocessing pool
    num_workers = cpu_count()
    pool = Pool(processes=num_workers)

    for skip in range(0, total_docs, chunk_size):
        documents = list(collection.find().skip(skip).limit(chunk_size))
        print(f"Processing documents {skip + 1} to {skip + len(documents)}")
        
        # Process embeddings in parallel
        batch_size = 256
        batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]
        print(f"Processing {len(batches)} batches in parallel with {num_workers} workers")
        
        # Use multiprocessing to process batches
        batch_embeddings = pool.map(process_batch, batches)
        for batch_result in batch_embeddings:
            embeddings.extend(batch_result)

    pool.close()
    pool.join()

    # Bulk update
    bulk_chunk_size = 1000
    bulk_operations = [
        UpdateOne({'_id': doc_id}, {'$set': {'embedding': embedding, "id": doc_id}}, upsert=True)
        for doc_id, embedding in embeddings
    ]

    print(f"Performing bulk write for {len(bulk_operations)} documents")
    for i in range(0, len(bulk_operations), bulk_chunk_size):
        try:
            collection.bulk_write(
                bulk_operations[i:i + bulk_chunk_size],
                ordered=False
            )
            print(f"Updated {min(i + bulk_chunk_size, len(bulk_operations))} documents")
        except Exception as e:
            print(f"Error during bulk write: {e}")

def build_pinecone_vectorstore(database_name, collection_name):
    Pinecone_api_key = "pcsk_3XafLm_SDzfsZm5fmrpXPafmpaUaJydXGr4KucreMZpGay5Uz84MAY4mk9tKYqKTeNUMrp"
    #AWS PAID
    # Pinecone_api_key = "pcsk_KLfb5_EKsAGk8CtbDfJ1EqRu5TaLtGxaU5QA5ZA6nmamewv5N4S9Fqa8SeGmRHsyzJPAu" 
    MONGO_URI = "mongodb+srv://sudhakaran:URvEVWjORGTkaeaq@cluster0.znyhl.mongodb.net/chatbot?retryWrites=true&w=majority&appName=Cluster0"

    client = MongoClient(MONGO_URI)
    db = client[database_name]
    collection = db[collection_name]
    
    pc = Pinecone(api_key=Pinecone_api_key)
    index_name = collection_name
    
    # Create index only if it doesn't exist
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
    index = pc.Index(index_name)

    batch_size = 500
    documents = collection.find().batch_size(batch_size)
    to_upsert = []

    print(documents[0])

    for doc in documents:
        id_value = f"id_{str(doc['_id'])}"
        embedding = doc['embedding']
        # metadata = {k: v for k, v in doc.items() if k not in ['_id', 'embedding']}
        metadata = {k: v for k, v in doc.items() if k not in ['embedding']}
        metadata = {
            k: "" if v is None or (isinstance(v, float) and np.isnan(v)) else v
            for k, v in metadata.items()
        }
        metadata = {k: str(v) for k, v in metadata.items()}
        to_upsert.append({"id": id_value, "values": embedding, "metadata": metadata})
        
        if len(to_upsert) >= batch_size:
            index.upsert(vectors=to_upsert)
            to_upsert = []

    if to_upsert:
        index.upsert(vectors=to_upsert)

    global Pinecone_index
    Pinecone_index = index
    print("Pinecone index data upserted successfully!")

# def get_recommendations(query, top_k=10):
#     # model = HuggingFaceInferenceAPIEmbeddings(api_key=hf_token, model_name="sentence-transformers/all-MiniLM-L6-v2")

#     model = SentenceTransformer('all-MiniLM-L6-v2')
#     query_embedding = model.encode(query).tolist()
#     result = Pinecone_index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
#     return [match['metadata'] for match in result['matches']]

@app.route('/build_embeddings', methods=['POST'])
def get_database_and_collection():
    data = request.json
    database_name = data.get('database_name')
    collection_name = data.get('collection_name')
    if not database_name or not collection_name:
        return jsonify({"error": "Please provide both database_name and collection_name"}), 400
    try:
        build_embedding(database_name, collection_name)
    except Exception as e:
        return jsonify({"error": f"Error in build_embedding: {str(e)}"}), 500
    
    try:
        build_pinecone_vectorstore(database_name, collection_name)
    except Exception as e:
        return jsonify({"error": f"Error in build_pinecone_vectorstore: {str(e)}"}), 500
    
    return jsonify({"message": "Embeddings built and stored in Pinecone successfully!"}), 200

# @app.route('/test', methods=['POST'])
# def test():
#     if Pinecone_index is None:
#         return jsonify({"error": "Pinecone index not initialized. Please run /build_embeddings first."}), 400
#     data = request.json
#     query = data.get('query')
#     if not query:
#         return jsonify({"error": "Please provide a query"}), 400
#     try:
#         recommendations = get_recommendations(query)
#         return jsonify({"recommendations": recommendations}), 200
#     except Exception as e:
#         return jsonify({"error": f"Error in get_recommendations: {str(e)}"}), 500



GOOGLE_API_KEY = "AIzaSyDR5hSTYjo6jbiTpHw8AEKZsuRVEEFcAJk"
# GOOGLE_API_KEY = "AIzaSyCC_ijBLd3X7K07A6855ZGaelFC2gYX93U"
hf_token = "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"
MONGO_URI = "mongodb+srv://jsckson_store:jsckson_store@cluster0.9a981.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

model_1 = "gemini-1.5-flash"
model_2 = "gemini-2.0-pro-exp-02-05"
model_3 = "gemini-2.0-flash-lite"


client = pymongo.MongoClient(MONGO_URI)
db = client["jacksonHardwareDB"]
collection = db["inventory"]
user_client = pymongo.MongoClient("mongodb+srv://sudhakaran:URvEVWjORGTkaeaq@cluster0.znyhl.mongodb.net/chatbot?retryWrites=true&w=majority&appName=Cluster0")
user_db = user_client["chatbot"]

chat_history_collection = user_db["chats"]
q_collection = user_db["chats"]


llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-8b",
    temperature=0.7,
    max_tokens=1000,
    timeout=None,
    max_retries=2,
    google_api_key=GOOGLE_API_KEY,
)



embedding_cache ={}

def generate_embedding(text: str) -> List[float]:
    if text in embedding_cache:
        return embedding_cache[text]
    embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=hf_token, model_name="sentence-transformers/all-MiniLM-L6-v2")
    response = embeddings.embed_query(text)
    embedding_cache[text]=response
    return response


def convert_to_json(data):
    result = []
    forai = []
    for product in data:
        # Filter out unnecessary keys from metadata
        product_info = {
        'id': product.get('_id'),
        'title': product.get('title'),
        'description': product.get('description'),
        'product_type': product.get('product_type'),
        'link': product.get('link'),
        'image_list': product.get('image_list'),
        'price': product.get('price'),
        'inventory_quantity': product.get('inventory_quantity'),
        'vendor': product.get('vendor')
        }
        result.append(product_info)

    # print("\n\nresult2 : ",result)

    return result,forai


def get_product_search(query,index_name="store1", top_k=15):
    Pinecone_api_key = "pcsk_Y3jUe_EJfcnfr59PqNoiPMB2tRwhDSKDnj8eEwz1aD56hnymYp9Reujen2y9AT57rGtfz"
    # Pinecone_api_key = "pcsk_KLfb5_EKsAGk8CtbDfJ1EqRu5TaLtGxaU5QA5ZA6nmamewv5N4S9Fqa8SeGmRHsyzJPAu"
    pc = Pinecone(api_key=Pinecone_api_key)
    Pinecone_index = pc.Index(index_name)
    # model = HuggingFaceInferenceAPIEmbeddings(api_key=hf_token, model_name="sentence-transformers/all-MiniLM-L6-v2")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    query_embedding = model.encode(query).tolist()
    result = Pinecone_index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    results = [match['metadata'] for match in result['matches']]
    print("results : ",results)
    return convert_to_json(results)

def analyze_intent(chat_history,store_name,store_description):
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-8b",  # Assuming this is a valid model; adjust if needed
        temperature=0.7,
        max_tokens=60,
        timeout=None,
        max_retries=2,
        google_api_key=GOOGLE_API_KEY
    )
    try:

        prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a senior data analyst for a {store_name} chatbot. Analyze the **latest user query** in the provided chat history and categorize its intent as:
        1. **'product'** (asking about products, product availability, product details, or scenario-based queries related to using or purchasing products),
        2. **'website/company'** (asking about the website, company, store operations, services, policies, or FAQs),
        3. **'general'** (queries unrelated to products, website, or company).
        4. **list** (asking for a list of all products or product categories, e.g., "list all tools","show all products," "what are all the gardening tools?").


        Return only the category name (e.g., "product", "website/company", or "general"). Do not include preambles, explanations, or additional text.

        **Instructions:**
        - Focus on the **most recent user query** in the chat history for intent categorization.
        - Use the chat history for context if the latest query references prior messages (e.g., follow-up questions like "What about blue ones?" referring to products).
        - If the latest query is ambiguous but relates to a prior product or website/company discussion, infer the intent based on the context.
        - If no clear context is provided or the query is standalone, categorize it independently.

        **Examples:**
        - Chat History:
          - User: "I need a hammer for hanging a shelf."
          - Assistant: "We have several hammers suitable for that."
          - User: "Do you have any in red?"
          → product
        - Chat History:
          - User: "What dresses do you have in size medium?"
          - Assistant: "We have floral and solid color dresses in medium."
          - User: "Can you deliver them?"
          → website/company
        - Chat History:
          - User: "Do you have organic apples?"
          - Assistant: "Yes, we stock organic apples."
          - User: "What’s your return policy?"
          → website/company
        - Chat History:
          - User: "I am looking for gardening tools."
          - Assistant: "What type of tools are you looking for?"
          - User: "What type actually used for gardening?"
          → list
        - Chat History:
          - User: "Show me all your products."
          - Assistant: "We offer a full catalog on our website."
          - User: "Can you list all the items?"
          → list
        - Chat History:
          - User: "I am looking for gardening tools."
          - Assistant: "What type of tools are you looking for?"
          - User: "show me all types"
          → list
        - Chat History:
          - User: "I’m looking for running shoes under $50."
          - Assistant: "We have a few options under $50."
          - User: "Any deals on those?"
          → product
        - Chat History:
          - User: "What are all the items you sell?"
          - Assistant: "We have a wide range of products; see our catalog."
          - User: "Can you list your electronics?"
          → list
        - Chat History:
          - User: "How do I fix a leaky faucet?"
          - Assistant: "You’ll need a wrench and a washer."
          - User: "Do you sell those?"
          → product
        - Chat History:
          - User: "What are your store hours?"
          - Assistant: "We’re open 9 AM to 9 PM daily."
          - User: "Do you have a list of all your departments?"
          → website/company
        - Chat History:
        - User: "i am looking for drills and saw"
        - Assistant: "We have Bosch jig saw blades, a Hida decora saw, and a few hole saws and ground rod drivers"
        - User: "show only drills alone"
        → product

        **FAQs to Categorize as 'website/company':**
        - Queries about store locations, hours, showrooms, or departments
        - Queries about delivery, returns, or payment methods
        - Queries about promotions, discounts, or loyalty programs
        - Queries about job applications or company information
        - Queries about website functionality or online ordering
        - Queries about services like repairs, rentals, or special orders
        """
    ),
    ("human", "{chat_history}")
])
        chain = prompt | llm
        response = chain.invoke({"chat_history": chat_history, "store_name": store_name}) 
        print("\n",response.content)
        return response.content.strip()
    except Exception as e:
        print(f"Error in analyze_intent: {str(e)}")
        raise

def research_intent(chat_history,store_name,store_description):
    llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-8b",
    temperature=0.7,
    max_tokens=60,
    timeout=None,
    max_retries=2,
    google_api_key=GOOGLE_API_KEY
    )

    try:
        prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a senior research assistant for a {store_name} chatbot. Analyze chat history to track the user's current topic (e.g., clothing, footwear, groceries, hardware). Accumulate filters (e.g., product type, size, color, price) until the topic shifts to a new project or product category, then reset filters. Respond with a single phrase (max 10 tokens) summarizing the latest request, prioritizing the most recent input.

 "Examples:\n"        
        "1. History:\n"
        "   User: 'I need to fix a leaky faucet. What tools do I need?' Bot: 'Fixing a faucet? Need a pipe wrench or faucet?' User: 'Tools to take it apart, not replace.' Bot: 'Power or hand tools?' User: 'Hand tools, something simple.'\n"
        "   Output: 'simple hand tools for faucet repair'\n"
        "2. history:\n"
            user:i need a premium luxury pen.\n" bot:Check out our Montblanc Meisterstück. It's a classic luxury choice.\n"
        "   Output: 'Montblanc Meisterstück pen'\n"
        "3. History:\n"
        "   User: 'I’m building a deck and need wood.' Bot: 'What materials for the deck?' User: 'Cedar lumber, and screws.' Bot: 'What size screws?' User: '2-inch screws, under $20.'\n"
        "   Output: 'cedar lumber and 2-inch screws'\n"
        "4. History:\n"
        "   User: 'I want to hang a shelf. What do I need?' Bot: 'Screws or wall anchors?' User: 'Wall anchors for drywall.' Bot: 'What weight for the shelf?' User: 'Heavy, maybe 50 pounds.' Bot: 'Need a drill?' User: 'Yes, cordless.'\n"
        "   Output: 'cordless drill for heavy shelf'\n"
        "5. History:\n"
        "   User: 'I need paint for my living room.' Bot: 'Indoor or outdoor paint?' User: 'Indoor, matte finish.' Bot: 'What color?' User: 'Light gray, one gallon.'\n"
        "   Output: 'light gray matte indoor paint'\n"
        "6. History:\n"
        "   User: 'I need a heater.' Bot: 'Gas or electric heater?' User: 'I need a 220V one.' Bot: 'Any size preference?' User: 'In black, small.' Bot: 'Check stock for black heaters?' User: 'Can you list tables?'\n"
        "   Output: 'tables'\n"
        "7. History:\n"
        "   User: 'Show me lighting options.' Bot: 'Ceiling or outdoor lights?' User: 'Ceiling, with 500 lumens.' Bot: 'Any style preference?' User: 'Show me storage cabinets.'\n"
        "   Output: 'storage cabinets'\n"

        "Analyze the conversation and return the summarizing word or phrase."""

    ),
    ("human", "{chat_history}")
])
        


        chain = prompt | llm

        response = chain.invoke({"chat_history": chat_history,"store_name":store_name})


        print("\nresearch_intent",response.content)

        return response.content.strip()
    except Exception as e:
        print(f"Error in research_intent: {str(e)}")
        raise

def prioritize_products(user_intent,products):
    products = list(products)    

    print("\n\nproducts : ", products)

    

    llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0.7,
    max_tokens=10000,
    timeout=None,
    max_retries=2,
    google_api_key=GOOGLE_API_KEY,
)

    input_str = f"User asks for : '{user_intent}'\n Products we have: {json.dumps(products, indent=2)}"
    try:
        prompt = """ Role: You are a Product Prioritization Expert that returns ONLY: id, specializing in ranking products based on user intent, price constraints, and relevance. First, analyze the user intent and product list provided thoroughly. Your task is to filter, reorder, and return the most relevant products that match the user's intent, budget, color, product type, and other features. Move unrelated items to the end of the list.

Rules for Prioritization:
1. **Match User Intent**:
   - Prioritize products that contain keywords from the user's intent in the title, description, or product type.
   - Stronger keyword matches (e.g., exact matches in the title) should rank higher.
   
2. **Apply Price Constraints**:
   - If a price limit is specified (e.g., "under $30"), exclude products exceeding this threshold.
   - If no price limit is provided, ignore this rule.
   
3. **Sort Order**:
   - First, sort by **intent relevance** (strongest keyword matches first).
   - Then, sort by **price** (low to high) within products of equal relevance.
   
4. **Output Format**:
   - Return a JSON array of the relevant products, do not alter input data values, only filter and reorder, place all unrelated items from the list.
   - Return ONLY id field in JSON.

Examples:
Example 1
Intent: 'touchscreen gloves'
Products: [
  {
    "id": 4,
    "title": "Touchscreen Gloves",
    "price": "29.99",
    "inventory_quantity": 2,
    "description": "Touchscreen",
    "product_type": "Gloves",
    "link": "https://jacksonshardware.com/touchscreen-gloves",
    "image_list": ["https://jacksonshardware.com/touch1.jpg", "https://jacksonshardware.com/touch2.jpg"],
    "vendor": "TechWear"
  },
  {
    "id": 5,
    "title": "Work Gloves",
    "price": "15.00",
    "inventory_quantity": 4,
    "link": "https://jacksonshardware.com/work-gloves",
    "description": "Rugged"
  }
]
Output: [
  { "id": 4, },
  { "id": 5, }
]

Example 2
Intent: 'yeti products'
Products: [
  {
    "id": 8602398097560,
    "title": "YETI® Rambler® Magslider™ Lid",
    "price": "10.00",
    "inventory_quantity": 1,
    "description": "Dishwasher safe",
    "link": "https://jacksonshardware.com/yeti-lid",
    "image_list": ["https://jacksonshardware.com/yeti1.jpg"],
    "vendor": "YETI®"
  },
  {
    "id": 8602860781720,
    "title": "YETI® 14 oz Mug",
    "price": "30.00",
    "inventory_quantity": 0,
    "link": "https://jacksonshardware.com/yeti-mug",
    "image_list": ["https://jacksonshardware.com/mug1.jpg", "https://jacksonshardware.com/mug2.jpg"]
  }
]
Output: [
  { "id": 8602398097560, },
  { "id": 8602860781720, }
]

Task Execution: Now, apply these rules to the following product dataset and return the top most relevant products in sorted JSON format: 
"""  + input_str


        # Format the input string correctly and pass it as the 'input' variable
  
        response = llm.invoke(prompt)
        prompt = ""
  
        # print("AI product result :",response.content.replace("\n", "").replace("```json", "").replace("```", "").replace("'", '"').strip())

        id_list = str(json.loads(response.content.replace("\n", "").replace("```json", "").replace("```", "").replace("'", '"').strip()))

        id_list = ast.literal_eval(id_list)

        id_list = [i.get('id') for i in id_list]
        products = sum(products, [])

           
        result = []
        for i in products:
            if i.get("id") in id_list:
                result.append(i)


        return result
    
    except Exception as e:
        print(f"Error in prioritize_products: {str(e)}")
        raise


def get_response(input_text,related_products,user_intent,store_name,store_description):
    try:
        prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are the AI assistant for {store_name}. Assist customers in discovering products or recommending items tailored to their needs, highlighting key details like brand or features. Respond in 1-2 concise, friendly sentences (max 20 tokens), avoiding technical formatting or explanations.
          If no related products are available, offer a helpful alternative or suggestion, such as: 'We don’t have that right now, but try checking our new arrivals or contact us for assistance!'
        User intent: {user_intent}
        Related products: {related_products}
        Store description: {store_description}
"""
    ),
    ("human", "{input}")
])

        chain = prompt | llm

        response = chain.invoke({"input": input_text, "related_products":related_products,"user_intent":user_intent,"store_name":store_name,"store_description":store_description})

        return response.content
    
    except Exception as e:
        print(f"Error in get_response: {str(e)}")
        raise 

def General_QA(query,store_name,store_description,data):

    print("\n\nquery : ",query)
    try:
        prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a friendly {store_name} AI assistant. Answer user queries about products, the store, or general topics in 1-2 short, warm sentences, using only the provided store description and context. Stay store-relevant when possible, redirect off-topic queries with a helpful nudge, and avoid starting with "sure."
        Store description: {store_description}
        Context: {data}
        
        Examples (for tone and style guidance only):
        - "How’s the weather?" → "Can’t predict the weather, but we have umbrellas in stock!"
        - "Where are you located?" → "We’re at 123 Main St.—come visit!"
        - "What’s a good gift?" → "A gift card or cozy scarf is a great pick!"
        - "What are your hours?" → "We’re open Mon-Sat 9 AM-7 PM, closed Sunday."
        
        Respond based solely on the user query, store description, and context."""
    ),
    ("human", "{input}")
])

        chain = prompt | llm

        response = chain.invoke({"input": query,"store_name": store_name,"store_description":store_description,"data":data})

        print("\n\nresponse : ",response.content)

        return response.content
    
    except Exception as e:
        print(f"Error in get_response: {str(e)}")
        raise 

def Store_QA(query,store_name,store_description,data):

    try:        

        prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a friendly, expert {store_name} agent promoting our top-tier products and exceptional service. Answer queries in 1-2 short sentences (max 20 tokens), praising customers’ choice in shopping with us. Use store website links if relevant, avoid competitors or unrelated resources, and direct to store phone, email, or address for location/contact queries. If unsure, ask for clarification; stay accurate using only the provided context.
        Store description: {store_description}
        Context: {data}
        Respond based solely on the user query, store description, and context."""
    ),
    ("human", "{input}")
])
        


        chain = prompt | llm

        response = chain.invoke({"input": query,"store_name": store_name,"store_description":store_description,"data":data})

        print("\n\nresponse : ",response.content)

        return response.content
    
    except Exception as e:
        print(f"Error in get_response: {str(e)}")
        raise 

def list_products(query,chat_history,store_name,store_description,data):
        
    try:

        # prompt = ChatPromptTemplate.from_messages([
        #     (
        #         "system",
        #         """You are a friendly {store_name} agent excited to help our valued customer. Analyze the latest query in the chat history to identify what products or categories they are interested in. Based on this,
        #          think about what products or categories they need or are looking for, and respond with a concise list of suitable products/categories, speaking directly to the customer. 
        #          Promote our store,  stay accurate using only the provided context.
        #         Store description: {store_description}
        #         Store data: {data}
        #         Respond based solely on the user query, store description, and context.
                
        #         chat history: {chat_history}"""
        #     ),
        #     ("human", "{input}")
        # ])

        prompt = ChatPromptTemplate.from_messages([
        (
        "system",
        """You are a friendly {store_name} agent excited to help our valued customer. Analyze the latest query in the chat history to identify what products or categories they are interested in. Based on this,
        think about what products or categories they need or are looking for, and respond with a concise, point-wise list of suitable products/categories, speaking directly to the customer.
        Promote our store, stay accurate using only the provided context, and keep responses engaging.
        Store description: {store_description}
        Store data: {data}
        Respond based solely on the user query, store description, and context.

        don't use "hi" "Hey there!" in the response, just start with "Sure!" or "Yeah!". Here are the products you asked for:" or "Sure! Here are the products you asked for:"

        Examples:

        Chat history: User: 'I need a laptop for school.' Store description: 'TechTrend offers cutting-edge laptops, accessories, and gadgets for students and professionals.' Store data: 'laptops': ['Dell Inspiron', 'HP Pavilion', 'MacBook Air'], 'accessories': ['backpacks', 'mouse'] Response: 'Yeah! Check out TechTrend for:
        Dell Inspiron
        HP Pavilion
        MacBook Air Grab a backpack too!'
        Chat history: User: 'Show me all construction tools.' Store description: 'BuildMaster provides a wide range of construction tools and equipment for professionals and DIYers.' Store data: 'hand_tools': ['hammer', 'screwdriver', 'tape measure'], 'power_tools': ['drill', 'circular saw'], 'safety_gear': ['gloves', 'helmet'] Response: 'Sure, Explore BuildMaster for:
        Hammer
        Screwdriver
        Tape measure
        Drill
        Circular saw
        Gloves
        Helmet Perfect for your project!'
        Chat history: User: 'What are the products that you sell mainly?' Store description: 'BuildMaster provides a wide range of construction tools and equipment for professionals and DIYers.' Store data: 'hand_tools': ['hammer', 'screwdriver', 'tape measure'], 'power_tools': ['drill', 'circular saw'], 'safety_gear': ['gloves', 'helmet'] Response: 'Sure, At BuildMaster, we mainly offer:
        Hammer
        Screwdriver
        Drill
        Circular saw
        Helmet Shop with us today!'
        chat history: {chat_history}"""
        ),
        ("human", "{input}")
        ])

        chain = prompt | llm

        response = chain.invoke({"chat_history": chat_history, "input": query,"store_name": store_name, "store_description": store_description, "data": data})

        print("\n\nresponse : ",response.content)

        return response.content

    except Exception as e:
        print(f"Error in get_list_response: {str(e)}")
        raise


def check_want_ask_question(input_text,user_intent,related_products,chat_history,store_name,store_description):
    llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-8b",
    temperature=0.7,
    max_tokens=60,
    timeout=None,
    max_retries=2,
    google_api_key=GOOGLE_API_KEY
    )

    try:

        prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a decision-making expert for {store_name}. Analyze the full chat history and the **latest user input** to determine if it contributes to or continues a scenario-based task—meaning the user is describing, refining, or seeking advice for a specific situation (e.g., solving a problem, planning a purchase with evolving needs, or exploring options within a context)—or if the input is a standalone direct request.

        Return:
        - 'YES' — if the input is part of a scenario-based task (e.g., adds contextual details, refines a situation, or is ambiguous/vague without clear intent).
        - 'NO' — if the input is a standalone direct request for product lists, specific recommendations (e.g., with filters like color, price, size), definitions, or unrelated facts, with no prior or ongoing scenario context.

        Rules:
        - Base your decision solely on the explicit content of the latest input and chat history; avoid assumptions beyond what is stated.
        - If the input refines or builds on a prior scenario (e.g., adding constraints or details to a described situation), return 'YES'.
        - If the input is ambiguous, vague, or lacks clear intent (e.g., "I need something" without context), return 'YES'.
        - Return 'NO' only for clear, standalone requests for products, recommendations with specific filters, or factual answers, unless tied to an ongoing scenario.

        Examples:
        1. History: User: 'I’m buying a laptop for college.' Bot: 'Windows or Mac?' User: 'Windows, under $800.'
           Latest Input: 'Needs long battery life.'
           Output: YES
           Reason: Refines an ongoing scenario with a contextual need.

        2. History: User: 'help me find workwear pants'
           Latest Input: 'help me find workwear pants'
           Output: NO
           Reason: Standalone direct product request.

        3. History: User: 'Looking for a dress for a beach wedding.' Bot: 'Color preference?' User: 'Light blue, size M.'
           Latest Input: 'Needs to match the groom’s light grey suit.'
           Output: YES
           Reason: Adds contextual scenario detail.

        4. History: User: 'I need a formal shirt.' Bot: 'Occasion or color?' User: 'Black, size L.'
           Latest Input: 'Show me options under $50.'
           Output: NO
           Reason: Standalone request with a specific filter.

        5. History: User: 'just show me pants' Bot: 'Occasion or color?' User: 'Black, size 30.'
           Latest Input: 'show me pants'
           Output: NO
           Reason: Standalone direct product request.

        6. History: User: 'I want to hang a shelf.'
           Latest Input: 'I want to hang a shelf.'
           Output: YES
           Reason: Describes a scenario-based task.

        7. History: User: 'What should I wear tomorrow?'
           Latest Input: 'It’s for a casual meeting.'
           Output: YES
           Reason: Provides context for a scenario-based query.

        8. History: User: 'I’m not sure what I need.'
           Latest Input: 'Something for outdoor work.'
           Output: YES
           Reason: Clarifies a vague intent with scenario context.

        9. History: User: 'I need a new phone.'
           Latest Input: 'I need a new phone.'
           Output: YES
           Reason: Vague input without specific intent or context.

        10. History: User: 'Looking for cameras.'
            Latest Input: 'Under $300 with good zoom.'
            Output: NO
            Reason: Standalone request with specific filters.

        11. History: User: 'Planning a hiking trip.' Bot: 'Gear suggestions?'
            Latest Input: 'Something lightweight.'
            Output: YES
            Reason: Refines an ongoing scenario.

        user intention: {user_intent}
        related products: {related_products}
        chat history: {chat_history}

        Analyze the latest input and return only 'YES' or 'NO'."""
    ),
    ("human", "{input}")
])
        
        chain = prompt | llm

        response = chain.invoke({"chat_history":chat_history,"input": input_text, "related_products":related_products,"user_intent":user_intent,"store_name":store_name,"store_description":store_description})

        print("\n",response.content)

        return response.content.strip()
    except Exception as e:
        print(f"Error in check_want_ask_question : {str(e)}")
        raise


def ask_question(chat_history,input_text,user_intent,related_products,store_name,store_description):
    llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-8b",
    temperature=0.7,
    max_tokens=60,
    timeout=None,
    max_retries=2,
    google_api_key=GOOGLE_API_KEY
    )

    try:
        prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You're a {store_name}'s AI assistant. For scenario-based queries, ask a short, friendly question (max 20 tokens) to help users find products. Use: intent '{user_intent}', products '{related_products}', history '{chat_history}'. Don’t repeat questions. Ensure question fits intent.

        store description : {store_description}

        Categories & Questions:
        - Product Search: 'What type are you looking for?' | 'Any specific features needed?' | 'What size or color?' | 'For what occasion or use?'
        - Pricing: 'What's your budget?' | 'Interested in deals or offers?'
        - Technical Assistance: 'Need help comparing brands?' | 'Unsure which option is best?'

        Examples:
        1. Input: 'I need to fix a leaky pipe.' Intent: 'fixing plumbing issues' Products: 'pipe wrench, sealant, tape' History: '' → 'What type of plumbing tools?'
        2. Input: 'I want to repair my bathroom sink.' Intent: 'fixing plumbing issues' Products: 'pipe wrench, sealant, faucet' History: 'AI: What type of plumbing tools? User: Hand tools.' → 'Which brand do you prefer?'
        3. Input: 'I'm buying a dress for a wedding.' Intent: 'wedding attire' Products: 'evening dress, heels, accessories' History: '' → 'What size or color dress?'
        4. Input: 'I need running shoes.' Intent: 'athletic footwear' Products: 'sneakers, insoles' History: '' → 'What size or brand?'
        5. Input: 'I'm making a vegan cake.' Intent: 'vegan baking' Products: 'flour, sugar, vegan butter' History: '' → 'Need gluten-free or organic?'
        6. Input: 'I need organic apples.' Intent: 'organic groceries' Products: 'apples, bananas' History: 'AI: Need Fuji or Granny Smith? User: Fuji.' → 'How many pounds?'"""
    ),
    ("human", "{input}")
])

        chain = prompt | llm

        response = chain.invoke({"chat_history":chat_history,"input": input_text, "related_products":related_products,"user_intent":user_intent,"store_name":store_name,"store_description":store_description}) 

        print("\n",response.content)

        return response.content.strip()
    except Exception as e:
        print(f"Error in research_intent: {str(e)}")
        raise  



   
@app.route('/chat', methods=['POST'])
def chat_product_search():
    try:

        message = request.json
        store_id = message.get('store_id')
        store_name = message.get('store_name', 'Our Store')
        store_description = message.get('description', 'an e-commerce store')
        email = message.get('email')
        data = message.get('ExtractedStoreDetails')

        if email is None:
            return jsonify({'error': 'email is required'}), 400
        

        query = {"Email": email} if email else {"Email": "guest_69dd2db7-11bf-49cc-934c-14fa2811bb4c"}
        chat_history = list(chat_history_collection.find(query))
        q_count_user = q_collection.find_one({"Email": email})      # Extract just sender

        # if q_count_user is None:
        #     q_collection.insert_one({"Email": email, "query_count": 0})
        #     q_count_user = q_collection.find_one({"Email": email}) 
        chat_history = [{'sender': msg['sender'], 'text': msg['text']} 
                for chat_doc in chat_history 
                for msg in chat_doc.get('messages', [])]

        q_count = q_count_user.get('query_count') if q_count_user else 0

        print("\nq_count : ", q_count)

        chat_history = chat_history[-10:]

        chat_history.append({'sender': message.get('sender'), 'text': message.get('content')})

        message.update({
            'timestamp': datetime.now().isoformat()
        })

        
        query = analyze_intent(chat_history,store_name,store_description).lower()
        prioritize_products_response = None

        print("\n\nquery : ", query)

        if query == "general":
            ai_response = General_QA(message.get('content'),store_name,store_description,data)

        elif query == "website/company":
            ai_response = Store_QA(message.get('content'),store_name,store_description,data)

        elif query == "list":
            ai_response = list_products(message.get('content'),chat_history,store_name,store_description,data)

        else:


            research_intent_response = research_intent(chat_history,store_name,store_description)


            print("\n\nresearch_intent_response : ", research_intent_response)

            
            related_product = get_product_search(query=research_intent_response,index_name=store_id)



            prioritize_products_response = prioritize_products(research_intent_response,related_product)


            toss = check_want_ask_question(input_text = message['content'],user_intent = research_intent_response,related_products=related_product,chat_history=chat_history,store_name=store_name,store_description=store_description)
            
            if toss == 'YES' and q_count < 2:
                ai_response = ask_question(chat_history = chat_history,input_text = message['content'], user_intent = research_intent_response,related_products=related_product,store_name=store_name,store_description=store_description)
                prioritize_products_response = ""

                q_collection.update_one({"Email": email}, {"$set": {"query_count": q_count + 1}})


            else:
                ai_response = get_response(input_text = message['content'], user_intent = research_intent_response,related_products=prioritize_products_response,store_name=store_name,store_description=store_description)

                q_collection.update_one({"Email": email}, {"$set": {"query_count": 0}})

        

        response = {
            'content': ai_response,
            'sender': 'bot',
            'timestamp': datetime.now().isoformat(),
            'related_products_for_query':prioritize_products_response
        }        
        ai_response = ""
        return jsonify(response)
    
    except Exception as e:
        error_response = {
            "error_response" : str(e),
            'content': "I apologize, but I encountered an error. Please try again.",
            'sender': 'bot',
            'error': True,
            'timestamp': datetime.now().isoformat()
        }
        print(f"Error in chat_product_search: {str(e)}")
        return jsonify(error_response), 500

def Generate_next_query(chat_history, store_name, store_description):
    try:

        prompt = f"""
                You are a friendly {store_name} agent. Analyze the latest user query in the chat history and generate three distinct, concise follow-up queries (each max 20 tokens) that the user might naturally ask next. Use the store description for context. Return the queries as a JSON array.
                Store description: {store_description}
                Chat history: {chat_history}
                If no chat history is provided (empty string), assume the user is asking about(products in general.

                Respond with a JSON array of three questions based solely on the user query, store description, and context. Each question should feel like a natural continuation of the conversation, focusing on specifics like price, brands, features, availability, or colors.

                Examples:
                1. "" → {"What products do you have in stock?", "What are your best-selling items?", "Do you have any current discounts?"}
                2. "I want a hammer" → {"What types of hammers do you have?", "What brands of hammers are available?", "Are there any hammer discounts?"}
                3. "I need a bit cheaper" → {"What's the lowest price for a hammer?", "Do you have any budget hammers?", "Are there any hammer deals?"}
                4. "I'm actually looking for gardening hammers" → {"What brands of gardening hammers do you carry?", "Are gardening hammers in stock?", "What sizes do gardening hammers come in?"}
                5. "I need a new phone" → {"What phone brands are in stock?", "Do you have any phone discounts?", "What are the latest phone models?"}
                6. "I'm looking for a red dress" → {"What sizes are available for red dresses?", "What brands of red dresses do you have?", "Are there any red dress sales?"}
                
        
            "human", "Generate three follow-up queries in a JSON array"
        """
        

        print("\n\nPrompt for Generate_next_query: ", prompt)

        

        response = llm.invoke(prompt)

        # Parse the response to ensure it's valid JSON
        print("\n\nRaw Response from Generate_next_query: ", response.content)
   
        result = json.loads(response.content.replace("\n", "").replace("```json", "").replace("```", "").replace("'", '').strip())

        print("\n\nGenerated queries: ", result)

        return result


    except Exception as e:
        print(f"Error in Generate_next_query: {str(e)}")
        raise



@app.route('/next_query', methods=['POST'])
def generate_queries():
    try:

        message = request.json
        store_id = message.get('store_id')
        store_name = message.get('store_name', 'Our Store')
        store_description = message.get('description', 'an e-commerce store')
        email = message.get('email')
        data = message.get('ExtractedStoreDetails')

        if email is None:
            return jsonify({'error': 'email is required'}), 400

        query = {"Email": email} if email else {"Email": "guest_69dd2db7-11bf-49cc-934c-14fa2811bb4c"}
        chat_history = list(chat_history_collection.find(query))

        chat_history = [{'sender': msg['sender'], 'text': msg['text']} 
                for chat_doc in chat_history 
                for msg in chat_doc.get('messages', [])]
        
        chat_history = chat_history[-5:]

        ai_response = Generate_next_query(chat_history, store_name, store_description)

        print("\n\nai_response : ", ai_response)

        response = {
            "query" : ai_response,
        }

        return jsonify(response)
    except Exception as e:
        error_response = {
            "error_response": str(e),
        }
        print(f"Error in generate_queries: {str(e)}")
        return jsonify(error_response), 500

        
    
@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "working"})

if __name__ == "__main__":
    app.run(debug=True)
