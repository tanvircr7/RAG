
def get_pc_apikey():
    from dotenv import load_dotenv
    import os

    load_dotenv()  # Load environment variables from .env file

    pc_apikey = os.getenv('PINECONE_API_KEY')
    return pc_apikey

def get_openai_apikey():
    from dotenv import load_dotenv
    import os

    load_dotenv()  # Load environment variables from .env file

    openai_apikey = os.getenv('OPENAI_API_KEY')
    
    return openai_apikey

def insert_or_fetch_embeddings(index_name, docs):
  import pinecone
  from langchain_community.vectorstores import Pinecone
  from langchain_openai import OpenAIEmbeddings
  # from pinecone import PodSpec
  from pinecone import Pinecone, ServerlessSpec
  from langchain_pinecone import PineconeVectorStore

  from uuid import uuid4

  from langchain_core.documents import Document

  import time

  # pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
  pc = pinecone.Pinecone(api_key=get_pc_apikey())

  # from openai import OpenAI
  # client = OpenAI()

  embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=512, api_key=get_openai_apikey())  # 512 works as well

  existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

  # loading from existing index
  if index_name in pc.list_indexes().names():
      print(f'Index {index_name} already exists. Loading embeddings ... ', end='')
  else:
      # creating the index and embedding the docs into the index
      print(f'Creating index {index_name} and embeddings ...', end='')

      # creating a new index
      pc.create_index(
          name=index_name,
          dimension=512,
          metric='cosine',
          # spec=PodSpec(environment='gcp-starter')
          spec=ServerlessSpec(
              cloud="aws",
              region="us-east-1"
          )
      )

      print("index created....")

  # processing the input documents, generating embeddings using the provided `OpenAIEmbeddings` instance,
  # inserting the embeddings into the index and returning a new Pinecone vector store object.
  index = pc.Index(index_name)
  vector_store = PineconeVectorStore(index=index, embedding=embeddings) # Initialize with the index and embedding
  uuids = [str(uuid4()) for _ in range(len(docs))]
  vector_store.add_documents(docs, ids=uuids)
  print('Ok')
  return vector_store


def delete_pinecone_index(index_name='all'):
    import pinecone
    pc = pinecone.Pinecone(api_key=get_pc_apikey())

    if index_name == 'all':
        indexes = pc.list_indexes().names()
        print('Deleting all indexes ... ')
        for index in indexes:
            pc.delete_index(index)
        print('Ok')
    else:
        print(f'Deleting index {index_name} ...', end='')
        pc.delete_index(index_name)
        print('Ok')
