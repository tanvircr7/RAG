def get_openai_apikey():
    from dotenv import load_dotenv
    import os

    load_dotenv()  # Load environment variables from .env file

    openai_apikey = os.getenv('OPENAI_API_KEY')
    return openai_apikey

def load_document(file):
    print(f"File received: {file}")  # Print the file path for debugging
    import os
    name, extension = os.path.splitext(file)
    print(f"File name: {name}, File extension: {extension}")  # Print file name and extension

    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        print(f"Loading PDF file: {file}")  # Indicate that the PDF loader is being used
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        print(f"Loading DOCX file: {file}")  # Indicate that the DOCX loader is being used
        loader = Docx2txtLoader(file)
    elif extension == '.txt':
        from langchain.document_loaders import TextLoader
        print(f"Loading TXT file: {file}")  # Indicate that the TXT loader is being used
        loader = TextLoader(file)
    else:
        print(f"Document format {extension} is not supported!")  # Error for unsupported file format
        return None

    # Attempt to load the document and print the output
    try:
        data = loader.load()
        print(f"Document loaded successfully")  # Print the first two elements of the loaded data for inspection
    except Exception as e:
        print(f"Error loading document: {e}")  # Print any error during loading
        return None

    return data


def load_from_arxiv(query):
  '''
   load from arxiv
   load_from_arxiv('pokemon')
  '''
  from langchain_community.document_loaders import ArxivLoader

  loader = ArxivLoader(query=query, load_max_doc=2)
  # docs = loader.load()
  docs = loader.get_summaries_as_docs()
  # print(docs[0].page_content[:100])
  print(docs[0].page_content)
  print(docs[0].metadata)



# wikipedia
def load_from_wikipedia(query, lang='en', load_max_docs=2):
    '''
   load from arxiv
   load_from_wikipedia('pokemon')
  '''
    from langchain.document_loaders import WikipediaLoader
    loader = WikipediaLoader(query=query, lang=lang, load_max_docs=load_max_docs)
    data = loader.load()
    return data


def chunk_data(data, chunk_size=256, chunk_overlap=20):
#   print("chunked1")
  from langchain.text_splitter import RecursiveCharacterTextSplitter
  # Define separators for splitting (optional)
  separators = ["\n\n", "\n", " ", ""]
  text_splitter = RecursiveCharacterTextSplitter(
      separators=separators,
      chunk_size=chunk_size,
      chunk_overlap=chunk_overlap,
  )
#   print("chunked2")
  chunks = text_splitter.split_documents(data)
#   print("chunked3")
  return chunks


from langchain.schema import Document

def convert_chunks_to_documents(chunks, metadata=None):
    """
    Converts chunks into a list of Document objects with metadata.

    Args:
    - chunks: List of chunks (list of strings or objects with `page_content`).
    - metadata: A dictionary containing metadata to attach to each document.

    Returns:
    - List of Document objects.
    """
    documents = []
    for i, chunk in enumerate(chunks):
        content = chunk.page_content if hasattr(chunk, 'page_content') else chunk
        doc_metadata = metadata.copy() if metadata else {}
        doc_metadata['chunk_index'] = i  # Add chunk-specific metadata
        documents.append(Document(page_content=content, metadata=doc_metadata))
    return documents


def ask_and_get_answer(vector_store, q, k=10):
  from langchain.chains import RetrievalQA
  from langchain_openai import ChatOpenAI

  llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1, api_key=get_openai_apikey())

  retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})

  chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

  answer = chain.invoke(q)
  return answer


def calculate_embedding_cost(texts):
    import tiktoken

    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])

    print(f'Total Tokens: {total_tokens}')
    print(f'Embedding Cost in USD: {total_tokens / 1000 * 0.0004: .6f}')

    return total_tokens, total_tokens / 1000 * 0.0004


def ask_questions_with_memory(vector_store, q, k):
    from langchain_openai import ChatOpenAI
    from langchain.chains import ConversationalRetrievalChain  # Import class for building conversational AI chains
    from langchain.memory import ConversationBufferMemory  # Import memory for storing conversation history

    # Instantiate a ChatGPT LLM (temperature controls randomness)
    llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.2, api_key=get_openai_apikey())

    # Configure vector store to act as a retriever (finding similar items, returning top 12)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})


    # Create a memory buffer to track the conversation
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    crc = ConversationalRetrievalChain.from_llm(
        llm=llm,  # Link the ChatGPT LLM
        retriever=retriever,  # Link the vector store based retriever
        memory=memory,  # Link the conversation memory
        chain_type='stuff',  # Specify the chain type
        verbose=False  # Set to True to enable verbose logging for debugging
    )


    result = crc.invoke({'question': q})

    return result