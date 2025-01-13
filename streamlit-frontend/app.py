import streamlit as st

from utils import (
    load_document,
    chunk_data,
    convert_chunks_to_documents,
    ask_and_get_answer,
    calculate_embedding_cost,
    ask_questions_with_memory,
)

from pinecone_helpers import (
    insert_or_fetch_embeddings,
    delete_pinecone_index,
)

# clear the chat history from streamlit session state
def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']

    # Delete files in the input_files directory
    input_files_dir = '../database/input_files/'
    if os.path.exists(input_files_dir):
        for file in os.listdir(input_files_dir):
            file_path = os.path.join(input_files_dir, file)
            try:
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
            except Exception as e:
                print(f"Could not delete file {file_path}: {e}")


# Define a callback function to clear the input
# def submit_question():
#     st.session_state.user_question = ""
#     st.session_state.user_question = st.session_state.input_question
#     st.session_state.input_question = ""


if __name__ == "__main__":
    import os

    # loading the OpenAI api key from .env
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)

    # st.image('cute-robot.jpeg', width=100)
    st.subheader('LLM Question-Answering with OpenAI and Pinecone')
    with st.sidebar:
        # text_input for the OpenAI API key (alternative to python-dotenv and .env)
        api_key = st.text_input('OpenAI API Key:', type='password')
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key
            # print("not using entered openai api key")

        pc_api_key = st.text_input('Pinecone API Key:', type='password')
        if pc_api_key:
            os.environ['PINECONE_API_KEY'] = pc_api_key
            # print("not using entered pinecone api key")


        # file uploader widget
        uploaded_file = st.file_uploader('Upload a file:', type=['pdf', 'docx', 'txt'])

        # chunk size number widget
        chunk_size = st.number_input('Chunk size:', min_value=100, 
                                     max_value=2048, 
                                     value=512, 
                                     on_change=clear_history
                                     )

        # k number input widget
        k = st.number_input('k', min_value=1, 
                            max_value=20, 
                            value=3, 
                            on_change=clear_history
                            )

        # add data button widget
        add_data = st.button('Add Data', 
                             on_click=clear_history
                             )

        if uploaded_file and add_data: # if the user browsed a file
            with st.spinner('Reading the file ...'):

                # writing the file from RAM to the current directory on disk
                bytes_data = uploaded_file.read()
                file_name = os.path.join('database', 'input_files', uploaded_file.name)

                print(file_name)

                # Create the necessary directories if they do not exist
                os.makedirs(os.path.dirname(file_name), exist_ok=True)

                with open(file_name, 'wb') as f:
                    f.write(bytes_data)

                # Pipeline
                data = load_document(file_name)
                chunks = chunk_data(data, chunk_size=chunk_size)
                docs = convert_chunks_to_documents(chunks)
                st.write(f'Chunk size: {chunk_size}, Chunks: {len(chunks)}')

                try:
                    tokens, embedding_cost = calculate_embedding_cost(chunks)
                    st.write(f'Embedding cost: ${embedding_cost:.4f}')
                except Exception as e:
                    st.error(f"An error occurred: {e}")

                # creating the embeddings and returning the Chroma vector store
                delete_pinecone_index()

                try: 
                    print('1')
                    vector_store = insert_or_fetch_embeddings('askadocument', docs)
                    print('w')

                    # saving the vector store in the streamlit session state (to be persistent between reruns)
                    st.session_state.vs = vector_store
                    print('w3')

                    st.success('File uploaded and embedded successfully.')
                except Exception as e:
                    st.error(f"An error occurred: {e}")


    # user's question text input widget
    # default key for the input field in session state
    # if "user_question" not in st.session_state:
    #     st.session_state.user_question = ""

    # q = st.text_input('Ask a question about the content of your file:', 
    #                   key='input_question', 
    #                   on_change=submit_question) 
    q = st.text_input("Ask a question about the content of your file:")

    if q: # if the user entered a question and hit enter
        if 'vs' in st.session_state: # if there's the vector store (user uploaded, split and embedded a file)
            vector_store = st.session_state.vs
            st.write(f'k: {k}')
            # answer = ask_and_get_answer(vector_store, q, k)
            answer = ask_questions_with_memory(vector_store, q, k)

            # text area widget for the LLM answer
            # st.text_area('LLM Answer: ', value=answer['result'])
            st.text_area('LLM Answer: ', value=answer['answer'])

            st.divider()

            # if there's no chat history in the session state, create it
            if 'history' not in st.session_state:
                st.session_state.history = ''

            # the current question and answer
            # the current question and answer
            # value = f'Q: {q} {" " * 5} \nA: {answer['result']}'
            value = f'Q: {q} {" " * 5} \nA: {answer['answer']}'

            st.session_state.history = f'{value} \n {"-" * 100} \n {st.session_state.history}'
            h = st.session_state.history

            # text area widget for the chat history
            st.text_area(label='Chat History', value=h, key='history', height=400)