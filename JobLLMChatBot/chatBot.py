import streamlit as st
from utils import write_message
from agent import generate_response
from langchain_core.exceptions import OutputParserException
from tools.score import calculate_compatibility_score

#Configure the default settings in the JobGPT Landing Page
st.set_page_config("Job GPT", page_icon="💼")
st.title("Job Search: Graph GPT")

#Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi, I'm a Job Search Chatbot!  How can I assist you today?"},
    ]

# Define the maximum token length for the model
MAX_TOKEN_LENGTH = 16384

# Create handle submit function to interact with agent and underlying architecture to handle the query submitted by user
def handle_submit(message):
    # Tokenize the message and check its length
    tokens = message.split()  # Assuming each word is a token
    if len(tokens) > MAX_TOKEN_LENGTH:
        st.error(f"Input exceeds maximum token length of {MAX_TOKEN_LENGTH}. Please shorten your message.")
        return
    
    try:
        #Writing the response
        with st.spinner('Thinking...'):
              #call the agent
              response = generate_response(message)
              write_message('assistant', response)
    except OutputParserException as e:
        st.error(f"An error occurred while parsing the output: {e}")
        # Log the error for further inspection
        st.error(f"Full error details: {e}")
    except ValueError as e:
        st.error(f"An unexpected error occurred: {e}")
        # Log the error for further inspection
        st.error(f"Full error details: {e}")

#Display message in Session State
for message in st.session_state.messages:
    write_message(message['role'], message['content'], save=False)

#Handle any user input
if question := st.chat_input("Job Search"):
    # Display user message in chat message container
    write_message('user', question)

    # Generate a response
    handle_submit(question)

if score_question := st.chat_input("Add your skills, experience, or any job-related question to get 'Job Compatibility Score':"):
    write_message('user', score_question)
    output = calculate_compatibility_score(score_question)
    st.write(f"Your job compatibility score is: {output[0]['output']} \n {output[1]}")