import streamlit as st
from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx

# write message is a helper function to save messages to session state and write to the UI 

def write_message(role, content, save=True):
    #Add a message to session state
    if save:
        st.session_state.messages.append({"role": role, "content": content})

    # Write message to the UI
    with st.chat_message(role):
        st.markdown(content)

# Function to get session id from Streamlit
def get_session_id():
    return get_script_run_ctx().session_id
