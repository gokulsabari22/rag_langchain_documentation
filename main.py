from core import run_llm
from langchain.chains import ConversationalRetrievalChain
import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
from const import INDEX_NAME

load_dotenv()

st.header("Langchain Helper Bot")

if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []

if "chat_answer_history" not in st.session_state:
    st.session_state["chat_answer_history"] = []

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

prompt = st.text_input("Prompt", placeholder="Enter your prompt here")

if prompt:
    with st.spinner(text="Generating Response"):
        generated_response = run_llm(query=prompt, chat_history=st.session_state["chat_history"])
        formatted_response = (f"{generated_response}")
        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answer_history"].append(formatted_response)
        st.session_state["chat_history"].append((prompt, formatted_response))

# if st.session_state["chat_answer_history"]:
for user_query, generated_response in zip(st.session_state["user_prompt_history"], st.session_state["chat_answer_history"]):
    user_message_key = f"user_message_{user_query}"
    response_message_key = f"response_message_{generated_response}"
    message(user_query, is_user=True, key=user_message_key)
    message(generated_response, key=response_message_key)
        