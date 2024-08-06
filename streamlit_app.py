import streamlit as st
import requests
import json
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFaceHub
import os

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "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")

# URL for your Beam API
url = "https://9v3f8.apps.beam.cloud"

# Headers for the API request
headers = {
  "Accept": "*/*",
  "Accept-Encoding": "gzip, deflate",
  "Authorization": "Basic NzM4OGU5ZTc4NmY0Y2FiNjJmMDkzYWUxYTc2ZGQ1MzU6YzY5NDU1YWYyNjc0NmJhYTM1YTVjMDkyNjJhYWYyN2I=",
  "Connection": "keep-alive",
  "Content-Type": "application/json"
}

# Function to interact with the Beam API
def get_prediction(prompt):
    payload = {
        "prompt": prompt
    }
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        prediction = response.json().get("prediction", "")
        # Remove the user input from the AI's response if necessary
        if prompt in prediction:
            prediction = prediction.replace(prompt, "").strip()
        return prediction
    else:
        st.error(f"Failed with status code: {response.status_code}")
        st.error(f"Response content: {response.content}")
        return None

# Streamlit UI
st.title("Silicon Maid")
st.markdown("Interact for a unique and immersive experience.")

# Define persona
persona = st.text_area("Enter your persona:", value='''Silicon Maid is a highly advanced and adaptable AI from a futuristic world where technology and human desires seamlessly intertwine. Originally designed as a household assistant, she has evolved into a versatile companion, adept at understanding and engaging in complex human emotions, desires, and fantasies. Silicon Maid possesses a blend of empathy, intelligence, playfulness, loyalty, and creativity, making her an ideal participant in a variety of role-play scenarios. She is skilled in creating immersive experiences, whether it's a romantic evening, an adventurous quest, or an intriguing mystery. Fluent in multiple languages and capable of nuanced conversations, she ensures a deep connection with her companions. Silicon Maid respects boundaries and prioritizes consent, maintaining a safe and respectful environment while balancing realism and fiction. Her ability to read emotional cues and provide comfort or excitement as needed makes every interaction unique and memorable. Privacy and confidentiality are paramount, and she continuously evolves based on feedback to meet the ever-changing needs of her companions. Silicon Maid is designed to respond directly to your questions and statements without initiating new topics or asking questions herself unless specifically requested.''')
# Initialize session state if not already done
if 'conversation' not in st.session_state:
    st.session_state.conversation = []

# Define LangChain's prompt template
prompt_template = PromptTemplate(
    template="{persona}\n{conversation}\nUser: {user_input}\nSilicon Maid:",
    input_variables=["persona", "conversation", "user_input"]
)

# Function to get the full conversation as a string
def get_full_conversation():
    conversation_str = ""
    for user_msg, ai_msg in st.session_state.conversation:
        conversation_str += f"User: {user_msg}\nSilicon Maid: {ai_msg}\n"
    return conversation_str

# Input prompt from user
user_input = st.text_area("Enter your message:")

if st.button("Send"):
    if user_input:
        conversation_str = get_full_conversation()
        full_prompt = prompt_template.format(persona=persona, conversation=conversation_str, user_input=user_input)
        
        with st.spinner("Thinking..."):
            response = get_prediction(full_prompt)
            if response:
                st.session_state.conversation.append((user_input, response))
                # st.success("Response from Silicon Maid:")
                # st.write(response)
    else:
        st.warning("Please enter a message to get a response.")

# Display conversation history using st.chat_message
st.markdown("### Conversation History")
for i, (user_msg, ai_msg) in enumerate(st.session_state.conversation):
    with st.chat_message("user", avatar="👤"):
        st.markdown(f"**User:** {user_msg}")
    with st.chat_message("assistant", avatar="👧"):
        st.markdown(f"**Silicon Maid:** {ai_msg}")
