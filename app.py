from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

import streamlit as st

st.set_page_config(page_title="LLM Chat with Memory")
st.title("🧠 Chat with AI (Memory enabled)")

# Load model
@st.cache_resource
def load_model():
    pipe = pipeline("text-generation", model="tiiuae/falcon-rw-1b", max_new_tokens=200)
    return HuggingFacePipeline(pipeline=pipe)

llm = load_model()

# Memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory()

chain = ConversationChain(llm=llm, memory=st.session_state.memory, verbose=False)

# UI
user_input = st.text_input("💬 You:", key="input")
if st.button("Send") and user_input:
    response = chain.run(user_input)
    st.markdown(f"**🤖 AI:** {response}")
