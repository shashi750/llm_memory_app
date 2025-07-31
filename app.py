import streamlit as st
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# Set Streamlit app title
st.set_page_config(page_title="LLM Chat with Memory")
st.title("🧠 Chat with AI (with Memory)")

# Load the HuggingFace model (text2text)
@st.cache_resource
def load_model():
    gen = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=100)
    return HuggingFacePipeline(pipeline=gen)

llm = load_model()

# Initialize memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory()

# Initialize chain
chain = ConversationChain(llm=llm, memory=st.session_state.memory, verbose=False)

# User input
user_input = st.text_input("💬 You:", key="input")

if st.button("Send") and user_input:
    response = chain.run(user_input)
    st.markdown(f"**🤖 AI:** {response}")
