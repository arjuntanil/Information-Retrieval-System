import streamlit as st
from src.helper import get_pdf_text, get_text_chunks, get_vector_store, get_conversational_chain

def user_input(user_question):
    """Handles user questions and displays the conversation."""
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chatHistory = response["chat_history"]
    for i, message in enumerate(st.session_state.chatHistory):
        if i % 2 == 0:
            st.write("User: " + message.content)
        else:
            st.write("Reply: " + message.content)

def main():
    st.set_page_config("Information Retrieval")
    st.header("Information Retrieval System")

    user_question = st.text_input("Ask a question about your documents:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chatHistory" not in st.session_state:
        st.session_state.chatHistory = None

    if user_question:
        if st.session_state.conversation is None:
            st.warning("Please upload and process your documents first.")
        else:
            user_input(user_question)

    with st.sidebar:
        st.title("Menu : ")
        pdf_docs = st.file_uploader("Upload your PDF Documents", accept_multiple_files=True, type=["pdf"])
        if st.button("Submit & Process Documents"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vector_store = get_vector_store(text_chunks)
                    st.session_state.conversation = get_conversational_chain(vector_store)
                    st.success("Documents processed successfully!")
            else:
                st.error("Please upload at least one PDF file.")

if __name__ == "__main__":
    main()
