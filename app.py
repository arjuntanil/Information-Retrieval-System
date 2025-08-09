import streamlit as st
from src.helper import (
    get_pdf_text, get_text_chunks, get_vector_store, get_conversational_chain,
    generate_quiz
)

def user_input(user_question):
    """Handles user questions and stores the conversation."""
    if st.session_state.last_question != user_question:  # Only process new questions
        response = st.session_state.conversation({"question": user_question})
        st.session_state.chatHistory = response["chat_history"]
        st.session_state.last_response = response
        st.session_state.last_question = user_question
        st.session_state.mode = "chat"

def main():
    st.set_page_config("Information Retrieval")
    st.header("📄 Information Retrieval System")

    # Initialize session state variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chatHistory" not in st.session_state:
        st.session_state.chatHistory = None
    if "raw_text" not in st.session_state:
        st.session_state.raw_text = ""
    if "last_response" not in st.session_state:
        st.session_state.last_response = None
    if "mode" not in st.session_state:
        st.session_state.mode = None  # can be "chat" or "quiz"
    if "last_question" not in st.session_state:
        st.session_state.last_question = None
    if "quiz_generated" not in st.session_state:
        st.session_state.quiz_generated = False

    # Input box for user question
    user_question = st.text_input("Ask a question about your documents:")

    # Handle new question only when user presses Enter
    if user_question and st.session_state.conversation is not None:
        if st.session_state.mode != "quiz":  # Prevent re-running on quiz mode
            user_input(user_question)

    # Show chat history only if in chat mode
    if st.session_state.mode == "chat" and st.session_state.last_response:
        for i, message in enumerate(st.session_state.chatHistory):
            if i % 2 == 0:
                st.write("User: " + message.content)
            else:
                st.write("Reply: " + message.content)

    with st.sidebar:
        st.title("📌 Menu")
        pdf_docs = st.file_uploader("Upload your PDF Documents", accept_multiple_files=True, type=["pdf"])

        if st.button("Submit & Process Documents"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    st.session_state.raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(st.session_state.raw_text)
                    vector_store = get_vector_store(text_chunks)
                    st.session_state.conversation = get_conversational_chain(vector_store)
                    st.session_state.mode = "chat"  # Default to chat mode after processing
                    st.session_state.last_question = None  # Reset last question
                    st.session_state.quiz_generated = False  # Reset quiz flag
                    st.success("✅ Documents processed successfully!")
            else:
                st.error("Please upload at least one PDF file.")

        st.subheader("📝 AI Tools")
        if st.button("Generate Quiz"):
            if st.session_state.raw_text:
                with st.spinner("Generating quiz..."):
                    st.session_state.mode = "quiz"  # Switch to quiz mode
                    quiz_content = generate_quiz(st.session_state.raw_text)
                    st.session_state.quiz_generated = True
                    st.session_state.last_question = None  # Clear last question
                    
                    # Display quiz in the sidebar
                    st.subheader("Generated Quiz")
                    st.markdown(quiz_content)
            else:
                st.warning("Please process a PDF first.")

if __name__ == "__main__":
    main()