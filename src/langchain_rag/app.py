import streamlit as st

from langchain_rag.main import agent

# Configure the Streamlit page settings
st.set_page_config(
    page_title="LangchainRAG Chat",
    page_icon="🤖",
    layout="wide",  # Options: 'centered' or 'wide'
    initial_sidebar_state="auto",
)


def main():
    """Main function to run the LangchainRAG Streamlit application."""
    st.header("LangchainRAG")

    app_description = (
        "This application leverages a large language model enhanced with a web search tool "
        "to answer your questions related to __Langchain__. Simply type your query below, and the agent will perform "
        "a web search and generate an answer based on its findings."
    )
    st.markdown(app_description)

    # Get user query from the chat input
    user_query = st.chat_input("Your question")

    if user_query:
        # Display the user's query in the chat
        with st.chat_message("human"):
            st.markdown(user_query)

        # Call the agent to process the query and generate a response
        with st.spinner("Searching...", show_time=True):
            response = agent(user_query)

        # Display the agent's response in the chat
        with st.chat_message("ai"):
            st.markdown(response)


if __name__ == "__main__":
    main()
