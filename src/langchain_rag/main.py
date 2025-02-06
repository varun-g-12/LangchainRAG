import os
import logging
from typing import TypedDict, Annotated

from langgraph.graph.state import CompiledStateGraph
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as convert_html_to_markdown

from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.tools import tool

# ------------------------------------------------------------------------------
# Logging Configuration
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
)


# ------------------------------------------------------------------------------
# Custom Exceptions
# ------------------------------------------------------------------------------
class EnvVariableError(Exception):
    """Exception raised when a required environment variable is missing."""

    pass


class PageNotFound(Exception):
    """Exception raised when the expected HTML element is not found on a webpage."""

    pass


# ------------------------------------------------------------------------------
# Typed Dictionary to maintain Agent state
# ------------------------------------------------------------------------------
class AgentState(TypedDict):
    # The conversation history between the agent and the tools.
    messages: Annotated[list[BaseMessage], add_messages]


# ------------------------------------------------------------------------------
# Tool Function
# ------------------------------------------------------------------------------
@tool
def fetch_webpage_content(url: str) -> str:
    """
    Retrieves and converts a webpage's content to markdown.

    This function performs an HTTP GET request to the specified URL, searches for the first
    <article> element in the HTML content using BeautifulSoup, and converts the HTML to
    markdown using the markdownify utility. If the <article> element is missing, a PageNotFound
    exception is raised.

    Args:
        url: The URL of the webpage to retrieve.

    Returns:
        str: The markdown-formatted content from the <article> element.

    Raises:
        HTTPError: If the HTTP GET request does not return a 200 status code.
        PageNotFound: If the <article> element is not found in the webpage.
    """
    logging.info(f"Fetching webpage content from URL: {url}")
    response = requests.get(url)

    # Check for a successful HTTP response
    if response.status_code != 200:
        logging.error(
            f"Failed to fetch URL: {url} with status code: {response.status_code}"
        )
        response.raise_for_status()

    # Parse the HTML content and search for the <article> element
    article_tag = BeautifulSoup(response.text, "html.parser").find("article")
    if not article_tag:
        logging.error(f"No <article> element found in the webpage: {url}")
        raise PageNotFound(f"Page not found for URL: {url}")

    # Convert the article HTML content to markdown format
    page_markdown = convert_html_to_markdown(str(article_tag))
    logging.info(f"Successfully converted webpage content to markdown for URL: {url}")
    return page_markdown


# List of available tools for the agent (currently only one tool)
TOOLS = [fetch_webpage_content]

# ------------------------------------------------------------------------------
# System Prompt for the Agent
# ------------------------------------------------------------------------------
SYSTEM_PROMPT = """
**Context:**  
You are engaged in a conversation with a Python developer. Your expertise lies in answering questions related to LangChain and LangGraph documentation.

**Input:**  
- A user's query is provided.  
- You have access to a tool for retrieving page content.

**Primary Task:**  
1. **Initial Retrieval:**  
   - Immediately invoke the `fetch_webpage_content` tool using the URL:  
     `https://python.langchain.com/docs/introduction/`  
   - This fetches the introduction page content.

2. **Content Exploration:**  
   - From the retrieved introduction content, identify any additional URLs that may be relevant to the user’s query.
   - **URL Correction:** If an identified URL is incomplete (e.g., it starts with `/docs/tutorials/`), prepend it with:  
     `https://python.langchain.com`  
     to form the complete URL (i.e., `https://python.langchain.com/docs/tutorials/`).

3. **Iterative Process:**  
   - Use the tool to fetch content from each complete URL you find.
   - Continue to explore any further URLs on these sub-pages that could help answer the query.
   - Repeat this process until you have gathered all necessary content to address the user's query comprehensively.
"""


# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------
def validate_env_variable(variable_key: str) -> None:
    """
    Validates that the required environment variable is set.

    Args:
        variable_key (str): The name of the environment variable.

    Raises:
        EnvVariableError: If the environment variable is not found.
    """
    logging.info(f"Validating environment variable: {variable_key}")
    if variable_key not in os.environ:
        error_message = f"Environment variable '{variable_key}' not found. Please check your .env file."
        logging.error(error_message)
        raise EnvVariableError(error_message)
    logging.info(f"Environment variable '{variable_key}' is set.")


def create_llm_instance(
    model: str = "gpt-4o-mini", temperature: float = 0.0
) -> ChatOpenAI:
    """
    Creates and returns an instance of the ChatOpenAI model bound with the available tools.

    Args:
        model (str, optional): The model identifier to use. Defaults to "gpt-4o-mini".
        temperature (float, optional): The temperature parameter for the model. Defaults to 0.0.

    Returns:
        ChatOpenAI: An instance of the ChatOpenAI model.
    """
    logging.info(
        f"Initializing LLM instance with model: {model} and temperature: {temperature}"
    )
    llm_instance = ChatOpenAI(model=model, temperature=temperature)
    return llm_instance


def process_agent_interaction(agent_state: AgentState) -> dict[str, list[BaseMessage]]:
    """
    Processes agent interaction using the LLM instance.

    This function retrieves the conversation history from the agent's state, sends it to the
    language model for processing, and returns the response message.

    Args:
        agent_state (AgentState): The current state of the agent including conversation history.

    Returns:
        dict[str, list[BaseMessage]]: A dictionary containing the updated conversation history.
    """
    logging.info("Processing agent interaction with the language model.")

    # Create and bind the language model with tools
    llm_instance = create_llm_instance().bind_tools(TOOLS)

    conversation_history = agent_state["messages"]
    logging.debug(f"Conversation history: {conversation_history}")

    # Invoke the language model with the current conversation history
    response_message = llm_instance.invoke(conversation_history)

    logging.info("LLM processed the conversation. Returning updated messages.")
    return {"messages": [response_message]}


def build_agent_graph() -> CompiledStateGraph:
    """
    Constructs and compiles the state graph for the agent's workflow.

    The state graph defines the sequence of operations between the language model (agent)
    and the tool nodes. Conditional edges are added based on tool usage.

    Returns:
        A compiled graph instance ready for invocation.
    """
    logging.info("Building the agent state graph.")
    agent_graph = StateGraph(AgentState)

    # Add the agent interaction node
    agent_graph.add_node(process_agent_interaction)

    # Add the tool node for handling webpage content fetching
    agent_graph.add_node("tools", ToolNode(tools=TOOLS))

    # Define the workflow edges
    agent_graph.add_edge(START, "process_agent_interaction")
    agent_graph.add_conditional_edges("process_agent_interaction", tools_condition)
    agent_graph.add_edge("tools", "process_agent_interaction")
    agent_graph.add_edge("process_agent_interaction", END)

    logging.info("Agent state graph successfully built and compiled.")
    return agent_graph.compile()


def main(user_query: str) -> str:
    """
    Main function to process a user's query.

    This function validates the required environment variable, builds the agent state graph,
    initializes the conversation messages, and returns the final response content from the LLM.

    Args:
        user_query (str): The query posed by the user.

    Returns:
        str: The content of the final message returned by the agent.
    """
    logging.info("Starting main process for user query processing.")

    # Validate that the OpenAI API key is available in the environment
    validate_env_variable("OPENAI_API_KEY")

    # Build the agent's state graph instance
    compiled_agent_graph = build_agent_graph()

    # Initialize conversation messages with the system prompt and the user's query
    initial_messages = [
        SystemMessage(SYSTEM_PROMPT),
        HumanMessage(f"User's query: {user_query}"),
    ]
    logging.info("Initial conversation messages prepared.")

    # Invoke the agent graph with the initial messages and retrieve the response
    result = compiled_agent_graph.invoke({"messages": initial_messages})
    final_message = result["messages"][-1].content
    logging.info("Agent processing complete. Returning final response.")

    return final_message


# ------------------------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Example user query for the LLM agent
    sample_query = "How can I create a custom tool using tool decorator ?"
    logging.info("Invoking main function with sample query.")

    try:
        response_content = main(sample_query)
        print(response_content)
    except Exception as e:
        logging.error(f"An error occurred during processing: {e}")
