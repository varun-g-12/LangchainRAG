import json
import logging
import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai.chat_models import ChatOpenAI
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_rag.tool import search_tool

# Configure the module logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Constants
TOOLS = [search_tool]
MODEL = "gpt-4o-mini"
TEMP = 0.0
SYSTEM_PROMPT = (
    "You are having conversion with python developer. You will be having access to websearch tool and user's query.\n"
    "Your task is to do the websearch and answer the user's question."
)


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def load_env_variables(key: str = "ENV_VARIABLES") -> None:
    """Loads environment variables from a .env file and updates os.environ with JSON config."""
    logger.info("Loading environment variables using dotenv.")
    load_dotenv()
    if key not in os.environ:
        error_msg = f"Environment variable '{key}' not found."
        logger.error(error_msg)
        raise ValueError(error_msg)

    config = os.getenv(key)
    if config:
        try:
            config_json = json.loads(config)
            os.environ.update(config_json)
            logger.info("Environment variables loaded and updated successfully.")
        except json.JSONDecodeError as e:
            logger.error(
                "Failed to decode JSON from environment variable '%s': %s", key, e
            )
            raise


def brain(state: AgentState) -> dict[str, BaseMessage]:
    """Processes the input messages using the language model and returns the result."""
    message_count = len(state["messages"])
    logger.info("Executing brain function with %d message(s).", message_count)

    llm = ChatOpenAI(model=MODEL, temperature=TEMP).bind_tools(TOOLS)
    response = llm.invoke(state["messages"])
    logger.info("Language model invocation complete. Received response.")

    return {"messages": response}


def get_graph_instance() -> CompiledStateGraph:
    """Creates and compiles the state graph for the agent workflow."""
    logger.info("Initializing the state graph instance.")
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node(brain)
    graph.add_node("tools", ToolNode(TOOLS))
    logger.info("Added nodes: 'brain' and 'tools'.")

    # Add edges between nodes
    graph.add_edge(START, "brain")
    logger.info("Added edge: START -> brain.")

    graph.add_conditional_edges("brain", tools_condition)
    logger.info(
        "Added conditional edge(s) from 'brain' to 'tools' based on tools_condition."
    )

    graph.add_edge("tools", "brain")
    logger.info("Added edge: tools -> brain.")

    graph.add_edge("tools", END)
    logger.info("Added edge: tools -> END.")

    compiled_graph = graph.compile()
    logger.info("State graph compiled successfully.")
    return compiled_graph


def agent(query: str) -> str:
    """Main agent function that processes the user query through the graph."""
    logger.info("Agent received query: '%s'", query)
    load_env_variables()

    messages = [SystemMessage(SYSTEM_PROMPT), HumanMessage(f"User's query: {query}")]
    logger.info("Created system and human messages for the query.")

    graph = get_graph_instance()
    logger.info("Graph instance obtained. Invoking the graph with prepared messages.")

    result = graph.invoke({"messages": messages})
    logger.info("Graph invocation complete.")

    final_content = result["messages"][-1].content
    logger.info("Final response message extracted from the graph output.")
    return final_content


def main() -> None:
    """Interactive loop to handle user queries."""
    from langchain_community.callbacks import get_openai_callback

    logger.info("Starting interactive agent loop.")
    try:
        while True:
            query = input("question: ").strip()
            if query.lower() in ["q", "quit", "exit"]:
                logger.info("Exit command received. Terminating interactive loop.")
                break

            with get_openai_callback() as cb:
                response = agent(query)
                logger.info("Agent produced a response.")
                print(cb)

            print("answer:\n")
            print(response)
    except Exception as e:
        logger.exception("An error occurred in the interactive loop: %s", e)


if __name__ == "__main__":
    main()
