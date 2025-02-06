from typing import Dict, List, TypedDict
import json
from uuid import uuid4
from dotenv import load_dotenv

from langchain.utils import parse_json_markdown
from langgraph.graph import add_messages
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool


from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

from flask import Flask, request, jsonify
from flask_cors import CORS
from langgraph.graph.message import Annotated, BaseMessage, Sequence

_ = load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)


DB_URI = "postgresql://fabda@localhost:5432/agentshop?sslmode=disable"


# Load inventory data
with open("app/inventory.json", "r") as f:
    inventory = json.load(f)


# Define state type
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    matches: List[Dict]
    selected_item: Dict
    next_step: str
    require_user_input: bool


# Initialize LLM
llm = ChatOpenAI(temperature=0, model="gpt-4o-mini-2024-07-18")


# Node 1: Greeting
def greet(state: AgentState) -> AgentState:
    print("-------------------- ENTER GREET ---------------------")
    greet_message = SystemMessage(
        content="You are an ecommerce chatbot assistant. Generate a friendly greeting asking what the user would like to purchase today."
    )

    response = llm.invoke([greet_message])

    return {
        "messages": [response],
        "next_step": "check_intent",
        "require_user_input": False,
        "matches": [],
        "selected_item": {},
    }


# Node 2: Intent Classification
def check_intent(state: AgentState) -> AgentState:
    print("----------------ENTER CHECK_INTENT--------------")
    # print(state)
    last_message = state["messages"][-2]
    # breakpoint()
    # if not last_message or not isinstance(last_message, HumanMessage):
    #     state["require_user_input"] = True
    #     return state

    # TODO: Is this -2 or -1?
    print("LAST USER MESSAGE:", last_message.content)

    response = llm.invoke(
        [
            SystemMessage(
                content="You are an ecommerce chatbot assistant. Generate a friendly greeting asking what the user would like to purchase today."
            ),
            AIMessage(
                content=f"""
        Determine if the following message is about a product or an intent to purchase a product or not.
        Message: {last_message.content}
        Return only 'purchase' or 'not_purchase'.

        Example: 
        1. "I would like to buy an iPhone 14" -> "purchase"
        2. "Iphone" -> "purchase"
        3. "What is the weather today?" -> "not_purchase"
        """
            ),
        ]
    )

    print("INTENT RESPONSE:", response.content)

    if (
        not response
        or not response.content
        or type(response.content) != str
        or response.content.strip() == "not_purchase"
    ):
        state["next_step"] = "handle_non_purchase"
        return state

    state["next_step"] = "search_inventory"
    return state


# Node 3: Search Inventory
def search_inventory(state: AgentState) -> AgentState:
    print("-------------ENTER SEARCH_INVENTORY--------------")
    last_message = state["messages"][-2]
    print("LAST USER MESSAGE:", last_message.content)

    # Use LLM to search inventory
    prompt = f"""
    Given the user query: {last_message.content}
    And the following inventory: {json.dumps(inventory)}
    Return the top 3 matching items in JSON format: [{{"name": "", "description": "", "price": "", "delivery": ""}}]
    Only return the JSON array, nothing else.
    """

    print("PROMPT:", prompt)

    response = llm.invoke([AIMessage(content=prompt)])

    print("RESPONSE SEARCH INVENTORY:", response.content)

    breakpoint()

    if not response or not response.content or type(response.content) != str:
        state["next_step"] = "handle_no_matches"
        return state

    try:
        matches = parse_json_markdown(response.content)
        state["matches"] = matches
        state["next_step"] = "present_options" if matches else "handle_no_matches"
    except json.JSONDecodeError:
        state["matches"] = []
        state["next_step"] = "handle_no_matches"

    return state


# Node 4: Present Options
def present_options(state: AgentState) -> AgentState:
    print("-------------ENTER PRESENT_OPTIONS--------------")
    matches = state["matches"]

    prompt = f"""
    Present these product matches to the user and ask them to select one by number (1-3):
    {json.dumps(matches)}
    Format the response nicely with prices and descriptions.
    """

    response = llm.invoke([AIMessage(content=prompt)])
    state["messages"] = [response]
    state["next_step"] = "handle_selection"
    state["require_user_input"] = True
    return state


# Node 5: Handle Selection
def handle_selection(state: AgentState) -> AgentState:
    print("-------------ENTER HANDLE_SELECTION--------------")
    last_user_message = state["messages"][-1]

    if not last_user_message or not isinstance(last_user_message, HumanMessage):
        state["messages"] = [
            AIMessage(content="Please enter a valid number between 1 and 3.")
        ]
        state["next_step"] = "present_options"
        return state

    try:
        selection = int(last_user_message.content) - 1
        if 0 <= selection < len(state["matches"]):
            state["selected_item"] = state["matches"][selection]

            prompt = f"""
            Ask the user to confirm their selection of:
            {json.dumps(state["selected_item"])}
            Ask them to respond with Y or N.
            """

            response = llm.invoke([AIMessage(content=prompt)])
            state["messages"] = [response]
            state["next_step"] = "confirm_purchase"
        else:
            state["messages"] = [
                AIMessage(
                    content="Invalid selection. Please choose a number between 1 and 3."
                )
            ]
            state["next_step"] = "present_options"
    except ValueError:
        state["messages"] = [
            AIMessage(content="Please enter a valid number between 1 and 3.")
        ]
        state["next_step"] = "present_options"

    return state


# Node 6: Handle Non-Purchase Intent
def handle_non_purchase(state: AgentState) -> AgentState:
    print("-------------ENTER HANDLE_NON_PURCHASE--------------")
    response = llm.invoke(
        [
            AIMessage(
                content="Generate a message explaining that this agent can only assist with product purchases, and ask what they would like to purchase. The message format is catered for chatbot responses."
            )
        ]
    )
    state["messages"] = [response]
    state["next_step"] = "check_intent"
    state["require_user_input"] = True
    return state


# Node 7: Handle No Matches
def handle_no_matches(state: AgentState) -> AgentState:
    print("-------------ENTER HANDLE_NO_MATCHES--------------")
    response = llm.invoke(
        [
            AIMessage(
                content="Generate a message informing the user that no matches were found and asking them what else they would like to purchase. The message format is catered for chatbot responses."
            )
        ]
    )
    state["messages"] = [response]
    state["next_step"] = "check_intent"
    state["require_user_input"] = True
    return state


# Create the graph
def create_graph(checkpointer):
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("greet", greet)
    graph.add_node("check_intent", check_intent)
    graph.add_node("search_inventory", search_inventory)
    graph.add_node("present_options", present_options)
    graph.add_node("handle_selection", handle_selection)
    graph.add_node("handle_non_purchase", handle_non_purchase)
    graph.add_node("handle_no_matches", handle_no_matches)

    # Add edges
    graph.add_edge("greet", "check_intent")
    graph.add_conditional_edges(
        "check_intent",
        lambda x: x["next_step"],
        {
            "search_inventory": "search_inventory",
            "handle_non_purchase": "handle_non_purchase",
        },
    )
    graph.add_conditional_edges(
        "search_inventory",
        lambda x: x["next_step"],
        {
            "present_options": "present_options",
            "handle_no_matches": "handle_no_matches",
        },
    )
    graph.add_edge("present_options", "handle_selection")
    graph.add_edge("handle_non_purchase", "check_intent")
    graph.add_edge("handle_no_matches", "check_intent")

    # Add Entrypoints and Finish points
    graph.set_entry_point("greet")
    graph.set_finish_point("handle_selection")

    agent_bot = graph.compile(checkpointer=checkpointer)

    print(agent_bot.get_graph().draw_ascii())
    return agent_bot


# Initialize the connection pool globally.
pool = ConnectionPool(
    conninfo=DB_URI,
    max_size=20,
    kwargs={"autocommit": True, "prepare_threshold": 0},
)

checkpointer = PostgresSaver(pool)
checkpointer.setup()

# Create the agent bot only once.
agent_bot = create_graph(checkpointer)


# Flask routes
@app.route("/agent", methods=["POST"])
def agent():
    """
    This endpoint is called repeatedly by the client with user input.
    JSON Example:
    {
      "session_id": "...",        # optional, if not provided we create a new session
      "nessage": "iPhone 14"   # or "1", or "y", etc. based on the conversation
    }
    """

    if agent_bot is None:
        return jsonify({"response": "An error occurred."})
    data = request.json or {}
    message = data.get("message", "")
    session_id = data.get("session_id")

    content = "" if not session_id else HumanMessage(content=message)

    if not session_id:
        session_id = str(uuid4())
    #
    config: RunnableConfig = {
        "configurable": {"thread_id": session_id},
    }

    # checkpoint = checkpointer.get(config)
    # print("CHECKPOINT:", checkpoint)
    final_state = None
    for chunk in agent_bot.stream({"messages": [content]}, config):
        final_state = list(chunk.values())[-1]
        print(final_state["messages"][-1].pretty_print())
        if final_state["require_user_input"] is True:
            break
    #
    if not final_state:
        return jsonify({"response": "An error occurred."})

    # Extract the last message
    last_message = final_state["messages"][-1].content

    return jsonify(
        {
            "response": last_message,
            "session_id": session_id,
            "state": {
                "matches": final_state["matches"],
                "selected_item": final_state["selected_item"],
                "next_step": final_state["next_step"],
            },
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
