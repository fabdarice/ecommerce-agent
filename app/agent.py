from typing import Annotated, Any, Dict, List, TypedDict
from langgraph.graph import Graph
from langgraph.prebuilt import ToolMessage
from operator import itemgetter
import json
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Load inventory data
with open("inventory.json", "r") as f:
    inventory = json.load(f)


# Define state type
class AgentState(TypedDict):
    messages: List[BaseMessage]
    intent: str
    matches: List[Dict]
    selected_item: Dict
    next_step: str


# Initialize LLM
llm = ChatOpenAI(temperature=0)


# Node 1: Greeting
def greet(state: AgentState) -> AgentState:
    response = llm.invoke(
        [
            HumanMessage(
                content="You are an ecommerce assistant. Generate a friendly greeting asking what the user would like to purchase today."
            )
        ]
    )
    state["messages"].append(response)
    state["next_step"] = "check_intent"
    return state


# Node 2: Intent Classification
def check_intent(state: AgentState) -> AgentState:
    last_user_message = state["messages"][-2].content  # Get the user's last message

    response = llm.invoke(
        [
            HumanMessage(
                content=f"""
        Determine if the following message is about purchasing a product or not.
        Message: {last_user_message}
        Return only 'purchase' or 'not_purchase'.
        """
            )
        ]
    )

    state["intent"] = response.content.strip()
    state["next_step"] = (
        "search_inventory" if state["intent"] == "purchase" else "handle_non_purchase"
    )
    return state


# Node 3: Search Inventory
def search_inventory(state: AgentState) -> AgentState:
    last_user_message = state["messages"][-2].content

    # Use LLM to search inventory
    prompt = f"""
    Given the user query: {last_user_message}
    And the following inventory: {json.dumps(inventory)}
    Return the top 3 matching items in JSON format: [{{"name": "", "description": "", "price": "", "delivery": ""}}]
    Only return the JSON array, nothing else.
    """

    response = llm.invoke([HumanMessage(content=prompt)])

    try:
        matches = json.loads(response.content)
        state["matches"] = matches
        state["next_step"] = "present_options" if matches else "handle_no_matches"
    except json.JSONDecodeError:
        state["matches"] = []
        state["next_step"] = "handle_no_matches"

    return state


# Node 4: Present Options
def present_options(state: AgentState) -> AgentState:
    matches = state["matches"]

    prompt = f"""
    Present these product matches to the user and ask them to select one by number (1-3):
    {json.dumps(matches)}
    Format the response nicely with prices and descriptions.
    """

    response = llm.invoke([HumanMessage(content=prompt)])
    state["messages"].append(response)
    state["next_step"] = "handle_selection"
    return state


# Node 5: Handle Selection
def handle_selection(state: AgentState) -> AgentState:
    last_user_message = state["messages"][-2].content

    try:
        selection = int(last_user_message) - 1
        if 0 <= selection < len(state["matches"]):
            state["selected_item"] = state["matches"][selection]

            prompt = f"""
            Ask the user to confirm their selection of:
            {json.dumps(state["selected_item"])}
            Ask them to respond with Y or N.
            """

            response = llm.invoke([HumanMessage(content=prompt)])
            state["messages"].append(response)
            state["next_step"] = "confirm_purchase"
        else:
            state["messages"].append(
                AIMessage(
                    content="Invalid selection. Please choose a number between 1 and 3."
                )
            )
            state["next_step"] = "present_options"
    except ValueError:
        state["messages"].append(
            AIMessage(content="Please enter a valid number between 1 and 3.")
        )
        state["next_step"] = "present_options"

    return state


# Node 6: Handle Non-Purchase Intent
def handle_non_purchase(state: AgentState) -> AgentState:
    response = llm.invoke(
        [
            HumanMessage(
                content="Generate a polite message explaining that this agent can only assist with product purchases, and ask what they would like to purchase."
            )
        ]
    )
    state["messages"].append(response)
    state["next_step"] = "check_intent"
    return state


# Node 7: Handle No Matches
def handle_no_matches(state: AgentState) -> AgentState:
    response = llm.invoke(
        [
            HumanMessage(
                content="Generate a message informing the user that no matches were found and asking them what else they would like to purchase."
            )
        ]
    )
    state["messages"].append(response)
    state["next_step"] = "check_intent"
    return state


# Create the graph
def create_graph():
    workflow = Graph()

    # Add nodes
    workflow.add_node("greet", greet)
    workflow.add_node("check_intent", check_intent)
    workflow.add_node("search_inventory", search_inventory)
    workflow.add_node("present_options", present_options)
    workflow.add_node("handle_selection", handle_selection)
    workflow.add_node("handle_non_purchase", handle_non_purchase)
    workflow.add_node("handle_no_matches", handle_no_matches)

    # Add edges
    workflow.add_edge("greet", "check_intent")
    workflow.add_conditional_edges(
        "check_intent",
        lambda x: x["next_step"],
        {
            "search_inventory": "search_inventory",
            "handle_non_purchase": "handle_non_purchase",
        },
    )
    workflow.add_conditional_edges(
        "search_inventory",
        lambda x: x["next_step"],
        {
            "present_options": "present_options",
            "handle_no_matches": "handle_no_matches",
        },
    )
    workflow.add_edge("present_options", "handle_selection")
    workflow.add_edge("handle_non_purchase", "check_intent")
    workflow.add_edge("handle_no_matches", "check_intent")

    return workflow.compile()


# Initialize the graph
graph = create_graph()


# Flask routes
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    message = data.get("message", "")

    # Initialize or get conversation state
    state = {
        "messages": [HumanMessage(content=message)],
        "intent": "",
        "matches": [],
        "selected_item": {},
        "next_step": "greet" if not message else "check_intent",
    }

    # Run the graph
    final_state = graph(state)

    # Extract the last message
    last_message = final_state["messages"][-1].content

    return jsonify(
        {
            "response": last_message,
            "state": {
                "intent": final_state["intent"],
                "matches": final_state["matches"],
                "selected_item": final_state["selected_item"],
                "next_step": final_state["next_step"],
            },
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
