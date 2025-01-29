from typing import Dict, List, TypedDict
from langgraph.graph import StateGraph
import json
from uuid import uuid4
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from flask import Flask, request, jsonify
from flask_cors import CORS

_ = load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)


# Load inventory data
with open("app/inventory.json", "r") as f:
    inventory = json.load(f)


# Define state type
class AgentState(TypedDict):
    messages: List[BaseMessage]
    matches: List[Dict]
    selected_item: Dict
    next_step: str
    require_user_input: bool


# Initialize LLM
llm = ChatOpenAI(temperature=0, model="gpt-4o-mini-2024-07-18")


# Node 1: Greeting
def greet(state: AgentState) -> AgentState:
    response = llm.invoke(
        [
            SystemMessage(
                content="You are an ecommerce assistant. Generate a friendly greeting asking what the user would like to purchase today. The conversation is via chatbox."
            )
        ]
    )
    state["messages"].append(response)
    state["next_step"] = "check_intent"
    state["require_user_input"] = True
    return state


# Node 2: Intent Classification
def check_intent(state: AgentState) -> AgentState:
    # TODO: Is this -2 or -1?
    last_user_message = state["messages"][-1].content  # Get the user's last message

    response = llm.invoke(
        [
            AIMessage(
                content=f"""
        Determine if the following message is about purchasing a product or not.
        Message: {last_user_message}
        Return only 'purchase' or 'not_purchase'.
        """
            )
        ]
    )

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
    last_user_message = state["messages"][-2].content

    # Use LLM to search inventory
    prompt = f"""
    Given the user query: {last_user_message}
    And the following inventory: {json.dumps(inventory)}
    Return the top 3 matching items in JSON format: [{{"name": "", "description": "", "price": "", "delivery": ""}}]
    Only return the JSON array, nothing else.
    """

    response = llm.invoke([HumanMessage(content=prompt)])

    if not response or not response.content or type(response.content) != dict:
        state["next_step"] = "handle_no_matches"
        return state

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

    response = llm.invoke([AIMessage(content=prompt)])
    state["messages"].append(response)
    state["next_step"] = "handle_selection"
    state["require_user_input"] = True
    return state


# Node 5: Handle Selection
def handle_selection(state: AgentState) -> AgentState:
    last_user_message = state["messages"][-1]

    if not last_user_message or not isinstance(last_user_message, HumanMessage):
        state["messages"].append(
            AIMessage(content="Please enter a valid number between 1 and 3.")
        )
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
            AIMessage(
                content="Generate a polite message explaining that this agent can only assist with product purchases, and ask what they would like to purchase."
            )
        ]
    )
    state["messages"].append(response)
    state["next_step"] = "check_intent"
    state["require_user_input"] = True
    return state


# Node 7: Handle No Matches
def handle_no_matches(state: AgentState) -> AgentState:
    response = llm.invoke(
        [
            AIMessage(
                content="Generate a message informing the user that no matches were found and asking them what else they would like to purchase."
            )
        ]
    )
    state["messages"].append(response)
    state["next_step"] = "check_intent"
    state["require_user_input"] = True
    return state


# Create the graph
def create_graph():
    workflow = StateGraph(AgentState)

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

    # Add Entrypoints and Finish points
    workflow.set_entry_point("greet")
    workflow.set_finish_point("handle_selection")

    return workflow.compile()


# Initialize the graph
graph = create_graph()

SESSION_STATES: Dict[str, AgentState] = {}


# Flask routes
@app.route("/agent", methods=["POST"])
def agent():
    """
    This endpoint is called repeatedly by the client with user input.
    JSON Example:
    {
      "session_id": "...",        # optional, if not provided we create a new session
      "user_input": "iPhone 14"   # or "1", or "y", etc. based on the conversation
    }
    """
    data = request.json or {}
    message = data.get("message", "")
    session_id = data.get("session_id")

    # If no session_id, create a new conversation session
    if not session_id or session_id not in SESSION_STATES:
        # Initialize or get conversation state
        initial_state: AgentState = {
            "messages": [
                SystemMessage(
                    content="You are an ecommerce assistant. Generate a friendly greeting asking what the user would like to purchase today. The conversation is via chatbox."
                )
            ],
            "matches": [],
            "selected_item": {},
            "next_step": "greet" if not message else "check_intent",
            "require_user_input": False,
        }

        session_id = str(uuid4())
        SESSION_STATES[session_id] = initial_state

    state = SESSION_STATES[session_id]
    state["messages"].append(HumanMessage(content=message))

    # Run the graph
    step_gen = graph.stream(state)
    final_state = None

    # Consume all steps in the generator to get the final state
    for step_state in step_gen:
        # Each step_state will be like {"node_name": actual_state}
        # Get the actual state from the dictionary
        print("ENTER HERE", step_state)
        final_state = list(step_state.values())[0]
        if final_state["require_user_input"] is True:
            break

    if not final_state:
        return jsonify({"response": "An error occurred."})

    print("FINAL STATE:", final_state)

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
