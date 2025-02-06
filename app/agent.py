from pydantic import BaseModel, ValidationError
from typing import List, Literal, Optional, TypedDict
import json
from uuid import uuid4
from dotenv import load_dotenv

from langgraph.graph import add_messages
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool
from langchain_core.load.dump import dumps
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser

from flask import Flask, request, jsonify
from flask_cors import CORS
from langgraph.graph.message import Annotated, BaseMessage, Sequence

_ = load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)


DB_URI = "postgresql://fabda@localhost:5432/agentshop?sslmode=disable"


class IntentResponse(BaseModel):
    intent: str


class Inventory(BaseModel):
    name: str
    description: str
    price: str
    delivery: str


class SearchInventoryResponse(BaseModel):
    matches: List[Inventory]


class InventorySelectionResponse(BaseModel):
    selection: int


class SelectionConfirmationResponse(BaseModel):
    selection: Literal["Y", "N", "y", "n"]


# Load inventory data
with open("app/inventory.json", "r") as f:
    inventory = json.load(f)


# Define state type
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    matches: List[Inventory]
    selected_item: Optional[Inventory]
    next_step: str
    require_user_input: bool


# Initialize LLM
llm = ChatOpenAI(temperature=0, model="gpt-4o-mini-2024-07-18")


# Node 1: Greeting
def greet(state: AgentState) -> AgentState:
    print("-------------------- ENTER GREET ---------------------")

    # Check if a human response has already been provided.
    if (
        state["messages"]
        and isinstance(state["messages"][-1], HumanMessage)
        and state["messages"][-1].content != ""
    ):
        state["next_step"] = "check_intent"
        state["require_user_input"] = False
        return state

    greet_message = SystemMessage(
        content="You are an ecommerce chatbot assistant. Generate a friendly greeting asking what the user would like to purchase today."
    )

    response = llm.invoke([greet_message])

    return {
        "messages": [response],
        "next_step": "check_intent",
        "require_user_input": True,
        "matches": [],
        "selected_item": None,
    }


# Node 2: Intent Classification
def check_intent(state: AgentState) -> AgentState:
    print("----------------ENTER CHECK_INTENT--------------")
    last_message = state["messages"][-1]
    print("LAST USER MESSAGE:", last_message.content)
    pydantic_parser = PydanticOutputParser(pydantic_object=IntentResponse)

    response = llm.invoke(
        [
            SystemMessage(
                content="You are an ecommerce chatbot assistant. Generate a friendly greeting asking what the user would like to purchase today."
            ),
            AIMessage(
                content=f"""
                    Determine if the following message is about a product or an intent to purchase a product or not.
                    Message: {last_message.content}
                    Return only 'purchase' or 'not_purchase' in the following format {pydantic_parser.get_format_instructions()}.
                """
            ),
        ]
    )

    try:
        if type(response.content) != str:
            state["next_step"] = "handle_non_purchase"
            return state

        parsed_output = pydantic_parser.parse(response.content)
        if parsed_output.intent == "purchase":
            state["next_step"] = "search_inventory"
        else:
            state["next_step"] = "handle_non_purchase"
    except ValidationError as e:
        print("Validation error:", e)
        state["next_step"] = "handle_non_purchase"

    return state


# Node 3: Search Inventory
def search_inventory(state: AgentState) -> AgentState:
    print("-------------ENTER SEARCH_INVENTORY--------------")
    last_message = state["messages"][-2]
    print("LAST USER MESSAGE:", last_message.content)

    pydantic_parser = PydanticOutputParser(pydantic_object=SearchInventoryResponse)

    # Use LLM to search inventory
    prompt = f"""
        Given the user query: {last_message.content}
        And the following inventory: {json.dumps(inventory)}
        Return up to 3 matching items in the following format: {pydantic_parser.get_format_instructions()}.
        Only return the items that match the user query by matching on the name or description.
    """

    response = llm.invoke([AIMessage(content=prompt)])

    if not response or not response.content or type(response.content) != str:
        state["next_step"] = "handle_no_matches"
        return state

    try:
        matches = pydantic_parser.parse(response.content).matches
        state["matches"] = matches
        state["next_step"] = "present_options" if matches else "handle_no_matches"
        state["messages"] = [response]
    except Exception:
        state["next_step"] = "handle_no_matches"
        state["matches"] = []

    return state


# Node 4: Present Options
def present_options(state: AgentState) -> AgentState:
    print("-------------ENTER PRESENT_OPTIONS--------------")
    # Check if a human response has already been provided.
    if state["messages"] and isinstance(state["messages"][-1], HumanMessage):
        state["next_step"] = "handle_selection"
        state["require_user_input"] = False
        return state

    matches = state["matches"]

    prompt = f"""
    Present these product matches to the user and ask them to select one by number (1-{len(state['matches'])}):
    {dumps(matches)}
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
            AIMessage(
                content=f"Please enter a valid number between 1 and {len(state['matches'])}."
            )
        ]
        state["next_step"] = "present_options"
        return state

    try:
        pydantic_parser = PydanticOutputParser(
            pydantic_object=InventorySelectionResponse
        )
        prompt = f"""
            You have presented three options to the user {dumps(state['matches'])}.
            Extract the number selected by the user from the following message {last_user_message.content}
            Format the response with the following format {pydantic_parser.get_format_instructions()}.
        """
        response = llm.invoke([AIMessage(content=prompt)])
        if type(response.content) != str:
            raise Exception("Response content is not a string.")

        selection = pydantic_parser.parse(response.content).selection
        if 0 < selection <= len(state["matches"]):
            state["selected_item"] = state["matches"][selection - 1]

            pydantic_parser = PydanticOutputParser(
                pydantic_object=SelectionConfirmationResponse
            )
            prompt = f"""
            The user has selected the following item: {dumps(state["selected_item"])}.
            Ask the user to confirm their selection by responding with Y or N.
            Format the response with the following format {pydantic_parser.get_format_instructions()}.
            """

            response = llm.invoke([AIMessage(content=prompt)])

            if type(response.content) != str:
                raise Exception("Response content is not a string.")

            confirmation = pydantic_parser.parse(response.content).selection
            if confirmation.lower() == "y":
                state["next_step"] = "purchase_item_onchain"
            else:
                state["next_step"] = "present_options"
        else:
            raise Exception("Number invalid.")
    except Exception:
        state["messages"] = [
            AIMessage(
                content=f"Please enter a valid number between 1 and {len(state['matches'])}."
            )
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


# Node 8: Purchase Item Onchain via CB Commerce
def purchase_item_onchain(state: AgentState) -> AgentState:
    print("-------------ENTER PURCHASE_ITEM_ONCHAIN--------------")
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
    graph.add_node("purchase_item_onchain", purchase_item_onchain)

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
    graph.add_conditional_edges(
        "handle_selection",
        lambda x: x["next_step"],
        {
            "present_options": "present_options",
            "purchase_item_onchain": "purchase_item_onchain",
        },
    )
    graph.add_edge("present_options", "handle_selection")
    graph.add_edge("handle_non_purchase", "check_intent")
    graph.add_edge("handle_no_matches", "check_intent")

    # Add Entrypoints and Finish points
    graph.set_entry_point("greet")
    graph.set_finish_point("purchase_item_onchain")

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
    print("EXIT API RESPONSE", final_state)

    return jsonify(
        {
            "response": last_message,
            "session_id": session_id,
            # "state": {
            #     "matches": final_state["matches"],
            #     "selected_item": (
            #         final_state["selected_item"]
            #         if "selected_item" in final_state
            #         else None
            #     ),
            #     "next_step": final_state["next_step"],
            # },
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
