from decimal import Decimal
from langgraph.graph.state import Command
from langgraph.types import Interrupt, interrupt
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

from app.services.aave import Aave
from app.services.commerce import CommerceService, TransferIntent
from app.services.web3 import Web3


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
    url: str


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
    user_input: str
    matches: List[Inventory]
    selected_item: Optional[Inventory]
    next_step: str
    aave_tx: Optional[str]


# Initialize LLM
llm = ChatOpenAI(temperature=0, model="gpt-4o-mini-2024-07-18")

commerce_svc = CommerceService()

web3 = Web3()
aave = Aave(web3)


# Node 1: Greeting
def greet(state: AgentState) -> AgentState:
    print("-------------------- ENTER GREET ---------------------")

    return {
        "messages": [],
        "user_input": "",
        "next_step": "check_intent",
        "matches": [],
        "selected_item": None,
        "aave_tx": None,
    }


# Node 2: Intent Classification
def check_intent(state: AgentState) -> AgentState:
    print("----------------ENTER CHECK_INTENT--------------")
    state["user_input"] = interrupt(
        "Hello there! Welcome to Shop Pal. I’m here to help you find the perfect item. What would you like to purchase today?."
    )
    pydantic_parser = PydanticOutputParser(pydantic_object=IntentResponse)

    response = llm.invoke(
        [
            SystemMessage(
                content="You are an ecommerce chatbot assistant. Generate a friendly greeting asking what the user would like to purchase today."
            ),
            AIMessage(
                content=f"""
                    Determine if the following message is about a product or an intent to purchase a product or not.
                    Message: {state['user_input']}
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

    pydantic_parser = PydanticOutputParser(pydantic_object=SearchInventoryResponse)

    # Use LLM to search inventory
    prompt = f"""
        Given the user query: {state['user_input']}
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

    matches = state["matches"]

    prompt = f"""
You are given a list of product options. Your task is to present these options to the user in a nicely formatted manner that can be rendered by our front end. For each product, include the following details:
- **Name**
- **Price**
- **Delivery** details
- **Image URL** (use a placeholder or real URL if available)

After listing the products, prompt the user to select one by entering the corresponding number (1-{len(state['matches'])}).

Please format your response using Markdown.

Here are the product options:
    {dumps(matches)}

Now, generate a response that lists these options formatted in Markdown and asks the user to select one by number (1-3).
    """

    response = llm.invoke([AIMessage(content=prompt)])
    state["messages"] = [response]
    state["next_step"] = "handle_selection"
    return state


# Node 5: Handle Selection
def handle_selection(state: AgentState) -> AgentState:
    print("-------------ENTER HANDLE_SELECTION--------------")
    last_user_message = state["messages"][-1]
    user_selection_input = interrupt(last_user_message.content)

    try:
        pydantic_parser = PydanticOutputParser(
            pydantic_object=InventorySelectionResponse
        )
        prompt = f"""
            You have presented three options to the user {dumps(state['matches'])}.
            Extract the number selected by the user from the following message {user_selection_input}.
            Format the response with the following format {pydantic_parser.get_format_instructions()}.
        """
        response = llm.invoke([AIMessage(content=prompt)])
        if type(response.content) != str:
            raise Exception("Response content is not a string.")

        selection = pydantic_parser.parse(response.content).selection
        if 0 < selection <= len(state["matches"]):
            state["selected_item"] = state["matches"][selection - 1]
        else:
            raise Exception("Number invalid.")
    except Exception as e:
        print("Error: ", e)
        state["messages"] = [
            AIMessage(
                content=f"Please enter a valid number between 1 and {len(state['matches'])}."
            )
        ]
        state["next_step"] = "greet"

    return state


def handle_selection_confirmation(state: AgentState):
    print("-------------ENTER HANDLE_SELECTION CONFIRMATION --------------")
    if not state["selected_item"]:
        state["next_step"] = "present_options"
        return state

    user_selection_input = interrupt(
        f"You have selected '{state['selected_item'].name}<br><br>![{state['selected_item'].name}]({state['selected_item'].url}).<br><br>Would you like to confirm your selection?<br><br>Please reply with 'Y' for Yes or 'N' for No."
    )

    try:
        if user_selection_input.lower() == "y":
            if float(web3.balances("usdc")) < float(state["selected_item"].price):  # type: ignore
                state["next_step"] = "not_enough_funds"
            else:
                state["next_step"] = "purchase_item_onchain"
        elif user_selection_input.lower() == "n":
            state["next_step"] = "present_options"
        else:
            raise Exception("Number invalid.")
    except Exception as e:
        print("Error: ", e)
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
    state["next_step"] = "greet"
    # state["require_user_input"] = True
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
    state["next_step"] = "greet"
    # state["require_user_input"] = True
    return state


# Node 9: Purchase Item Onchain via CB Commerce
def purchase_item_onchain(state: AgentState) -> AgentState:
    print("-------------ENTER PURCHASE_ITEM_ONCHAIN--------------")
    if state["selected_item"] is None:
        return state
    charge_id: str = commerce_svc.create_charge(state["selected_item"])
    transfer_intent: TransferIntent = commerce_svc.hydrate_charge(
        charge_id, web3.address
    )
    print(f"Transfer Intent: {transfer_intent}")
    tx = web3.invoke_transfers_contract(
        transfer_intent.metadata.contract_address,
        transfer_intent.to_onchain_params,
    )
    print(f"Submitting Transaction Onchain: {tx}")
    tx.wait()
    print(
        f"Payment Complete. Receipt: https://commerce.coinbase.com/pay/{charge_id}/receipt"
    )
    msg = f"""
**Payment Complete!**

You will receive your items within {state['selected_item'].delivery}.
<br><br>
**Receipt:** [Click here to view your receipt](https://commerce.coinbase.com/pay/{charge_id}/receipt)
        """

    msg = (
        msg + f"<br>**Aave Withdrawal**: [Transaction]({state['aave_tx']})"
        if state.get("aave_tx")
        else msg
    )

    state["messages"] = [AIMessage(msg)]
    return state


# Node 8:  not enough funds
def not_enough_funds(state: AgentState) -> AgentState:
    # Retrieve the current balances
    current_wallet_balance = web3.balances("usdc")
    current_aave_deposit = aave.get_usdc_deposit_amount()
    user_confirm_aave = interrupt(
        f"**Insufficient funds for purchase**<br><br>"
        f"Your current USDC wallet balance: {current_wallet_balance} USDC<br><br>"
        f"I've detected that you currently have **{current_aave_deposit} USDC deposited on Aave**.<br> "
        "Would you like to withdraw funds from Aave to proceed with the purchase?<br><br>"
        "Please reply with 'Y' for Yes or 'N' for No."
    )

    try:
        if user_confirm_aave.lower() == "y":
            state["next_step"] = "withdraw_aave"
        elif user_confirm_aave.lower() == "n":
            state["next_step"] = "greet"
        else:
            raise Exception("Invalid option.")
    except Exception:
        state["messages"] = [AIMessage(content=f"Please enter yes or no.")]
        state["next_step"] = "present_options"

    return state


def withdraw_aave(state: AgentState) -> AgentState:
    current_aave_deposit = aave.get_usdc_deposit_amount()
    tx = aave.withdraw_usdc(current_aave_deposit)
    state["aave_tx"] = tx.transaction_link

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
    graph.add_node("handle_selection_confirmation", handle_selection_confirmation)
    graph.add_node("handle_non_purchase", handle_non_purchase)
    graph.add_node("handle_no_matches", handle_no_matches)
    graph.add_node("not_enough_funds", not_enough_funds)
    graph.add_node("withdraw_aave", withdraw_aave)
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
        "handle_selection_confirmation",
        lambda x: x["next_step"],
        {
            "present_options": "present_options",
            "purchase_item_onchain": "purchase_item_onchain",
            "not_enough_funds": "not_enough_funds",
        },
    )
    graph.add_conditional_edges(
        "not_enough_funds",
        lambda x: x["next_step"],
        {
            "greet": "greet",
            "withdraw_aave": "withdraw_aave",
        },
    )
    graph.add_edge("present_options", "handle_selection")
    graph.add_edge("handle_selection", "handle_selection_confirmation")
    graph.add_edge("withdraw_aave", "purchase_item_onchain")
    graph.add_edge("handle_non_purchase", "greet")
    graph.add_edge("handle_no_matches", "greet")

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


checkpointer = PostgresSaver(pool)  # type: ignore
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

    stream_content = (
        {"messages": HumanMessage(content="")}
        if not session_id or session_id == ""
        else Command(resume=message)
    )

    if not session_id:
        session_id = str(uuid4())
    #
    config: RunnableConfig = {
        "configurable": {"thread_id": session_id},
    }
    response_message = ""
    for chunk in agent_bot.stream(stream_content, config):
        if "__interrupt__" in chunk:
            response_message = chunk["__interrupt__"][0].value
        else:
            final_state = list(chunk.values())[-1]
            if final_state["messages"]:
                print(final_state["messages"][-1].pretty_print())
                response_message = final_state["messages"][-1].content

    return jsonify(
        {
            "messages": [{"role": "assistant", "content": response_message}],
            "session_id": session_id,
            "address": web3.address,
            "balance": web3.balances("usdc"),
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
