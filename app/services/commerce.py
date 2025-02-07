import os
import requests
from dataclasses import dataclass
from typing import Dict, Any, Tuple
from datetime import datetime, timezone


@dataclass
class CallData:
    deadline: int
    fee_amount: int
    id: str
    operator: str
    prefix: str
    recipient: str
    recipient_amount: int
    recipient_currency: str
    refund_destination: str
    signature: str


@dataclass
class Metadata:
    chain_id: int
    contract_address: str
    sender: str


@dataclass
class TransferIntent:
    call_data: CallData
    metadata: Metadata

    @staticmethod
    def extract(payload: Dict[str, Any]) -> "TransferIntent":
        transfer_intent_data = payload.get("transfer_intent", {})
        call_data = transfer_intent_data.get("call_data", {})
        metadata = transfer_intent_data.get("metadata", {})
        # Convert deadline to epoch seconds
        deadline_epoch = int(
            datetime.fromisoformat(call_data["deadline"].replace("Z", ""))
            .replace(tzinfo=timezone.utc)
            .timestamp()
        )

        call_data_obj = CallData(
            deadline=deadline_epoch,
            fee_amount=int(call_data["fee_amount"]),
            id=call_data["id"],
            operator=call_data["operator"],
            prefix=call_data["prefix"],
            recipient=call_data["recipient"],
            recipient_amount=int(call_data["recipient_amount"]),
            recipient_currency=call_data["recipient_currency"],
            refund_destination=call_data["refund_destination"],
            signature=call_data["signature"],
        )

        metadata_obj = Metadata(
            chain_id=metadata["chain_id"],
            contract_address=metadata["contract_address"],
            sender=metadata["sender"],
        )

        return TransferIntent(call_data=call_data_obj, metadata=metadata_obj)

    @property
    def to_onchain_params(self) -> dict:
        """
        Converts the TransferIntent into a tuple structure that matches the subsidizedTransferToken ABI.
        """
        return {
            "_intent": [
                str(self.call_data.recipient_amount),
                str(self.call_data.deadline),
                self.call_data.recipient,
                self.call_data.recipient_currency,
                self.call_data.refund_destination,
                str(self.call_data.fee_amount),
                self.call_data.id,  # Convert hex string to bytes
                self.call_data.operator,
                self.call_data.signature,
                self.call_data.prefix,  # Convert string to bytes
            ]
        }


api_url = "https://api.commerce.coinbase.com/charges"


class CommerceService:
    def __init__(self) -> None:
        self.api_key = os.getenv("COINBASE_COMMERCE_API_KEY")
        pass

    def create_charge(self, item) -> str:
        print(f"call {api_url}")

        headers = {"Content-Type": "application/json", "X-CC-Api-Key": self.api_key}
        data = {
            "name": item.name,
            "description": item.description,
            "pricing_type": "fixed_price",
            "local_price": {"amount": item.price, "currency": "usd"},
        }

        try:
            # Sending the POST request
            response = requests.post(api_url, headers=headers, json=data)
            # Output the response
            print("Response Charge JSON:", response.json())
            charge_id = response.json()["data"]["id"]
            print("Charge ID:", charge_id)

            return charge_id
        except requests.RequestException as e:
            print(f"Error creating Coinbase charge: {e}")
            raise e

    def transact_onchain(self, charge_id: str, sender: str) -> TransferIntent:
        print(f"call {api_url}/{charge_id}/hydrate")
        headers = {"Content-Type": "application/json", "X-CC-Api-Key": self.api_key}
        data = {"chain_id": 8453, "sender": sender}

        try:
            response = requests.put(
                f"{api_url}/{charge_id}/hydrate",
                headers=headers,
                json=data,
            )

            print("Response Hydrate JSON:", response.json())
            print("Web3 Data", response.json()["data"]["web3_data"])
            web3_data = response.json()["data"]["web3_data"]
            return TransferIntent.extract(web3_data)
        except requests.RequestException as e:
            print(f"Error creating Coinbase charge: {e}")
            raise e
