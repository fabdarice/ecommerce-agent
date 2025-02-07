from typing import List, Dict, Any
import os

from cdp import Cdp, Wallet

from app.utils.abi import TRANSFER_ABI


class Web3:
    def __init__(self) -> None:
        api_key_name = os.getenv("COINBASE_CDP_API_KEY_NAME")
        api_key_private_key = os.getenv("COINBASE_CDP_SECRET")
        wallet_id = os.getenv("COINBASE_CDP_WALLET_ID")

        if not api_key_name or not api_key_private_key:
            raise ValueError("Missing API key name or private key")

        self.cdp = Cdp.configure(api_key_name, api_key_private_key)

        if os.path.exists(".seed.json") and wallet_id:
            print("Loading existing wallet")
            self.wallet = Wallet.fetch(wallet_id)
            self.wallet.load_seed(".seed.json")
        else:
            print("Creating new wallet")
            self.wallet = Wallet.create(network_id="base-mainnet")
            data = self.wallet.export_data()

            print(data.to_dict())
            self.wallet.save_seed(".seed.json", encrypt=True)

    @property
    def address(self):
        if self.wallet and self.wallet.default_address:
            return self.wallet.default_address.address_id
        raise ValueError("No wallet found")

    def balances(self, currency=None):
        return self.wallet.balance(currency) if currency else self.wallet.balances()

    def invoke_transfers_contract(
        self,
        contract_address: str,
        args: Any,
    ):
        return self.wallet.invoke_contract(
            contract_address=contract_address,
            abi=TRANSFER_ABI,
            method="transferTokenPreApproved",
            args=args,
        )
