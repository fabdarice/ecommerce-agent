from decimal import Decimal

from cdp import ContractInvocation

from app.services.web3 import USDC_CONTRACT_ADDRESS, Web3

# Aave V3 Pool contract on Base
AAVE_V3_POOL = "0xA238Dd80C259a72e81d7e4664a9801593F98d1c5"
USDC_A_TOKEN = "0x4e65fE4DbA92790696d040ac24Aa414708F5c0AB"  # aUSDC token address

# Add this to your existing ABI imports
AAVE_V3_POOL_ABI = [
    {
        "inputs": [
            {"internalType": "address", "name": "asset", "type": "address"},
            {"internalType": "address", "name": "account", "type": "address"},
        ],
        "name": "getUserAccountData",
        "outputs": [
            {
                "internalType": "uint256",
                "name": "totalCollateralBase",
                "type": "uint256",
            },
            {"internalType": "uint256", "name": "totalDebtBase", "type": "uint256"},
            {
                "internalType": "uint256",
                "name": "availableBorrowsBase",
                "type": "uint256",
            },
            {
                "internalType": "uint256",
                "name": "currentLiquidationThreshold",
                "type": "uint256",
            },
            {"internalType": "uint256", "name": "ltv", "type": "uint256"},
            {"internalType": "uint256", "name": "healthFactor", "type": "uint256"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "address", "name": "asset", "type": "address"},
            {"internalType": "uint256", "name": "amount", "type": "uint256"},
            {"internalType": "address", "name": "to", "type": "address"},
        ],
        "name": "withdraw",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "nonpayable",
        "type": "function",
    },
]


class Aave:
    def __init__(self, web3_instance: Web3):
        self.web3 = web3_instance

    def get_usdc_deposit_amount(self) -> Decimal:
        """
        Read the currently deposited amount of USDC in Aave V3 market
        Returns the amount in USDC (decimal adjusted)
        """
        try:
            ausdc_balance = self.web3.wallet.balance(USDC_A_TOKEN)
            return ausdc_balance

        except Exception as e:
            print(f"Error reading USDC deposit amount: {str(e)}")
            raise

    def withdraw_usdc(self, amount: Decimal) -> ContractInvocation:
        """
        Withdraw a specific amount of USDC from Aave V3
        Args:
            amount: Amount of USDC to withdraw (in USDC, not wei)
        Returns:
            Transaction details
        """
        try:
            # Convert amount to wei (USDC has 6 decimals)
            amount_wei = int(amount * Decimal(10**6))

            # Call withdraw function on Aave Pool
            tx = self.web3.wallet.invoke_contract(
                contract_address=AAVE_V3_POOL,
                abi=AAVE_V3_POOL_ABI,
                method="withdraw",
                args={
                    "asset": USDC_CONTRACT_ADDRESS,
                    "amount": str(amount_wei),
                    "to": self.web3.address,
                },
            )
            tx.wait()
            return tx

        except Exception as e:
            print(f"Error withdrawing USDC: {str(e)}")
            raise
