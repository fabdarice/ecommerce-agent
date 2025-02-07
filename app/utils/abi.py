TRANSFER_ABI = [
    {
        "inputs": [
            {
                "internalType": "contract IUniversalRouter",
                "name": "_uniswap",
                "type": "address",
            },
            {"internalType": "contract Permit2", "name": "_permit2", "type": "address"},
            {"internalType": "address", "name": "_initialOperator", "type": "address"},
            {
                "internalType": "address",
                "name": "_initialFeeDestination",
                "type": "address",
            },
            {
                "internalType": "contract IWrappedNativeCurrency",
                "name": "_wrappedNativeCurrency",
                "type": "address",
            },
        ],
        "stateMutability": "nonpayable",
        "type": "constructor",
    },
    {"inputs": [], "name": "AlreadyProcessed", "type": "error"},
    {"inputs": [], "name": "ExpiredIntent", "type": "error"},
    {
        "inputs": [
            {"internalType": "address", "name": "attemptedCurrency", "type": "address"}
        ],
        "name": "IncorrectCurrency",
        "type": "error",
    },
    {"inputs": [], "name": "InexactTransfer", "type": "error"},
    {
        "inputs": [
            {"internalType": "uint256", "name": "difference", "type": "uint256"}
        ],
        "name": "InsufficientAllowance",
        "type": "error",
    },
    {
        "inputs": [
            {"internalType": "uint256", "name": "difference", "type": "uint256"}
        ],
        "name": "InsufficientBalance",
        "type": "error",
    },
    {
        "inputs": [{"internalType": "int256", "name": "difference", "type": "int256"}],
        "name": "InvalidNativeAmount",
        "type": "error",
    },
    {"inputs": [], "name": "InvalidSignature", "type": "error"},
    {"inputs": [], "name": "InvalidTransferDetails", "type": "error"},
    {
        "inputs": [
            {"internalType": "address", "name": "recipient", "type": "address"},
            {"internalType": "uint256", "name": "amount", "type": "uint256"},
            {"internalType": "bool", "name": "isRefund", "type": "bool"},
            {"internalType": "bytes", "name": "data", "type": "bytes"},
        ],
        "name": "NativeTransferFailed",
        "type": "error",
    },
    {"inputs": [], "name": "NullRecipient", "type": "error"},
    {"inputs": [], "name": "OperatorNotRegistered", "type": "error"},
    {"inputs": [], "name": "PermitCallFailed", "type": "error"},
    {
        "inputs": [{"internalType": "bytes", "name": "reason", "type": "bytes"}],
        "name": "SwapFailedBytes",
        "type": "error",
    },
    {
        "inputs": [{"internalType": "string", "name": "reason", "type": "string"}],
        "name": "SwapFailedString",
        "type": "error",
    },
    {
        "anonymous": False,
        "inputs": [
            {
                "indexed": False,
                "internalType": "address",
                "name": "operator",
                "type": "address",
            },
            {
                "indexed": False,
                "internalType": "address",
                "name": "feeDestination",
                "type": "address",
            },
        ],
        "name": "OperatorRegistered",
        "type": "event",
    },
    {
        "anonymous": False,
        "inputs": [
            {
                "indexed": False,
                "internalType": "address",
                "name": "operator",
                "type": "address",
            }
        ],
        "name": "OperatorUnregistered",
        "type": "event",
    },
    {
        "anonymous": False,
        "inputs": [
            {
                "indexed": True,
                "internalType": "address",
                "name": "previousOwner",
                "type": "address",
            },
            {
                "indexed": True,
                "internalType": "address",
                "name": "newOwner",
                "type": "address",
            },
        ],
        "name": "OwnershipTransferred",
        "type": "event",
    },
    {
        "anonymous": False,
        "inputs": [
            {
                "indexed": False,
                "internalType": "address",
                "name": "account",
                "type": "address",
            }
        ],
        "name": "Paused",
        "type": "event",
    },
    {
        "anonymous": False,
        "inputs": [
            {
                "indexed": True,
                "internalType": "address",
                "name": "operator",
                "type": "address",
            },
            {
                "indexed": False,
                "internalType": "bytes16",
                "name": "id",
                "type": "bytes16",
            },
            {
                "indexed": False,
                "internalType": "address",
                "name": "recipient",
                "type": "address",
            },
            {
                "indexed": False,
                "internalType": "address",
                "name": "sender",
                "type": "address",
            },
            {
                "indexed": False,
                "internalType": "uint256",
                "name": "spentAmount",
                "type": "uint256",
            },
            {
                "indexed": False,
                "internalType": "address",
                "name": "spentCurrency",
                "type": "address",
            },
        ],
        "name": "Transferred",
        "type": "event",
    },
    {
        "anonymous": False,
        "inputs": [
            {
                "indexed": False,
                "internalType": "address",
                "name": "account",
                "type": "address",
            }
        ],
        "name": "Unpaused",
        "type": "event",
    },
    {
        "inputs": [],
        "name": "owner",
        "outputs": [{"internalType": "address", "name": "", "type": "address"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "pause",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "paused",
        "outputs": [{"internalType": "bool", "name": "", "type": "bool"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "permit2",
        "outputs": [
            {"internalType": "contract Permit2", "name": "", "type": "address"}
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "registerOperator",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "address", "name": "_feeDestination", "type": "address"}
        ],
        "name": "registerOperatorWithFeeDestination",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "renounceOwnership",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "address", "name": "newSweeper", "type": "address"}
        ],
        "name": "setSweeper",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {
                "components": [
                    {
                        "internalType": "uint256",
                        "name": "recipientAmount",
                        "type": "uint256",
                    },
                    {"internalType": "uint256", "name": "deadline", "type": "uint256"},
                    {
                        "internalType": "address payable",
                        "name": "recipient",
                        "type": "address",
                    },
                    {
                        "internalType": "address",
                        "name": "recipientCurrency",
                        "type": "address",
                    },
                    {
                        "internalType": "address",
                        "name": "refundDestination",
                        "type": "address",
                    },
                    {"internalType": "uint256", "name": "feeAmount", "type": "uint256"},
                    {"internalType": "bytes16", "name": "id", "type": "bytes16"},
                    {"internalType": "address", "name": "operator", "type": "address"},
                    {"internalType": "bytes", "name": "signature", "type": "bytes"},
                    {"internalType": "bytes", "name": "prefix", "type": "bytes"},
                ],
                "internalType": "struct TransferIntent",
                "name": "_intent",
                "type": "tuple",
            },
            {
                "components": [
                    {"internalType": "address", "name": "owner", "type": "address"},
                    {"internalType": "bytes", "name": "signature", "type": "bytes"},
                ],
                "internalType": "struct EIP2612SignatureTransferData",
                "name": "_signatureTransferData",
                "type": "tuple",
            },
        ],
        "name": "subsidizedTransferToken",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {
                "components": [
                    {
                        "internalType": "uint256",
                        "name": "recipientAmount",
                        "type": "uint256",
                    },
                    {"internalType": "uint256", "name": "deadline", "type": "uint256"},
                    {
                        "internalType": "address payable",
                        "name": "recipient",
                        "type": "address",
                    },
                    {
                        "internalType": "address",
                        "name": "recipientCurrency",
                        "type": "address",
                    },
                    {
                        "internalType": "address",
                        "name": "refundDestination",
                        "type": "address",
                    },
                    {"internalType": "uint256", "name": "feeAmount", "type": "uint256"},
                    {"internalType": "bytes16", "name": "id", "type": "bytes16"},
                    {"internalType": "address", "name": "operator", "type": "address"},
                    {"internalType": "bytes", "name": "signature", "type": "bytes"},
                    {"internalType": "bytes", "name": "prefix", "type": "bytes"},
                ],
                "internalType": "struct TransferIntent",
                "name": "_intent",
                "type": "tuple",
            },
            {"internalType": "uint24", "name": "poolFeesTier", "type": "uint24"},
        ],
        "name": "swapAndTransferUniswapV3Native",
        "outputs": [],
        "stateMutability": "payable",
        "type": "function",
    },
    {
        "inputs": [
            {
                "components": [
                    {
                        "internalType": "uint256",
                        "name": "recipientAmount",
                        "type": "uint256",
                    },
                    {"internalType": "uint256", "name": "deadline", "type": "uint256"},
                    {
                        "internalType": "address payable",
                        "name": "recipient",
                        "type": "address",
                    },
                    {
                        "internalType": "address",
                        "name": "recipientCurrency",
                        "type": "address",
                    },
                    {
                        "internalType": "address",
                        "name": "refundDestination",
                        "type": "address",
                    },
                    {"internalType": "uint256", "name": "feeAmount", "type": "uint256"},
                    {"internalType": "bytes16", "name": "id", "type": "bytes16"},
                    {"internalType": "address", "name": "operator", "type": "address"},
                    {"internalType": "bytes", "name": "signature", "type": "bytes"},
                    {"internalType": "bytes", "name": "prefix", "type": "bytes"},
                ],
                "internalType": "struct TransferIntent",
                "name": "_intent",
                "type": "tuple",
            },
            {
                "components": [
                    {
                        "components": [
                            {
                                "components": [
                                    {
                                        "internalType": "address",
                                        "name": "token",
                                        "type": "address",
                                    },
                                    {
                                        "internalType": "uint256",
                                        "name": "amount",
                                        "type": "uint256",
                                    },
                                ],
                                "internalType": "struct ISignatureTransfer.TokenPermissions",
                                "name": "permitted",
                                "type": "tuple",
                            },
                            {
                                "internalType": "uint256",
                                "name": "nonce",
                                "type": "uint256",
                            },
                            {
                                "internalType": "uint256",
                                "name": "deadline",
                                "type": "uint256",
                            },
                        ],
                        "internalType": "struct ISignatureTransfer.PermitTransferFrom",
                        "name": "permit",
                        "type": "tuple",
                    },
                    {
                        "components": [
                            {
                                "internalType": "address",
                                "name": "to",
                                "type": "address",
                            },
                            {
                                "internalType": "uint256",
                                "name": "requestedAmount",
                                "type": "uint256",
                            },
                        ],
                        "internalType": "struct ISignatureTransfer.SignatureTransferDetails",
                        "name": "transferDetails",
                        "type": "tuple",
                    },
                    {"internalType": "bytes", "name": "signature", "type": "bytes"},
                ],
                "internalType": "struct Permit2SignatureTransferData",
                "name": "_signatureTransferData",
                "type": "tuple",
            },
            {"internalType": "uint24", "name": "poolFeesTier", "type": "uint24"},
        ],
        "name": "swapAndTransferUniswapV3Token",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {
                "components": [
                    {
                        "internalType": "uint256",
                        "name": "recipientAmount",
                        "type": "uint256",
                    },
                    {"internalType": "uint256", "name": "deadline", "type": "uint256"},
                    {
                        "internalType": "address payable",
                        "name": "recipient",
                        "type": "address",
                    },
                    {
                        "internalType": "address",
                        "name": "recipientCurrency",
                        "type": "address",
                    },
                    {
                        "internalType": "address",
                        "name": "refundDestination",
                        "type": "address",
                    },
                    {"internalType": "uint256", "name": "feeAmount", "type": "uint256"},
                    {"internalType": "bytes16", "name": "id", "type": "bytes16"},
                    {"internalType": "address", "name": "operator", "type": "address"},
                    {"internalType": "bytes", "name": "signature", "type": "bytes"},
                    {"internalType": "bytes", "name": "prefix", "type": "bytes"},
                ],
                "internalType": "struct TransferIntent",
                "name": "_intent",
                "type": "tuple",
            },
            {"internalType": "address", "name": "_tokenIn", "type": "address"},
            {"internalType": "uint256", "name": "maxWillingToPay", "type": "uint256"},
            {"internalType": "uint24", "name": "poolFeesTier", "type": "uint24"},
        ],
        "name": "swapAndTransferUniswapV3TokenPreApproved",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {
                "internalType": "address payable",
                "name": "destination",
                "type": "address",
            }
        ],
        "name": "sweepETH",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {
                "internalType": "address payable",
                "name": "destination",
                "type": "address",
            },
            {"internalType": "uint256", "name": "amount", "type": "uint256"},
        ],
        "name": "sweepETHAmount",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "address", "name": "_token", "type": "address"},
            {"internalType": "address", "name": "destination", "type": "address"},
        ],
        "name": "sweepToken",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "address", "name": "_token", "type": "address"},
            {"internalType": "address", "name": "destination", "type": "address"},
            {"internalType": "uint256", "name": "amount", "type": "uint256"},
        ],
        "name": "sweepTokenAmount",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "sweeper",
        "outputs": [{"internalType": "address", "name": "", "type": "address"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [
            {
                "components": [
                    {
                        "internalType": "uint256",
                        "name": "recipientAmount",
                        "type": "uint256",
                    },
                    {"internalType": "uint256", "name": "deadline", "type": "uint256"},
                    {
                        "internalType": "address payable",
                        "name": "recipient",
                        "type": "address",
                    },
                    {
                        "internalType": "address",
                        "name": "recipientCurrency",
                        "type": "address",
                    },
                    {
                        "internalType": "address",
                        "name": "refundDestination",
                        "type": "address",
                    },
                    {"internalType": "uint256", "name": "feeAmount", "type": "uint256"},
                    {"internalType": "bytes16", "name": "id", "type": "bytes16"},
                    {"internalType": "address", "name": "operator", "type": "address"},
                    {"internalType": "bytes", "name": "signature", "type": "bytes"},
                    {"internalType": "bytes", "name": "prefix", "type": "bytes"},
                ],
                "internalType": "struct TransferIntent",
                "name": "_intent",
                "type": "tuple",
            }
        ],
        "name": "transferNative",
        "outputs": [],
        "stateMutability": "payable",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "address", "name": "newOwner", "type": "address"}],
        "name": "transferOwnership",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {
                "components": [
                    {
                        "internalType": "uint256",
                        "name": "recipientAmount",
                        "type": "uint256",
                    },
                    {"internalType": "uint256", "name": "deadline", "type": "uint256"},
                    {
                        "internalType": "address payable",
                        "name": "recipient",
                        "type": "address",
                    },
                    {
                        "internalType": "address",
                        "name": "recipientCurrency",
                        "type": "address",
                    },
                    {
                        "internalType": "address",
                        "name": "refundDestination",
                        "type": "address",
                    },
                    {"internalType": "uint256", "name": "feeAmount", "type": "uint256"},
                    {"internalType": "bytes16", "name": "id", "type": "bytes16"},
                    {"internalType": "address", "name": "operator", "type": "address"},
                    {"internalType": "bytes", "name": "signature", "type": "bytes"},
                    {"internalType": "bytes", "name": "prefix", "type": "bytes"},
                ],
                "internalType": "struct TransferIntent",
                "name": "_intent",
                "type": "tuple",
            },
            {
                "components": [
                    {
                        "components": [
                            {
                                "components": [
                                    {
                                        "internalType": "address",
                                        "name": "token",
                                        "type": "address",
                                    },
                                    {
                                        "internalType": "uint256",
                                        "name": "amount",
                                        "type": "uint256",
                                    },
                                ],
                                "internalType": "struct ISignatureTransfer.TokenPermissions",
                                "name": "permitted",
                                "type": "tuple",
                            },
                            {
                                "internalType": "uint256",
                                "name": "nonce",
                                "type": "uint256",
                            },
                            {
                                "internalType": "uint256",
                                "name": "deadline",
                                "type": "uint256",
                            },
                        ],
                        "internalType": "struct ISignatureTransfer.PermitTransferFrom",
                        "name": "permit",
                        "type": "tuple",
                    },
                    {
                        "components": [
                            {
                                "internalType": "address",
                                "name": "to",
                                "type": "address",
                            },
                            {
                                "internalType": "uint256",
                                "name": "requestedAmount",
                                "type": "uint256",
                            },
                        ],
                        "internalType": "struct ISignatureTransfer.SignatureTransferDetails",
                        "name": "transferDetails",
                        "type": "tuple",
                    },
                    {"internalType": "bytes", "name": "signature", "type": "bytes"},
                ],
                "internalType": "struct Permit2SignatureTransferData",
                "name": "_signatureTransferData",
                "type": "tuple",
            },
        ],
        "name": "transferToken",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {
                "components": [
                    {
                        "internalType": "uint256",
                        "name": "recipientAmount",
                        "type": "uint256",
                    },
                    {"internalType": "uint256", "name": "deadline", "type": "uint256"},
                    {
                        "internalType": "address payable",
                        "name": "recipient",
                        "type": "address",
                    },
                    {
                        "internalType": "address",
                        "name": "recipientCurrency",
                        "type": "address",
                    },
                    {
                        "internalType": "address",
                        "name": "refundDestination",
                        "type": "address",
                    },
                    {"internalType": "uint256", "name": "feeAmount", "type": "uint256"},
                    {"internalType": "bytes16", "name": "id", "type": "bytes16"},
                    {"internalType": "address", "name": "operator", "type": "address"},
                    {"internalType": "bytes", "name": "signature", "type": "bytes"},
                    {"internalType": "bytes", "name": "prefix", "type": "bytes"},
                ],
                "internalType": "struct TransferIntent",
                "name": "_intent",
                "type": "tuple",
            }
        ],
        "name": "transferTokenPreApproved",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "unpause",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "unregisterOperator",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {
                "components": [
                    {
                        "internalType": "uint256",
                        "name": "recipientAmount",
                        "type": "uint256",
                    },
                    {"internalType": "uint256", "name": "deadline", "type": "uint256"},
                    {
                        "internalType": "address payable",
                        "name": "recipient",
                        "type": "address",
                    },
                    {
                        "internalType": "address",
                        "name": "recipientCurrency",
                        "type": "address",
                    },
                    {
                        "internalType": "address",
                        "name": "refundDestination",
                        "type": "address",
                    },
                    {"internalType": "uint256", "name": "feeAmount", "type": "uint256"},
                    {"internalType": "bytes16", "name": "id", "type": "bytes16"},
                    {"internalType": "address", "name": "operator", "type": "address"},
                    {"internalType": "bytes", "name": "signature", "type": "bytes"},
                    {"internalType": "bytes", "name": "prefix", "type": "bytes"},
                ],
                "internalType": "struct TransferIntent",
                "name": "_intent",
                "type": "tuple",
            },
            {
                "components": [
                    {
                        "components": [
                            {
                                "components": [
                                    {
                                        "internalType": "address",
                                        "name": "token",
                                        "type": "address",
                                    },
                                    {
                                        "internalType": "uint256",
                                        "name": "amount",
                                        "type": "uint256",
                                    },
                                ],
                                "internalType": "struct ISignatureTransfer.TokenPermissions",
                                "name": "permitted",
                                "type": "tuple",
                            },
                            {
                                "internalType": "uint256",
                                "name": "nonce",
                                "type": "uint256",
                            },
                            {
                                "internalType": "uint256",
                                "name": "deadline",
                                "type": "uint256",
                            },
                        ],
                        "internalType": "struct ISignatureTransfer.PermitTransferFrom",
                        "name": "permit",
                        "type": "tuple",
                    },
                    {
                        "components": [
                            {
                                "internalType": "address",
                                "name": "to",
                                "type": "address",
                            },
                            {
                                "internalType": "uint256",
                                "name": "requestedAmount",
                                "type": "uint256",
                            },
                        ],
                        "internalType": "struct ISignatureTransfer.SignatureTransferDetails",
                        "name": "transferDetails",
                        "type": "tuple",
                    },
                    {"internalType": "bytes", "name": "signature", "type": "bytes"},
                ],
                "internalType": "struct Permit2SignatureTransferData",
                "name": "_signatureTransferData",
                "type": "tuple",
            },
        ],
        "name": "unwrapAndTransfer",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {
                "components": [
                    {
                        "internalType": "uint256",
                        "name": "recipientAmount",
                        "type": "uint256",
                    },
                    {"internalType": "uint256", "name": "deadline", "type": "uint256"},
                    {
                        "internalType": "address payable",
                        "name": "recipient",
                        "type": "address",
                    },
                    {
                        "internalType": "address",
                        "name": "recipientCurrency",
                        "type": "address",
                    },
                    {
                        "internalType": "address",
                        "name": "refundDestination",
                        "type": "address",
                    },
                    {"internalType": "uint256", "name": "feeAmount", "type": "uint256"},
                    {"internalType": "bytes16", "name": "id", "type": "bytes16"},
                    {"internalType": "address", "name": "operator", "type": "address"},
                    {"internalType": "bytes", "name": "signature", "type": "bytes"},
                    {"internalType": "bytes", "name": "prefix", "type": "bytes"},
                ],
                "internalType": "struct TransferIntent",
                "name": "_intent",
                "type": "tuple",
            }
        ],
        "name": "unwrapAndTransferPreApproved",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {
                "components": [
                    {
                        "internalType": "uint256",
                        "name": "recipientAmount",
                        "type": "uint256",
                    },
                    {"internalType": "uint256", "name": "deadline", "type": "uint256"},
                    {
                        "internalType": "address payable",
                        "name": "recipient",
                        "type": "address",
                    },
                    {
                        "internalType": "address",
                        "name": "recipientCurrency",
                        "type": "address",
                    },
                    {
                        "internalType": "address",
                        "name": "refundDestination",
                        "type": "address",
                    },
                    {"internalType": "uint256", "name": "feeAmount", "type": "uint256"},
                    {"internalType": "bytes16", "name": "id", "type": "bytes16"},
                    {"internalType": "address", "name": "operator", "type": "address"},
                    {"internalType": "bytes", "name": "signature", "type": "bytes"},
                    {"internalType": "bytes", "name": "prefix", "type": "bytes"},
                ],
                "internalType": "struct TransferIntent",
                "name": "_intent",
                "type": "tuple",
            }
        ],
        "name": "wrapAndTransfer",
        "outputs": [],
        "stateMutability": "payable",
        "type": "function",
    },
    {"stateMutability": "payable", "type": "receive"},
]
