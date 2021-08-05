# Import dependencies
import subprocess
import json
from dotenv import load_dotenv
from web3 import Web3
from web3.middleware import geth_poa_middleware
from constants import *
import os
from bit import *
from bit.network import NetworkAPI
from web3 import Web3
from eth_account import Account

# Web3 connection and loading mnemonic
# Nodes runing with POW
w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:8545/0x24E7bc25E627e6210a3a15C6ed9354ef11B8E58E"))
w3.middleware_onion.inject(geth_poa_middleware, layer=0)
# Load and set environment variables
load_dotenv()
mnemonic=os.getenv("mnemonic")

# Create a function called `derive_wallets`
def derive_wallets(mnemonic, coin, numderive):
    command=f'php ./hd-wallet-derive/hd-wallet-derive.php -g --mnemonic="{mnemonic}" --numderive="{numderive}" --coin="{coin}" --format=json'
    p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    (output, err) = p.communicate()
    p_status = p.wait()
    keys = json.loads(output)
    return keys

# Create a dictionary object called coins to store the output from `derive_wallets`.
coins = {BTC, ETH, BTCTEST}
numderive = 3
# Setting the dictionarry
keys = {}
for coin in coins:
    keys[coin]= derive_wallets(os.getenv('mnemonic'), coin, numderive=3)

# Creating a private keys object
eth_PrivateKey = keys[ETH][0]['privkey']
btc_PrivateKey = keys[BTCTEST][0]['privkey']
btc2_PrivateKey = keys[BTCTEST][1]['privkey']

# Create a function called `priv_key_to_account` that converts privkey strings to account objects.
def priv_key_to_account(coin, priv_key):
    if coin == ETH:
        return Account.privateKeyToAccount(priv_key)
    if coin == BTCTEST:
        return PrivateKeyTestnet(priv_key)
# Create a function called `create_tx` that creates an unsigned transaction appropriate metadata.
def create_tx(coin, account, recipient, amount):
    global trx_data
    if coin == ETH:
        gasEstimate = w3.eth.estimateGas(
            {"from": account.address, "to": recipient, "value": amount}
        )
        trx_data = {
            "to": recipient,
            "from": account.address,
            "value": amount,
            "gasPrice": w3.eth.gasPrice,
            "gas": gasEstimate,
            "nonce": w3.eth.getTransactionCount(account.address)
        }
        return trx_data

    if coin == BTCTEST:
        return PrivateKeyTestnet.prepare_transaction(account.address, [(recipient, amount, BTC)])

    # Create a function called `send_tx` that calls `create_tx`, signs and sends the transaction.
def send_tx(coin, account, recipient, amount):
    if coin == "eth":
        trx_eth = create_tx(coin, account, recipient, amount)
        sign = account.signTransaction(trx_eth)
        result = w3.eth.sendRawTransaction(sign.rawTransaction)
        print(result.hex())
        return result.hex()
    else:
        trx_btctest = create_tx(coin, account, recipient, amount)
        sign_trx_btctest = account.sign_transaction(trx_btctest)

        NetworkAPI.broadcast_tx_testnet(sign_trx_btctest)
        return sign_trx_btctest

# create ETH, BTCTEST accounts
eth_acc = priv_key_to_account(ETH, eth_PrivateKey)
btc_acc = priv_key_to_account(BTCTEST,btc_PrivateKey)
btc2_acc = priv_key_to_account(BTCTEST,btc2_PrivateKey)

#send some btc back to the faucet's address
create_tx(BTCTEST,btc_acc,"mkHS9ne12qx9pS9VojpwU5xtRd4T7X7ZUt", 0.00001)
send_tx(BTCTEST,btc_acc,"mkHS9ne12qx9pS9VojpwU5xtRd4T7X7ZUt", 0.00001)

# send some btc the second account
create_tx(BTCTEST,btc_acc,"mwUZnn6tmNAunkjPhFuvbkU4bgC8Qixo7Z", 0.00001)
send_tx(BTCTEST,btc_acc,"mwUZnn6tmNAunkjPhFuvbkU4bgC8Qixo7Z", 0.00001)

if(w3.isConnected()):
    create_tx(ETH,eth_acc, "0x24E7bc25E627e6210a3a15C6ed9354ef11B8E58E", 1992)
    send_tx(ETH,eth_acc, "0x24E7bc25E627e6210a3a15C6ed9354ef11B8E58E", 1992)
