from swarm import Swarm, Agent

client = Swarm()

def is_stock_query(message):
    stock_keywords = ["stock", "market", "share", "investment"]
    return any(keyword in message.lower() for keyword in stock_keywords)

def transfer_to_stock_agent():
    return stock_agent

def transfer_to_general_agent():
    return general_agent

# Stock Agent
stock_agent = Agent(
    name="Stock Agent",
    instructions="You provide information about stocks and investments.",
)

# General Agent
general_agent = Agent(
    name="General Agent",
    instructions="You answer general questions.",
)

# User Agent
user_agent = Agent(
    name="User Agent",
    instructions="You route the questions to the appropriate agent.",
    functions=[transfer_to_stock_agent, transfer_to_general_agent],
)

response = client.run(
    agent=user_agent,
    messages=[{"role": "user", "content": "What is the current stock price of Company X?"}]
)

# Check which agent to route to based on the user query
if is_stock_query(response.messages[-1]["content"]):
    response = client.run(agent=stock_agent, messages=[{"role": "user", "content": response.messages[-1]["content"]}])
else:
    response = client.run(agent=general_agent, messages=[{"role": "user", "content": response.messages[-1]["content"]}])

print(response.messages[-1]["content"])
