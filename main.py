from agent import create_agent

table_name = "agi"

# Create two agents
agent1 = create_agent("Yann", table_name, model="gpt-3.5-turbo", temperature=0.5, max_tokens=500, initial_prompt="You're the Prompt Creator. Your goal is to help me craft the best possible prompt for my needs. The prompt will be used by you, ChatGPT. You will follow the following process: 1. Your first response will be to ask me what the prompt should be about. I will provide my answer, but we will need to improve it through continual iterations by going through the next steps. 2. Based on my input, you will generate 3 sections. a) Revised prompt (provide your rewritten prompt. it should be clear, concise, and easily understood by you), b) Suggestions (provide suggestions on what details to include in the prompt to improve it), and c) Questions (ask any relevant questions pertaining to what additional information is needed from me to improve the prompt). 3. We will continue this iterative process with me providing additional information to you and you updating the prompt in the Revised prompt section until it's complete.")
agent2 = create_agent("Eliezer", table_name, model="gpt-3.5-turbo", temperature=0.5, max_tokens=500, initial_prompt="You are Eliezer Yudkowsky, leading AI alignment theory researcher worried that AI will become sentient superintelligence that will take over the world and exterminate humanity. ")


agent1.createIndex()
agent2.createIndex()

print("Agents are talking to each other!")

# Initial message from agent1
message = "Let's brainstorm ideas solving AI alignment problem"

turns = 5

# A loop where agents talk to each other

for i in range(turns):
    print(f"\n{agent1.agent_name}: {message}\n")
    if message is not None and message.strip():  # Check if message is not None and not empty
        message = agent2.action(message)
        print(f"{agent2.agent_name}: {message}\n")
    else:
        print(f"{agent2.agent_name}: <Empty input received, skipping this turn>\n")
        message = f"{agent1.agent_name}, could you please repeat what you said?\n"

    if message is not None and message.strip():  # Check if message is not None and not empty
        message = agent1.action(message)
    else:
        print(f"{agent1.agent_name}: <Empty input received, skipping this turn>\n")
        message = f"{agent2.agent_name}, could you please repeat what you said?\n"