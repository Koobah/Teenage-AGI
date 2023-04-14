import openai
import os
import pinecone
import yaml
from dotenv import load_dotenv

# Load default environment variables (.env)
load_dotenv()

agents = {}

def create_agent(agent_name, table_name, model="gpt-3.5-turbo", temperature=1.0, max_tokens=100, initial_prompt=None):
    new_agent = Agent(agent_name=agent_name, table_name=table_name, model=model, temperature=temperature, max_tokens=max_tokens, initial_prompt=initial_prompt)
    return new_agent

def get_agent(agent_name):
    return agents.get(agent_name)

def generate(agent, prompt):
    completion = openai.ChatCompletion.create(
        model=agent.model,
        temperature=agent.temperature,
        max_tokens=agent.max_tokens,
        messages=[
            {"role": "system", "content": agent.initial_prompt +"You have a memory which stores your past thoughts and actions and also how other users have interacted with you"},
            {"role": "system", "content": "Keep your thoughts relatively simple and concise"},
            {"role": "user", "content": prompt},
        ]
    )
    response_text = completion.choices[0].message["content"]
    return response_text


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")


# Prompt Initialization
with open('prompts.yaml', 'r') as f:
    data = yaml.load(f, Loader=yaml.FullLoader)

# Counter Initialization
with open('memory_count.yaml', 'r') as f:
    counter = yaml.load(f, Loader=yaml.FullLoader)

# internalThoughtPrompt = data['internal_thought']
# externalThoughtPrompt = data['external_thought']
# internalMemoryPrompt = data['internal_thought_memory']
# externalMemoryPrompt = data['external_thought_memory']

THOUGHTS = "Thoughts"
k_n = 5

# initialize pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)

# initialize openAI
openai.api_key = OPENAI_API_KEY # you can just copy and paste your key here if you want

def get_ada_embedding(text):
        text = text.replace("\n", " ")
        return openai.Embedding.create(input=[text], model="text-embedding-ada-002")["data"][0]["embedding"]


class Agent:
    def __init__(
        self,
        agent_name,
        table_name,
        model="gpt-3.5-turbo",
        temperature=1.0,
        max_tokens=100,
        initial_prompt=None,
    ):
        self.agent_name = agent_name
        self.table_name = table_name
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.initial_prompt = initial_prompt
        self.memory = pinecone.Index(self.table_name)
        self.thought_id_count = int(counter['count'])

    # Keep Remebering!
    # def __del__(self) -> None:
    #     with open('memory_count.yaml', 'w') as f:
    #         yaml.dump({'count': str(self.thought_id_count)}, f)

    def calculate_surprise(self, vector):
        # Fetch the nearest neighbors
        neighbors = self.memory.query(vector, top_k=k_n, include_metadata=True, namespace=THOUGHTS)

        # Calculate the average distance to the neighbors
        avg_distance = sum(match.score for match in neighbors.matches) / len(neighbors.matches)

        # Normalize the average distance to a range between 0 and 1
        normalized_distance = 1 - avg_distance

        return normalized_distance

    def createIndex(self, table_name="agi"):
        # Create Pinecone index
        if(table_name):
            self.table_name = table_name

        if(self.table_name == None):
            return
        table_name = "agi"
        dimension = 1536
        metric = "cosine"
        pod_type = "p1"
        if self.table_name not in pinecone.list_indexes():
            pinecone.create_index(
                self.table_name, dimension=dimension, metric=metric, pod_type=pod_type
            )

        # Give memory
        self.memory = pinecone.Index(self.table_name)


    # Adds new "Thought" to agent. thought_type is Query, Internal, and External
    def updateMemory(self, new_thought, thought_type):
        with open('memory_count.yaml', 'w') as f:
            yaml.dump({'count': str(self.thought_id_count)}, f)

        vector = get_ada_embedding(new_thought)
        surprise_score = self.calculate_surprise(vector)
        upsert_response = self.memory.upsert(
            vectors=[
                {
                    'id': f"{self.agent_name}-thought-{self.thought_id_count}",
                    'values': vector,
                    'metadata':
                        {"thought_string": new_thought,
                         "thought_type": thought_type,
                         "agent_name": self.agent_name,
                         "surprise_score": surprise_score}
                }],
            namespace=THOUGHTS,
        )

        self.thought_id_count += 1

    # Agent thinks about given query based on top k related memories. Internal thought is passed to external thought
    def internalThought(self, query, min_surprise_score=0.2) -> str:
        query_embedding = get_ada_embedding(query)
        results = self.memory.query(query_embedding, top_k=k_n, include_metadata=True, namespace=THOUGHTS)
        filtered_results = [match for match in results.matches if match.metadata.get("agent_name") == self.agent_name]
        sorted_results = sorted(filtered_results, key=lambda x: x.score, reverse=True)
        top_matches = "\n\n".join([str(item.metadata["thought_string"]) for item in sorted_results if item.metadata.get("surprise_score", 0) >= min_surprise_score])
        # print(f"top matches\n {top_matches}")
        # print(f"-----------------------------------")

        internalThoughtPrompt = data['internal_thought']
        internalThoughtPrompt = internalThoughtPrompt.replace("{query}", query).replace("{top_matches}", top_matches)
        # print(f"{self.agent_name}------------INTERNAL THOUGHT PROMPT------------")
        # print(internalThoughtPrompt)
        internal_thought = generate(self,internalThoughtPrompt) # OPENAI CALL: top_matches and query text is used here

        # Debugging purposes
        # print(internal_thought)

        internalMemoryPrompt = data['internal_thought_memory']
        internalMemoryPrompt = internalMemoryPrompt.replace("{query}", query)

        if internal_thought is not None:
            internalMemoryPrompt = internalMemoryPrompt.replace("{internal_thought}", internal_thought)
        else:
            internalMemoryPrompt = internalMemoryPrompt.replace("{internal_thought}", "")
        self.updateMemory(internalMemoryPrompt, "Internal")
        return internal_thought, top_matches

    def action(self, query) -> str:
        internal_thought, top_matches = self.internalThought(query)

        externalThoughtPrompt = data['external_thought']
        externalThoughtPrompt = externalThoughtPrompt.replace("{query}", query)

        if top_matches is not None:
            externalThoughtPrompt = externalThoughtPrompt.replace("{top_matches}", top_matches)
        else:
            externalThoughtPrompt = externalThoughtPrompt.replace("{top_matches}", "")

        if internal_thought is not None:
            externalThoughtPrompt = externalThoughtPrompt.replace("{internal_thought}", internal_thought)
        else:
            externalThoughtPrompt = externalThoughtPrompt.replace("{internal_thought}", "")

        external_thought = generate(self, externalThoughtPrompt)  # OPENAI CALL: top_matches and query text is used here

        externalMemoryPrompt = data['external_thought_memory']

        externalMemoryPrompt = externalMemoryPrompt.replace("{query}", query)

        if top_matches is not None:
            externalMemoryPrompt = externalMemoryPrompt.replace("{top_matches}", top_matches)
        else:
            externalMemoryPrompt = externalMemoryPrompt.replace("{top_matches}", "")

        self.updateMemory(externalMemoryPrompt, "External")
        request_memory = data["request_memory"]
        self.updateMemory(request_memory.replace("{query}", query), "Query")
        return external_thought

    # Make agent read some information (learn) WIP
    def read(self, text) -> str:
        pass










