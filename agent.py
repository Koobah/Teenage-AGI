import openai
import os
import pinecone
import yaml
from dotenv import load_dotenv
from sklearn.cluster import KMeans
import numpy as np

# Load default environment variables (.env)
load_dotenv()

agents = {}


def create_agent(agent_name, table_name, model="gpt-3.5-turbo", temperature=1.0, max_tokens=100, initial_prompt=None):
    new_agent = Agent(agent_name=agent_name, table_name=table_name, model=model, temperature=temperature,
                      max_tokens=max_tokens, initial_prompt=initial_prompt)
    return new_agent


def get_agent(agent_name):
    return agents.get(agent_name)


def generate(agent, prompt):
    completion = openai.ChatCompletion.create(
        model=agent.model,
        temperature=agent.temperature,
        max_tokens=agent.max_tokens,
        messages=[
            {"role": "system",
             "content": agent.initial_prompt + "You have a memory which stores your past thoughts and actions and also how other users have interacted with you"},
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
openai.api_key = OPENAI_API_KEY  # you can just copy and paste your key here if you want


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
            n_clusters=5,  # Add the n_clusters attribute here

    ):
        self.agent_name = agent_name
        self.table_name = table_name
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.initial_prompt = initial_prompt
        self.memory = pinecone.Index(self.table_name)
        self.thought_id_count = int(counter['count'])
        self.n_clusters = n_clusters
        self.cluster_update_interval = 10

    # Keep Remebering!
    # def __del__(self) -> None:
    #     with open('memory_count.yaml', 'w') as f:
    #         yaml.dump({'count': str(self.thought_id_count)}, f)

    def createIndex(self, table_name="agi"):
        # Create Pinecone index
        if (table_name):
            self.table_name = table_name

        if (self.table_name == None):
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

    def calculate_surprise(self, memory_embedding, k=5):
        results = self.memory.query(memory_embedding, top_k=k, include_metadata=True, namespace=THOUGHTS)
        filtered_results = [match for match in results.matches if match.metadata.get("agent_name") == self.agent_name]
        sorted_results = sorted(filtered_results, key=lambda x: x.score, reverse=True)
        if not sorted_results:
            return 1.0
        return 1 - sorted_results[0].score

    # Adds new "Thought" to agent. thought_type is Query, Internal, and External
    def updateMemory(self, new_thought, thought_type, cluster_label):

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
                         "cluster_label": cluster_label}  # Add the cluster_label to the metadata
                }],
            namespace=THOUGHTS,
        )

        self.thought_id_count += 1
        if self.thought_id_count % self.cluster_update_interval == 0:
            self.cluster_memories()

    # Agent thinks about given query based on top k related memories. Internal thought is passed to external thought
    def internalThought(self, query, min_surprise_score=0.2) -> str:
        query_embedding = get_ada_embedding(query)

        # Retrieve more results
        results = self.memory.query(query_embedding, top_k=k_n * self.n_clusters, include_metadata=True,
                                    namespace=THOUGHTS)
        filtered_results = [match for match in results.matches if match.metadata.get("agent_name") == self.agent_name]
        sorted_results = sorted(filtered_results, key=lambda x: (x.metadata.get("cluster_label", 0), x.score),
                                reverse=True)

        # Select top k_n results with different cluster labels
        top_matches = []
        seen_clusters = set()
        for match in sorted_results:
            if len(top_matches) >= k_n:
                break
            if match.metadata.get("cluster_label", 0) not in seen_clusters:
                top_matches.append(match.metadata["thought_string"])  # Append the thought string instead of metadata
                seen_clusters.add(match.metadata.get("cluster_label", 0))

        top_matches_str = "\n\n".join(top_matches)  # Convert the list of top_matches to a single string

        internalThoughtPrompt = data['internal_thought']
        internalThoughtPrompt = internalThoughtPrompt.replace("{query}", query).replace("{top_matches}",
                                                                                        top_matches_str)
        internal_thought = generate(self, internalThoughtPrompt)  # OPENAI CALL: top_matches and query text is used here

        internalMemoryPrompt = data['internal_thought_memory']
        internalMemoryPrompt = internalMemoryPrompt.replace("{query}", query)

        if internal_thought is not None:
            internalMemoryPrompt = internalMemoryPrompt.replace("{internal_thought}", internal_thought)
        else:
            internalMemoryPrompt = internalMemoryPrompt.replace("{internal_thought}", "")
        self.updateMemory(internalMemoryPrompt, "Internal", cluster_label=match.metadata.get("cluster_label", 0))
        return internal_thought, top_matches_str

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

        self.updateMemory(externalMemoryPrompt, "External", cluster_label=0)

        request_memory = data["request_memory"]
        self.updateMemory(request_memory.replace("{query}", query), "Query", cluster_label=0)
        return external_thought

    # Make agent read some information (learn) WIP
    def read(self, text) -> str:
        pass

    def cluster_memories(self, n_clusters=5):
        # Fetch all memories and their embeddings
        all_memories = self.memory.fetch_all(include_metadata=True, namespace=THOUGHTS)

        # Extract the embeddings and metadata
        embeddings = [memory.values for memory in all_memories]
        metadata = [memory.metadata for memory in all_memories]

        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters)
        labels = kmeans.fit_predict(embeddings)

        # Update memory metadata with cluster labels
        for i, memory_id in enumerate(all_memories.ids):
            metadata[i]["cluster_label"] = labels[i]
            self.memory.update_metadata(memory_id, metadata[i], namespace=THOUGHTS)
