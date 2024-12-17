import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define the embedding layer
class Embedding(nn.Module):
    def __init__(self, input_size, embedding_dim):
        super(Embedding, self).__init__()
        self.embedding = nn.Linear(input_size, embedding_dim)

    def forward(self, x):
        return self.embedding(x)

# Define the attention mechanism
class Attention(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(Attention, self).__init__()
        self.Wa = nn.Linear(embedding_dim + hidden_dim, hidden_dim)
        self.va = nn.Parameter(torch.randn(hidden_dim))
        self.Wc = nn.Linear(embedding_dim + hidden_dim, hidden_dim)
        self.vc = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, embedded_inputs, hidden_state):
        # embedded_inputs: (batch_size, num_inputs, embedding_dim)
        # hidden_state: (batch_size, hidden_dim)

        batch_size, num_inputs, _ = embedded_inputs.size()

        # Concatenate embedded inputs and hidden state
        hidden_state = hidden_state.unsqueeze(1).repeat(1, num_inputs, 1)
        concatenated = torch.cat((embedded_inputs, hidden_state), dim=2)

        # Calculate attention scores
        ut = torch.tanh(self.Wa(concatenated))  # (batch_size, num_inputs, hidden_dim)
        ut = ut.matmul(self.va)  # (batch_size, num_inputs)
        at = torch.softmax(ut, dim=1)  # (batch_size, num_inputs)

        # Calculate context vector
        context_vector = at.unsqueeze(2) * embedded_inputs  # (batch_size, num_inputs, embedding_dim)
        context_vector = context_vector.sum(dim=1)  # (batch_size, embedding_dim)

        return at, context_vector

# Define the decoder network
class Decoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.rnn = nn.GRUCell(embedding_dim, hidden_dim)
        self.attention = Attention(embedding_dim, hidden_dim)
        self.Wc = nn.Linear(embedding_dim + hidden_dim, hidden_dim)
        self.vc = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, embedded_inputs, hidden_state):
        # embedded_inputs: (batch_size, num_inputs, embedding_dim)
        # hidden_state: (batch_size, hidden_dim)

        batch_size, num_inputs, _ = embedded_inputs.size()

        # Calculate attention weights and context vector
        at, context_vector = self.attention(embedded_inputs, hidden_state)

        # Concatenate embedded inputs and context vector
        context_vector = context_vector.unsqueeze(1).repeat(1, num_inputs, 1)
        concatenated = torch.cat((embedded_inputs, context_vector), dim=2)

        # Calculate output scores
        ut = torch.tanh(self.Wc(concatenated))  # (batch_size, num_inputs, hidden_dim)
        ut = ut.matmul(self.vc)  # (batch_size, num_inputs)
        output_probs = torch.softmax(ut, dim=1)  # (batch_size, num_inputs)

        return output_probs, hidden_state, at

# Example usage:
# Assuming you have input data 'inputs' and the input and embedding sizes
input_size = # ...
embedding_dim = 128
hidden_dim = 256

# Instantiate the embedding layer
embedding = Embedding(input_size, embedding_dim)

# Embed the input data
embedded_inputs = embedding(inputs)

# Instantiate the decoder network
decoder = Decoder(embedding_dim, hidden_dim)

# Initialize hidden state
hidden_state = torch.zeros(batch_size, hidden_dim)

# Perform decoding steps
for t in range(sequence_length):
    output_probs, hidden_state, at = decoder(embedded_inputs, hidden_state)

    # Sample an action from the output probabilities
    action = torch.distributions.Categorical(output_probs).sample()

    # ... use the action to update the environment ...

# ... (Embedding, Attention, and Decoder classes from previous response) ...

class VRPEnvironment:
    def __init__(self, num_nodes, capacity):
        self.num_nodes = num_nodes
        self.capacity = capacity
        self.reset()

    def reset(self):
        # Generate random customer and depot locations in [0, 1] x [0, 1]
        self.node_locations = torch.rand(self.num_nodes, 2)

        # Generate random demands in {1, ..., 9}
        self.demands = torch.randint(1, 10, (self.num_nodes,))

        # Initialize vehicle location at the depot (node 0)
        self.current_location = 0
        self.remaining_load = self.capacity
        self.visited = torch.zeros(self.num_nodes, dtype=torch.bool)
        self.tour_length = 0

        return self.get_state()

    def get_state(self):
        # Create a state representation (e.g., concatenate location, demand, and remaining load)
        state = torch.cat([
            self.node_locations,
            self.demands.unsqueeze(1),
            torch.tensor([self.remaining_load]).repeat(self.num_nodes, 1),
            self.visited.unsqueeze(1).float()
        ], dim=1)
        return state

    def step(self, action):
        # Move to the selected node
        next_location = action.item()
        self.tour_length += torch.norm(self.node_locations[self.current_location] - self.node_locations[next_location])
        self.current_location = next_location

        # Update demands and vehicle load
        if next_location != 0:  # If not the depot
            self.demands[next_location] = max(0, self.demands[next_location] - self.remaining_load)
            self.remaining_load = max(0, self.remaining_load - self.demands[next_location])
            self.visited[next_location] = True

        # Check if all demands are satisfied
        done = torch.all(self.demands == 0)

        # Calculate reward (negative tour length for minimization)
        reward = -self.tour_length

        return self.get_state(), reward, done, {}

    def apply_mask(self, logits):
        # Mask infeasible actions
        mask = torch.ones_like(logits)

        # (i) Mask nodes with zero demand
        mask[self.demands == 0] = 0

        # (ii) Mask all customers if the vehicle's load is 0
        if self.remaining_load == 0:
            mask[1:] = 0  # Mask all except the depot

        # (iii) Mask customers with demand greater than the vehicle's load
        mask[self.demands > self.remaining_load] = 0

        # Apply mask to logits
        logits = logits.masked_fill(mask == 0, float('-inf'))
        return logits

# Example usage:
num_nodes = 5
capacity = 20
embedding_dim = 128
hidden_dim = 256

# Instantiate the environment, embedding layer, and decoder network
env = VRPEnvironment(num_nodes, capacity)
embedding = Embedding(env.get_state().size(1), embedding_dim)
decoder = Decoder(embedding_dim, hidden_dim)

# Initialize hidden state
hidden_state = torch.zeros(1, hidden_dim)  # Batch size = 1 for single instance

# Perform decoding steps with masking
state = env.reset()
done = False
while not done:
    embedded_state = embedding(state)
    output_probs, hidden_state, _ = decoder(embedded_state.unsqueeze(0), hidden_state)

    # Apply mask to the output probabilities
    logits = torch.log(output_probs)
    masked_logits = env.apply_mask(logits)
    output_probs = torch.softmax(masked_logits, dim=1)

    # Choose action (greedy decoding)
    action = torch.argmax(output_probs)

    # Take action in the environment
    state, reward, done, _ = env.step(action)

# Print the final tour length
print("Tour Length:", env.tour_length)