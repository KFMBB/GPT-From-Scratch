import torch # Pytorch module
import torch.nn as nn
import torch.nn.functional as F

# Let us define the Hyperparameters
# The number of data instances forward propagated at once to be processed in parallel, defines the batch size.
batch_size = 32 # In usual practice we assign the epoch size as a number squared. (like 64, 128, etc.)
# The maximum amount of data the model can use to infer/sample the next token, defines the context_length
context_length = 12 # In our case since our data is small the window the model needs doesn't have to be large
# The total number of training steps, is defined by the maX_iter.
max_iters = 3000 # It essentially determines how long we train the model
# Monitoring the loss per X iterations is defined by eval_interval.
eval_interval = 300 # Frequency of loss evaluation
# The number of batches used to estimate loss, is defioned by eval_iters.
eval_iters = 200 # Ensures stable loss evaluation by averaging across multiple batches
learning_rate = 1e-2 # Controls how much the model’s parameters are updated with each optimization step.
# For faster processing we look to use the GPU when training the model, if you don't have it's fine as the CPU is enough.
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# If you'd like to get the same results as me keep the seed as it is.
torch.manual_seed(101) # This seed keeps the randomly assigned numbers the same.

# The data we'll work is the Shakespeer data set.
with open('Input_Data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# The Bigram model predict the next token (in out case char) given the previous token.
# So we have to define the set of possible characters that may follow another character with respect to our data.
# This essentially defines the model's vocab
chars = sorted(list(set(text))) # The set to extract the unique elements, the list to use the sort func to store the data periodically.
vocab_size = len(chars)
# print('The character set is:', chars)
# print('The vocabulary size is:', vocab_size)
# In short: In a bigram model, we predict the next character using a probability distribution over vocab_size possible characters.

# Since we're planning to make computations on our data we have to define an encoding procedure to represent our data numerically.
# Below is a very simple code snippet that defines an encoding & decoding procedure for our data.
ctoi = {c:i for i,c in enumerate(chars)} # Map a character to an index
itoc = {i:c for i,c in enumerate(chars)} # Map an index to a character
encode = lambda s: [ctoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itoc[i] for i in l]) # decoder: take a list of integers, output a string
# Please feel free to try it out..

# Now that we have defined the vocab and the encoding and decoding procedure let us encode and split our data.
# When training a model, we want to measure its performance on unseen data to ensure it generalizes well.
# A common strategy is to split the dataset into: 1-Training set and 2-Test/Val set
# We'll do a 80% 20% split
size = int(0.8 * len(text))
data = torch.tensor(encode(text), dtype=torch.long, device=device)
train_data = data[:size]
test_data = data[size:]


def get_batch(split):
    # Decide which split we're picking
    if split == 'train':
        data = train_data
    else:
        data = test_data

    # Initialize batch_size number of starting indices, this helps us sample form random positions which helps the model generalize rather than learning purly sequential patterns.
    ix = torch.randint(len(data) - context_length, (batch_size,)) # In here we essentially pick random starting points batch_size times

    # Define x and y:
    # We will have batch_size number of sequences of context_length, (in our case a 32X8 matrix)
    x = torch.stack([data[i:i + context_length] for i in ix]) # Input sequences
    y = torch.stack([data[i + 1:i + context_length + 1] for i in ix])  # Target sequences
    # For faster computations we send them over to device memory
    x, y = x.to(device), y.to(device)
    return x, y

# A Bi-Gram model is an n-gram language model that predicts the probability of a word in a sentence based on the previous one word.
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Define the embeddings layer
        # This layer acts as a lookup table, it associates each token with a vocab_size-dimensional vector.
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    # Feed forward
    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx)  # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Initialize our simple bigram model:
model = BigramLanguageModel(vocab_size)
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Start training
for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets, you can play with this py tuning the hyperparameters
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train') # Get the 32X8 matrix repsenting the data going into the model

    # evaluate the loss
    logits, loss = model(xb, yb) # Feed forward
    optimizer.zero_grad(set_to_none=True) # Reset gradients before computing new ones
    loss.backward() # Compute gradients
    optimizer.step() # Update model parameters

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

