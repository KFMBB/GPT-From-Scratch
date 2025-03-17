# GPT from Scratch

## Overview
This project is an educational implementation of a **GPT-like model** from scratch, based on **Andrej Karpathy's ng-video-lecture repository** ([source repo](https://github.com/karpathy/ng-video-lecture)). The goal is to gain a deep theoretical and coding understanding of **transformer-based language models**, moving beyond simple n-gram models like **bigram language models**.

## Features
- **Bigram Model**: Implemented a simple bigram-based language model as a baseline.
- **Transformer-Based GPT Model**:
  - Self-attention mechanism with multi-head support.
  - Positional encodings to maintain token order.
  - Causal masking to ensure autoregressive behavior.
  - Multi-layer perceptrons (MLP) for deeper transformations.
  - Dropout for regularization.
- **Training Pipeline**:
  - Implements data loading, batch generation, and loss evaluation.
  - Uses cross-entropy loss and AdamW optimizer for weight updates.
  - Monitors training and validation loss over time.
- **Text Generation**:
  - Uses an autoregressive approach to generate text iteratively.
  - Outputs dynamically sampled next tokens from learned distributions.

## Theoretical Understanding
This project moves beyond simple **bigram models**, which only consider one previous token, to a full-fledged **Transformer architecture**, allowing the model to:
- **Attend to all previous tokens** in a sentence using self-attention.
- **Learn long-range dependencies** instead of relying on fixed n-gram history.
- **Process text in parallel**, making training more efficient than recurrent models.

### Key Components Explained:
#### **1. Bigram Model**
The bigram model was used as a stepping stone before implementing the full GPT architecture. It uses a simple lookup table to predict the next token based only on the previous token.

#### **2. Transformer Attention Head (`Head` class)**
A single self-attention head was implemented to:
- Project input tokens into **queries, keys, and values**.
- Compute **scaled dot-product attention** scores.
- Apply a **causal mask** to prevent looking ahead.
- Normalize with **softmax** and apply learned attention weights.

#### **3. Multi-Head Self-Attention (`MultiHeadAttention` class)**
Multiple `Head` instances are stacked together to learn diverse attention patterns across the sequence.

#### **4. Transformer Block (`Block` class)**
Each Transformer block contains:
- **Layer Normalization** for stable training.
- **Self-Attention Mechanism** to contextualize token embeddings.
- **Feedforward Neural Network (MLP)** for further transformations.

## Training Process
The training follows a standard deep learning workflow:
1. **Data Preparation**:
   - Tokenization using character-level encoding.
   - Training and validation splits.
   - Mini-batch sampling for efficient training.
2. **Training Loop**:
   - Forward pass through transformer layers.
   - Cross-entropy loss computation.
   - Backpropagation and weight updates using AdamW optimizer.
3. **Evaluation & Loss Monitoring**:
   - Periodic validation loss checks.
   - Learning rate tuning for stability.
4. **Text Generation**:
   - Uses trained GPT model to generate sequences token by token.

## How to Use
### **1. Install Dependencies**
Ensure you have Python 3.8+ and install necessary libraries:
```bash
pip install torch numpy tqdm
```

### **2. Clone the Repository**
```bash
git clone <your-github-repo-url>
cd <your-project-directory>
```

### **3. Train the Model**
```bash
python train.py
```

### **4. Generate Text**
```python
from model import GPTLanguageModel
model = GPTLanguageModel.load('model_checkpoint.pth')
context = torch.tensor([[0]])  # Example start token
print(model.generate(context, max_new_tokens=100))
```

## References
This implementation was inspired by:
- **Andrej Karpathy's ng-video-lecture repository**: [GitHub](https://github.com/karpathy/ng-video-lecture)
- **"The Illustrated Transformer"**: Jay Alammar’s blog on Transformer models ([link](http://jalammar.github.io/illustrated-transformer/))


## Contributing
Feel free to fork, modify, and submit pull requests to improve the model or add new features!

