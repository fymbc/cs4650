# %% [markdown]
# <a href="https://colab.research.google.com/github/fymbc/cs4650/blob/main/PA2.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# # Part of Speech Tagging with LSTM, Fine-tuned BERT
# **CS 4650 "Natural Language Processing" Project 2**  
# Georgia Tech, Spring 2025 (Instructor: Weicheng Ma)
# 
# **To start, first make a copy of this notebook to your local drive, so you can edit it.**
# 
# If you want GPUs (which will improve training speed), you can always change your instance type to GPU by going to Runtime -> Change runtime type -> Hardware accelerator.
# 
# 

# %% [markdown]
# ## 1. Basic POS Tagger  [15 points]
# 
# In this assignment, we will train LSTM-based POS-taggers, and evaluate their performance. We will use English text from the Wall Street Journal, marked with POS tags such as `NNP` (proper noun) and `DT` (determiner).

# %% [markdown]
# ### 1.1 Setup

# %%
!curl -L -o train.txt "https://www.dropbox.com/scl/fi/nqtk53b2ihqzf6hugolms/train.txt?rlkey=y7003b74z7gp06e8qa2gfgd4r&st=37lt5q66&dl=0"

# %%
# ===========================================================================
# Run some setup code for this notebook. Don't modify anything in this cell.
# ===========================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import random

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ===========================================================================
# A quick note on CUDA functionality (and `.to(model.device)`):
# CUDA is a parallel GPU platform produced by NVIDIA and is used by most GPU
# libraries in PyTorch. CUDA organizes GPUs into device IDs (i.e., "cuda:X" for GPU #X).
# "device" will tell PyTorch which GPU (or CPU) to place an object in. Since
# collab only uses one GPU, we will use 'cuda' as the device if a GPU is available
# and the CPU if not. You will run into problems if your tensors are on different devices.
# ===========================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %% [markdown]
# You can check to make sure a GPU is available using the following code block.
# 
# 
# ```py
# # If the below message is shown, it means you are using a CPU.
# /bin/bash: nvidia-smi: command not found
# ```
# 
# 
# 
# 

# %%
gpu_info = !nvidia-smi
gpu_info = '\n'.join(gpu_info)
if gpu_info.find('failed') >= 0:
  print('Select the Runtime > "Change runtime type" menu to enable a GPU accelerator, ')
  print('and then re-execute this cell.')
else:
  print(gpu_info)

# %% [markdown]
# ### 1.2 Preparing Data
# 
# `train.txt`: The training data is present in this file. This file contains sequences of words and their respective tags. The data is split into 80% training and 20% development to train the model and tune the hyperparameters, respectively. See `load_tag_data` for details on how to read the training data.

# %%
# ===========================================================================
# Run some preprocessing code for our dataset. Don't modify anything in this cell.
# ===========================================================================

def load_tag_data(tag_file):
    all_sentences = []
    all_tags = []
    sent = []
    tags = []
    with open(tag_file, 'r') as f:
        for line in f:
            if line.strip() == "":
                all_sentences.append(sent)
                all_tags.append(tags)
                sent = []
                tags = []
            else:
                word, tag, _ = line.strip().split()
                sent.append(word)
                tags.append(tag)
    return all_sentences, all_tags

train_sentences, train_tags = load_tag_data('train.txt')

unique_tags = set([tag for tag_seq in train_tags for tag in tag_seq])

# Create train-val split from train data
train_val_data = list(zip(train_sentences, train_tags))
random.shuffle(train_val_data)
split = int(0.8 * len(train_val_data))
training_data = train_val_data[:split]
val_data = train_val_data[split:]

print("Train Data: ", len(training_data))
print("Val Data: ", len(val_data))
print("Total tags: ", len(unique_tags))

# %% [markdown]
# ### 1.3 Word-to-Index and Tag-to-Index mapping
# In order to work with text in Tensor format, we need to map each word to an index.

# %%
# ===========================================================================
# Don't modify anything in this cell.
# ===========================================================================

word_to_idx = {}
for sent in train_sentences:
    for word in sent:
        if word not in word_to_idx:
            word_to_idx[word] = len(word_to_idx)

tag_to_idx = {}
for tag in unique_tags:
    if tag not in tag_to_idx:
        tag_to_idx[tag] = len(tag_to_idx)

idx_to_tag = {}
for tag in tag_to_idx:
    idx_to_tag[tag_to_idx[tag]] = tag

print("Total tags", len(tag_to_idx))
print("Vocab size", len(word_to_idx))

# %%
def prepare_sequence(sent, idx_mapping):
    idxs = [idx_mapping[word] for word in sent]
    return torch.tensor(idxs, dtype=torch.long)

# %% [markdown]
# ### 1.4 Set up model
# We will build and train a Basic POS Tagger which is an LSTM model to tag the parts of speech in a given sentence. Here we define a few default hyperparameters for your model.

# %%
EMBEDDING_DIM = 4
HIDDEN_DIM = 8
LEARNING_RATE = 0.1
LSTM_LAYERS = 1
DROPOUT = 0
EPOCHS = 10

# %% [markdown]
# ### 1.5 Define Model [5 points]
# 
# The model takes as input a sentence as a tensor in the index space. This sentence is then converted to embedding space where each word maps to its word embedding. The word embeddings is learned as part of the model training process. These word embeddings act as input to the LSTM which produces a representation for each word. Then the representations of words are passed to a Linear layer.

# %%
class BasicPOSTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        """
        Define and initialize anything needed for the forward pass.

        You are required to create a model with:
          an embedding layer: that maps words to the embedding space
          an LSTM layer: that takes word embeddings as input and outputs hidden states
          a linear layer: maps from hidden state space to tag space
        """
        super(BasicPOSTagger, self).__init__()

        ### BEGIN YOUR CODE ###
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, LSTM_LAYERS, batch_first=True)
        self.linear = nn.Linear(hidden_dim, tagset_size)
        ### END YOUR CODE ###

    def forward(self, sentence):
        """
        Implement the forward pass.

        Given a tokenized index-mapped sentence as the argument,
        compute the corresponding raw scores for tags (without softmax)

        returns:: tag_scores (Tensor)
        """
        tag_scores = None

        ### BEGIN YOUR CODE ###
        embeddings = self.embedding(sentence)
        output, (h_n, c_n) = self.lstm(embeddings)
        tag_scores = self.linear(output)

        ### END YOUR CODE ###

        return tag_scores

# %% [markdown]
# ### 1.6 Training [5 points]
# 
# We define train and evaluate procedures that allow us to train our model using our created train-val split.

# %%
def train(epoch, model, loss_function, optimizer):
    model.train()
    train_loss = 0
    train_examples = 0
    for sentence, tags in training_data:
        """
        Implement the training method

        Hint: you can use the prepare_sequence method for creating index mappings
        for sentences. Find the gradient with respect to the loss and update the
        model parameters using the optimizer.
        """

        ### BEGIN YOUR CODE ###

        # Zero out the parameter gradients
        optimizer.zero_grad()

        # Prepare input data (sentences and gold labels)
        input = prepare_sequence(sentence, word_to_idx)
        target = prepare_sequence(tags, tag_to_idx)

        # Do forward pass with current batch of input
        tag_scores = model(input)

        # Get loss with model predictions and true labels
        loss = loss_function(tag_scores.view(-1, tag_scores.shape[-1]), target.view(-1))
        loss.backward()

        # Update model parameters
        optimizer.step()

        # Increase running total loss and the number of past training samples
        train_loss += loss.item()
        train_examples += len(sentence)

        ### END YOUR CODE ###

    avg_train_loss = train_loss / train_examples
    avg_val_loss, val_accuracy = evaluate(model, loss_function)

    print(f"Epoch: {epoch}/{EPOCHS}\tAvg Train Loss: {avg_train_loss:.4f}\tAvg Val Loss: {avg_val_loss:.4f}\t Val Accuracy: {val_accuracy:.0f}")

def evaluate(model, loss_function):
    """
    returns:: avg_val_loss (float)
    returns:: val_accuracy (float)
    """
    model.eval()
    correct = 0
    val_loss = 0
    val_examples = 0
    with torch.no_grad():
        for sentence, tags in val_data:
            """
            Implement the evaluate method

            Find the average validation loss along with the validation accuracy.
            Hint: To find the accuracy, argmax of tag predictions can be used.s
            """
            ### BEGIN YOUR CODE ###

            # Prepare input data (sentences and gold labels)
            input = prepare_sequence(sentence, word_to_idx)
            target = prepare_sequence(tags, tag_to_idx)

            # Do forward pass with current batch of input
            tag_scores = model(input)

            # Get loss with model predictions and true labels
            loss = loss_function(tag_scores.view(-1, tag_scores.shape[-1]), target.view(-1))

            # Get the predicted labels
            _, predicted = torch.max(tag_scores, dim=1)
            
            # Get number of correct prediction
            correct += (predicted.view(-1) == target.view(-1)).sum().item()

            # Increase running total loss and the number of past valid samples
            val_loss += loss.item()
            val_examples += len(sentence)

            ### END YOUR CODE ###
    val_accuracy = 100. * correct / val_examples
    avg_val_loss = val_loss / val_examples
    return avg_val_loss, val_accuracy

# %%
"""
Initialize the model, optimizer and the loss function
"""
### BEGIN YOUR CODE ###
model = BasicPOSTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_idx), len(tag_to_idx))
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), LEARNING_RATE, momentum=0.8)

### END YOUR CODE ###

for epoch in range(1, EPOCHS + 1):
    train(epoch, model, loss_function, optimizer)

# %% [markdown]
# *Hint: Under the default hyperparameter setting, after 5 epochs you should be able to get at least `0.75` accuracy on the validation set.*

# %% [markdown]
# ### 1.7 Error analysis [5 points]
# 
# In this step, we will analyze what kind of errors it was making on the validation set.
# 
# Step 1, write a method to generate predictions from the validation set. For every sentence, get its words, predicted tags (model_tags), and the ground truth tags (gt_tags). To make the next step easier, you may want to concatenate words from all sentences into a very long list, and same for model_tags and gt_tags.
# 
# 
# Step 2, analyze what kind of errors the model was making. For example, it may frequently label NN as VB. Let's get the top-10 most frequent types of errors, each of their frequency, and some example words. One example is at below. It is interpreted as the model predicts NNP as VBG for 626 times, five random example words are shown.
# 
# ```
# ['VBG', 'NNP', 626, ['Rowe', 'Livermore', 'Parker', 'F-16', 'HEYNOW']]
# ```

# %%
def generate_predictions(model, val_data):
    """
    Generate predictions for val_data

    Create lists of words, tags predicted by the model and ground truth tags.
    Hint: It should look very similar to the evaluate function.

    returns:: word_list (str list)
    returns:: model_tags (str list)
    returns:: gt_tags (str list)
    """
    ### BEGIN YOUR CODE ###
    model.eval()
    word_list = []
    model_tags = []
    gt_tags = []

    with torch.no_grad():
       for sentence, tags in val_data:
          input = prepare_sequence(sentence, word_to_idx)
          tag_scores = model(input)

          _, predicted = torch.max(tag_scores, dim=1)

          word_list.extend(sentence)
          model_tags.extend([idx_to_tag[idx.item()] for idx in predicted])
          gt_tags.extend(tags)
    ### END YOUR CODE ###

    return word_list, model_tags, gt_tags

def error_analysis(word_list, model_tags, gt_tags):
    """"
    Carry out error analysis

    From those lists collected from the above method, find the
    top-10 tuples of (model_tag, ground_truth_tag, frequency, example words)
    sorted by frequency

    returns: errors (list of tuples)
    """
    ### BEGIN YOUR CODE ###
    error_count = {}
    error_eg = {}

    for i in range(len(word_list)):
       pred_tag = model_tags[i]
       gt_tag = gt_tags[i]
       word = word_list[i]

       if pred_tag != gt_tag:
        key = (pred_tag, gt_tag)
          
        if key in error_count:
            error_count[key] += 1
            if len(error_eg[key]) < 5:
                    error_eg[key].append(word)
        else:
           error_count[key] = 1
           error_eg[key] = [word]
    
    errors = []
    for key in error_count:
        errors.append((key[0], key[1], error_count[key], error_eg[key]))

    errors.sort(key=lambda x: x[2], reverse=True)
    ### END YOUR CODE ###

    return errors

word_list, model_tags, gt_tags = generate_predictions(model, val_data)
errors = error_analysis(word_list, model_tags, gt_tags)

for i in errors[:10]:
  print(i)

# %% [markdown]
# **Report your findings here.**  
# What kinds of errors did the model make and why do you think it made them?
# 
# It frequently misclassifies certain noun forms and adjectives. For instance, proper nouns were misclassified as adjectives, and plural nouns were misclassified as singular nouns. It likely made these errors because many words can have multiple possible POS tags depending on context, and hence it struggles with ambiguous words. Another possible factor is that the model only uses one LSTM layer with no bidirectional processing, therefore not being able to capture sufficient context.

# %% [markdown]
# ## 2. Hyper-parameter Tuning [10 points]
# 
# In order to improve your model performance, try making some modifications on `EMBEDDING_DIM`, `HIDDEN_DIM`, and `LEARNING_RATE`.

# %%
YOUR_EMBEDDING_DIM = 32
YOUR_HIDDEN_DIM = 64
YOUR_LEARNING_RATE = 0.01

# Set three hyper-parameters. Initialize the model, optimizer and the loss function
# Hint, you may want to use reduction='sum' in the CrossEntropyLoss function

### BEGIN YOUR CODE ###
model = BasicPOSTagger(YOUR_EMBEDDING_DIM, YOUR_HIDDEN_DIM, len(word_to_idx), len(tag_to_idx))
loss_function = nn.CrossEntropyLoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=YOUR_LEARNING_RATE, momentum=0.9)

### END YOUR CODE ###

for epoch in range(1, EPOCHS + 1):
    train(epoch, model, loss_function, optimizer)

# %% [markdown]
# ## 3. Character-level POS Tagger  [15 points]
# 
# Use the character-level information to augment word embeddings. For example, words that end with -ing or -ly give quite a bit of information about their POS tags. To incorporate this information, run a character-level LSTM on every word to create a character-level representation of the word. Take the last hidden state from the character-level LSTM as the representation and concatenate with the word embedding (as in the `BasicPOSTagger`) to create a new word representation that captures more information.

# %%
# Create char to index mapping
char_to_idx = {}
unique_chars = set()
MAX_WORD_LEN = 0

for sent in train_sentences:
    for word in sent:
        for c in word:
            unique_chars.add(c)
        if len(word) > MAX_WORD_LEN:
            MAX_WORD_LEN = len(word)

for c in unique_chars:
    char_to_idx[c] = len(char_to_idx)
char_to_idx[' '] = len(char_to_idx)


# %% [markdown]
# ### An Aside on Padding
# 
# #### How to do padding correctly for the characters?
# 
# 
# Assume we have got a sentence ["We", "love", "NLP"]. You are supposed to first prepend a certain number of blank characters to each of the words in this sentence.
# 
# How to determine the number of blank characters we need? The calculation of MAX_WORD_LEN is here for help (which we already provide in the starter code). For the given sentence, MAX_WORD_LEN equals 4. Therefore we prepend two blank characters to "We", zero blank character to "love", and one blank character to "NLP". So the resultant padded sentence we get should be ["  We", "love", " NLP"].
# 
# Then, we feed all characters in ["  We", "love", " NLP"] into a char-embedding layer, and get a tensor of shape (3, 4, char_embedding_dim). To make this tensor's shape proper for the char-level LSTM (nn.LSTM), we need to transpose this tensor, i.e. swap the first and the second dimension. So we get a tensor of shape (4, 3, char_embedding_dim), where 4 corresponds to seq_len and 3 corresponds to batch_size.
# 
# The last thing you need to do is to obtain the last hidden state from the char-level LSTM, and concatenate it with the word embedding, so that you can get an augmented representation of that word.
# 
# ![padding](https://raw.githubusercontent.com/chaojiang06/chaojiang06.github.io/master/TA/spring2022_CS4650/char_padding.png)
#   *An illustration for left padding characters*
# 
# #### Why doing the padding?
# Someone may ask why we want to do such a kind of padding, instead of directly passing each of the character sequences of each word one by one through an LSTM, to get the last hidden state. The reason is that if you don't do padding, then that means you can only implement this process using "for loop". For CharPOSTagger, if you implement it using "for loop", the training time would be approximately 150s (GPU) / 250s (CPU) per epoch, while it would be around 30s (GPU) / 150s (CPU) per epoch if you do the padding and feed your data in batches. Therefore, we strongly recommend you learn how to do the padding and transform your data into batches. In fact, those are quite important concepts which you should get yourself familar with, although it might take you some time.
# 
# #### Why doing *left* padding?
# Our hypothesis is that the suffixes of English words (e.g., -ly, -ing, etc) are more indicative than prefixes for the part-of-speech (POS). Though LSTM is supposed to be able to handle long sequences, it still lose information along the way and the information closer to the last state (which you use as char-level representations) will be retained better.
# 
# #### How to understand the dimention change?
# Assume we have got a sentence with 3 words ["We", "love", "NLP"], and assume the dimension of character embedding is 2, the dimension of word embedding is 4, the dimension of word-level LSTM's hidden layer is 5, the dimension of character-level LSTM's hidden layer is 6.
# 
# In `BasicPOSTagger`, the dimension change would be:
# 
# - ------ input ------> $(3\times 1\times 4)$
# - -- word-level LSTM --> $(3\times 1\times 5)$
# - ----- linear layer -----> $(3\times 1\times 44)$
# 
# In `CharPOSTagger`, after padding, character embedding, and swapping, the dimension change would be:
# 
# - ------ input ------> $($ MAX_WORD_LEN $\times 3\times 2)$
# -  -- character-level LSTM --> $($ MAX_WORD_LEN $\times 3\times 6)$
# - -- Take the last hidden state --> $(3\times 6)$
# - -- concatenate with word embedings --> $(3\times 1\times 10)$
# - -- word-level LSTM --> $(3\times 1\times 5)$
# - -- linear layer --> $(3\times 1\times 44)$.

# %%
EMBEDDING_DIM = 4
HIDDEN_DIM = 8
LEARNING_RATE = 0.1
LSTM_LAYERS = 1
DROPOUT = 0
EPOCHS = 10
CHAR_EMBEDDING_DIM = 4
CHAR_HIDDEN_DIM = 4

# %% [markdown]
# ### 3.1 Define Model [5 points]

# %%
class CharPOSTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, char_embedding_dim,
                 char_hidden_dim, char_size, vocab_size, tagset_size):
        """
        Define and initialize anything needed for the forward pass.

        You are required to create a model with:
          an embedding layer for word: that maps words to their embedding space
          an embedding layer for character: that maps characters to their embedding space
          a character-level LSTM layer: that finds the character-level embedding for a word
          a word-level LSTM layer: that takes the concatenated representation per word (word embedding + char-lstm) as input and outputs hidden states
          a linear layer: maps from hidden state space to tag space
        """
        super(CharPOSTagger, self).__init__()

        ### BEGIN YOUR CODE ###
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.char_embedding = nn.Embedding(char_size, char_embedding_dim)

        self.char_lstm = nn.LSTM(char_embedding_dim, char_hidden_dim, LSTM_LAYERS, batch_first=True)
        self.word_lstm = nn.LSTM(embedding_dim + char_hidden_dim, hidden_dim, LSTM_LAYERS, batch_first=True)

        self.linear = nn.Linear(hidden_dim, tagset_size)
        ### END YOUR CODE ###

    def forward(self, sentence, chars):
        tag_scores = None
        """
        Implement the forward pass.

        Given a tokenized index-mapped sentence and a character sequence as the arguments,
        find the corresponding raw scores for tags (without softmax)

        returns:: tag_scores (Tensor)
        """

        ### BEGIN YOUR CODE ###
        word_embeddings = self.word_embedding(sentence).unsqueeze(0)
        batch_size, seq_len, max_word_len = chars.shape
        chars = chars.view(-1, max_word_len)
        char_embeddings = self.char_embedding(chars)

        c_out, (c_hn, c_cn) = self.char_lstm(char_embeddings)
        c_hn = c_hn.squeeze(0)

        c_hn = c_hn.view(batch_size, seq_len, -1)


        combined = torch.cat((word_embeddings, c_hn), dim=2)

        w_out, (w_hn, w_cn) = self.word_lstm(combined)

        tag_scores = self.linear(w_out)
        ### END YOUR CODE ###

        return tag_scores

# %% [markdown]
# ### 3.2 Training [5 points]

# %%
def train_char(epoch, model, loss_function, optimizer):
    model.train()
    train_loss = 0
    train_examples = 0
    for sentence, tags in training_data:
        """
        Implement the training method

        Hint: you can use the prepare_sequence method for creating index mappings
          for sentences. For constructing character input, you may want to left pad
          each word to MAX_WORD_LEN first, then use prepare_sequence method to create
          index  mappings.
        """

        ### BEGIN YOUR CODE ###

        # Zero out the parameter gradients
        optimizer.zero_grad()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        # Prepare input data (sentences, characters, and gold labels)
        input_words = prepare_sequence(sentence, word_to_idx)

        input_chars = [[char_to_idx[c] if c in char_to_idx else char_to_idx[' '] for c in word] for word in sentence]
        max_word_len = max(len(word) for word in sentence)
        input_chars = [[0] * (max_word_len - len(chars)) + chars for chars in input_chars]
        input_chars = torch.tensor(input_chars).unsqueeze(0)
        
        target = prepare_sequence(tags, tag_to_idx)

        # Do forward pass with current batch of input
        tag_scores = model(input_words, input_chars)

        # Get loss with model predictions and true labels
        loss = loss_function(tag_scores.view(-1, tag_scores.shape[-1]), target.view(-1))

        # Update model parameters
        loss.backward()
        optimizer.step()

        # Increase running total loss and the number of past training samples
        train_loss += loss.item()
        train_examples += len(sentence)

        ### END YOUR CODE ###

    avg_train_loss = train_loss / train_examples
    avg_val_loss, val_accuracy = evaluate_char(model, loss_function)

    print(f"Epoch: {epoch}/{EPOCHS}\tAvg Train Loss: {avg_train_loss:.4f}\tAvg Val Loss: {avg_val_loss:.4f}\t Val Accuracy: {val_accuracy:.0f}")

def evaluate_char(model, loss_function):
    """
    returns:: avg_val_loss (float)
    returns:: val_accuracy (float)
    """
    model.eval()
    correct = 0
    val_loss = 0
    val_examples = 0
    with torch.no_grad():
        for sentence, tags in val_data:
            """
            Implement the evaluate method. Find the average validation loss
            along with the validation accuracy.

            Hint: To find the accuracy, argmax of tag predictions can be used.
            """

            ### BEGIN YOUR CODE ###

            # Prepare input data (sentences, characters, and gold labels)
            input_words = prepare_sequence(sentence, word_to_idx)

            input_chars = [[char_to_idx[c] if c in char_to_idx else char_to_idx[' '] for c in word] for word in sentence]
            max_word_len = max(len(word) for word in sentence)
            input_chars = [[0] * (max_word_len - len(chars)) + chars for chars in input_chars]
            input_chars = torch.tensor(input_chars).unsqueeze(0)

            target = prepare_sequence(tags, tag_to_idx)

            # Do forward pass with current batch of input
            tag_scores = model(input_words, input_chars)

            # Get loss with model predictions and true labels
            loss = loss_function(tag_scores.view(-1, tag_scores.shape[-1]), target.view(-1))

            # Get the predicted labels
            _, predicted = torch.max(tag_scores, dim=2)

            # Get number of correct prediction
            correct += (predicted.view(-1) == target.view(-1)).sum().item()

            # Increase running total loss and the number of past valid samples
            val_loss += loss.item()
            val_examples += len(sentence)

            ### END YOUR CODE ###
    val_accuracy = 100. * correct / val_examples
    avg_val_loss = val_loss / val_examples
    return avg_val_loss, val_accuracy

# %%
# Initialize the model, optimizer and the loss function
# Hint, you may want to use reduction='sum' in the CrossEntropyLoss function

### BEGIN YOUR CODE ###
model = CharPOSTagger(EMBEDDING_DIM, HIDDEN_DIM, CHAR_EMBEDDING_DIM, CHAR_HIDDEN_DIM, len(char_to_idx), len(word_to_idx), len(tag_to_idx))
loss_function = nn.CrossEntropyLoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), LEARNING_RATE, momentum=0.9)
### END YOUR CODE ###

for epoch in range(1, EPOCHS + 1):
    train_char(epoch, model, loss_function, optimizer)

# %% [markdown]
# *Hint: Under the default hyperparameter setting, after 5 epochs you should be able to get at least `0.85` accuracy on the validation set.*

# %% [markdown]
# ### 3.3 Error analysis [5 points]
# Write a method to generate predictions for the validation set.
# Create lists of words, tags predicted by the model and ground truth tags.
# 
# Then use these lists to carry out error analysis to find the top-10 types of errors made by the model.
# 
# This part is very similar to part 1.7. You may want to refer to your implementation there.

# %%
def generate_predictions(model, val_data):
    """
    Generate predictions for val_data

    Create lists of words, tags predicted by the model and ground truth tags.
    Hint: It should look very similar to the evaluate function.

    returns:: word_list (str list)
    returns:: model_tags (str list)
    returns:: gt_tags (str list)
    """
    ### BEGIN YOUR CODE ###
    model.eval()
    word_list = []
    model_tags = []
    gt_tags = []
    with torch.no_grad():
        for sentence, tags in val_data:
            input = prepare_sequence(sentence, word_to_idx)
            
            input_chars = [[char_to_idx[c] if c in char_to_idx else char_to_idx[' '] for c in word] for word in sentence]
            max_word_len = max(len(word) for word in sentence)
            input_chars = [[0] * (max_word_len - len(chars)) + chars for chars in input_chars]
            input_chars = torch.tensor(input_chars).unsqueeze(0)
            tag_scores = model(input, input_chars)
            
            _, predicted = torch.max(tag_scores, dim=2)
            predicted = predicted.view(-1)

            word_list.extend(sentence)
            model_tags.extend([idx_to_tag[idx.item()] for idx in predicted])
            gt_tags.extend(tags)


    ### END YOUR CODE ###

    return word_list, model_tags, gt_tags

def error_analysis(word_list, model_tags, gt_tags):
    """
    Carry out error analysis

    From those lists collected from the above method, find the
    top-10 tuples of (model_tag, ground_truth_tag, frequency, example words)
    sorted by frequency

    returns: errors (list of tuples)
    """
    ### BEGIN YOUR CODE ###
    error_count = {}
    error_eg = {}

    for i in range(len(word_list)):
        pred_tag = model_tags[i]
        gt_tag = gt_tags[i]
        word = word_list[i]

        if pred_tag != gt_tag:
            key = (pred_tag, gt_tag)
            
            if key in error_count:
                error_count[key] += 1
                if len(error_eg[key]) < 5:
                    error_eg[key].append(word)
            else:
                error_count[key] = 1
                error_eg[key] = [word]
    
    errors = []
    for key in error_count:
        errors.append((key[0], key[1], error_count[key], error_eg[key]))

    errors.sort(key=lambda x: x[2], reverse=True)
    ### END YOUR CODE ###

    return errors

word_list, model_tags, gt_tags = generate_predictions(model, val_data)
errors = error_analysis(word_list, model_tags, gt_tags)

for i in errors[:10]:
  print(i)

# %% [markdown]
# **Report your findings here.**  
# What kinds of errors does the character-level model make as compared to the original model, and why do you think it made them?
# 
# The character-level model frequently misclassifies common nouns (NN) as proper nouns (NNP) and vice versa. This suggests that it may rely on capitalization patterns but struggles with unusual or context-dependent proper nouns. The original model on the other hand misclassifies named entities possibly due to limited exposure to specific proper names in the training data.

# %% [markdown]
# ## 4. Fine-tuned BERT POS Tagger [Extra Credit - 5 points]
# 
# In the above sections, we trained sequence-based models for POS tagging on a fairly limited dataset of *labeled* part of speech data. However, we can imagine the model is having to both learn the basics of language *and* part of speech tagging simultaneously. Perhaps, we can use a model pre-trained on a much larger corpus of language, and *fine-tune* the model on our specific task.
# 
# For this, we can use **BERT** (see [*Pre-training of Deep Bidirectional Transformers for Language Understanding*](https://aclanthology.org/N19-1423.pdf) NAACL, 2019). BERT introduces a method of pre-training a transformer encoder and fine-tuning the encoder on downstream tasks, and is extrordinarily infuential in NLP research and engineering (e.g., [`bert-base-uncased`](https://huggingface.co/bert-base-uncased) has 45M downloads per month from Huggingface). The core idea is *transfer learning*, or that pre-training on a self-supervised mask language modeling objective can help with our downstream language task of POS tagging. For a step-by-step introduction to the BERT architecture, please see Jay Almmar's [The Illustrated BERT](http://jalammar.github.io/illustrated-bert/).
# 
# This section will walk you through the use of the popular **Huggingface Transformers** library (see [*Transformers: State-of-the-Art Natural Language Processing*](https://aclanthology.org/2020.emnlp-demos.6), the [HuggingFace Documentation](https://huggingface.co/transformers/) and [Abhishek Mishra's HF tutorial](https://github.com/abhimishra91/transformers-tutorials)), which is a widely used library for distributing and using transformer models. Luckily, we can think of the HuggingFace library as a wrapper on top of PyTorch, so these sections should look familiar to your work so far.
# 
# **For this extra credit section, we will use a pre-trained BERT model, and fine-tune it on the POS tagging task.**

# %% [markdown]
# ### 4.1 Install `transformers` and download DistilBERT
# 
# For your fine-tuning code to run a bit faster, we will use a smaller "distilled" version of BERT called **DistilBERT** (see [*DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter*](https://arxiv.org/abs/1910.01108)). Fortunately with the `transformers` library, we could swap out the underlying model with no code changes to our dataloaders, architecture or traning setup!

# %%
!pip install -qU tokenizers transformers

# %%
# If you are interested in what other models are available, you can find a
# list of model names here (e.g., roberta-base, bert-base-uncased):
# https://huggingface.co/transformers/pretrained_models.html

from transformers import DistilBertModel, DistilBertTokenizerFast
bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# %%
# Let's take a look at our DistilBERT architecture
bert_model

# %% [markdown]
# ### 4.2 Load the dataset with a PyTorch dataloader
# 
# Please take a look at the `bert-base-cased` tokenizer on the [Tokenizer Playground](https://huggingface.co/spaces/Xenova/the-tokenizer-playground). Our goal will be to predict the POS of each word, but BERT is trained on sub-word tokens, so we need to segment our dataset such that **only the first token of each word is classified**.

# %%
from torch.utils.data import Dataset

class POSDataset(Dataset):
  def __init__(self, data, tokenizer, max_len):
    self.data = data
    self.tokenizer = tokenizer
    self.max_len = max_len

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    """
    Given an index, return the value in your training data (self.data). Make
    sure the full output dict from self.tokenizer is returned, with an additional
    value for your labels.

    Remember! Your BERT tokenizer will give multiple tokens to words with the
    same POS tag. We want the FIRST token be given the tag and all other tokens
    to be given -100.

    Hint: You may use the prepare_sequence() function from earlier sections
    Hint: Our training data is already tokenized, so you may find the `is_split_into_words=True`
      and `return_offsets_mapping=True` arguments helpful for getting the token offsets.
    Hint: When using the tokenizer, you can also use padding='max_length' for [PAD]
      tokens to be added for you.
    """
    encoding = None

    ### BEGIN YOUR CODE ###

    # Get the sentence and POS tags
    sentence, tags = self.data[index]
    tag_indices = [tag_to_idx[tag] for tag in tags]

    # Use the BERT tokenizer (self.tokenizer) to encode the sentence. Make sure to
    # truncate the sentence if it is longer than self.max_len, and pad the sentence if it
    # is less than self.max_len.
    encoding = self.tokenizer(
            sentence,
            is_split_into_words=True,
            return_offsets_mapping=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
    
    input_ids = encoding["input_ids"].squeeze(0)
    attention_mask = encoding["attention_mask"].squeeze(0)
    offset_mapping = encoding["offset_mapping"].squeeze(0)

    # Create token labels, where the first token of each word is the POS tag, and
    # all others are -100.
    labels = torch.full((self.max_len,), -100, dtype=torch.long)

    # Add the token labels back to the tokenized dict
    word_idx = -1
    for i, (start, end) in enumerate(offset_mapping):
        if start == 0 and end != 0:
            word_idx += 1
            if word_idx < len(tag_indices):
                labels[i] = tag_indices[word_idx]

    # Make sure both your encoded sentence, labels and attention mask are PyTorch tensors
    encoding["labels"] = labels
    encoding["input_ids"] = input_ids
    encoding["attention_mask"] = attention_mask

    ### END YOUR CODE ###

    return encoding

# %%
# Use your POSDataset class to create a train and test set
MAX_LEN = 128

# Further split your train data into train/test. You now have train/test/val.
train_test_data, split = training_data, int(0.7 * len(training_data))
random.shuffle(train_test_data)
split_training_data, split_test_data = train_test_data[:split], train_test_data[split:]

training_set = POSDataset(split_training_data, tokenizer, MAX_LEN)
testing_set = POSDataset(split_test_data, tokenizer, MAX_LEN)
validation_set = POSDataset(val_data, tokenizer, MAX_LEN)

# %%
# Print a few values from your Dataloader!
print(training_set.__getitem__(0)['input_ids'])
print(training_set.__getitem__(0)['labels'])

# %%
# Create PyTorch dataloaders from the POSDataset
from torch.utils.data import DataLoader

training_loader = DataLoader(training_set, batch_size=64, shuffle=True)
testing_loader = DataLoader(testing_set, batch_size=64, shuffle=True)
validating_loader = DataLoader(validation_set, batch_size=8, shuffle=True)

# %% [markdown]
# ### 4.3 Define your `BertForPOSTagging` Model
# 
# Now we will modify BERT by extending the `DistilBertModel` class for our task.

# %%
class BertForPOSTagging(DistilBertModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        ### BEGIN YOUR CODE ###
        ### END YOUR CODE ###

        self.post_init()

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass through your model. Returns output logits for each POS
        label and the loss (if labels is not None)

        Hint: You may use nn.CrossEntropyLoss() to calculate your loss.
        """
        loss, logits = None, None

        ### BEGIN YOUR CODE ###



        ### END YOUR CODE ###

        if loss is not None:
          return loss, logits
        return logits

# %%
model = BertForPOSTagging.from_pretrained(
    'distilbert-base-uncased',
    num_labels=len(tag_to_idx)
).to(device)

MAX_GRAD_NORM = 10
EPOCHS = 5

optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-04)

# %% [markdown]
# ### 4.4 Training and Evaluation
# 
# Now we have instantiated our model, please create the train loop!
# 
# *Hint: If your implementation is correct, you can expect a validation accuracy of `0.88`*

# %%
# DistilBERT will take up a lot of memory (particularly during development)
# use this to check the amount of memory you currently have. (Note: you should
# be able to fine-tune with ~5 GB of GPU memory)
print(f"Currently allocated GPU memory: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB / {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# Hint: use `torch.cuda.empty_cache()` to clear the CUDA cache

# %%
def train(epoch):
    train_loss = 0
    train_examples, train_steps = 0, 0

    model.train()
    model.zero_grad()

    for idx, batch in enumerate(training_loader):
        ids = batch['input_ids'].to(device, dtype=torch.long)
        mask = batch['attention_mask'].to(device, dtype=torch.long)
        labels = batch['labels'].to(device, dtype=torch.long)

        ### BEGIN YOUR CODE ###



        ### END YOUR CODE ###

        train_steps += 1
        train_examples += labels.size(0)

    avg_train_loss = train_loss / train_steps
    avg_val_loss, val_accuracy = evaluate_bert(model)

    print(f"Epoch: {epoch}/{EPOCHS}\tAvg Train Loss: {avg_train_loss:.4f}\tAvg Val Loss: {avg_val_loss:.4f}\t Val Accuracy: {val_accuracy:.0f}")

def evaluate_bert(model):
    correct, val_loss, val_examples = 0, 0, 0

    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(validating_loader):
            """
            Implement the evaluate method. Find the average validation loss
            along with the validation accuracy.

            Remember! You have labeled only the first token of each word. Make
            sure you only calculate accuracy on values which are not -100.
            """
            ids = batch['input_ids'].to(device, dtype=torch.long)
            mask = batch['attention_mask'].to(device, dtype=torch.long)
            labels = batch['labels'].to(device, dtype=torch.long)

            ### BEGIN YOUR CODE ###


            # Compute training accuracy

            # Only compute accuracy at active labels

            # Get the predicted labels

            # Get number of correct predictions

            # Increase running total loss and the number of past valid samples


            ### END YOUR CODE ###

    val_accuracy = 100 * correct / val_examples
    avg_val_loss = val_loss / val_examples
    return avg_val_loss, val_accuracy

# %%
for epoch in range(EPOCHS):
    train(epoch)

# %% [markdown]
# ### 4.5 Inference
# 
# Good job! Now we can use our fine-tuned BERT model for POS tagging.
# 
# In fact, if you have a fine-tuned transformer model (such as in a final project), you could directly upload the model to HuggingFace for others to use (see [this group](https://huggingface.co/QCRI/bert-base-multilingual-cased-pos-english), which fine-tuned on a much larger corpus of POS tags).

# %%
def generate_prediction(model, sentence):
    """
    Given a sentence, generate a full prediction of POS tags.

    In this case, you are given a full sentence (not array of tokens), so you
    will need to use your tokenizer differently.

    Return your prediction in the format:
      [(token 1, POS prediction 1), (token 2, POS prediction 2), ...]

    E.g., "The imperatives that" => [('the', 'DT'), ('imperative', 'NNS'), ('that', 'WDT')]
    """
    prediction = []

    ### BEGIN YOUR CODE ###



    ### END YOUR CODE ###

    return prediction

# %%
sentence = "The imperatives that can be obeyed by a machine that has no limbs are bound to be of a rather intellectual character."
print(generate_prediction(model, sentence))

# %% [markdown]
# ## 5. Submit Your Homework
# This is the end of Project 2. Congratulations!
# 
# Now, follow the steps below to submit your homework in Gradescope:
# 
# 1. Rename this ipynb file to 'CS4650_p2_GTusername.ipynb'. We recommend ensuring you have removed any extraneous cells & print statements, clearing all outputs, and using the Runtime --> Run all tool to make sure all output is update to date. Additionally, leaving comments in your code to help us understand your operations will assist the teaching staff in grading. It is not a requirement, but is recommended.
# 2. Click on the menu 'File' --> 'Download' --> 'Download .py'.
# 3. Click on the menu 'File' --> 'Download' --> 'Download .ipynb'.
# 4. Download the notebook as a .pdf document. Make sure the output from Parts 1.6 & 2 & 3 are captured so we can see how the loss and accuracy changes while training.
# 5. Upload all 3 files to Gradescope. Double check the files start with `CS4650_p2_*`, capitalization matters.
# 
# 


