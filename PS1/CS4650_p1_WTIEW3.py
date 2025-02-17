# %% [markdown]
# <a href="https://colab.research.google.com/github/fymbc/cs4650/blob/main/PS1" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# # Headline Classification with Neural BOW, LSTM
# **CS 4650 "Natural Language Processing" Project 1**  
# Georgia Tech, Spring 2025 (Instructor: Weicheng Ma)
# 
# Welcome to the first full programming project for CS 4650! **To start, first make a copy of this notebook to your local drive, so you can edit it.**
# 
# If you want GPUs (which will improve training times), you can always change your instance type by going to Runtime -> Change runtime type -> Hardware accelerator.
# 
# **In this project, we will be using PyTorch.** If you are new to PyTorch, or simply want a refresher, we recommend you start by looking through these [Introduction to PyTorch](https://sites.cc.gatech.edu/classes/AY2021/cs7650_fall/slides/Introduction_to_PyTorch.pdf) slides and this interactive [PyTorch basics notebook](http://bit.ly/pytorchbasics). Additionally, this [text sentiment](http://bit.ly/pytorchexample) notebook will provide some insight into working with PyTorch with a specific NLP task.

# %% [markdown]
# ## 1. Load and preprocess data [10 points]
# This project will be modeling a *classification task* for headlines from [The Onion](https://www.theonion.com), a satirical news website. Our dataset contains headlines and whether they belong to The Onion or CNN. Given a headline, we want to predict whether it is Onion or not.
# 
# The following cell loads, pre-processes and tokenizes our OnionOrNot dataset.

# %%
!curl -so OnionOrNot.csv https://raw.githubusercontent.com/lukefeilberg/onion/master/OnionOrNot.csv

# %%
# ===========================================================================
# Run some setup code for this notebook. Don't modify anything in this cell.
# ===========================================================================

import torch
import random, sys

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

# Check what version of Python is running
print(sys.version)

# %% [markdown]
# ### 1.1 Dataset preprocessing functions
# The following cell define some methods to clean the dataset, but feel free to take a look to see some of the operations it's doing.
# 

# %%
# ===========================================================================
# Run some preprocessing code for our dataset. Don't modify anything in this cell.
# This code was adapted from fast-bert.
# ===========================================================================

import re
import html

def spec_add_spaces(t: str) -> str:
    "Add spaces around / and # in `t`. \n"
    return re.sub(r"([/#\n])", r" \1 ", t)

def rm_useless_spaces(t: str) -> str:
    "Remove multiple spaces in `t`."
    return re.sub(" {2,}", " ", t)

def replace_multi_newline(t: str) -> str:
    return re.sub(r"(\n(\s)*){2,}", "\n", t)

def fix_html(x: str) -> str:
    "List of replacements from html strings in `x`."
    re1 = re.compile(r"  +")
    x = (
        x.replace("#39;", "'")
        .replace("amp;", "&")
        .replace("#146;", "'")
        .replace("nbsp;", " ")
        .replace("#36;", "$")
        .replace("\\n", "\n")
        .replace("quot;", "'")
        .replace("<br />", "\n")
        .replace('\\"', '"')
        .replace(" @.@ ", ".")
        .replace(" @-@ ", "-")
        .replace(" @,@ ", ",")
        .replace("\\", " \\ ")
    )
    return re1.sub(" ", html.unescape(x))

def clean_text(input_text):
    text = fix_html(input_text)
    text = replace_multi_newline(text)
    text = spec_add_spaces(text)
    text = rm_useless_spaces(text)
    text = text.strip()
    return text

# %% [markdown]
# ### 1.2 Tokenize using NLTK
# 
# We will use our rule-based `clean_text` function to clean our raw text, then use the popular NLTK [punkt tokenizer](https://www.nltk.org/_modules/nltk/tokenize/punkt.html) to convert text to individual sub-words. This will take a while because you have to download the pre-trained punkt tokenizer.
# 
# *If you are interested: There's a [long and diverse history of converting raw text to "tokens"](https://arxiv.org/abs/2112.10508), and many available methods/algorithms (you can experiment with some recently trained ones, trained on a dynamic programming-based method called BPE, [here](https://huggingface.co/spaces/Xenova/the-tokenizer-playground)).*

# %%
# ===========================================================================
# Tokenize using punkt. Don't modify anything in this cell.
# ===========================================================================

import pandas as pd
import nltk
from tqdm import tqdm

nltk.download('punkt_tab')
nltk.download('punkt')
df = pd.read_csv("OnionOrNot.csv")
df["tokenized"] = df["text"].apply(lambda x: nltk.word_tokenize(clean_text(x.lower())))

# %% [markdown]
# We will use `pandas`, a popular library for data analysis and table manipulation, in this project to manage the dataset. For more information on usage, please refer to the [Pandas documentation](https://pandas.pydata.org/docs/).
# 
# The primary data structure in Pandas is a `DataFrame`. The following cell will print out the basic information contained in our `DataFrame` structure, and the first few rows of our dataset.

# %%
# View the first few entries of our dataset
df.head()

# %% [markdown]
# Try to guess some examples! Is the task more difficult than you expected?
# 
# DataFrames can be indexed using [`.iloc[]`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.iloc.html). `iloc` uses interger based indexing and supports a single integer (`df.iloc[42]`), a list of integers (`df.iloc[[1, 5, 42]]`), or a slice (`df.iloc[7:42]`).

# %%
# E.g., get row 42 of our dataset
df.iloc[42]

# %% [markdown]
# ### 1.3 Split the dataset into training, validation, and testing

# %% [markdown]
# **Train/Test/Val Split** - Now that we've loaded this dataset, we need to split the data into train, validation, and test sets.
# 
# A good explanation of why we need these different sets can be found in $\S$2.2.5 of [Eisenstein](https://github.com/jacobeisenstein/gt-nlp-class/blob/master/notes/eisenstein-nlp-notes.pdf) but our high-level goal is to have a generalized model and have confidence in our results.
# 
# 
# The *training set* is used to fit our model's learned parameters (weights and biases) to the task. The *validation  set* (sometimes called development set) is used to verify our training jobs are minimizing loss on an unseen subset of the data and can also be used to help choose hyperparameters for our training setup. The *test set* is used to provide a final evaluation of our trained model (unbiased by development or training decisions), ideally providing some insight into how the model will perform in a scenario we cannot perfectly represent in our data (i.e., the real world). *Each of these sets should be disjoint from the others*, to prevent any leackage that could introduce bias in our evaluation metrics (in this case accuracy).
# 
# **Model Vocabulary** - We cannot directly feed sub-word token strings into a model! We need to create a "vocab map", which contains an ID for each unique token in our Onion dataset. This will be used as a "lookup" in the next few sections, since your PyTorch implementation will require first converting your Onion token representations to a list of sub-word IDs.
# 
# **In the following cell, please implement `split_train_val_test` and `generate_vocab_map`.**

# %%
# ===========================================================================
# Set constants for PAD and UNK. You will use these values, but DO NOT change
# them, or import additional packages.

from collections import Counter
PADDING_VALUE = 0
UNK_VALUE     = 1

# ===========================================================================


def split_train_val_test(df, props=[.8, .1, .1]):
    """
    This method takes a dataframe and splits it into train/val/test splits.
    It uses the props argument to split the dataset appropriately.

    Args:
      df (pd.DataFrame): A dataset as a Pandas DataFrame
      props (list): Proportions for each split in the order of [train, validation, test].
                    the last value of the props array is repetitive, but we've kept it for clarity.

    Returns:
      train_df (pd.DataFrame): Train DataFrame split.
      val_df (pd.DataFrame): Validation DataFrame split.
      test_df (pd.DataFrame): Test DataFramem split.
    """
    assert round(sum(props), 2) == 1 and len(props) >= 2
    train_df, test_df, val_df = None, None, None

    ### BEGIN YOUR CODE (~3-5 lines) ###
    ### Hint: You can use df.iloc to slice into specific indexes or ranges.
    length = len(df)
    train_df = df.iloc[:int(props[0]*length)]
    val_df = df.iloc[int(props[0]*length):int(props[0]*length + props[1]*length)]
    test_df = df.iloc[int(props[0]*length + props[1]*length):]
    ### END YOUR CODE ###

    return train_df, val_df, test_df


def generate_vocab_map(df, cutoff=2):
    """
    This method takes a dataframe and builds a vocabulary to unique number map.
    It uses the cutoff argument to remove rare words occuring <= cutoff times.
    *NOTE*: "" and "UNK" are reserved tokens in our vocab that will be useful
    later. You'll also find the Counter imported for you to be useful as well.

    Args:
      df (pd.DataFrame): The entire dataset this mapping is built from
      cutoff (int): We exclude words from the vocab that appear less than or
                    eq to cutoff

    Returns:
      vocab (dict[str, int]):
        In vocab, each str is a unique token, and each dict[str] is a
        unique integer ID. Only elements that appear > cutoff times appear
        in vocab.

      reversed_vocab (dict[int, str]):
        A reversed version of vocab, which allows us to retrieve
        words given their unique integer ID. This map will
        allow us to "decode" integer sequences we'll encode using
        vocab!
    """

    vocab          = {"": PADDING_VALUE, "UNK": UNK_VALUE}
    reversed_vocab = None

    ### BEGIN YOUR CODE (~5-15 lines) ###
    ### Hint: Start by iterating over df["tokenized"]
    c = Counter()

    for i in df["tokenized"]:
        c.update(i)

    for i in c:
        if c[i] > cutoff:
            vocab[i] = len(vocab)

    reversed_vocab = {v:k for k, v in vocab.items()}   

    ### END YOUR CODE ###

    return vocab, reversed_vocab

# %% [markdown]
# With the methods you have implemented above, we can now split the dataset into training, validation, and testing sets and generate our dictionaries mapping from word tokens to IDs (and vice versa).
# 
# *Note: The props list currently being used splits the dataset so that 80% of samples are used to train, and the remaining 20% are evenly split between training and validation. How you split your dataset is itself a major choice and something you would need to consider in your own projects. Can you think of why?*

# %%
df                         = df.sample(frac=1)
train_df, val_df, test_df  = split_train_val_test(df, props=[.8, .1, .1])
train_vocab, reverse_vocab = generate_vocab_map(train_df)

# %%
# ===========================================================================
# This line of code will help test your implementation, the expected output is
# the same distribution used in 'props' in the above cell. Try out some
# different values to ensure it works, but for submission ensure you use
# [.8, .1, .1]
# ===========================================================================

(len(train_df) / len(df)), (len(val_df) / len(df)), (len(test_df) / len(df))

# %% [markdown]
# ### 1.4 Building a Dataset Class

# %% [markdown]
# PyTorch has custom Dataset Classes that have very useful extentions, we want to turn our current pandas DataFrame into a subclass of Dataset so that we can iterate and sample through it for minibatch updates. **In the following cell, fill out the `HeadlineDataset` class.** Refer to PyTorch documentation on [Dataset Classes](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
# for help.

# %%
# ===========================================================================
# Please do not change, or import additional packages.
from torch.utils.data import Dataset
# ===========================================================================

class HeadlineDataset(Dataset):
  """
  This class takes a Pandas DataFrame and wraps in a PyTorch Dataset.
  Read more about Torch Datasets here:
  https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
  """

  def __init__(self, vocab, df, max_length=50):
    """
    Initialize this class with appropriate instance variables

    We would *strongly* recommend storing the dataframe itself as an instance
    variable, and keeping this method very simple. Leave processing to
    __getitem__.

    Sometimes, however, it does make sense to preprocess in __init__. If you
    are curious as to why, read the aside at the bottom of this cell.
    """

    ### BEGIN YOUR CODE (~3 lines) ###
    self.vocab = vocab
    self.df = df
    self.maxLength = max_length
    return
    ### END YOUR CODE ###

  def __len__(self):
    """
    Return the length of the dataframe instance variable
    """
    df_len = None

    ### BEGIN YOUR CODE (~1 line) ###
    df_len = len(self.df)
    ### END YOUR CODE ###

    return df_len

  def __getitem__(self, index: int):
    """
    Converts a dataframe row (row["tokenized"]) to an encoded torch LongTensor,
    using our vocab map created using generate_vocab_map. Restricts the encoded
    headline length to max_length.

    The purpose of this method is to convert the row - a list of words - into
    a corresponding list of numbers.

    i.e. using a map of {"hi": 2, "hello": 3, "UNK": 0}
    this list ["hi", "hello", "NOT_IN_DICT"] will turn into [2, 3, 0]

    Returns:
      tokenized_word_tensor (torch.LongTensor):
        A 1D tensor of type Long, that has each token in the dataframe mapped to
        a number. These numbers are retrieved from the vocab_map we created in
        generate_vocab_map.

        **IMPORTANT**: if we filtered out the word because it's infrequent (and
        it doesn't exist in the vocab) we need to replace it w/ the UNK token.

      curr_label (int):
        Binary 0/1 label retrieved from the DataFrame.

    """
    tokenized_word_tensor = None
    curr_label            = None

    ### BEGIN YOUR CODE (~3-7 lines) ###
    temp = []
    for i in self.df.iloc[index]['tokenized']:
        temp.append(self.vocab.get(i, UNK_VALUE))
    if len(temp) > self.maxLength:
        temp = temp[:self.maxLength]
    else:
        temp = temp + [PADDING_VALUE] * (self.maxLength - len(temp))
    tokenized_word_tensor = torch.LongTensor(temp)
    curr_label = self.df.iloc[index]['label']
    ### END YOUR CODE ###

    return tokenized_word_tensor, curr_label


# ===========================================================================
# Completely optional aside on preprocessing in __init__.
#
# Sometimes the compute bottleneck actually ends up being in __getitem__.
# In this case, you'd loop over your dataset in __init__, passing data
# to __getitem__ and storing it in another instance variable. Then,
# you can simply return the preprocessed data in __getitem__ instead of
# doing the preprocessing.
#
# There is a tradeoff though: can you think of one?
# ===========================================================================

# %%
from torch.utils.data import RandomSampler

train_dataset = HeadlineDataset(train_vocab, train_df)
val_dataset   = HeadlineDataset(train_vocab, val_df)
test_dataset  = HeadlineDataset(train_vocab, test_df)

# Now that we're wrapping our dataframes in PyTorch datsets, we can make use of
# PyTorch Random Samplers, they'll define how our DataLoaders sample elements
# from the HeadlineDatasets
train_sampler = RandomSampler(train_dataset)
val_sampler   = RandomSampler(val_dataset)
test_sampler  = RandomSampler(test_dataset)

# %% [markdown]
# ### 1.5 Finalizing our DataLoader

# %% [markdown]
# We can now use PyTorch `DataLoader` to batch our data for us. **In the following cell, please implement `collate_fn`.** Refer to PyTorch documentation on [`DataLoader`](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) for help.

# %%
# ===========================================================================
# Please do not change, or import additional packages.
from torch.nn.utils.rnn import pad_sequence
# ===========================================================================

def collate_fn(batch, padding_value=PADDING_VALUE):
  """
  This function is passed as a parameter to Torch DataSampler. collate_fn collects
  batched rows, in the form of tuples, from a DataLoader and applies some final
  pre-processing.

  Objective:
    In our case, we need to take the batched input array of 1D tokenized_word_tensors,
    and create a 2D tensor that's padded to be the max length from all our tokenized_word_tensors
    in a batch. We're moving from a Python array of tuples, to a padded 2D tensor.

    *HINT*: you're allowed to use torch.nn.utils.rnn.pad_sequence (ALREADY IMPORTED)

    Finally, you can read more about collate_fn here: https://pytorch.org/docs/stable/data.html

  Args:
    batch: PythonArray[tuple(tokenized_word_tensor: 1D Torch.LongTensor, curr_label: int)]
           len(batch) == BATCH_SIZE

  Returns:
    padded_tokens: 2D LongTensor of shape (BATCH_SIZE, max len of all tokenized_word_tensor))
    y_labels: 1D FloatTensor of shape (BATCH_SIZE)

  """
  padded_tokens, y_labels = None, None

  ### BEGIN YOUR CODE (~4-8 lines) ###
  token_tensor = [i[0] for i in batch]
  labels = [i[1] for i in batch]
  padded_tokens = pad_sequence(token_tensor, batch_first = True, padding_value=padding_value)
  y_labels = torch.FloatTensor(labels)
  ### END YOUR CODE ###

  return padded_tokens, y_labels

# %%
from torch.utils.data import DataLoader
BATCH_SIZE = 16

train_iterator = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, collate_fn=collate_fn)
val_iterator   = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler, collate_fn=collate_fn)
test_iterator  = DataLoader(test_dataset, batch_size=BATCH_SIZE, sampler=test_sampler, collate_fn=collate_fn)

# %%
# ===========================================================================
# Use this to test your collate_fn implementation.
#
# You can look at the shapes of x and y or put print statements in collate_fn
# while running this snippet
# ===========================================================================

for x, y in test_iterator:
    print(x, y)
    print(f'x: {x.shape}')
    print(f'y: {y.shape}')
    break
test_iterator = DataLoader(test_dataset, batch_size=BATCH_SIZE, sampler=test_sampler, collate_fn=collate_fn)

# %% [markdown]
# ## 2. Modeling [10 points]
# Now that we have a clean dataset and a useful PyTorch `DataLoader` object, we can begin building a model for our task! In the following code block, you will build a feed-forward neural network implementing a neural bag-of-words baseline, `NBOW-RAND`, described in $\S$2.1 of [this paper](https://www.aclweb.org/anthology/P15-1162.pdf). You may find [the PyTorch `torch.nn` docs](https://pytorch.org/docs/stable/nn.html) useful for understanding the different layers and [this PyTorch sequence models tutorial](https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html) for how to put together `torch.nn` layers.
# 
# The core intuition behind `NBOW-RAND` is that after we embed each word for our input, we average the embeddings to produce a single vector that hopefully averages the information across all embeddings. Formally, we first convert each document of length $n$ tokens into a matrix of $n\times d$, where $d$ is the dimension of the token embedding. Then we average all embeddings to produce a vector of length $d$.
# 
# If you are new to PyTorch, ensuring your matrix operations are correct is often the most common source of errors. Keep in mind how the dimensions change and what each axes represents. Your documents will be passed in as minibatches, so be careful when selecting which axes to apply certain operations. Feel free to experiment with the architecture of this network outside of the basic `NBOW-RAND` setup (such as adding in other layers) to see how this changes your results.

# %% [markdown]
# ### 2.1 Define the NBOW model class

# %%
# ===========================================================================
# Please do not change, or import additional packages.
import torch.nn as nn
# ===========================================================================

class NBOW(nn.Module):
  def __init__(self, vocab_size, embedding_dim):
    """
    Instantiate layers for your model.
    Your model architecture will be a feed-forward neural network.

    You will need 3 nn.Modules at minimum
     1. An embeddings layer (see nn.Embedding)
     2. A linear layer (see nn.Linear)
     3. A sigmoid output (see nn.Sigmoid)

    HINT: In the forward step, the BATCH_SIZE is the first dimension.
    """
    super().__init__()

    ### BEGIN YOUR CODE (~4 lines) ###
    self.embedding = nn.Embedding(vocab_size, embedding_dim)
    self.linear = nn.Linear(embedding_dim, 1)
    self.sigmoid = nn.Sigmoid()

    ### END YOUR CODE ###

  def forward(self, x):
    """
    Complete the forward pass of the model.

    Use the output of the embedding layer to create the average vector,
    which will be input into the linear layer.

    Args:
      x: 2D LongTensor of shape (BATCH_SIZE, max len of all tokenized_word_tensor))
         This is the same output that comes out of the collate_fn function you completed
    """
    ### BEGIN YOUR CODE (~4-5 lines) ###
    x = self.embedding(x)
    x = torch.mean(x, dim = 1)
    x = self.linear(x)
    x = self.sigmoid(x)

    return x
    ### END YOUR CODE ###


# %% [markdown]
# ### 2.2 Initialize the NBOW classification model
# 
# Since the NBOW model is rather basic, there is only one meaningful hyperparameter w.r.t. model architecture: the size of the embedding dimension (`embedding_dim`). (We also see a `vocab_size` parameter here, but this only a by-product on our cutoff for infrequent tokens, there also may more hyperparameters if you modified the architecture, such as adding a linear layer).
# 
# Remember the CUDA discussion in the first cell of this notebook? Here the `.to(device)` is where that discussion becomes relevant (if `device=='cuda'`, PyTorch will perform the matrix operations on GPU). If you recieve a mismatch error, your tensors may be on different devices.

# %%
model = NBOW(
  vocab_size    = len(train_vocab.keys()),
  embedding_dim = 300
).to(device)

# %% [markdown]
# ### 2.3 Instantiate the loss function and optimizer

# %% [markdown]
# Please select and instantiate an appropriate loss function and optimizer.
# 
# *Hint: What loss functions are availible for binary classification? Feel free to look at the [torch.nn docs on loss functions](https://pytorch.org/docs/stable/nn.html#loss-functions) for help!*

# %%
# While we import Adam for you, you may try / import other optimizers as well
from torch.optim import Adam

criterion, optimizer = None, None

### BEGIN YOUR CODE ###
criterion = nn.BCELoss()
optimizer = Adam(model.parameters())
### END YOUR CODE ###

# %% [markdown]
# Now that we have a NBOW model, a loss function, optimizer and dataset, we can begin training!

# %% [markdown]
# ## 3. Training and Evaluation [10 points]
# We will now instantiate a `train_loop`, and a `val_loop` to evaluate our model at each epoch.
# 
# **Fill out the train and test loops below. Treat real headlines as `False`, and Onion headlines as `True`.**

# %%
def train_loop(model, criterion, optim, iterator):
  """
  Returns the total loss calculated from criterion
  """
  model.train()
  total_loss = 0
  for x, y in tqdm(iterator):
    ### BEGIN YOUR CODE (~6 lines) ###
    x, y = x.to(device), y.to(device)
    optim.zero_grad()
    y_pred = model(x)
    loss = criterion(y_pred, y.unsqueeze(1))
    loss.backward()
    optim.step()

    total_loss += loss.item()
    ### END YOUR CODE ###

  return total_loss


def val_loop(model, iterator):
  """
  Returns:
    true (List[bool]): All the ground truth values taken from the dataset iterator
    pred (List[bool]): All model predictions.
  """
  true, pred = [], []

  ### BEGIN YOUR CODE (~8 lines) ###
  model.eval()

  for x, y in tqdm(iterator):
    y_pred = model(x)
    pred.extend(y_pred.round().tolist())
    true.extend(y.bool().tolist())

  ### END YOUR CODE ###

  return true, pred

# %% [markdown]
# ### 3.1 Define the evaluation metrics

# %% [markdown]
# We will also need evaluation metrics to tell us how well our model is doing on the validation set at each epoch and later how well the model does on the held-out test set. You may find $\S$4.4.1 of Eisenstein useful for these questions.
# 
# **Complete the functions in the following cell.**

# %%
# Note: You will not need to import anything to implement these functions.

def accuracy(true, pred):
  """
  Calculate the ratio of correct predictions.

  Args:
    true (List[bool]): ground truth
    pred (List[bool]): model predictions

  Returns:
    acc (float): percent accuracy with range [0, 1]
  """
  acc = None
  ### BEGIN YOUR CODE (~2-5 lines) ###
  pred = [i[0] for i in pred]
  num_correct = 0

  for i in range(len(true)):
      if true[i] == pred[i]:
         num_correct += 1
  acc = num_correct / len(true)


  ### END YOUR CODE ###
  return acc


def binary_f1(true, pred, selected_class=True):
  """
  Calculate F-1 scores for a binary classification task.

  Args:
    true (List[bool]): ground truth
    pred (List[bool]): model predictions
    selected_class (bool): the selected class the F-1 is being calculated for.

  Returns:
    f1 (float): F-1 score between [0, 1]
  """
  f1 = None
  ### BEGIN YOUR CODE (~10-15 lines) ###
  tp, fp, fn = 0, 0, 0
  pred = [i[0] for i in pred]

  for i in range(len(true)):
      if pred[i] == selected_class:
        if true[i] == selected_class:
           tp += 1
        else:
           fp += 1
      elif true[i] == selected_class:
        fn += 1
  
  precision = tp / (tp + fp) if (tp + fp) > 0 else 0
  recall = tp / (tp + fn) if (tp + fn) > 0 else 0

  f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

  ### END YOUR CODE ###
  return f1


def binary_macro_f1(true, pred):
  """
  Calculate averaged F-1 for all selected (true/false) classes.

  Args:
    true (List[bool]): ground truth
    pred (List[bool]): model predictions
  """
  averaged_macro_f1 = None
  ### BEGIN YOUR CODE (~1 line) ###
  averaged_macro_f1 = (binary_f1(true, pred, True) + binary_f1(true, pred, False)) / 2

  ### END YOUR CODE ###
  return averaged_macro_f1

# %%
# ===========================================================================
# To test your eval implementation, we will evaluate the untrained model on our
# dev dataset. It will do pretty poorly (it's untrained), but the exact performance
# will be random since the initialization of the model parameters is random.
# ===========================================================================

true, pred = val_loop(model, val_iterator)
print(f'Binary Macro F1: {binary_macro_f1(true, pred)}')
print(f'Accuracy: {accuracy(true, pred)}')

# %% [markdown]
# ## 4. Full Training Run [1 point]
# Now we can perform a full run and see the model fit our loss. If everything goes correctly, you should be able to achieve a validation F1 score of at least `0.80`
# 
# **Feel free to adjust the number of epochs to prevent overfitting or underfitting and to play with your model hyperparameters/optimizer & loss function.**

# %%
TOTAL_EPOCHS = 10
for epoch in range(TOTAL_EPOCHS):
    train_loss = train_loop(model, criterion, optimizer, train_iterator)
    true, pred = val_loop(model, val_iterator)
    print(f"EPOCH: {epoch}")
    print(f"TRAIN LOSS: {train_loss}")
    print(f"VAL F-1: {binary_macro_f1(true, pred)}")
    print(f"VAL ACC: {accuracy(true, pred)}")

# %% [markdown]
# We can also look at the models performance on the held-out test set, using the same `val_loop` from earlier.

# %%
true, pred = val_loop(model, test_iterator)
print(f"TEST F-1: {binary_macro_f1(true, pred)}")
print(f"TEST ACC: {accuracy(true, pred)}")

# %% [markdown]
# ## 5. Analysis [5 points]
# While modeling and accuracy are a great signal that our model is working in our specific task setup, an inspection of what the model is classifying (particularly its errors), can allow us to hypothesize about what is going on, why it works, and how to improve.
# 
# 

# %% [markdown]
# ### 5.1 Impact of Vocab Size
# **Question:** *What happens to the vocab size as you change the cutoff in the cell below? Can you explain this in the context of [Zipf's Law](https://en.wikipedia.org/wiki/Zipf%27s_law)?*
# 
# **Answer:** The size of the vocab decreases as cutoff increases. This is because increasing the cutoff results in only including words that appear at least a certain number of times, hence excluding rare words. Zipf's law notes that frequency of any word is inversely proportional to its rank in the frequency table, where vast majority of words will have lower frequencies and drop-off as we move down the list of ranked words.

# %%
tmp_vocab, _ = generate_vocab_map(train_df, cutoff = 3)
len(tmp_vocab)

# %% [markdown]
# ### 5.2 Error Analysis
# 
# *Can you describe what cases the model is getting wrong in the witheld test-set?*
# 
# To do this, you will need to create a new `val_train_loop_incorrect` which returns incorrect sequences **and** you will need to decode these sequences back into words. You have already created a map that can convert encoded sequences back to regular English (`reverse_vocab`).

# %%
def val_train_loop_incorrect(model, iterator):
  """
  Implement this however you like! It should look very similar to val_loop.
  Pass the test_iterator through this function to look at errors in the test set.
  """

  model.eval()
  
  incorrect = []

  for x, y in tqdm(iterator):
    y_pred = model(x)
    y_pred = y_pred.round()
    
    for i in range(len(y_pred)):
      if y_pred[i] != y[i]:
          indices = x[i].tolist()
          indices = [i for i in indices if i != PADDING_VALUE]
          words = " ".join([reverse_vocab.get(i, "UNK") for i in indices])
          
          incorrect.append(words)

  return incorrect

# %%
val_train_loop_incorrect(model, test_iterator)

# %% [markdown]
# Now that we have our incorrect sequences:   
# **Question:** *Can you describe what cases the model is getting wrong in the witheld test-set?*
# 
# **Answer:** The model seems to be replacing important entities/terms with 'UNK', due to low word frequency, which causes it to be unable to maintain context in sentences and hence incorrect outputs. This could point at a possibility where the cutoff was too high and key words were excluded from the vocab.

# %% [markdown]
# ## 6. LSTM Model [Extra credit, 4 points]

# %% [markdown]
# ### 6.1 Define the RecurrentModel class
# Something that has been overlooked in this project (and a significant limitation of the bag-of-words approach) is the sequential structure of language: a word typically only has a clear meaning because of its relationship to the words before and after it in the sequence, and the feed-forward network of Part 2 cannot model this type of data. A solution to this, is the use of [recurrent neural networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/). These types of networks not only produce some output given some step from a sequence, but also update their internal state, hopefully "remembering" some information about the previous steps in the input sequence. Of course, they do have their own faults, but we'll cover this more thoroughly later in the semester.
# 
# Your task for the extra credit portion of this assignment, is to implement such a model below using a LSTM. Instead of averaging the embeddings as with the FFN in Part 2, you'll instead feed all of these embeddings to a LSTM layer, get its final output, and use this to make your prediction for the class of the headline.

# %%
class RecurrentModel(nn.Module):
  def __init__(self, vocab_size, embedding_dim, hidden_dim, \
                num_layers=1, bidirectional=True):
    """
    Instantiate layers for your model

    Your model architecture will be an optionally bidirectional LSTM, followed
    by a linear + sigmoid layer.

    You will need 4 nn.Modules:
      1. An embeddings layer (see nn.Embedding)
      2. A bidirectional LSTM (see nn.LSTM)
      3. A Linear layer (see nn.Linear)
      4. A sigmoid output (see nn.Sigmoid)

    HINT: In the forward step, the BATCH_SIZE is the first dimension.
    HINT: Think about what happens to the linear layer's hidden_dim size
          if bidirectional is True or False.
    """
    super().__init__()

    ### BEGIN YOUR CODE (~4 lines) ###
    self.embedding = nn.Embedding(vocab_size, embedding_dim)
    self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)
    self.linear = nn.Linear(2 * hidden_dim if bidirectional else hidden_dim, 1)
    self.sigmoid = nn.Sigmoid()
    ### END YOUR CODE ###

  def forward(self, x):
    """
    Complete the forward pass of the model.

    Use the last timestep of the output of the LSTM as input to the linear
    layer. This will only require some indexing into the correct return
    from the LSTM layer.

    Args:
      x: 2D LongTensor of shape (BATCH_SIZE, max len of all tokenized_word_tensor))
         This is the same output that comes out of the collate_fn function you completed-
    """
    ### BEGIN YOUR CODE (~4-5 lines) ###
    x = self.embedding(x)

    _, (hidden, _) = self.lstm(x)

    if self.lstm.bidirectional:
        last_state = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=-1)
    else:
        last_state = hidden[-1, :, :]
    x = self.linear(last_state)
    x = self.sigmoid(x)
    return x
    ### END YOUR CODE ###

# %% [markdown]
# Now that the `RecurrentModel` is defined, we will reinitialize our dataset iterators.

# %%
train_iterator = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, collate_fn=collate_fn)
val_iterator   = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler, collate_fn=collate_fn)
test_iterator  = DataLoader(test_dataset, batch_size=BATCH_SIZE, sampler=test_sampler, collate_fn=collate_fn)

# %% [markdown]
# ### 6.2 Initialize the LSTM classification model
# 
# Next we need to initialize our new LSTM model, as well as define it's optimizer and loss function as we did for the FFNN. Feel free to use the same optimizer you did above, or see how this model reacts to different optimizers/learning rates than the FFNN.  

# %%
lstm_model = RecurrentModel(vocab_size    = len(train_vocab.keys()),
                            embedding_dim = 300,
                            hidden_dim    = 300,
                            num_layers    = 1,
                            bidirectional = True).to(device)

# %%
lstm_criterion, lstm_optimizer = None, None

### BEGIN YOUR CODE ###
lstm_criterion = nn.BCELoss()
lstm_optimizer = Adam(model.parameters())
### END YOUR CODE ###

# %% [markdown]
# ### 6.3 Training and Evaluation
# 
# Because the only difference between this model and the FFN is the internal structure, we can use the same methods as above to evaluate and train it. You should be able to achieve a validation F-1 score of at least `0.80` if everything went correctly.
# 
# **Feel free to adjust the number of epochs to prevent overfitting or underfitting and to play with your model hyperparameters/optimizer & loss function.**

# %%
# ===========================================================================
# Pre-train to see what accuracy we can get with random parameters
# ===========================================================================

true, pred = val_loop(lstm_model, val_iterator)
print(f'Binary Macro F1: {binary_macro_f1(true, pred)}')
print(f'Accuracy: {accuracy(true, pred)}')

# %%
# ===========================================================================
# Train your LSTM model
# ===========================================================================

TOTAL_EPOCHS = 10
for epoch in range(TOTAL_EPOCHS):
    train_loss = train_loop(lstm_model, lstm_criterion, lstm_optimizer, train_iterator)
    true, pred = val_loop(lstm_model, val_iterator)
    print(f"EPOCH: {epoch}")
    print(f"TRAIN LOSS: {train_loss}")
    print(f"VAL F-1: {binary_macro_f1(true, pred)}")
    print(f"VAL ACC: {accuracy(true, pred)}")

# %%
# ===========================================================================
# Evaluate your model on the held-out test set
# ===========================================================================

true, pred = val_loop(lstm_model, test_iterator)
print(f"TEST F-1: {binary_macro_f1(true, pred)}")
print(f"TEST ACC: {accuracy(true, pred)}")

# %% [markdown]
# ## 7. Submit Your Homework
# This is the end of Project 1. Congratulations!  
# 
# Now, follow the steps below to submit your homework in [Gradescope](https://www.gradescope.com/courses/944807):
# 
# 1. Rename this ipynb file to `CS4650_p1_GTusername.ipynb`. Make sure all cells have been run. We recommend ensuring you have removed any extraneous cells & print statements, clearing all outputs, and using the Runtime --> Run all tool to make sure all output is update to date.
# 2. Click on the menu 'File' --> 'Download' --> 'Download .py'.
# 3. Click on the menu 'File' --> 'Download' --> 'Download .ipynb'.
# 4. Download the notebook as a .pdf document. Make sure the output from Parts 4 & 6.3 are captured so we can see how the loss, F1, & accuracy changes while training.
# 5. Upload all 3 files to Gradescope. Double check the files start with `CS4650_p1_*`, capitalization matters.


