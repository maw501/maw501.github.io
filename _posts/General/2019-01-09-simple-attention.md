---
layout: post
title: The how of attention for sentiment analysis (with PyTorch code)
date: 2019-01-09
use_math: true
tags: ['attention', 'deep_learning']
image: "attention.png"
comments: true
---
We are going to take a look at the attention mechanism in relation to sentiment analysis focusing on the classic implementation from [Bahdanau 2015](https://arxiv.org/abs/1409.0473), Neural Machine Translation by Jointly Learning to Align and Translate.

In particular we will focus on breaking down the calculation into several easier to understand steps.

<!--more-->
<hr class="with-margin">

The aim of this post is not to explain 'what' or 'why' of attention but rather 'how' it works. If you would like either of the former please see, for example, this excellent article [here](http://www.wildml.com/2016/01/attention-and-memory-in-deep-learning-and-nlp/) of which there are many.

If you just want to see the code you can jump to it by clicking the content page heading below.

<hr class="with-margin">
<div class="list-of-contents">
  <h4>Contents</h4>
  <ul></ul>
</div>

<hr class="with-margin">
<h4 class="header" id="intro">Overview</h4>

##### Problem set-up

Classically attention was introduced in terms of translating from one language to another word by word. Typically we have a input vector $x$ of length $T$ and wanted to decode it into a another vector $y$ of potentially different length. In our example we are going to be thinking about sentiment analysis so for a single sentence (or mini-batch of sentences, which is our input) we simply need to output a single number which indicates the sentiment of that sentence.

##### Goal

We wish to direct our model to focus on some part of the sentence. We can think of this as trying to end up with a tensor of the same length as our sentence which will provide a weighting for each word - we can think of this as how much attention we pay to each word when calculating the sentiment.

Note: the terms tensor and vector will be used interchangeably where appropriate.

##### Parameters

$\text{bs}$ = batch size

$\text{max_len}$ = the length of each sentence (padded if less than max_len)

$\text{hidden_dim}$ = the dimensionality of the LSTM whose output we pass into the attention layer (note we pass in a tensor with shape $2 * \text{hidden_dim}$ as we generally use a bi-directional LSTM, not explained here. But explained [here](https://towardsdatascience.com/introduction-to-sequence-models-rnn-bidirectional-rnn-lstm-gru-73927ec9df15))

In this example we will have $\text{bs} = 32, \text{max_len}  = 70 $ and $\text{hidden_dim} = 75$.

<hr class="with-margin">
<h4 class="header" id="attention">Walking through attention with pictures</h4>

The input to our attention function is of shape: $(\text{bs}, \text{max_len}, 2 * \text{hidden_dim})$

Note our input to the attention mechanism isn't the raw sentence embeddings (if it was it would be of shape: $(\text{bs}, \text{max_len}, \text{emb_dim})$) as we are assuming it's been through a bi-directional LSTM first which has encoded each word into tensors of dimension $2 * \text{hidden_dim}$.

##### Viewing the input

This is what our input tensor to the attention function looks like:

<p align="center">
    <img src="/assets/img/attention_input.jpg" alt="Image" width="600" height="400" />
</p>

<em class="figure">Fig. 1: input tensor to attention block</em>

We have a batch size of 32, a sentence length of 70 and each word is encoded into 150 dimensions. I've coloured a few strips where the red strip represents the encodings of the first words of all sentences in our mini-batch, the green strip is the second word's encodings across all examples in the mini-batch etc...

##### Unpacking each word and calculating 'alignment'

<blockquote class="tip">
<strong>Step summary:</strong> linearly weight our hidden dimensions to collapse them to a single dimension.
</blockquote>

This is the point whereby typical explanations of attention get themselves in a lather by proclaiming something along the lines of that we are training a 'feedforward neural network which is jointly trained with the whole system'.

Whilst this is technically true in my view this masks the true understanding which is fairly natural.

Recall our goal is to weight each word in the sentence according to how important it is in determining the sentiment. Well, if this is the case it would be nice to have a tensor that is of shape $(\text{bs}, \text{max_len})$ in order to weight each word in the sentence. In other words, for each of the 32 sentences in our mini-batch we wish to obtain a vector which weights how much importance each of the 70 words contributes towards determining sentiment.

To do this we need to perform some opThis requires the above reading to have been doneeration that can get us to a tensor of the shape we want: $(\text{bs}, \text{max_len})$ - there are many ways to do this but we will focus on a way which uses the information we have to hand from our input tensor $x$. As Montell Jordan said, [this is how we do it](https://www.youtube.com/watch?v=0hiUuL5uTKc):

<p align="center">
    <img src="/assets/img/attention_reshape_weight.png" alt="Image" width="600" height="400" />
</p>

<em class="figure">Fig. 2: showing the main attention calculation</em>

We first reshape our 3d tensor by stacking all the encodings on top of each other as shown above. The red tensor is the encodings for all the first words in our sequence and of shape (32, 150). We do the same for all 70 words in our sequence to end up with a tensor of shape (70 * 32, 150).

Next we perform matrix multiplication with a vector of shape (150, 1) to end up with an output of shape (70 * 32, 1). We then reshape this back to a tensor of shape (32, 70) which is our output, let's call it $e_{ij}$ to match the orinal paper's notation.

It's important to realise that the reshaping was just a convenience thing to do and really all we did was take a linear combination of the 150 hidden dimensions using a weight vector $w$.

This tensor $e_{ij}$ is now the shape we want and can be thought of as giving the 'energy' of each word in determining sentiment.

##### Let's call it a neural network

<blockquote class="tip">
<strong>Step summary:</strong> apply a non-linearity then softmax activation to get a probability distribution for each mini-batch example.
</blockquote>

Recall that a single layer neural network is just a function of the form: $\sigma \,(Xw + b)$ for some non-linear activation $\sigma$.

Well, we've just done the matrix multiplication bit so let's now apply a non-linear function to $e_{ij}$ and we can the declare it a neural network. The activation we apply is a $\text{tanh}$ function followed by a softmax. Applying the softmax over the $\text{max_len}$ dimension forces the model to favour a particular part of the sentence to focus on as softmax will tend to favour one large activation. Let's call our output $a$:

$$ a = \dfrac{\exp(e_{ij})}{\sum_{k=1}^{T} \exp(e_{ik})}$$

Note: $a$ is still have a tensor of shape $(\text{bs}, \text{max_len})$ but as we've passed it through a softmax each row now sums to 1 and can be thought of as a probability distribution telling us where to focus for each input sentence.

##### The output (take expectations)

<blockquote class="tip">
<strong>Step summary:</strong> calculate the expectation over each word in the input sentence.
</blockquote>

For each example in our original mini-batch of inputs, $x$, we now multiply it (element-wise) by $a$ and then take the sum over each sentence as shown below

<p align="center">
    <img src="/assets/img/attention_output.png" alt="Image" width="600" height="400" />
</p>

<em class="figure">Fig. 3: the attention output</em>

Given we can think of each row of $a$ as representing a probability distribution then multiplying $x$ by this and summing can be thought of (is the same) as calculating expectation over all words in the sentence of a word being important in determining sentiment. By calculating the expectation over all words in the sentence we allow them all to contribute (even though it's likely only a few will) - this is called soft attention.

Enough explanations, let's see the code!

<hr class="with-margin">
<h4 class="header" id="code">Function for single forward step</h4>

It's perhaps easier to initially present attention as a function that performs a single attention forward step - the below is commented to match the above explanation.

The below code is also on my github with a simple example [here](https://github.com/maw501/blog-notebooks/blob/master/attention_example.ipynb).

<pre><code class="language-python">import torch
def attention(x, weight, hidden_dim, max_len):
    '''
    Calculates attention on a input tensor x
    Inputs:
        x: torch tensor of shape (bs, max_len, 2*hidden_dim)
        weight: torch tensor of shape (2*hidden_dim, 1)
        hidden_dim: 2*hidden_dim
        max_len: max_len
    Returns:
        att_x: torch tensor of shape (bs, 2*hidden_dim)
    '''
    # matrix multiplication and reshaping:
    eij = torch.mm(
        x.contiguous().view(-1, 2*hidden_dim), # bs*maxlen, 2*hidden_dim
        weight  # 2*hidden_dim, 1
    ).view(-1, max_len)  # bs, max_len

    # tanh non-linearity then softmax:
    eij = torch.tanh(eij)
    a = torch.exp(eij)  # bs, max_len        
    a = a / torch.sum(a, 1, keepdim=True) + 1e-10 # bs, max_len

    # Calculate expectation by summing across max_len dim to
    # get single number per hidden dim:
    weighted_input = x * torch.unsqueeze(a, -1)  # bs, max_len, 2*hidden_dim
    att_x = torch.sum(weighted_input, 1)  # bs, 2*hidden_dim
    return att_x
</code></pre>

<hr class="with-margin">
<h4 class="header" id="code_cls">Attention PyTorch class</h4>

Below is the implementation as a PyTorch class.

The below code is also on my github with a simple example [here](https://github.com/maw501/blog-notebooks/blob/master/attention_example.ipynb).

<pre><code class="language-python">import torch
import torch.nn as nn
class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, \**kwargs):
        super(Attention, self).__init__(\**kwargs)

        self.supports_masking = True
        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0

        weight = torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)

        if bias: self.b = nn.Parameter(torch.zeros(step_dim))

    def forward(self, x, mask=None):
        '''x is bs, max_len, 2*hidden_dim'''
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim),
            self.weight  
        ).view(-1, step_dim)  

        if self.bias: eij = eij + self.b  

        eij = torch.tanh(eij)
        a = torch.exp(eij)     
        if mask is not None: a = a * mask
        a = a / torch.sum(a, 1, keepdim=True) + 1e-10

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)  
</code></pre>
