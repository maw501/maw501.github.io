---
layout: post
title: The how of attention for sentiment analysis (with PyTorch code)
date: 2019-01-09
use_math: true
tags: ['attention', 'deep_learning']
image: "attention.png"
comments: true
---
We are going to take a look at the attention mechanism in relation to sentiment analysis focusing on the classic implementation from [Bahdanau 2015](https://arxiv.org/abs/1409.0473), *Neural Machine Translation by Jointly Learning to Align and Translate*.

In particular we will focus on breaking down the calculation into several easier to understand steps.

<!--more-->
<hr class="with-margin">

The aim of this post is not to explain in detail the 'what' or 'why' of attention but rather 'how' the calculation works. We will give a brief explanation of what attention tries to accomplish but if you want more details, see, for example, the excellent article [here](http://www.wildml.com/2016/01/attention-and-memory-in-deep-learning-and-nlp/) of which there are many more similar.

If you just want to see the code you can jump to it by clicking the content page heading below - the below explanation is quite a detailed walk-through. There is also a section now with a few reminders of some of the mathematics based on feedback.

<hr class="with-margin">
<div class="list-of-contents">
  <h4>Contents</h4>
  <ul></ul>
</div>

<hr class="with-margin">
<h4 class="header" id="intro">Quick recap: simple NLP model & what is attention?</h4>

<blockquote class="tip">
<strong>TLDR:</strong> attention is a layer within an NLP model which calculates how much 'attention' the model should pay to each word when deciding the sentiment of the sentence by learning a probability distribution over words in the sentence. This is $a$ in Fig. 0 below.
</blockquote>

Below is the structure of a simple NLP model:
<pre><code class="language-python">import packages
def dummy_NLP_model_with_attention(x):
    '''This is pseudo-code. the input x has shape: bs, max_len'''
    x_embedded = get_embedding(x)  # bs, max_len, emb_dim
    x_lstm = bi_directional_lstm(x_embedded)  # bs, max_len, 2*hidden_size
    att_x = attention(x_lstm)  # bs, 2*hidden_size  <--- WHERE WE WILL FOCUS
    out = relu(linear_layer(att_x))  # bs, 1
    return out
</code></pre>

A mini-batch of sentences $x$ come into the model and each word retrieves its embedding vector, this gives the tensor `x_embedded`. This tensor is then passed into a bi-directional LSTM that runs over each sentence forwards and backwards and maps the embedding dimensionality into a new dimensionality that we get to choose - this is often called the hidden size of the LSTM.

We then pass this `x_lstm` tensor to our attention block which works out how much attention we should pay to each word in the sentence (which is of length `max_len`). It does this by calculating a probability distribution over all the words in our sentences for each example - we then take the expectation of our original input with these learned distributions.

Notice that the output from this no longer has a dimension for the sentence length - this is the calculation we are going to explain in this blog post and is a direct consequence of the expectation calculation we mentioned in the previous paragraph. Also notice that for each example in our mini-batch we output a single number which is our sentiment prediction.

The below shows the basic steps for a single example of batch size 1 (excluding the last linear layer as I ran out of room) for the above NLP model. The probability distribution $a$ is learned as part of the `attention` function for each example in a mini-batch.

<p align="center">
    <img src="/assets/img/simple_nlp.jpg" alt="Image" width="600" height="800" />
</p>

<em class="figure">Fig. 0: a simple NLP model</em>

##### What if we excluded the attention layer?

If in the above model we decided to exclude the attention layer we would need to still find a way to reduce the `x_lstm` tensor over the sentence dimension as in sentiment analysis our goal is to summarise the whole sentence into a single output number expressing sentiment. This could be done by a pooling layer before it's passed into the linear layer for output which maps the hidden dimensions of the LSTM to a single number. For example:

<pre><code class="language-python">import packages
def dummy_NLP_model_no_attention(x):
    '''This is pseudo-code. the input x has shape: bs, max_len'''
    x_embedded = get_embedding(x)  # bs, max_len, emb_dim
    x_lstm = bi_directional_lstm(x_embedded)  # bs, max_len, 2*hidden_size
    pool_x = avg_pool(x_lstm, 1)  # bs, 2*hidden_size
    out = relu(linear_layer(pool_x))  # bs, 1
    return out
</code></pre>

By using average pooling we would be weighting each word in the sentence equally - attention will explicitly learn the weighting for each word via the probability distribution $a$ as previously discussed.

Note: what we are calling `att_x` can be thought of as a context vector. As we are doing sentiment analysis we have a single context vector for each mini-batch example but in the original paper (cited above) on machine translation they output a context vector for each of words in the decoder sentence.

* **Sentiment analysis**: sentences of length `max_len`, output a single number.  
  * The context vector looks over the whole input sentence.
* **Translation**: sentences of length `max_len`, output a fixed length vector translation.
  * For each output word we have a context vector that has looked over the whole input sentence - we thus have as many context vectors as output words.

<hr class="with-margin">
<h4 class="header" id="intro">Overview for sentiment analysis</h4>

##### Problem set-up

Classically attention was introduced in terms of translating from one language to another word by word. Typically we have a input vector $x$ of length $T$ and wanted to decode it into another vector $y$ of potentially different length. In our example we are going to be thinking about sentiment analysis so for a single sentence (or mini-batch of sentences, which is our input) we simply need to output a single number which indicates the sentiment of that sentence.

Note: the terms tensor and vector will be used interchangeably where appropriate.

##### Parameters

`bs` = batch size

`max_len` = the length of each sentence (padded if less than max_len)

`hidden_dim` = the dimensionality of the LSTM whose output we pass into the attention layer (note we pass in a tensor with shape 2 * `hidden_dim` as we generally use a bi-directional LSTM, not explained here. But explained [here](https://towardsdatascience.com/introduction-to-sequence-models-rnn-bidirectional-rnn-lstm-gru-73927ec9df15))

In this example we will have `bs` = 32, `max_len` = 70 and `hidden_dim` = 75.

<hr class="with-margin">
<h4 class="header" id="attention">Walking through attention with pictures</h4>

The input to our attention function is of shape: (`bs`, `max_len`, 2 * `hidden_dim`).

Note our input to the attention mechanism isn't the raw sentence embeddings (if it was it would be of shape: (`bs`, `max_len`, `emb_dim`) as we are assuming it's been through a bi-directional LSTM first which has encoded each word into tensors of dimension: 2 * `hidden_dim`.

##### Unpacking each word and calculating 'alignment'

<blockquote class="tip">
<strong>Step summary:</strong> linearly weight our hidden dimensions to collapse them to a single dimension.
</blockquote>

This is the point whereby typical explanations of attention get themselves in a lather by proclaiming something along the lines of that we are training a 'feedforward neural network which is jointly trained with the whole system'.

Whilst this is technically true in my view this masks the true understanding which is fairly natural.

Recall our goal is to weight each word in the sentence according to how important it is in determining the sentiment. Well, if this is the case it would be nice to have a tensor that is of shape (`bs`, `max_len`) in order to weight each word in the sentence. In other words, for each of the 32 sentences in our mini-batch we wish to obtain a vector which weights how much importance each of the 70 words contributes towards determining sentiment.

To do this we need to perform some operation that can get us to a tensor of the shape we want: (`bs`, `max_len`) - there are many ways to do this but we will focus on a way which uses the information we have to hand from our input tensor $x$. As Montell Jordan said, [this is how we do it](https://www.youtube.com/watch?v=0hiUuL5uTKc):

<p align="center">
    <img src="/assets/img/attention_reshape_weight.png" alt="Image" width="600" height="400" />
</p>

<em class="figure">Fig. 2: showing the main attention calculation</em>

We first reshape our 3d tensor by stacking all the encodings on top of each other as shown above. The red tensor is the encodings for all the first words in our sequence and of shape (32, 150). We do the same for all 70 words in our sequence to end up with a tensor of shape (70 * 32, 150).

Next we perform matrix multiplication with a vector of shape (150, 1) to end up with an output of shape (70 * 32, 1). We then reshape this back to a tensor of shape (32, 70) which is our output, let's call it $e_{ij}$ to match the original paper's notation.

It's important to realise that the reshaping was just a convenience thing to do and really all we did was take a linear combination of the 150 hidden dimensions using a weight vector $w$.

This tensor $e_{ij}$ is now the shape we want and can be thought of as giving the 'energy' of each word in determining sentiment.

##### Let's call it a neural network

<blockquote class="tip">
<strong>Step summary:</strong> apply a non-linearity then soft-max activation to get a probability distribution for each mini-batch example.
</blockquote>

Recall that a single layer neural network is just a function of the form: $\sigma \,(Xw + b)$ for some non-linear activation $\sigma$.

Well, we've just done the matrix multiplication bit so let's now apply a non-linear function to $e_{ij}$ and we can the declare it a neural network. The activation we apply is a $\text{tanh}$ function followed by a soft-max which forces each row of the resulting tensor to now sum to 1 (without changing its shape).
* Note: we applied the soft-max over the `max_len` dimension forces the model to favour a particular part of the sentence to focus on as soft-max will tend to favour one large activation. Please see the section at the very end for a refresher on soft-max if you are rusty.

Let's call our output $a$:

$$ a = \dfrac{\exp(e_{ij})}{\sum_{k=1}^{T} \exp(e_{ik})}$$

Note: $a$ is still have a tensor of shape (`bs`, `max_len`) but as we've passed it through a soft-max each row now sums to 1 and so can be thought of as a probability distribution telling us where to focus for each input sentence.

##### The output (take expectations)

<blockquote class="tip">
<strong>Step summary:</strong> calculate the expectation over each word in the input sentence.
</blockquote>

For each example in our original mini-batch of inputs, $x$, we now multiply it (element-wise) by $a$ and then take the sum over each sentence as shown below:

<p align="center">
    <img src="/assets/img/attention_output.png" alt="Image" width="600" height="400" />
</p>

<em class="figure">Fig. 3: the attention output</em>

The above is really just another way of viewing the same calculation we saw for `att_x` in Fig 0. except we are now doing it for a mini-batch and not a single example.

As we can think of each row of $a$ as representing a probability distribution then multiplying $x$ by this and summing can be thought of (is the same) as calculating expectation over all words in the sentence of a word being important in determining sentiment. By calculating the expectation over all words in the sentence we allow them all to contribute (even though it's likely only a few will) - this is called soft attention.

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
    eij = torch.exp(eij)  # bs, max_len        
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

        if bias: self.b = nn.Parameter(torch.zeros(1))

    def forward(self, x, mask=None):
        '''x is bs, max_len, 2*hidden_dim'''
        feature_dim = self.feature_dim

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim),
            self.weight  
        ).view(-1, self.step_dim)  

        if self.bias: eij = eij + self.b  

        eij = torch.tanh(eij)
        a = torch.exp(eij)     
        if mask is not None: a = a * mask
        a = a / torch.sum(a, 1, keepdim=True) + 1e-10

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)  
</code></pre>

<hr class="with-margin">
<h4 class="header" id="intro">A bit of mathematics to remember</h4>

##### Soft-max function

Recall that for a vector $\textbf{a}$ the soft-max function, $\sigma$ can be written as:

$$ \sigma (\textbf{a}) = \dfrac{\exp(a_i)}{\sum_{j=1}^{N} \exp(a_j)} $$

for $i = 1, ..., N$

In words: we take the exponential of every element in the vector of length $N$ then divide each word by the sum. This has two implications:

* The resulting vector now sums to 1 so can be thought of as a probability distribution.
* The fact we took the exponential of each element means small differences in the original vector are amplified. i.e. the soft-max function favours picking one element as the dominant one.

In attention we apply the soft-max over the words in our sentence, so we can think of it as emphasising a particular word to focus on.

##### Expectations

In the last steps of attention we multiply our original input $x$ by $a$ and then sum over each row (the sentence dimension). This is exactly calculating the expected value over each sentence as each row of $a$ is a probability distribution.
<hr class="with-margin">
