---
layout: post
title: VAEs - questions and answers when first encountering
date: 2019-03-18
use_math: true
tags: ['VAEs', 'bayesian', 'generative_modelling']
image: "vae.jpg"
comments: true
---
There are many excellent explanations that cover the topic of VAEs, however, many questions still arose for me whilst trying to get a better understanding of them. In this article I am going to list a lot of the questions that cropped up as sources of confusion as well as providing my answers.

<!--more-->
<hr class="with-margin">
As an interesting pedagogical musing, it's simple to now point to most of the sources (e.g. blogs, papers etc...) I originally read and can now just say "well, the answer was addressed here so you just needed to have read it more carefully". This is a key point, when we are first encountering something we are not able to spot patterns, links or appreciate the gravity of certain utterances. It's hard to resolve points readily as our working memory is already overloaded - usually it's only on the $n$th reading ($ n > 3$ for me) that things start coming together. I hope by laying out some of my confusions it will help others.

The goal in this blog is not to regurgitate the many explanations listed below but rather to ask specific questions about VAEs and try to answer them clearly. If you have no understanding of VAEs you might find the assumed knowledge a little high though hopefully will still benefit from the post.

The question and answers will be sorted loosely into 4 main areas:

* **Conceptual**: bigger picture stuff to just grasp an idea.
* **Theoretical**: more technical discussion about the details.
* **Mathematical**: includes derivations and explanations of key steps.
* **Practical**: concerned with actually training a VAE.

Please also note any errors are entirely my own and reflect my still developing understanding. If you have any additional questions please leave a message in the comments section and I'll do my best to answer or incorporate into the main article.

<hr class="with-margin">
<div class="list-of-contents">
  <h4>Contents</h4>
  <ul></ul>
</div>

<hr class="with-margin">
<h4 class="header" id="ref">References</h4>

We will start with the main references I used when reading about VAEs. Most of the questions I had when starting to understand VAEs were based on having read the below and trying to synthesize my knowledge:

* [Tutorial - What is a variational autoencoder?](https://jaan.io/what-is-variational-autoencoder-vae-tutorial/)
  * An excellent article by a student of David Blei which links the NN perspective of VAEs to that probabilistic interpretation (via graphical models). This was my first reading on the subject and is an accessible introduction.
* [Variational Autoencoders](https://www.jeremyjordan.me/variational-autoencoders/)
  * Another excellent article with some great images and explanations of latent variable models.
* [Tutorial on Variational Autoencoders](https://arxiv.org/abs/1606.05908) by Carl Doersch
  * A widely cited reference that "introduces the intuitions behind VAEs, explains the mathematics behind them, and describes some empirical behavior".
* [Notes on Variational Autoencoders](http://www.1-4-5.net/~dmm/ml/vae.pdf) by David Meyer
  * This article is dated before the above tutorial and contains some of the exact same wording - I'm unclear which came first but there are some nice pictures in the article.
* Lecture 13 from the 2017 Stanford course CS231n on generative modelling, slides [here](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture13.pdf) and video [here](https://www.youtube.com/watch?v=5WoItGTWV54).
  * A great overview with some really well put together slides, like the rest of the course.
* [Variational Autoencoders](http://bjlkeng.github.io/posts/variational-autoencoders/) by Brian Keng
  * A long and detailed blog post with two follow ups [here](http://bjlkeng.github.io/posts/a-variational-autoencoder-on-the-svnh-dataset/) and [here](http://bjlkeng.github.io/posts/semi-supervised-learning-with-variational-autoencoders/) which is extremely well written and like all of Brian's posts form an excellent base for digging deeper into a topic.
* [Density Estimation: Variational Autoencoders](http://ruishu.io/2018/03/14/vae/) by Rui Shu
  * A very well written blog with some excellent tips on the issues faced when training VAEs.
* [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) by Diederik P Kingma, Max Welling
  * The original paper introducing VAEs
* [Semi-Supervised Learning with Deep Generative Models](https://arxiv.org/abs/1406.5298) by Diederik P. Kingma, Danilo J. Rezende, Shakir Mohamed, Max Welling
  * A paper introducing VAEs in a semi-supervised learning fashion.

<hr class="with-margin">
<h4 class="header" id="notation">Notation summary</h4>

Statistics is notorious for notational inconsistency and given I'm collating references I'm conscious that I may be guilty of this myself. In order to aid understanding, here is a summary of most of the notation used in the post:

* $P(X)$: the true probability distribution of the data, we wish to maximize this.
* $P(z)$: the prior distribution of the latent variables.
* $Q(z \| X)$: the probability distribution over latent variables given data $X$. This will be learned by the encoder which has parameters $\phi$ and so is sometimes written as $Q_{\phi}(z∣X)$.
* $P(X \| z)$: the probability distribution over the data given latent variables $z$. This will be learned by the decoder which has parameters $\theta$ and so is sometimes written as $P_{\theta}(X∣z)$ or $P(X∣z; \theta)$.
* $\tilde{X}$: the reconstructed data from the decoder.

It's often (sloppily) said that $Q$ is the encoder and $P$ is the decoder - really it is meant that the encoder and decoder output parameters to the distributions $Q$ and $P$. We will clarify this further later.

<hr class="with-margin">
<h4 class="header" id="reminders">A few big picture reminders</h4>

It's useful to keep in mind that what we are trying to do with any generative modelling technique is to learn a parameterised probability distribution $P(X)$ for the input data $X$ such that we can sample from $P(X)$ once we've learned the model (we say model because we define a model to learn the distribution of interest). In particular we would like to be able to maximize the probability of $X$ under a generative process that assumes some latent variables, $z$:

$$P(X) = \int P(X|z; \theta) \, P(z) \, dz$$

There are two hard things about the above:

1. Deciding how to define $P(z)$
2. Computing the integral over all $z$

We will however start with a quick explanation of the two above points.

##### 1. Deciding how to define $P(z)$

VAEs make the (unusual) assumption that samples of $z$ can be drawn from a simple (in this case, normal) distribution. They are able to do this due to the fact that we can map any set of normally distributed variables to an arbitrarily complex distribution if we use a sufficiently complicated function.

For VAEs we choose the 'sufficiently complicated function' to be a neural network and so can choose $P(z)$ to be a multi-dimensional isotropic Gaussian (isotropic just meaning the covariance matrix is the identity i.e. all covariance terms are 0, $\Sigma = \sigma^2 I$).

We can therefore say $P(z)=\mathcal{N}(z \| 0, I)$ is the prior.

See 'What distributional assumptions are we making about the data?' for a further explanation.

##### 2. Computing the integral over all $z$

Conceptually we could try to compute $P(X)$ by sampling a large number of $z$ values $\\{z_{1}, \dots, z_{n}\\}$ from the prior and then calculating:

$$P(X) \approx \frac{1}{n} \sum_{i} P\left(X | z_{i}\right)$$

However we cannot compute this in any reasonable time due to the [curse of dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality).

VAEs get around this by observing that for most values of $z$, $P(X \| z)$ will be essentially zero, and hence contribute almost nothing to the estimate of $P(X)$. You can think of this as follows: imagine that the data $X$ is simple cartoon faces and each dimension of the latent variables $z$ correspond to interpretable features like "shape of head", "placement of eyes", "amount of smile" etc... Then there will only be certain values of each $z$ value that when combined with other $z$ values make a plausible looking face from the data $X$.

Quoting [Doersch](https://arxiv.org/abs/1606.05908):

*The key idea behind the variational autoencoder is to attempt to sample values of $z$ that are likely to have produced $X$, and compute $P(X)$ just from those. This means that we need a new function $Q(z \|X)$ which can take a value of $X$ and give us a distribution over $z$ values that are likely to produce $X$. Hopefully the space of $z$ values that are likely under $Q$ will be much smaller than the space of all $z$’s that are likely under the prior $P(z)$.*

##### A final point on the ELBO

As we are not actually going to use the prior $P(z)=\mathcal{N}(z \| 0, I)$ to sample $z$ from we need some way to relate the distribution of $z$ output from $Q$ to $P(X)$, which is the term we wish to maximize. This is where the oft cited ELBO derivation comes in and results in the final objective term we will optimise. The ELBO derivation is contained in [Doersch](https://arxiv.org/abs/1606.05908) and whilst it is reasonably straightforward we will still explain it below in the mathematics section.

<hr class="with-margin">
<h4 class="header" id="QandAconceptual">Q and A: conceptual</h4>

### Why is there so much notational overload?

Tell me about it.

You need to remember that the convention in VAEs is to call the encoder $Q$ and the decoder $P$ (remember this as EQDP).

However **$P$ is an overloaded term**, don't get confused...statisticians have a habit of calling any probability distribution they don't know $P$, and sometimes they use lower-case too. For even more notational abuse, just think of Bayes' theorem...where each $p$ actually means a potentially different distribution!

Recall a probability distribution is a mathematical function which (usually) has some parameters. Those parameters themselves can be modelled by complicated functions which can lead to notational confusion. For example, consider the following expression for the data likelihood:

$$ P(X∣z; \theta) = \mathcal{N}(X| \, f(z;\theta), \, \sigma^2 * I)$$

In the above $P$ is now being used in a more general sense to just mean a probability distribution and the function $f$ is actually the decoder neural network which in this case outputs (only) the means for the probability distribution $P$. I'm highlighting this point as this is the actual notation often used simultaneously in the same paper and it's important to get straight what is going on.

Technically the encoder and decoder output parameters for a distribution which we then sample from to get either $z\|X$ or $X\|z$. However, confusingly (more on this later), we generally don't sample from $ P(X∣z; \theta) $ to get the reconstructed $X$. Thus sometimes people refer to the decoder as actually outputting the data $X\|z$ and sometimes to outputting the parameters to $ P(X∣z; \theta) $ - keep this in mind for now but don't dwell on it. For both $z\|X$ or $X\|z$ we eventually will have distributions whose parameters are complicated functions themselves with their own parameters.

Note: the **;** semi-colon you see being used above is notation to distinguish between different types of inputs, usually input variables and parameters.

#### What is the difference between the encoder, decoder, inference network and generator network?

* **Encoder or inference network**: these are two terms for the same thing. The encoder takes as input the data $X$ and trains a neural network with parameters denoted by $\phi$, outputting parameters to $Q_{\phi}(z∣X)$ which is a probability density function. Thus if $Q$ is a Gaussian distribution, the encoder neural network will learn a vector of means and variances whose length depends on the dimensionality of the latent variables (we choose this).
* **Decoder or generative network**: these are two terms for the same thing. The decoder takes as input latent variables $z$ and trains a neural network with parameters denoted by $\theta$, outputting parameters to $P_{\theta}(X∣z)$ which is a probability density function. Thus if $P$ is a Gaussian distribution, the decoder neural network will learn a vector of means and variances whose length depends on the dimensionality of the original data.

#### Why does the VAE graphical model notation only show the generator?

Commonly we see the graphical model for a VAE specified as something like Fig 0 which describes the decoder/generative network only:

<p align="center">
    <img src="/assets/img/vae_graph_model.png" alt="Image" width="250" height="400" />
</p>

<em class="figure">Fig. 0: Plate notation for VAEs. The N denotes the number of times we sample with the parameters fixed.</em>

Where is the encoder?

In somewhat circular feeling reasoning this is because this is the really the graphical model we assume. This model structure doesn't tell us how to learn $z$, only that it is assumed such $z$ exist.

Two points on this:
* VAEs learn $z$ via the encoder/inference network.
  * Technically they actually learn $z\|X$, this will be discussed later.
* Loosely speaking the ELBO derivation shows (amongst other things) us that learning $z$ via an encoder network we are able to still maximize the term we want, $P(X)$.

#### Why does the term variational appear in the name?

Because we are performing variational inference, which means we are using an approximation to the posterior of interest rather than trying to calculate the exact posterior directly. Recall this posterior is the output of the encoder and is over the latent variables, i.e. $Q(z \| X) $.

I quite liked [this](https://www.quora.com/What-is-variational-inference) quote:

*In short, variational inference is akin to what happens at every presentation you've attended. Someone in the audience asks the presenter a very difficult answer which he/she can't answer. The presenter conveniently reframes the question in an easier manner and gives an exact answer to that reformulated question rather than answering the original difficult question.*

<hr class="with-margin">
<h4 class="header" id="QandAtheoretical">Q and A: theoretical</h4>

#### What distributional assumptions are we making about the data?

With all the talk of Gaussians it's tempting to think VAEs place restrictive distributions on the data we wish to model. This is not the case.

<blockquote class="tip">
<strong>Short answer:</strong> the data (independent of the model) is given by $P(X)$ and VAEs generally only place distributional assumptions on $P(X|z; \theta)$, and $P(z)$.
</blockquote>

The longer answer starts with the fact that our encoder will be outputting parameters to a distribution $Q(z\|X)$ which is being pulled towards the prior, $P(z)$, which is an isotropic Gaussian. The data distribution $P(X)$ can be arbitrarily complex and it is known (trust me on this or see subsection below) that we can map any set of normally distributed variables to an arbitrarily complex distribution if we use a sufficiently complicated function (see Fig 2). The 'sufficiently complicated function' we use is a neural network, which are known to be [universal function approximators](http://neuralnetworksanddeeplearning.com/chap4.html).

Recall the data likelihood can be expressed as:

$$P(X∣z; \theta) = \mathcal{N}(X| \, f(z;\theta), \, \sigma^2 * I)$$

And $f$ is the neural network that allows us to learn the 'sufficiently complicated function' which models the mean of the data likelihood. To illustrate the distributional point that $P(X\| \, z; \theta)$ can be Gaussian even when $P(X)$ clearly isn't, consider the simple chart in Fig 1 below. Here the data $P(X)$ clearly isn't multivariate normal as the marginal distributions are bimodal in both the the $x_1$ and $x_2$ dimensions. However, it is not difficult to see that given a cluster assignment the data within a cluster is multivariate normal.

Thus $P(X\|z)$ in this case is now Gaussian, where $z$ would denote a per observation cluster assignment.

<p align="center">
    <img src="/assets/img/simple_cluster.png" alt="Image" width="500" height="400" />
</p>
<em class="figure">Fig. 1: Illustration of data being Gaussian conditioned upon cluster assignment.</em>

##### Mapping independent normal RVs to any function (illustration)

To see that we can go from independent normal random variables (RVs) in 2d to any complicated function consider the example below, from [Tutorial on Variational Autoencoders](https://arxiv.org/abs/1606.05908) by Carl Doersch.

We start with data samples from a Gaussian in 2d (i.e. these could be the $z$) and these samples are then mapped through the function:

$$g(z) = \dfrac{z}{10} + \dfrac{z}{||z||}$$

to form a ring. This is how VAEs can model arbitrarily complex distributions for the data, the deterministic function $g$ is a NN learned from the data(!).

<p align="center">
    <img src="/assets/img/2d_normal_RVs.png" alt="Image" width="800" height="400" />
</p>
<em class="figure">Fig. 2: Creating data of an arbitrary distribution from normally distributed $z$.</em>

Here is some numpy code showing the above data generation:

<pre><code class="language-python">import numpy as np
X = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], size=(250))

def euc_norm(X):
    '''Computes euclidean norm of array along axis 1'''
    return np.expand_dims(np.sqrt(np.sum(X**2, 1)), axis=1)

def arb_func(X):
    '''Arbitrary transformation'''
    return X/10 + X/euc_norm(X)

out = arb_func(X)
</code></pre>

#### Why do we need the KL term in the objective function?

The intuition surrounding this explanation is very interesting and worth the effort to understand.

<blockquote class="tip">
<strong>Short answer:</strong> the KL loss forces the encoder to distribute the latent representations for each $x_i$ around the centre of the (high-dimensional) latent space. In other words it stops the encoder outputting a different $\mu$ and $\sigma$ for every observation $i$ which would result in a non-smooth latent state representation and would be troublesome to sample from.
</blockquote>

A key benefit from using a VAE is to learn a smooth latent state representation of the input data from which we can sample.

Recall that what we output from the encoder are parameters $\mu_{z\|X}$ and $\sigma_{z\|X}^2$ which have a dimensionality equal to what we specify as the latent dimension. As the encoder is learning $Q(z\|X)$ then the parameters we get back will depend on the $X$ we put in. It is possible then that the encoder could just learn to output different $\mu_{z\|X}$ and $\sigma_{z\|X}^2$ for every example in $X$. Such a latent space representation would be non-continuous and thus hard to sample from.

Having $z$ depend on $X$ is what we want - it allows us to avoid the curse of dimensionality by allowing us to sample only from the $z$ that matter and makes the posterior tractable.

Fig 3. shows 3 cases:

<p align="center">
    <img src="/assets/img/vae_latent.png" alt="Image" width="750" height="300" />
</p>
<em class="figure">Fig. 3: Latent space representations for MNIST, [image credit](https://www.jeremyjordan.me/variational-autoencoders/).</em>

* The left image is if we only used the reconstruction loss when training the VAE (as in a normal autoencoder). In this case the latent representations have gaps in them and are clustered apart.
* The middle image shows what happens if we just use KL loss...the encoder is now essentially outputting $\mu_{z\|X} = 0$ and $\sigma_{z\|X}^2 = 1$ regardless of what $X$ we feed into the model.  In other words, irrespective of what an observation looks like, we encode it the same; and so we've failed to describe the original data.
  * This is closely related to the phenomenon of 'posterior collapse' which is explained later.
* The right hand image is when we use both reconstruction and KL loss as in a well-trained VAE - the model learns latent states for observations with distributions close to the prior ($z \sim \mathcal{N}(0, I)$) but deviating when necessary to describe salient features of the input.


For further reading on this topic consult the fantastic posts [here](https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf) and [here](https://www.jeremyjordan.me/variational-autoencoders/) which I have used as reference.

#### What is mean-field and amortized inference?

This is perhaps the area of VAEs that is least clearly explained in my view.

Recall that in a VAE we are approximating the true (unknown) posterior of the latent variables with a distribution output from the encoder, i.e. $Q_{\phi}(z∣X)$.

Bluntly speaking, in order to learn an approximate posterior (to get the $z_i$ when we sample) we need to learn some parameters. Broadly speaking there are two ways this can go:

1. **mean-field variational inference:** we can have parameters for each observation $i$ such that as the data grows the number of parameters we need to estimate grows.
2. **amortized variational inference:** we share ('amortize') the parameters we learn across many data points.

VAEs use amortized inference as the encoder (which is a NN) has a fixed number of parameters, $\phi$, which do not change in number as we scale the data. It is important not to get confused here: we sample from the learned encoder distribution $Q_{\phi}(z∣X)$ enough times to obtain a $z_i$ for each data point $x_i$ but the number of parameters we learn in order to output the mean and variances for $Q$ is fixed.

In other words amortized variational inference allows us to use a parameterized function that maps from the observation space of the data to the parameters of the approximate posterior distribution. The encoder neural network accepts an observation as input, and outputs the mean and variance parameter for the latent variable associated with that observation. We are then able to optimize the parameters of this neural network instead of the individual parameters of each observation.

*Do we give up anything by using amortized variational inference?*

Yes, the model is now less expressive as in addition to making the approximate posterior Gaussian, we now are imposing an additional constraint by using a NN which is sharing parameters (rather than having a parameter for every data point). This cost is known as the [amortization gap](https://arxiv.org/pdf/1801.03558.pdf). For a NN with infinite capacity, this gap would go away but this is not the case in any practical implementations (as all networks have finite capacity).

More reading [here](https://www.quora.com/What-is-amortized-variational-inference), [here (section 6)](https://arxiv.org/pdf/1711.05597.pdf) and [here](http://bjlkeng.github.io/posts/variational-bayes-and-the-mean-field-approximation/).

#### Why do VAEs assume $p(z)$ is an isotropic Gaussian - isn't this restrictive?

<blockquote class="tip">
<strong>Short answer:</strong> No, due to the fact that we can map any set of independent normally distributed variables to an arbitrarily complex distribution if we use a sufficiently complicated function.
</blockquote>

This is answered elsewhere in this post under the question: *What distributional assumptions are we making about the data?.*

The longer answer is beyond the scope of this blog post.

<hr class="with-margin">
<h4 class="header" id="QandAmath">Q and A: mathematical</h4>

#### How does the derivation of the ELBO help us?

The ELBO derivation is what links the term we wish to maximize but are unable to do so directly, $P(X)$, with an equivalent expression we can solve which is tractable.

Formally we will show that $\log P(X) \geq$ ELBO and so maximizing the ELBO (which is tractable) will also be maximizing $\log P(X).$

Before we start deriving the ELBO, recall the original statement for $P(X)$:

$$P(X) = \int P(X|z; \theta) \, P(z) \, dz = E_{z \sim P} P(X | z; \theta)$$

Also recall that we cannot sample from the prior $P(z)$ to compute the above and so will find some new distribution, $Q$ from which we will sample $z$ from - the justification of this is provided in section 2. In particular it is stated that we need a new function $Q(z \| X)$ which will take a value of $X$ and provide a distribution over z values that are likely to produce X. Using such a distribution we will be able to compute

$$E_{z \sim Q} P(X | z; \theta) $$

which is the expectation computed over $z$ which has distribution $Q$. In other words, we will be using $Q$ to approximate the true distribution of $z$ which we do not know.

How is $E_{z \sim Q} P(X \| z; \theta)$ related to $P(X)$? This is what the ELBO derivation shows. We will now walk through the derivation in [Doersch](https://arxiv.org/abs/1606.05908) but providing more details.

The start of the derivation seems a little bit like a leap of faith so just bear with me. We consider the KL divergence between the approximate distribution of $z$ we wish to find, $Q$ and the actual posterior of the latent variables $P(z \| X)$, which we do not know:

$$\mathcal{D}[Q(z) \| P(z | X)]=E_{z \sim Q}[\log Q(z)-\log P(z | X)]$$

the right-hand side is just the definition of the KL divergence (we don't explain this here). Also note that $Q$ at the moment doesn't depend on $X$ - we will address this later.

The next thing to note is that given we are using logs (maximizing the log of something is the same as maximizing the thing itself) we can write Bayes rule as:

$$\log P(z | X) = \log P(X | z) + \log P(z) - \log P(X) $$

where we've used the fact that $\log (ab) = \log(a) + \log(b)$ and $\log (a/b) = \log(a) - \log(b)$

Substituting this back in gives

$$\mathcal{D}[Q(z) \| P(z | X)]=E_{z \sim Q}[\log Q(z)- \log P(X | z) - \log P(z) ] + \log P(X)$$

where we have moved $P(X)$ outside the expectation as it doesn't depend on $z$. Using the linearity of expectations

$$E[A + B] = E[A] + E[B]$$

we have (after swapping sides for a few terms)

$$\log P(X) - \mathcal{D}[Q(z) \| P(z | X)] = E_{z \sim Q}[ \log P(X | z)] - \underbrace{E_{z \sim Q}[ \log Q(z)- \log P(z)]}_{\mathcal{D}[Q(z) \| P(z)]} $$

and can note the term on the right can be rewritten as a KL term. Note that $Q$ can be any distribution we choose (we just want to sample $z$ from it) and so it makes sense to make $Q$ actually depend on $X$. This leaves us with

$$\underbrace{\log P(X)}_\text{want to max}
 - \underbrace{\mathcal{D}[Q(z|X) \| P(z | X)]}_{\geq 0} =
 \overbrace{\underbrace{E_{z \sim Q}[ \log P(X | z)]}_\text{reconstruction loss}
 - \underbrace{\mathcal{D}[Q(z|X) \| P(z)]}_\text{KL loss}}^\text{ELBO}
 $$

Due to the fact the KL divergence is always non-negative we have shown that maximizing the ELBO is equivalent to maximizing $P(X)$ as ELBO $\leq \log P(X)$ (because we have to add a non-negative term to the ELBO to get $P(X)$).

So we can instead maximize the ELBO (which is tractable) in order to maximize $\log P(X)$.

<blockquote class="tip">
<strong>Upshot:</strong> We can maximize the ELBO (which is tractable) instead of directly trying to maximize $\log P(X)$.
</blockquote>

Of course, the above derivation doesn't immediately make obvious how we are going to end up with a VAE as an encoder-decoder model trained via stochastic gradient descent but it does provide the theoretical basis.

**Note:** ELBO stands for Evidence Lower BOund and refers to the fact the ELBO provides a lower bound for $ \log P(X)$, the log probability of the observed data.

#### What is the calculus of variations and how does it relate to VAEs?

Pending.

<hr class="with-margin">
<h4 class="header" id="QandApractical">Q and A: practical</h4>

#### What is the output of the decoder? And what if I have real-valued data, which loss function should I use?

This point confused me when training a VAE for a different dataset with real-valued output (i.e. numeric data).

Technically the decoder outputs parameters to $P_{\theta}(X∣z)$ (the data likelihood) and in order to generate new data $X$ we should sample from this distribution as shown in Fig 4.

<p align="center">
    <img src="/assets/img/vae_model_cs231n.png" alt="Image" width="750" height="275" />
</p>
<em class="figure">Fig. 4: VAE model, image from CS231n, lecture 13 2017, [here](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture13.pdf).</em>

However, if you've ever implemented a VAE you'll realise that no-one actually performs that sampling of $X$ from $P_{\theta}(X∣z)$ - this was a source of considerable confusion initially for me. Instead the decoder simply returns the approximate data $\tilde{X}$ as samples from the generator which is implicitly the same as returning the mean $f(z;\theta)$ from the data likelihood:

$$\tilde{X} = P(X∣z; \theta) = \mathcal{N}(X | \, f(z;\theta), \, \sigma^2 * I)$$

You can read more about this practice [here](http://ruishu.io/2018/03/14/vae/).

If you are inputting real-valued numeric data (i.e. simple structured data from a csv) you can use MSE as a loss function for the decoder. If you do this be sure to centre and scale the data appropriately and remove any sigmoid function from the end of the decoder.

#### What is posterior collapse and should I be scared of it?

Yes, probably.

This is an open area of research which I stumbled across when trying to train a VAE on a different dataset to what is provided in the standard tutorials. It's (apparently) well-known within the research community that VAEs are hard to train and suffer from a few practical problems. However, given most blog posts either don't provide practical implementations or use a simple MNIST dataset it's easy to not realise these issues exist.

<blockquote class="tip">
<strong>Short answer:</strong> Posterior (or component) collapse is when the KL term from the encoder falls to 0 whilst training the VAE - this implies that $Q(z|X)$ is equal to the prior $P(z)=\mathcal{N}(z | 0, I)$ regardless of what $X$ is.
</blockquote>

Why is it a bad thing that $Q(z\|X)$ equals the prior - isn't this what we are trying to achieve by minimizing the KL divergence between the two terms?

No. We wanted to learn a distribution $Q$ over $z$ values that depended on $X$ (hence are likely to have produced $X$) in order to save ourselves searching the whole $z$ space. If the output of the encoder is always the prior (i.e. $\mu_{z\|X} = 0$ and $ \Sigma_{z\|X} = 1$) regardless of what $X$ we feed in, how is the decoder going to decode these $z$ into reconstructed $X$ samples? It won't be able to and we will get nonsense out of the decoder.

The idea of minimizing the KL term $\mathcal{D}[Q(z\|X) \| P(z)]$ isn't to drive it to 0 but rather to use the prior as a regulariser on the structure of the latent space the encoder learns when it outputs parameters to $Q(z\|X)$. See the question above on: *Why do we need the KL term in the objective function?*

##### How can we solve posterior collapse?

I'm not going to provide a full answer to this, principally because it's an open area of research with no clear resolution. I will however point in the direction of resources I found useful when resolving the issue:

* [Variational Autoencoder and Extensions](http://videolectures.net/deeplearning2015_courville_autoencoder_extension/) by Aaron Courville
  * Skip to ~33 minutes to see the discussion on component collapse.
* [Lagging Inference Networks and Posterior Collapse in Variational Autoencoders](https://arxiv.org/abs/1901.05534) by Junxian He, Daniel Spokoyny, Graham Neubig, Taylor Berg-Kirkpatrick
  * A very recent paper discussing this topic though I didn't find their solution to work.
* [Preventing Posterior Collapse with delta-VAEs](https://deepmind.com/research/publications/preventing-posterior-collapse-delta-vaes/) by Deepmind

<hr class="with-margin">
