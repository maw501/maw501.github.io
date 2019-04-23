---
layout: post
title: ColumbiaX - ML - week 9
date: 2018-11-14
use_math: true
tags: ['collaborative_filtering', 'topic_modelling', 'LDA']
image: "collab_filt.jpeg"
comments: true
---
In week 9 we take a look at collaborative filtering, topic modelling and Latent Dirichlet Allocation (LDA) which delves into some of the technical aspects around matrix factorization.

<!--more-->
<hr class="with-margin">
This page is a summary of my notes for the above course, link [here](https://www.edx.org/course/machine-learning-columbiax-csmm-102x-4).

This is my first attempt at publishing course notes and I have no intention to make it comprehensive but rather to highlight the bits I feel are important and maybe explain some of the things I found a little trickier (or weren't explained to my taste). Understanding is deeply personal though if you spot any errors or have any questions please feel free to drop me an email.

<hr class="with-margin">
<div class="list-of-contents">
  <h4>Contents</h4>
  <ul></ul>
</div>

<hr class="with-margin">
<h4 class="header" id="intro">Week 9 (lectures 17 and 18): overview</h4>

Below is a brief summary of this week's topics:

* **Collaborative Filtering (CF):**
If our goal is to make, say, a film recommendation to users then there are several ways to go about this problem. CF is a method that uses the previous user input or behaviour to make a recommendation, it does not form *a priori* beliefs about users. For example we may define a measure of similarity between users and recommend films on the basis of how much our overlapping ratings agree.
  * In order to solve this problem we need to use a low-rank matrix factorization of our data given most users will not have rated most films and so we will have many (> 95%) missing entries in our data. The hope is that a low-rank matrix will help us fill in missing data by capturing the correlations between different users and films. We will do this using a probabilistic model.
* **Topic modelling and LDA:** This is the business of modelling the topics in a set of documents. A probabilistic topic model can learn distributions on words ('topics') across documents, learn a distribution of topics for each document and assign every word in a document to a topic. It does this with no regard to the order or structure of the text (this is called a 'bag of words') model and turns out to be sufficient for many tasks, such as sentiment analysis.
  * The standard topic model is LDA which is a generative statistical model that ([wikipedia](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation)): *"allows sets of observations to be explained by unobserved groups that explain why some parts of the data are similar. For example, if observations are words collected into documents, it posits that each document is a mixture of a small number of topics and that each word's presence is attributable to one of the document's topics"*.
* **Non-negative matrix factorization (NMF or NNMF):** NMF is a general technique where we factorize a matrix into the product of (usually) two other matrices such that all the matrices have no negative elements. This is closely related to what we are doing when we perform LDA and so we will present two methods for NMF that achieve a similar goal. NMF is generally not exactly solvable and so we find an approximate numerical solution instead.

<hr class="with-margin">
<h4 class="header" id="cf">Collaborative Filtering (CF)</h4>

<blockquote class="tip">
<strong>TLDR:</strong> CF methods are akin to <i>location-based</i> approaches whereby we embed users and objects into points in space, see Figure 1 below.
</blockquote>

<p align="center">
    <img src="/assets/img/collab_location.png" alt="Image" width="600" height="400" />
</p>

<em class="figure">Fig. 1: Koren, Y., Robert B., and Volinsky, C.. “Matrix factorization techniques for recommender systems.” Computer 42.8 (2009): 30-37.</em>

The way we learn such a representation (which may not have intuitive axis like above) is by using a matrix factorization of our data. We will now give a brief overview of this process saving any details for the end of the post (mathematical details section).
<hr class="with-margin">

##### Set-up

We start by assuming we have a matrix which we'll call $M$ of dimensions $N_1$ by $N_2$ where $N_1$ is the number of users and $N_2$ is the number of objects (in this example, films). Thus the entry $M_{ij}$ corresponds to the rating from the $i$th user for the $j$th film. Most of this matrix is empty and our goal is to fill it in in order to make recommendations to users for films they haven't yet seen.

<p align="center">
    <img src="/assets/img/matrix_fact.png" alt="Image" width="500" height="300" />
</p>

<em class="figure">Fig. 2: Matrix factorization, [source](https://www.datacamp.com/community/tutorials/matrix-factorization-names)</em>

Notice how the two matrices we will approximate $M$ by are of much lower rank. For instance, take the green 'Users' matrix. This still has $N_1$ rows but now only $d$ columns where $d$ is much less than the number of films, $N_2$. In this way what we are saying is the there was really a lot of correlation between the original $N_2$ films and we are going to capture that information by $d$ dimensions. i.e. you might think of the $d$ dimensions as now representing genres of films and so the Users matrix is now the preference of a user for a particular genre. Similarly for the Movies (Films!) matrix.

To make explicit:

* The reasons we wish to learn a low-rank factorization are that we have many missing values and we think many of the films will have correlated ratings so we don't need the full dimensionality of the matrix $M$.
* We wish to learn a user 'location' vector $u_i \in \mathbb{R}^d$ for user $i$ (called $f_i$ in Fig 2. above) and a film 'location' vector $v_j \in \mathbb{R}^d$  (called $f_j$ in Fig. 2 above) for user $j$.
<hr class="with-margin">

##### Probabilistic Matrix Factorization

The way we are going to solve the problem of learning the low rank factorization is by assuming a generative model for our data which although incorrect turns out to work well and is easy to implement.

First, some notation:

Call $\Omega$ the $(i, j)$ pairs of our data that are observed (not missing) and so $(i, j) \in \Omega$ if user $i$ rated film $j$. Further, let's call $\Omega_{u_i}$ the set of films rated by user $i$ and similarly call $\Omega_{v_i}$ the set of users who rated film $j$.

The generative model we assume (for $N_1$ users and $N_2$ films) is that we generate the user data as:

$$ u_i \sim N(0, \lambda^{-1}\,I), \, i = 1, ..., N_1$$

and similarly we generate the film data as:

$$ v_j \sim N(0, \lambda^{-1}\,I), \, j = 1, ..., N_2$$

This leads to a data distribution for the matrix $M_{ij}$ of:

$$ M_{ij} \sim N(u_i^T v_j, \sigma^2), \, \text{for each} \, (i, j) \in \Omega $$

To explain this a little further using the users as an example the above means we are assuming the distribution of ratings that a user will give to the films follows a Gaussian distribution. This is wrong (for example ratings are actually discrete) but it will turn out to still work well.

<blockquote class="tip">
<strong>Goal:</strong> we wish to solve the above generative model for $N_1$ user vectors, $u_i \in \mathbb{R}^d$ and $N_2$ film vectors, $v_j \in \mathbb{R}^d$. We do this by maximizing the joint data likelihood, $p(M_o, U, V)$.
</blockquote>

Here $M_o$ corresponds to the observed part of the data and $U$ and $V$ are the matrices holding all the $u_i$ and $v_j$.

This means we will be able to predict how user $i$ would now rate any film by taking the dot product of the learned user location vector $u_i$ with any film location vector $v_j$, as per $u_i^T v_j$.

##### Solving the matrix factorization

We are skipping a bit from the lecture here as the details are more worthy of an appendix (and we will give more later on) but we now we give a brief summary:

* We could use the EM algorithm for solving for all the $u$'s and $v$'s but it turns out we can find exact expressions for each $u_i$ and $v_j$ by using MAP to maximize the joint log likelihood.
* It turns out we can differentiate the joint log-likelihood to find the maximum i.e. the exact expressions for $u_i$ and $v_j$ are known though they depend on each other, so we have to use a coordinate ascent algorithm to solve (like for K-means and GMM).

Sketch overview of algorithm:

<hr class="with-margin">
* **Input:** an incomplete ratings matrix $M$ of rank $d$.
* **Output:** $N_1$ user vectors, $u_i \in \mathbb{R}^d$ and $N_2$ film vectors, $v_j \in \mathbb{R}^d$
* Initialize each $v_j$.

* For each iteration:
  * for $i = 1, ..., N_1$
    * update user location, $u_i$
  * for $j = 1, ..., N_2$
    * update film location, $v_j$

* Predict user $i$ rates film $j$ as $u_i^T v_j$ rounded to closest rating.

This solves the matrix factorization by maximizing the MAP objective function (joint log-likelihood). You can read more [here](https://lazyprogrammer.me/tutorial-on-collaborative-filtering-and-matrix-factorization-in-python/), [here](http://www.albertauyeung.com/post/python-matrix-factorization/) and [here](https://towardsdatascience.com/various-implementations-of-collaborative-filtering-100385c6dfe0).

<hr class="with-margin">
<h4 class="header" id="topic">Topic Modelling</h4>

Topic modelling is an intuitive concept which is usually handled via what we call a probabilistic topic model. This model can:

* Learn distributions on words (topics) shared by documents
* Learn a distribution of topics for each documents
* Assign every word in a document to a topic

As we are in the realm of unsupervised learning we don't know any of the above in advance and so we must somehow learn it from the data. To do this we will need to define a model as well as an algorithm to learn our objective. The standard model is one called Latent Dirichlet Allocation (LDA).

##### Latent Dirichlet Allocation (LDA)

LDA is two main things:

* A collection of distributions on words (topics).
  * i.e. the topic of sport might share the same overall vocabulary as the topic of history (the English language) but the distribution of words used in each topic is very different. So each topic captures a theme with a distribution on a coherent set of words.
* A distribution on topics for each document.

<hr class="with-margin">

LDA is also a generative model and works as follows (note: when you see the word Dirichlet below just think "this will return me a vector summing to 1 of the length I request" - we will explain more about the Dirichlet distribution later):

* Generate each topic which is a distribution on words according to a Dirichlet distribution. We will have $K$ topics each containing a word distribution. Each topic has a (vector/distribution) $\beta_k$ such that:

$$ \beta_k \sim Dirichlet(\gamma), \,\,\,\,\, k = 1, ..., K $$

* For each document, generate a distribution on topics according to a Dirichlet distribution. We will have $D$ documents. So each document has a (vector/distribution) $\theta_d$ such that:

$$ \theta_d \sim Dirichlet(\alpha), \,\,\,\,\, d = 1, ..., D $$

* For the $n$th word in the $d$ document:
  * Allocate the word to one of the $K$ topics (i.e. choose a topic).
    * We say $c_{dn} \sim Discrete(\theta_d) $
    * That is, $c_{dn}$ is a topic indicator from a discrete distribution using the distribution on topics for document $d$. So $c_{dn}$ will pick out one of the $K$ topics available to it where the probability of picking a particular topic is encoded in the distribution vector $\theta_d$.

  * Now generate the word from the selected topic.
    * That is, $x_{dn} \sim Discrete(\beta_{c_{dn}})$
    * Given the indicator $c_{dn}$ of which topic the word is coming from, choose a word according to a discrete choice from the vector $\beta_{c_{dn}}$.

<hr class="with-margin">

For example, say we generate word distributions for the 3 topics of sport, politics and history (LDA will learn how to do this!). Then for each document we have a probability distribution of that document's topic which might for a single document be something like [0.3, 0.4, 0.3] corresponding to the above 3 topics. Now to actually generate a word we assign the word to a topic based on some discrete distribution, let's say we end up with sport. We know from the sport topic generate a word from the sport word distribution. So we might end up with 'goal' as our generated word.


##### So how do we actually perform LDA?

In short, [Bayesian inference](https://en.wikipedia.org/wiki/Bayesian_inference). This is not something we covered in lectures except in a very cursory way to point out the following:

<blockquote class="tip">
<strong>Goal:</strong> we wish to find for a particular document the probability of a particular word being equal to some value.
</blockquote>

We can state this as:

$$ p(x_{dn} = i \mid \boldsymbol{\beta}, \theta_d) $$

In other words, given the $K$ topic distributions $\boldsymbol{\beta}$ (which are distributions on words) and the distribution of topics for this document $d$ what is the probability a word is equal to some value?

It turns out we can rewrite this term into the product of two matrices with non-negative entries as follows:

$$p(x_{dn} = i \mid \boldsymbol{\beta}, \theta_d) = \sum_{k=1}^K p(x_{dn} = i, c_{dn} = k \mid \boldsymbol{\beta}, \theta_d) $$

where we have added in the 'cluster' assignment (topic indicator) $c_{dn}$ and then just integrate it out by summing over all possible values of it. This gives:

$$p(x_{dn} = i \mid \boldsymbol{\beta}, \theta_d) = \sum_{k=1}^K \underbrace{p(x_{dn} = i \mid \boldsymbol{\beta}, c_{dn} = k)}_\text{$ = \beta_{ki}$} \, \underbrace{p(c_{dn} = k \mid \theta_d)}_\text{$\theta_{dk}$} $$

And if we let $B = [\beta_1, ..., \beta_K]$ and $\Theta = [\theta_1, ..., \theta_D]$ then we can say:

$$ p(x_{dn} = i \mid \boldsymbol{\beta}, \theta) = {(B\Theta)}_{id}$$

In other words the matrix multiplication ${(B\Theta)}$ gives us all the probabilities and we just read off the $(i, d)$th entry to get the probability that a given word is equal to some other word $i$ in document $d$.

Note: $B$ is of dimension $V$ by $K$ where $V$ is the size of our vocabulary and $\Theta$ is of dimension $K$ by $D$.

Solving this problem for the various distributions (the set of topics, their associated word probabilities, the topic of each word, and the particular topic mixture of each document) is what Bayesian inference does and we are not going to cover this now. I will come back to this in a later post as it uses techniques such as Gibbs sampling which are more advanced techniques.

<hr class="with-margin">
<h4 class="header" id="mathdetails">Some comments on the mathematics this week</h4>

##### A comment on matrix factorization, ridge regression and least squares

In the sketch algorithm we gave above for solving the probabilistic matrix factorization we can actually find an analytic expression for the updates $u_i$ and $v_j$ based on the generative model:

User location:

$$ u_i \sim N(0, \lambda^{-1}\,I), \, i = 1, ..., N_1$$

Film locations:

$$ v_j \sim N(0, \lambda^{-1}\,I), \, j = 1, ..., N_2$$

Recall that to find the maximum of the joint log-likelihood $L$, we differentiate $L$ with respect to both $u_i$ and $v_j$ and set the resulting equations to 0 and solve for $u_i$ and $v_j$. This gives the following (for the MAP) solution:

$$u_i = \bigl(\lambda \sigma^2 I + \sum_{j \in \Omega_{u_i}}\, v_j v_j^T \bigr)^{-1} \bigl( \sum_{j \in \Omega_{u_i}} M_{ij} v_j\bigr)$$

and

$$v_j = \bigl(\lambda \sigma^2 I + \sum_{i \in \Omega_{v_j}}\, u_i u_i^T \bigr)^{-1}  \bigl( \sum_{i \in \Omega_{v_j}} M_{ij} u_i\bigr) $$

Now, recall that the MAP solution for the optimal weight vector in ridge regression is:

$$w_{RR} = (\lambda I + X^TX)^{-1}X^Ty $$

And that in the matrix factorization problem we are trying to minimise (for a single entry) the sum of squares error:

$$ \dfrac{1}{\sigma_2}(M_{ij} - u_i^T v_j)^2$$

with a penalty $\lambda \|\|v_j\|\|^2$. Comparing the $w_{RR}$ solution to the equation for $u_i$ we can see that from the perspective of $v_j$ this is essentially the same as solving a ridge regression problem.

Given we solve $u_i$ for all $N_1$ times (one for each $i$) and $v_j$ is solved $N_2$ times we can think of this as solving $N_1 + N_2$ ridge regression problems.

Finally, note that if we removed the Gaussian priors on $u_i$ and $v_j$ then we'd need every user to have rated at least $d$ objects and every object to be rated by at least $d$ users (as this is the dimension of the latent space we are trying to factorize $M$ into) in order to invert the matrices to solve for $u_i$ and $v_j$. This almost certainly wouldn't be the case and so here it's necessary to have a prior.

##### The Dirichlet Distribution

I recall reading about the Dirichlet distribution a year or two ago and being perplexed by the level of complex explanations that existed. For example, take a look at [this](https://www.quora.com/What-is-an-intuitive-explanation-of-the-Dirichlet-distribution) response when asked for an 'intuitive' explanation. It clicked for me (as with most things) when I was recently looking to solve an actual problem I had - this will form the basis of our explanation below.

Let's say we have predictions from $n$ models and we wish to take a weighted average of them in order to form a single ensemble prediction. Weighting them equally would give each model $1/n$ weight. But what if we wanted just to randomly generate weightings and take the weighted average according to some arbitrary weighting provided by a vector $w$? Well this is precisely what the Dirichlet distribution does: it gives us vectors of the length of our choosing (here, $n$) such that the sum is equal to 1. In this sense the returned vector is a discrete probability distribution.

The main parameter of the Dirichlet distribution is a vector $\gamma$ and this controls how evenly we distribute the weights between a single entry and spread evenly as in the $1/n$ case. High constant values of $\gamma$ give roughly equal weighting to all entries, low values put a lot of weight on a single weight entry.

This is why we say the Dirichlet distribution is defined as a continuous distribution on discrete probability vectors.

##### Non-negative matrix factorization (NMF or NNMF)

Above we have talked about both probabilistic matrix factorization and LDA which were solved via MAP/coordinate ascent and Bayesian inference respectively. Both methods are solving something of the form $X = WH$ for factorizing matrices $W$ and $H$. NMF is a general technique which can solve such problems for $X$ with non-negative entries and no missing data (but can have many zeros/be sparse) e.g. word frequencies. The learned factorizing matrices $W$ and $H$ will both also have non-negative entries.  

We are not going to cover all the details except to say that NMF minimizes one of the two following objective functions.

The squared error objective:

$$\|X - WH \|^2 = \sum_i \sum_j (X_{ij} - {(WH)}_{ij})^2 $$

The divergence objective:

$$D(X \| WH) = -\sum_i \sum_j [X_{ij} \ln{(WH)}_{ij} - {(WH)}_{ij}] $$

where both are constrained such that $W$ and $H$ must contain non-negative values.

NMF uses a fast algorithm for optimizing both the above objectives. You can read more on [wikipedia](https://en.wikipedia.org/wiki/Non-negative_matrix_factorization) if you wish.

Final comment: as a result of the non-negativity constraint NMF learns what is called a 'parts-based' representation where each column of what $W$ captures is usually interpretable.

<hr class="with-margin">
<h4 class="header" id="unclear">Things I'm unclear on (or outstanding questions)</h4>

TBC

<hr class="with-margin">
<h4 class="header" id="textbooks">What did the textbooks say?</h4>


To be updated.
