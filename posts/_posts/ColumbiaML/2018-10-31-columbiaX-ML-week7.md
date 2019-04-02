---
layout: post
title: ColumbiaX - ML - week 7
date: 2018-10-31
use_math: true
tags: ['boosting', 'kmeans']
image: "boosting.png"
comments: true
---
In week 7 we cover boosting, a powerful algorithm that has had a lot of success in data mining with more advanced implementations in the manner of gradient boosting (e.g. libraries such as XGBoost and LightGBM). We then went onto start the next major topic of unsupervised learning by looking at the K-means algorithm.

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
<h4 class="header" id="intro">Week 7 (lectures 13 and 14): overview</h4>

Both of the major topics covered this week (as compared to, say, SVMs) sit on the end of the spectrum of things you are likely to encounter in your day to day life as a machine learning engineer and luckily they are both pretty intuitive topics. Unfortunately there were some items introduced in the lecture that I thought dwelt on arcane details - I will hopefully be able to clarify some of this below.

* **Boosting**
  * Boosting is essentially that combining a (large) collection of 'weak' classifiers together can outperform a single stronger classifier. The loose intuition is that each classifier is essentially trying to learn from the previous model's mistakes in a simple way.
* **Unsupervised Learning**
  * This is essentially the paradigm whereby we no longer have labels for our data but rather wish to learn some structure within the data as defined by a model. This doesn't have a clear performance metric but can be as simple as the...
* **K-means algorithm**
  * This is a simple algorithm that tries to group our data into K groups depending on how similar the data are. For instance, think of a child being asked to put a group of toy cars and teddy bears into two groups only being told to put similar things together and nothing else - it's likely they would sort them into cars and bears. Implicitly what the child is doing is using some notion of similarity to group (or cluster) the data - we will use this concept a lot in unsupervised learning.

We will focus first on boosting and then on K-means.

<hr class="with-margin">
<h4 class="header" id="boost">Understanding boosting</h4>

<blockquote class="tip">
<strong>TLDR:</strong> [AdaBoost] sample data with some probability -> fit weak model -> re-weight data based on errors -> repeat
</blockquote>

Boosting (though it differs in its various guises) can intuitively be thought of as having a bunch of weak models that are able to learn from one another's mistakes. Loosely speaking we fit weak classifiers that are fast to train sequentially and come up with some way to re-weight our data so we put more focus on the examples we got wrong i.e. each round incorrect examples have a higher probability of being sampled and included in the learner whereas correct examples have a lower probability. So the future weak classifiers/learners focus more on the examples that previous weak learners misclassified.

##### AdaBoost details

Here is an overview of the algorithm without all the formula. We fit a weak classifier $T$ times (sometimes called rounds).

<hr class="with-margin">

* Given data $(x_1, y_1)$, ... , $(x_n, y_n)$, $x \in X$, $y \in \\{-1, +1\\}$ set the initial weights of our data as $w_1(i) = \dfrac{1}{n}$ and:
  * For $t = 1, ..., T$
    * Sample a boostrap dataset $B_t$ of size n according to $w_t$. i.e. pick each $(x_i, y_i)$ with probability $w_t(i)$, note $\dfrac{1}{n}$ is just our starting distribution.
    * Learn a classifier $f_t$ using data in $B_t$
    * Update our error at round $t$, which we call $\epsilon_t$ and calculate a parameter $\alpha_t$ which we use to re-weight the data (i.e. update $w_t(i)$ for every data point $i$).
    * Scale our weights using $\alpha_t$ and normalise to ensure they sum to 1
      * Note: $\alpha_t$ is a so called 'alpha-weight' which is some measure of performance and the better a single learner performs the more it will contribute to the final weighted learner.
  * Once we've done, to predict a new data point we take a majority vote of the class over the weighted classifiers $f_t$

<hr class="with-margin">

The bits we've neglected to mention are how we calculate $\epsilon_t$ and $\alpha_t$ in order to update our weights. This is not important for an intuitive understanding of the algorithm.

However for completeness here are the rest of the formula for each round $t$:
  * Set $\epsilon_t = \sum_{i=1}^{n}w_t(i)\_{y_i \neq f_t(x_i)}$
    * i.e. we sum the weights for the examples we misclassified.
  * Set $\alpha_t = \dfrac{1}{2}\ln\Bigl(\dfrac{1-\epsilon_t}{\epsilon_t}\Bigr)$
    * The reasons for this refer to The Training Error theorem discussed later.
  * Set $ \beta_{t+1}(i) = w_{t}(i) e^{-\alpha_t y_i f_t(x_i)}$ and $w_{t+1}(i) = \dfrac{\beta_{t+1(i)}}{\sum_{j} \beta_{t+1}(j)}$
    * The first part of this just updates our weights temporarily and the second part assigns them once we've scaled them to sum to one.

##### Relating boosting to high dimensional feature mappings
We simply predict the class for a new data point $x_0$ as a majority vote over all of our weak learners $f_t$, expressed as (this is definitely one of those things that is easier to say in words than read formulae for):

$$f_{boost}(x_0) = sign \Bigl( \sum_{t=1}^{T} \alpha_t f_t(x_0)  \Bigr) $$

We note that $\alpha_t$ is a $T$ length vector with the weights for each boosting round. Thinking of any data point $x$ now we could write $\phi(x) = [f_1(x), ..., f_T(x)]^T$ where each $f_t(x) \in \\{-1, +1\\}$.
* Then following on from our work with hyperplanes we can think of $\phi(x)$ as the high-dimensional feature map for $x$.
* So if we call $ \gamma = [\alpha_1, ..., \alpha_T]^T $ this contains the weights for our hyperplane.
* Then we can now write our classifier as $f_{boost}(x_0) = sign \Bigl(\phi(x_0)^T \gamma \Bigr)$.
* Thus boosting is learning the feature mapping and hyperplane simultaneously(!).

##### Bagging vs. boosting vs. AdaBoost vs. gradient boosting (e.g. XGBoost)...

It can be a minefield navigating the terminology in machine learning, in particular when all the names seem to converge in the same space in your head it is easy to get confused. In particular, the differences between AdaBoost and a gradient boosting algorithm such as XGBoost are worth dwelling on and will be subject to a separate blog post.

* **Bagging**
  * Create $B$ datasets of our data by sampling, and fit a classifier to each which we ensemble by averaging. This can be done in parallel.
* **Boosting**
  * Create $B$ datasets of our data by weighting each dataset based on some measure of the previous classifiers' performance and fit a classifier to each dataset. This is done sequentially.
* **AdaBoost**
  * A specific implementation of the boosting algorithm which up-weights observations that have been misclassified in previous rounds. It is usually used in the context of binary classification.
    * In other words, each round AdaBoost is changing the sample distribution from which we sample our data by modifying the weights attached to each data point (the weights are the probability of a data point being chosen).
* **Gradient boosting**
  * A popular example of gradient boosting is  [XGBoost](https://xgboost.readthedocs.io/en/latest/tutorials/model.html).
  * Unlike AdaBoost, each round gradient boosting doesn't modify the sample distribution but rather fits on the current residuals of the model (which is another way to give more importance to the points the model is getting wrong). The contribution of the weak learner to the overall ensemble isn't based on its performance but instead using the loss gradients in order to minimize the overall error of the ensembled learner. In the above XGBoost link see the section entitled 'The Structure Score'.
  * [This](https://www.quora.com/What-is-the-difference-between-gradient-boosting-and-adaboost) answer also has some nice discussion.

##### More reading on Boosting
AdaBoost [tutorial](http://mccormickml.com/2013/12/13/adaboost-tutorial/).

<hr class="with-margin">
<h4 class="header" id="kmeans">K-means algorithm</h4>

This is such a popular and well documented algorithm I'm not going to spend much time discussing how it works but rather point out some of the more interesting details for those who are already familiar with it. If you are not try reading [this](https://www.naftaliharris.com/blog/visualizing-k-means-clustering/).

##### Summary of K-means algorithm
We have input data $x_1, ..., x_n$ with $x \in \mathbb{R}^d$ and would like to output a vector $c$ of cluster assignments for each point $n$. There are K possible clusters and each cluster has a mean (or centroid) $\mu_K \in \mathbb{R}^d$ which defines a cluster.

We pick an objective function that tells us how to assign points to a cluster and gives us a way of calculating a good centroid. It must also be easy to optimize. In words: we choose to find the best $\mu^\*$ and $c^\*$ where we find these by minimizing the sum of the squared Euclidean distances of each point to its cluster centroid.

Essentially the K-means algorithm does the following for each step (after randomly initializing the $\mu_K$):

<hr class="with-margin">

1. Assigns each data point $x_i$ to the closest cluster based on $\mu_K$
2. Updates each centroid $\mu_K$ to the mean of the points now assigned to it

<hr class="with-margin">

A few comments:

* This objective function depends heavily on how we choose K, this isn't easy but can be done via domain knowledge or other methods not discussed here depending on the reason for using K-means.
* This objective function is non-convex so we might not find the global minima.
* Each data points' distance to its centroid is all that counts towards the objective function (i.e. not how far it is from the other clusters)
* We use a technique called coordinate descent to perform the optimization (which is a fancy way of looking at the two steps outlined above).
  * The loose idea (see [wikipedia](https://en.wikipedia.org/wiki/Coordinate_descent)) is that the minimization of a multivariable function $F(x)$ can be achieved by minimizing it along one direction at a time, i.e., solving univariate (or at least much simpler) optimization problems in a loop.
  * In this example (as above) we first fix the centres of our clusters, then assign points and then with this cluster assignments fixed, update the centres of our clusters.
    * It turns out we can do this bit exactly (i.e. solve for one with the other fixed and vice-versa)

##### Convergence of the K-means algorithm

Each update step decreases the objective function vs. the previous step until the assignments stop changing (so the objective function is monotonically decreasing) however this does not mean we made it to the global optima! In practice we will run the algorithm many times with different restarts and then we'll use the one with the lowest loss (note this is holding K fixed).

##### Extending K-means -> K-medoids

We can replace the squared Euclidean distance used in K-means with any arbitrary distance function $D(x, \mu)$, this is called K-medoids clustering.

##### More reading on K-means
On the non-convexity [here](https://pafnuty.wordpress.com/2013/08/14/non-convex-sets-with-k-means-and-hierarchical-clustering/) and on its relation to the EM algorithm (!) [here](http://stanford.edu/~cpiech/cs221/handouts/kmeans.html).

<hr class="with-margin">
<h4 class="header" id="sec2">A comment on the mathematics for the week</h4>

##### Training Error Theorem

The main heavy bit of maths this week came in the form of proving something called the 'Training Error Theorem' for boosting (and AdaBoost in particular). This is a theoretical guarantee that says that under AdaBoost the training error will eventually go to 0 as we fit more rounds (in fact it decays exponentially fast). Of course, we care about the validation error not the training error so this feels to me a little like we've got our reasoning backwards.

<blockquote class="tip">
<strong>Key point:</strong> there are many nice theoretical ideas that do not work in practice and the reason The Training Error theorem is taught/presented seems to be because boosting happens to generalize well. This is nice but needn't be the case. Whilst it's important to focus on theoretical backing it's also important to see what works in practice and try to understand if there is any theoretical justification for it (or if we even care).
</blockquote>

As an example, many models are [universal function approximators](http://neuralnetworksanddeeplearning.com/chap4.html) but that doesn't mean they would work well in practice or be feasible to train. In such cases theoretical guarantees are worthless.

Having been through the details of the derivation I do not think it is worth me regurgitating the details here.

Some more reading [here](https://www.cs.cmu.edu/~aarti/Class/10701/slides/Lecture10.pdf) if you wish or a version of the proof [here](https://www.cs.princeton.edu/courses/archive/fall08/cos402/readings/boosting.pdf).

##### Coordinate descent

This is what we used to optimize K-means and [this](http://kldavenport.com/the-cost-function-of-k-means/) provides a nice explanation though [wikipedia](https://en.wikipedia.org/wiki/Coordinate_descent) is reasonably accessible on this topic.

<hr class="with-margin">
<h4 class="header" id="sec3">Things I'm unclear on (or outstanding questions)</h4>

TBC

<hr class="with-margin">
<h4 class="header" id="sec4">What did the textbooks say?</h4>


To be updated.
