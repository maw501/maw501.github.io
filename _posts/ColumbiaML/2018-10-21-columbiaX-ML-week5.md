---
layout: post
title: ColumbiaX - ML - week 5
date: 2018-10-21
use_math: true
image: "logreg.png"
comments: true
---

Week 5 introduced the topic of logistic regression in several guises before getting into more technical topics such as Laplace approximation, kernels and ultimately Gaussian Processes. Gaussian Processes in particular are quite tricky to understand initially and I've spent more than a couple of hours trying to become fully happy with various aspects of them.

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
<h4 class="header" id="intro">Week 5 (lectures 9 and 10): overview</h4>

Hopefully I can help clarify some of insights for others. First though, the summary:

* **Logistic Regression**
  * Logistic Regression is analogous to linear regression with some modifications for outputting a classification probability.
* **Bayesian Logistic Regression -> Laplace Approximation**
  * We can assume a Gaussian distribution for our parameters $w$. The main difference when moving to a Bayesian framework for logistic regression is that **we can no longer get an analytic expression for the posterior $p(w \mid x, y)$.** However we can approximate the posterior using something called Laplace Approximation.

* **Feature expansions**
  * Feature or basis expansions involve taking transformations of the **columns** of dataset (i.e. the covariates), usually in order to capture non-linear behaviour. We then fit a linear model to the enhanced set of features.
* **Kernels**
  * Kernels are a way of assessing the similarity of our data points to each other (i.e. the rows of our $X$ matrix) and are usually introduced in the context of SVMS. Assessing the similarity of our data points to each other is the basis of how methods such as k-NN and SVMs work for classification. When prediction time comes we will assign a new point to a class based on some notion of similarity to points we saw in our training data. Kernels allow us to use our existing data in a manner which is equivalent to carrying out the similarity calculation in a higher dimensional space without actually visiting that higher (possibly infinite) dimensional space. **Crazy nonsense eh?** Don't worry, we'll give a simple example below which will elucidate this notion which is notoriously badly taught in general.
    * **Side rant:** I actually have spent a fair bit of time getting my head ahead the kernel trick which (like most things in mathematics) once understood isn't actually that complicated. It's just that mathematicians on average seem incapable of introducing something neat (which obviously they themselves would usually never have discovered or been able to prove first hand) to students (who often are having many WTF moments) without overtly focusing on its technical beauty or the full generality of the idea which more often than not just obfuscates matters. Phrases like "kernels are symmetric functions that allow us to compute in a possibly infinite dimensional space without visiting that space" just raise too many confusing questions for students without making anything clear.
* **Gaussian Processes**
  * Gaussian Processes are a flexible technique whereby we impose minimal assumptions on our data but are still based on probability theory and so allow confidence estimates for new predictions.

<hr class="with-margin">
<h4 class="header" id="more_details">A more detailed overview of the week</h4>

Week 5 has some theoretically heavy topics in (most notably kernels and Gaussian Processes) which I will do my best to elaborate on in a little more detail, and hopefully with less dense presentation.

##### Logistic Regression is a discriminative model analogous to linear regression but for classification.

It learns a model of the form $\sigma(x^Tw + w_0)$ with no explicit assumptions on $w$ and $w_0$.

Recall previously in the context of binary classification that we chose to model $p(y\mid x)$ using Bayes rule as:

$$\dfrac{p(x \mid y) \, p(y)}{p(x)}$$

with a prior on $p(y)$ of a Bernoulli and $p(x \mid y)$ as we pleased e.g. a Gaussian. We then took the log of the ratio of each of these class probabilities and showed it could be expressed in the form $x^Tw + w_0$ where we had a closed expression for both $w$ and $w_0$ in terms of our prior and the data $x$. This led to either LDA or QDA depending on whether we assumed each class had the same variance or not. This is a **generative** model as we are modelling $x$. Logistic regression is a **discriminative** version of this where we relax the assumptions on the data $x$ and simply try to find a model directly from the of the form $p(y \mid x) = \sigma(x^Tw + w_0)$ where $\sigma$ is the sigmoid function

$$\sigma(t) = \dfrac{e^t}{1+e^t}$$

and we call $t=x^Tw+w_0$ the *link function*.
* Note if we absorb the term $w_0$ into our data then we can think just about $x^Tw$ and note that if $x^Tw > 0$ then $\sigma(x^Tw) > 0.5$ and we can predict the class of $y$ accordingly.
* There is no analytic solution to solving the maximum likelihood solution and so we use an iterative algorithm. See the mathematical details section.

##### Bayesian Logistic Regression: just like we put a prior on our $w$ parameters for linear regression we can regularize logistic regression in the same way

* When we choose to add a regularizing term to our logistic regression we know from previous work that if the term is an $L_2$ penalty this corresponds to a Gaussian prior distribution on $w$. What can we say about the posterior $p(w \mid x,y)$?
* It turns out in this case we can't calculate the posterior analytically as we can't calculate the denominator and so we approximate $p(w \mid x,y)$ with a Gaussian distribution using a technique called Laplace Approximation...

##### Laplace Approximation is a way of approximating the posterior distribution $p(w \mid x,y)$ analytically
The Laplace approximation framework aims to find a Gaussian approximation to a continuous distribution which in this case is our posterior $p(w \mid x,y)$. The method aims specifically at problems in which the distribution is uni-modal.

* In short this method expands the joint distribution $f(w) = \ln p(y, w \mid x)$ around a point $z = w_{MAP}$ using a second order Taylor series expansion.
* This allows us to approximate $p(w \mid x, y) \sim N(w_{MAP}, \Sigma)$ i.e. as a Gaussian. More details will be given below.

##### Feature expansions are simply adding new features to our dataset that are transformations of our original features/covariates.
Feature or basis expansions are usually introduced in quite a mathematically heavy way but for anyone who has done a little machine learning in the real world (or a kaggle competition) they are an entirely natural concept. We simply take various transformations of the columns of our data (e.g. append new columns to our design matrix $X$) and then fit a linear model to this expanded dataset. For both regression and classification this can greatly enhance the modelling capability of the linear model and find non-linear boundaries in the original feature space. We could in a loose sense think of these feature expansions as feature engineering which raises a natural question: **which expansion should we do?**
* Often the answer to this question requires specialist knowledge about the dataset in question and so one approach is to take many transformations and use an $L_1$ penalty in order to find a sparse subset of the higher dimensional space.
* We call $\phi(x)$ the mapping of the features to a higher dimensional space $\mathbb{R}^D$ where $D >d$ and $d$ is the number of columns we originally started out with.

##### Kernels are a way of assessing the similarity of our data points to each other (i.e. the rows of our $X$ matrix) - they also have the fearsome idea of the *kernel trick*

I have written a separate post on the kernel trick [here](../../../2018/10/19/The-kernel-trick) and kernels more generally [here](../../../2018/10/22/Why-Kernels-Matter).

##### Gaussian Processes (GPs)

These are getting their own blog post [here](../../../2018/10/22/Gaussian-Processes).

<hr class="with-margin">
<h4 class="header" id="math">Main mathematical ideas from the lectures</h4>

##### Logistic Regression algorithm
* **Input:** training data $(x_1, y_1), ..., (x_n, y_n)$ and step size $\eta >0$.
* Step 1: Set our weights at step 1 equal to the zero vector: $w^{(1)} = \vec{0}$
* Step 2: For step $t = 1, 2, ...$ do
  * Update $w^{(t+1)} = w^{(t)} + \eta \sum_{i=1}^{n} (1-\sigma_i(y_i \cdot w)) y_i x_i$
* In words this means we grab the probability of an observed point with $\sigma_i(y_i \cdot w)$ so $1 - \sigma_i(y_i \cdot w)$ is the probability we assigned the wrong label
* We sum the probabilities of being wrong over all of our data points and use this to weight our update to $w^{(t+1)}$.
* So the logistic regression update step is the same as for the perceptron except we are weighting by the probability we are wrong.

* **Recap:** recall the perceptron update is:  $w^{(t+1)} = w^{(t)} +\eta y_ix_i$ where we are only updating the examples $i$ that are misclassified. So those are the ones that 'move' the boundary around. More info on why this moves the boundary correctly is given [here](https://towardsdatascience.com/perceptron-learning-algorithm-d5db0deab975) and  [here](https://www.lucidarme.me/simplest-perceptron-update-rules-demonstration/).

##### Laplace Approximation

* This is a simple idea but gets a little ugly and I'm not going to overtly dwell on this as it's not really a machine learning technique but rather a more general method used to approximate integrals. It arises in the context of Bayesian logistic regression where we are trying to find the posterior $p(w \mid x, y)$ with $w \sim N(0, \lambda^{-1}I)$. Recall for logistic regression we have a joint likelihood of our data (see below) equal to: $\, \prod_{i = 1}^{n} \sigma(y_i \cdot w)$. The posterior thus becomes:

$$p(w \mid x, y) = \dfrac{\prod_{i = 1}^{n} \sigma(y_i \cdot w) \, p(w)}{\int \prod_{i = 1}^{n} \sigma(y_i \cdot w) \, p(w) dw} $$

* **Side point**: note from Bayes we have written the denominator as the numerator integrated over all parameter values. We have done this as from Bayes we have:

  $$ p(w \mid x, y) = \dfrac{p(y \mid x, w) p(w)}{p(y \mid x)}$$

  * We are calling $p(y \mid x, w) = \sigma(x_i;w) = \sigma(y_i \cdot w)$ in this notation, see below.
  * **Q: what is $p(y \mid x)$ equal to - why is it an integral???**
  * **Answer:** It's the actual probability of our data (not assuming any parameters from our model) - we don't know what this is!!! But given we have assumed a probabilistic model for our data depending on some parameters $w$ we can try to get $p(y \mid x)$ by integrating out the dependence on $w$. Unfortunately this is easier said than done.
* We then (details omitted) can take a second order expansion of $f(w) = \ln p(y, w \mid x)$ around a point of $w$ which we set to be $w_{MAP}$. This allows us to get an expression for $p(w \mid x, y) \sim N(\mu, \Sigma)$ after some work.

##### Kernels

Kernels in general are going to be dealt with more thoroughly in [this](../../../2018/10/22/Why-Kernels-Matter) post.

Kernels are symmetric functions taking in two vectors in $\mathbb{R}^d$ and computing the similarity between them according to some specified way such that the result is positive. Thus $K$ is a $n$ by $n$ matrix with the similarity of each data point to every other. In this sense $K$ satisfies the properties of a covariance matrix. A few points on kernels:
  * We can reformulate many existing linear models into an expression involving a kernel, for example ridge regression. This means that instead of predicting $y(\textbf{x}) = w^T \phi(\textbf{x})$ for a new point/row $\textbf{x}$ (after it may have been transformed by $\phi$) we can instead predict using:

    $$y(\textbf{x}) = \textbf{k}(\textbf{x})^T(K + \lambda I)^{-1}\textbf{y}$$

    * **Notice that there is no weight vector $w$ but the methods are identical**
    * $\textbf{k}(\textbf{x})$ is vector of length $n$ holding the result of our new points similarity given by the kernel function to each of the orginal $n$ points in our data set.
    * $K$ is the $n$ by $n$ matrix with the similarity of each data point from our training set to every other.
    * $\textbf{y}$ is the target from our training data of length $n$.
  * **Q: why would we do this?**
    * **Answer:** This type of kernel regression moves focus away from features and their parameters to the actual data points and their weights. In many machine learning methods there is a duality between feature weights and example weights. This duality allows us to use the kernel trick to expand the representational power of a model without (much) computational expense. All the information about features went into defining $K$, the kernel matrix. Further reading [here](https://alliance.seas.upenn.edu/~cis520/dynamic/2017/wiki/index.php?n=Lectures.Kernels) or Bishop Chapter 6.
  * We can compute new kernels by adding, multiplying or exponentiating existing kernels.

<hr class="with-margin">
<h4 class="header" id="detail">Some mathematical details</h4>

Here are some of the mathematical details from the week:

* **Logistic Regression notation**:
  * For two classes $y=+1$ or $y=-1$.
  * Define $\sigma_i(w) = \sigma(x_i^T;w) = \dfrac{e^{x_i^Tw}}{1+e^{x_i^Tw}}$
  * Call this $\sigma(x_i^T;w) = p(y_i \mid x_i, w)$
  * So the joint likelihood of our data is: $\, \prod_{i = 1}^{n} p(y_i \mid x_i, w)$
  * This is just the product of the probability of each point dependent on its $x_i$.
  * It turns out (for unimportant reasons) we can write this as $\, \prod_{i = 1}^{n} \sigma(y_i \cdot w)$ where we have $\sigma(y_i \cdot w) = \dfrac{e^{y_ix_i^Tw}}{1+e^{y_ix_i^Tw}}$.
    * Basically if our $y_i = +1$ we get the probability of the +1 class which we call $\sigma_i(w)$ and if $y=-1$ we get the probability of the negative class defined as $1-\sigma_i(w)$.
* **Mercer's theorem**
  * This theorem justifies our use of a kernel as a proxy for transforming two data points/rows $(x_i, x_j)$ into a higher dimensional space according to some transformation $\phi$ and then computing $\phi(x_i)^T \phi(x_j)$ (the dot product in the higher dimensional space). Mercer's theorem basically assures us that computing $\textbf{k}(x_i, x_i)$ with our original data $x_i, x_j$ is the same as performing $\phi(x_i)^T \phi(x_j)$ in some higher dimensional space $\mathbb{R}^D$ where $D>d$.

<hr class="with-margin">
<h4 class="header" id="sec3">Things I'm unclear on (or outstanding questions)</h4>

TBC

<hr class="with-margin">
<h4 class="header" id="sec4">What did the textbooks say?</h4>


To be updated.
