---
layout: post
title: ColumbiaX - ML - week 5
date: 2018-10-21
use_math: true
---

This page is a summary of my notes for the above course, link [here](https://www.edx.org/course/machine-learning-columbiax-csmm-102x-4).

This is my first attempt at publishing course notes and I have no intention to make it comprehensive but rather to highlight the bits I feel are important and maybe explain some of the things I found a little trickier (or weren't explained to my taste). Understanding is deeply personal though if you spot any errors or have any questions please feel free to drop me an email.

## Week 5 (lectures 9 and 10): overview

Week 5 introduced the topic of logistic regression in several guises before getting into more technical topics such as Laplace approximation, kernels and ultimately Gaussian Processes. Gaussian Processes in particular are quite tricky to understand initially and I've spent more than a couple of hours trying to become fully happy with various aspects of them. Hopefully I can help clarify some of insights for others. First though, the summary:

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

## Week 5 (lectures 9 and 10): more detailed overview

Week 5 has some theoretically heavy topics in (most notably Gaussian Processes) which I will do my best to elaborate on in a little more detail, and hopefully with less dense presentation.

#### Logistic Regression is a discriminative model analogous to linear regression but for classification.

It learns a model of the form $\sigma(x^Tw + w_0)$ with no explicit assumptions on $w$ and $w_0$.

Recall previously in the context of binary classification that we chose to model $p(y\mid x)$ using Bayes rule as $\dfrac{p(x \mid y) \, p(y)}{p(x)}$ with a prior on $p(y)$ of a Bernoulli and $p(x \mid y)$ as we pleased e.g. a Gaussian. We then took the log of the ratio of each of these class probabilities and showed it could be expressed in the form $x^Tw + w_0$ where we had a closed expression for both $w$ and $w_0$ in terms of our prior and the data $x$. This led to either LDA or QDA depending on whether we assumed each class had the same variance or not. This is a **generative** model as we are modelling $x$. Logistic regression is a **discriminative** version of this where we relax the assumptions on the data $x$ and simply try to find a model directly from the of the form $p(y \mid x) = \sigma(x^Tw + w_0)$ where $\sigma$ is the sigmoid function $\sigma(t) = \dfrac{e^t}{1+e^t}$ and we call $t=x^Tw+w_0$ the *link function*.
  * Note if we absorb the term $w_0$ into our data then we can think just about $x^Tw$ and note that if $x^Tw > 0$ then $\sigma(x^Tw) > 0.5$ and we can predict the class of $y$ accordingly.
  * There is no analytic solution to solving the maximum likelihood solution and so we use an iterative algorithm. See the mathematical details section.

#### Bayesian Logistic Regression: just like we put a prior on our $w$ parameters for linear regression we can regularize logistic regression in the same way

* When we choose to add a regularizing term to our logistic regression we know from previous work that if the term is an $L_2$ penalty this corresponds to a Gaussian prior distribution on $w$. What can we say about the posterior $p(w \mid x,y)$?
* It turns out in this case we can't calculate the posterior analytically as we can't calculate the denominator and so we approximate $p(w \mid x,y)$ with a Gaussian distribution using a technique called Laplace Approximation...

#### Laplace Approximation is a way of approximating the posterior distribution $p(w \mid x,y)$ analytically
The Laplace approximation framework aims to find a Gaussian approximation to a continuous distribution which in this case is our posterior $p(w \mid x,y)$. The method aims specifically at problems in which the distribution is uni-modal.

* In short this method expands the joint distribution $f(w) = \ln p(y, w \mid x)$ around a point $z = w_{MAP}$ using a second order Taylor series expansion.
* This allows us to approximate $p(w \mid x, y) \sim N(w_{MAP}, \Sigma)$ i.e. as a Gaussian. More details will be given below.

#### Feature expansions are simply adding new features to our dataset that are transformations of our original features/covariates.
Feature or basis expansions are usually introduced in quite a mathematically heavy way but for anyone who has done a little machine learning in the real world (or a kaggle competition) they are an entirely natural concept. We simply take various transformations of the columns of our data (e.g. append new columns to our design matrix $X$) and then fit a linear model to this expanded dataset. For both regression and classification this can greatly enhance the modelling capability of the linear model and find non-linear boundaries in the original feature space. We could in a loose sense think of these feature expansions as feature engineering which raises a natural question: **which expansion should we do?**
* Often the answer to this question requires specialist knowledge about the dataset in question and so one approach is to take many transformations and use an $L_1$ penalty in order to find a sparse subset of the higher dimensional space.
* We call $\phi(x)$ the mapping of the features to a higher dimensional space $\mathbb{R}^D$ where $D >d$ and $d$ is the number of columns we originally started out with.

#### Kernels are a way of assessing the similarity of our data points to each other (i.e. the rows of our $X$ matrix) - they also have the fearsome idea of the *kernel trick*

I have actually just written a separate post on the kernel trick [here](../19/The-kernel-trick).

#### Gaussian Processes (GPs)

In a loose sense Gaussian Processes can be thought of as a probabilistic non-parametric form of non-linear regression. Or at least that is how they are usually introduced. I hope to post a separate article on GPs with code examples that can better explain things.

Something I don't usually do (but in line with the fact I'm not wanting to explain everything from scratch but just give my take on things): preliminary reading [here](http://katbailey.github.io/post/gaussian-processes-for-dummies/), [here](http://platypusinnovation.blogspot.com/2016/05/a-simple-intro-to-gaussian-processes.html) and [here](http://keyonvafa.com/gp-tutorial/). There are also good lectures [here](https://www.youtube.com/watch?v=4vGiHC35j9s&list=PLE6Wd9FR--EdyJ5lbFl8UuGjecvVw66F6&t=0s&index=9) from none other than Nando de Freitas that cover Gaussian Processes a lot more thoroughly than we did in lectures.

##### Attempt at an explanation 1

We specify some prior on our data by defining a kernel function which tells us how similar two vectors in our input space are (i.e. how similar are two rows of our design matrix $X$ which are usually in $\mathbb{R}^d)$.
  * Point of confusion 1: all of the beginner tutorials for Gaussian Processes usually use a 1d example and non-linear regression and so it's easy to see when two points are near each other in $x$ space. But the $x$-axis could equally represent how close together our points are after they've come out of the kernel function. i.e. the $x$-axis now tells us how similar these points are in their $d$-dimensional space. We can have $n$ of these points (rows in our dataset).

So our kernel (which we haven't yet defined) computes the similarity between each of our $n$ data points and thus returns a $n$ by $n$ matrix. We then define an $n$ dimensional Gaussian $f \sim N(\mu_X, K_{XX})$ which is the prior probabilistic model of how our data is generated. $\mu_X$ is the mean function of each data point and is often assumed to be 0 for reasons not discussed here.

**We can now simulate new data from this function $f$**

At this point you probably hear phrases like "Gaussian Processes are simply an infinite-dimensional distribution over functions" which do nothing to aid understanding. "Ah yes" you say, "a distribution over functions, why I do that all the time".

The reason this phrase crops up is because theoretically we could have an infinite about of data points (we don't, we have a finite subset $n$) and we can sample from this prior $f \sim N(\mu_X, K_{XX})$ which turns out to be defining a distribution over the possible functions which define our data.

##### What is the kernel?

The kernel is actually the crucial thing that determines what sort of functions we end up with. It basically controls the smoothness and type of functions we can get from our prior. How do I choose the kernel? Read [this](http://www.cs.toronto.edu/~duvenaud/cookbook/index.html).

##### Right, so I think I get the kernel/distribution over function thing, now what?

Well the point is that after we have specified our prior and then have seen some data we can sample from our posterior and see what our functions values look like probabilistically. Furthermore, when we see new data we can define a probability distribution over the set of possible values we think the output will take. This is like forming a predictive distribution which we have seen before.

**Gaussian Processes section TBC**

##### Further reading
More links [here](http://people.ee.duke.edu/~lcarin/David1.27.06.pdf), [here](https://www.linkedin.com/pulse/machine-learning-intuition-gaussian-processing-chen-yang) and [here](https://www.eurandom.tue.nl/events/workshops/2010/YESIV/Prog-Abstr_files/Ghahramani-lecture2.pdf).

## Main mathematical ideas from the lectures

* **Logistic Regression algorithm**
  * **Input:** training data $(x_1, y_1), ..., (x_n, y_n)$ and step size $\eta >0$.
  * Step 1: Set our weights at step 1 equal to the zero vector: $w^{(1)} = \vec{0}$
  * Step 2: For step $t = 1, 2, ...$ do
    * Update $w^{(t+1)} = w^{(t)} + \eta \sum_{i=1}^{n} (1-\sigma_i(y_i \cdot w)) y_i x_i$
  * In words this means we grab the probability asssiss


## Some mathematical details

Here are some of the mathematical details from the week:

* **Logistic Regression notation**:
  * For two classes $y=+1$ or $y=-1$.
  * Define $\sigma_i(w) = \sigma(x_i^T;w) = \dfrac{e^{x_i^Tw}}{1+e^{x_i^Tw}}$
  * Call this $\sigma(x_i^T;w) = p(y_i \mid x_i, w)$
  * So the joint likelihood of our data is: $\, \prod_{i = 1}^{n} p(y_i \mid x_i, w)$
  * This is just the product of the probability of each point dependent on its $x_i$.
  * It turns out (for unimportant reasons) we can write this as $\, \prod_{i = 1}^{n} \sigma(y_i \cdot w)$ where we have $\sigma(y_i \cdot w) = \dfrac{e^{y_ix_i^Tw}}{1+e^{y_ix_i^Tw}}$.
    * Basically if our $y_i = +1$ we get the probability of the +1 class which we call $\sigma_i(w)$ and if $y=-1$ we get the probability of the negative class defined as $1-\sigma_i(w)$.

## Things I'm unclear on (or outstanding questions)

* TBC

## What did the textbooks say?

To be updated.
