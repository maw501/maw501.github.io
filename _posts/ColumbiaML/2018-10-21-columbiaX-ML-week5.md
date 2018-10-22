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



#### Gaussian Processes

* Have a notion of how similar two points are in $x$ space (which can be $d$-dimensional).
* For any subset of
Similar to KNN, the Gaussian Process is a lazy algorithm: we keep the training data, and fit a model for a specific input. Also like KNN, the shape of the model will come from the data. And as with KNN, the key part is the relationship between examples, which we haven't defined yet.
The kernel function takes in any two points, and outputs the covariance between them. That determines how strongly linked (correlated) these two points are.









## Main mathematical ideas from the lectures

* **Gaussian class conditional densities**
   * Defining $p(x \mid Y=y) = N(x \mid \mu_y, \Sigma_y)$ then we can calculate the MLE estimates of $(\mu_y, \Sigma_y)$ which are just the empirical mean and covariances of the corresponding class $y$.
* **Using log odds to make a binary classification decision**
  * For a given class we are approximating $p(Y=y \mid X=x)$ with $p(x \mid Y=y)\,p(Y=y)$. For two classes taking the natural log of the ratio of the probabilities is called the log odds:
    * E.g. Evaluate if: $\ln \dfrac{p(x \mid y=1)\,p(y=1)}{p(x \mid y=0)\,p(y=0)} > 0$
    * **Example for LDA:** i.e. $p(x \mid Y=y) = N(x \mid \mu_y, \Sigma)$ this evaluates to something a little ugly but the main point to note is that there is a term not involving $x$ which we call $w_0$ and a term involving $x^T$ multiplied by a vector not involving $x$, we call this $w$. Both $w$ and $w_0$ involve $\Sigma, \mu_1, \mu_0$ as well as $\pi_0, \pi_1$ which are the baseline priors e.g. $p(y=0), p(y=1)$.
    * So evaluating which class a point belongs to is equivalent to determining the sign of $x^Tx + w_0$ where in this case we have explicit formula for $w$ and $w_0$.
    * This produces a linear decision boundary.
    * Extend to QDA (still solvable analytically) by using different covariances for each class.
* **Hyperplanes**
  * The main idea is that $x^Tw + w_0$ gives a sense of distance from the hyperplane with the sign telling us which side we are on. The lecture notes have some nice illustrations of this that are much easier to follow than words so please refer to those.

## Some mathematical details

Here are some of the mathematical details from the week:

* **Notions of distance**:
  * The typical Euclidean distance in $\mathbb{R}^d$ is $\|\|u-v\|\|\_2$ but we can extend this notion to the $l_p$ distance for $p \in [1, \infty]$ as follows: $l_p = \|\|u-v\|\|\_p = \big(\sum_{i=1}^{d}\|u_i - v_i\|^p\big)^{\frac{1}{p}}$
    * Here the single $\|$ means take the absolute value of the resulting number when we subtract $v_i$ from $u_i$.
* **Probability details**
  * There were two of these introduced in the lecture and whilst I don't usually find some aspects of probability theory highly intuitive I do think the latter of the two results mentioned in class actually has a simpler explanation. It was presented as follows:
    * $ C = E[A \mid B] $ with $A$ and $B$ both random, so $C$ is random.
    * $E[C] = E[E[A \mid B]] = E[A]$ is the 'tower property' of expectation
    * **What does this mean intuitively?**
    * **Example**: let's say there is a factory that makes fancy new GPUs and depending on the result of a fair coin flip some burn out after an average of 10,000 hours (flip H) and some burn out after an average of 20,000 hours (flip T).
      * Let's call flipping H event $B$ and random variable $A$ how many hours before the GPU burns out.
        * Note: $B^c$ is thus the event of flipping T.
      * In this case $ C = E[A \mid B] $ is still a random variable as given we have a GPU there is still some randomness as $A$ is itself random. However we can think about what the **average** value of $C$ might be, that is, the average numbers of hours a GPU from this factory will last: $E[C] = E[A \mid B]P[B] + E[A \mid B^c]P[B^c] = 10,000\cdot0.5 + 20,000\cdot0.5 = 15,000$ hours. This is no longer random.
      * It is just the probability weighted average and is a completely natural thing to do!
* **Cosine similarity and hyperplanes**
  * Note: this was incorrectly stated as the cosine rule in lecture which means something else. Cosine similarity is a measure of how 'similar' two vectors are my calculating the angle between them - vectors that point in the similar directions have a higher cosine similarity.

## Things I'm unclear on (or outstanding questions)

* TBC

## What did the textbooks say?

To be updated.
