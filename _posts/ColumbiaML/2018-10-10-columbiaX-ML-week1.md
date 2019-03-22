---
layout: post
title: ColumbiaX - ML - week 1
date: 2018-10-10
use_math: true
image: "bivariategauss2.jpeg"
comments: true
---
We kick start the course by introducing probabilistic modelling, MLE, multivariate Gaussians as well as the notion of OLS (including its geometry).

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
<h4 class="header" id="intro">Week 1 (lectures 1 and 2): overview</h4>

There were a few key ideas for those who've perhaps done some machine learning but haven't dug behind the scenes. In the rough order they were introduced they were:

* The notion of a probabilistic model
* The concept of maximum-likelihood estimation (MLE)
* Defining a multivariate Gaussian distribution and showing how to find its parameters under MLE
* The notion of least squares (LS) and linear regression (LR)
* Polynomial regression
* Geometry of LS

<hr class="with-margin">
<h4 class="header" id="big">Week 1 (lectures 1 and 2): the big picture</h4>

I feel a little that the main story of week 1 is slightly hard to tell without referencing week 2 as we wandered off into a few technical derivations which made the story harder to decipher initially than I think was necessary. As a result I'm just going to state where we are initially heading.

##### We wish to contrast two different ways of looking at the problem of linear regression:

1. **LS approach:** assume a loss function and minimize it based on the linear model choice, there are no (explicit) assumptions about the data. We return a set of weights from the optimization.

2. **MLE approach:** A probabilistic approach where we find the set of parameters that makes the data most likely according to our chosen distribution family for generating the data. We can find the corresponding weights for LR with a further assumption about the distribution parameters.

It will turn out that the **'optimal'** set of weights from both approaches are the same given certain assumptions.

We also discussed how polynomial regression is an acceptable extension of linear regression as well as the geometry of least squares.

<hr class="with-margin">
<h4 class="header" id="math">Main mathematical ideas from the lectures</h4>

* **Differentiate the joint (log) likelihood function to find your MLE parameters**
  * We assume our data $x$ is generated according to some probability distribution $p(x \mid \theta)$. Further assuming the data is [iid](https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables) means we can compute the joint likelihood of our data as the product of the probability of each individual observation.
  * This is the product of many terms all less than one (they are probabilities) and this can be complicated to compute. We thus take logs to turn multiplication into addition and note this is still maximizing the same set of parameters. This is now the joint log likelihood.
  * For each parameter we wish to find we differentiate the joint log likelihood of our data with respect to that parameter, set this equation to 0 and find the value of the parameter at that point. This is where the gradient of the joint log likelihood is maximized (as well as the joint likelihood).
* **Rewriting the equations for LS linear regression in both vector and matrix form**
  * The advantage of this is we can treat our weight parameters as a single vector which we differentiate with respect to in one go. This requires a little knowledge of matrix calculus: i.e. some of the stuff from [here](https://en.wikipedia.org/wiki/Matrix_calculus) is useful when trying to derive $w_{LS}$ in matrix form as a few of the steps were skipped in the lecture.
* **The geometry of least squares linear regression**
  * This is quite a fun topic, particularly if you know a little linear algebra and grok the concept of matrix column and row spaces. This is perhaps worthy of a post on its own so I might come back to it at a later point.

<hr class="with-margin">
<h4 class="header" id="details">Some mathematical details</h4>

There were quite a few of these hiding behind the scenes (and some in plain sight) so I'm going to try to out a few of them now.

* Dividing by the determinant of the covariance matrix in the definition of a multivariate Gaussian which then in week 2 gets stated differently. For now all I'll say is that in the case where $\Sigma$ is diagonal the determinant is the product of the diagonal entries (here the individual variances) which is exactly what we wish to normalise our density function by.
* In the derivation of $\Sigma_{ML}$ we differentiated $\ln(\mid\Sigma\mid)$.   
  * Note: $\frac{\partial}{\partial A} \ln(\mid A \mid) = A^{-T}$
* In the derivation of $\Sigma_{ML}$ we also used something called the 'trace trick' to compute this derivation. More details [here](https://people.eecs.berkeley.edu/~jordan/courses/260-spring10/other-readings/chapter13.pdf) and [here](http://nonconditional.com/2012/06/the-trace-trick-for-gaussian-log-likelihood/).
* Note that $X^{T}X = \sum_{i=1}^{n} x_i x_i^{T}$
  * i.e. that matrix multiplication is the same as summing over all the individual outer products. This point will matter when we talk about active learning later in the course.

<hr class="with-margin">
<h4 class="header" id="sec3">Things I'm unclear on (or outstanding questions)</h4>

* The main thing appears to be the gap between the quiz difficulty which is so far quite straightforward vs. the technical details covered in lectures. This will be the topic of another post giving my initial course thoughts.
* There are also a few little points around the technical details of deriving the MLE solution and the LS solution.

<hr class="with-margin">
<h4 class="header" id="sec4">What did the textbooks say?</h4>


To be updated.
