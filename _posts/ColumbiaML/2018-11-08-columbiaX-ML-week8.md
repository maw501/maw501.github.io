---
layout: post
title: ColumbiaX - ML - week 8
date: 2018-11-08
use_math: true
tags: ['em_algorithm', 'gaussian_mixture', 'kmeans']
image: "gmm.png"
comments: true
---
In week 8 we take a look at the EM algorithm, an iterative algorithm which helps us find parameter estimates when we cannot directly solve the model equations to obtain solution. We also looked at soft vs. hard clustering and use the EM algorithm to solve a Gaussian Mixture Model.

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
<h4 class="header" id="intro">Week 8 (lectures 15 and 16): overview</h4>

The week was quite theoretically heavy but I think if we can step around that we can see that all topics discussed have some nice intuition as well as (easy to understand) conceptual explanations.

<blockquote class="tip">
<strong>Note:</strong> Given the density of the course notes on this topic I am not going to try to give full treatment to all the details (though may follow up on the EM algorithm in a separate post).
</blockquote>

* **EM algorithm**
  * The EM algorithm is a way of finding the (local) MLE of the parameters of a model when we cannot solve the equations directly. The EM algorithm does this by assuming the presence of latent (i.e. not directly observed) variables which turn out to make the problem more tractable. An example of this which we will get to later is that of a [mixture model](https://en.wikipedia.org/wiki/Mixture_model) whereby we assume each data point has some unobserved latent variable indicating the probability it belongs to a certain mixture/cluster.
  * The introduction from [wikipedia](https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm#Introduction) is pretty good.
* **Soft vs. hard clustering**
  * An example of soft clustering is (a logical extension of K-means) where instead of assigning each data point to a specific cluster with certainty we assign it with a probability - this is called weighted K-means. More [here](https://sandipanweb.wordpress.com/2017/03/19/hard-soft-clustering-with-k-means-weighted-k-means-and-gmm-em/).
* **Gaussian Mixture Models (GMMs)**
  * Mixture models in general are generative models (i.e. they define a probability distribution on the data) that are weighted combinations of simpler distributions which are all in the same distribution family with the weighting determined by a discrete probability distribution. GMMs use the underlying family of, you guessed it..., Gaussians.

We will focus primarily on the EM algorithm as this represents the bulk of the week's content.

<hr class="with-margin">
<h4 class="header" id="em">The EM algorithm</h4>

<blockquote class="tip">
<strong>TLDR:</strong> allows us to find (local) MLE of the parameters of a function we otherwise would find tricky to optimize.
</blockquote>

Explaining the EM algorithm will be a somewhat lengthy process. Our strategy will be as follows:
<hr class="with-margin">
* Set-up the discussion with a little motivation
* State the EM objective function
* Give an overview of the EM algorithm steps
* Take a breather to talk about imputing missing data using the EM algorithm
* Give a discussion of the EM algorithm for a Gaussian Mixture Model (GMM)

<hr class="with-margin">

##### Set-up

Suppose we are in the maximum likelihood world where we want to find the MLE solution for some set of parameters $\theta_1$:

$$ \theta_{1, ML} = \underset{\theta_1}{\operatorname{argmax}} \sum_{i=1}^{n} \ln p( x_i \mid \theta_1) $$

Recall this means we are maximizing the probability of our data given the probability model and its parameters. It turns out that the formulation above (not discussed why here) is often tricky to solve and (yes, really!) adding a second variable/set of parameters $\theta_2$ can make things easier. i.e. instead maximize:

$$ \sum_{i=1}^{n} \ln p( x_i, \theta_2 \mid \theta_1) $$

Note as $ \theta_2$ appears on the left of the conditioning meaning we need a prior distribution for it.

The EM algorithm is a technique for solving for $ \theta_{1, ML}$ by using this second equation statement involving $\theta_2$.

<blockquote class="tip">
<strong>Application:</strong> The EM algorithm can be used to find the MLE of the parameters of a model when we have missing data whereby we iterate back and forth between filling in the missing data and then updating our parameter estimates in light of the imputed data. We will briefly discuss this later.
</blockquote>

##### The EM objective function

We are just going to state this and then below we will discuss it (note rewriting the RHS to prove this equality is surprisingly about 2 lines):

$$ \ln p(x \mid \theta_1) = \underbrace{\int q(\theta_2) \ln \dfrac{p( x, \theta_2 \mid \theta_1)}{q(\theta_2)} \, d\theta_2 \,}_\text{call this term $\mathcal{L}(x, \theta_1)$}  + \, \underbrace{\int q(\theta_2) \ln \dfrac{q(\theta_2)}{p( \theta_2 \mid x, \theta_1)} \, d\theta_2}_\text{this is a KL divergence term} $$


i.e.

$$\ln p(x \mid \theta_1) = \mathcal{L}(x, \theta_1) + KL$$

A few comments:
* The EM algorithm will maximize the above objective function by using the RHS which is easier to work with than the LHS.
* The statement of the (log) probability in this way is nice because the last term on the right is actually the [Kullback-Leibler (KL) Divergence](https://medium.com/@cotra.marko/making-sense-of-the-kullback-leibler-kl-divergence-b0d57ee10e0a) between $q(\theta_2)$ and $p( \theta_2 \mid x, \theta_1)$. There are two facts about the KL divergence we need to know:
  1. It's always positive. This fact is useful in showing the EM algorithm actually works (i.e. gives monotonic improvement at each step)
  2. When the two distributions are equal (i.e. $q=p$) then the KL term is equal to 0.
* The mysterious distribution $q$ that has crept in is our prior assumption for the distribution of $\theta_2$. This can be *any* (continuous) distribution we want and it turns out not to matter as the EM algorithm actually tells us how to calculate this without us ever having to use it.
* The $\mathcal{L}(x, \theta_1)$ term will only depend on $\theta_1$ once we've integrated out $\theta_2$. We are assuming we can perform this integration.

<blockquote class="tip">
<strong>Warning:</strong> remember the two facts about the KL divergence, they are important(!).
</blockquote>

##### Overview of EM algorithm steps (slightly longer than intended)

Here is an overview of the algorithm from a step $t$ to step $t+1$. This little overview will also be the sketch proof that the EM algorithm improves $\ln p(x \mid \theta_1)$ monotonically for the sequence $\theta_1^{(t)}$ over all $t$:

<hr class="with-margin">
We start at step $t$ with:

* $\ln p(x \mid \theta_1^{(t)}) = \mathcal{L}(x, \theta_1^{(t)}) + KL_t$
* Next set $q = p$ which causes the $KL$ term to disappear by the property of KL.
* **E-step** of the EM algorithm: Evaluating the integral $\mathcal{L}(x, \theta_1^{(t)})$ has the term $q(\theta_2)$ in it which we have set equal to $p( \theta_2 \mid x, \theta_1)$
  * It's called the E-step as integrating (summing in a discrete world) is actually computing the (log) likelihood using the current estimate for the parameters (when we set $q(\theta_2) = p( \theta_2 \mid x, \theta_1)$).
* We now have (note the $t$ subscript on $\mathcal{L}$ as we've calculated it for a specific step):

$$\ln p(x \mid \theta_1^{(t)}) = \mathcal{L}_t(x, \theta_1^{(t)})$$

* **M-step** of the EM algorithm: find the value of our parameters $\theta_1$ that maximizes $\mathcal{L}_t(x, \theta_1^{(t)})$ as this expression still depends on $\theta_1$. These are our new $\theta_1$ values for the next step $t+1$.
* This means:

$$\ln p(x \mid \theta_1^{(t)}) \leq \mathcal{L}_t(x, \theta_1^{(t+1)})$$

* Now here comes a funky bit. If we add any postive term to the RHS of the above inequality the inequality still holds. Thus we cleverly choose to add the KL term at $t+1$ back to it which is positive as we haven't set $q=p$ there yet. We now have:

$$\ln p(x \mid \theta_1^{(t)}) \leq \mathcal{L}_t(x, \theta_1^{(t+1)}) + KL_{t+1} = \ln p(x \mid \theta_1^{(t+1)})$$

Thus:

$$\ln p(x \mid \theta_1^{(t)}) \leq \ln p(x \mid \theta_1^{(t+1)})$$

This shows the EM algorithm will increase for every step $t$ as we find values that increase $\ln p(x \mid \theta_1^{(t)})$ for each $t$.

<hr class="with-margin">

Now, if you are anything like me you might be a little confused about how this actually works in practice. I will now give a quick intuitive example of an application with no equations before we dive into the GMM example.

##### Imputing missing data using the EM algorithm (no equations)

Suppose we have some data which we assume is generated according to a Gaussian distribution with some unknown mean and variance. Under this Gaussian assumption we can calculate the MLE estimate of the mean and variance parameters (that is, find the mean and variance that make our data most probable). However if we have missing data this isn't simple unless we somehow impute the data. The EM algorithm will allow us to do both of these tasks together.

Essentially we work as follows:

* Start off with some estimates of the mean and covariance.
* Using these parameter values, fill in the missing data by computing the conditional expectation of the data given the observed data and the current parameter values. As we've assumed a Gaussian we can do this analytically. This is the E-step.
* Now we have a full set of data, go back and calculate which parameter values for the mean and variance maximize the probability of the data. This is the M-step.
* We keep iterating through this process until we converge to a (usually local) maximum.

<hr class="with-margin">
<h4 class="header" id="emgmm">The EM algorithm for Gaussian Mixture Models</h4>

Writing an algorithm to solve a GMM using EM was actually this week's project and so I'm not going (or allowed) to give the numpy code example applying EM to GMMs which really helps to clarify the equations. As a compromise the scikit-learn documentation is pretty [readable](https://scikit-learn.org/stable/modules/mixture.html).

<hr class="with-margin">

For observation $i$:

1. Assign to a cluster $c_i \sim Discrete(\pi)$ based on some discrete distribution.
2. Generate the data value using $x_i \sim N(\mu_{c_i}, \Sigma_{c_i})$
<hr class="with-margin">

##### Definitions

$\pi$ is a $K$-dimensional probability distribution which is the probability of being in cluster $K$. e.g. $\pi = [0.5,0.5]$ for $K=2$ is the same as flipping a fair coin to see which cluster we end up in. Each cluster's data generated by a Gaussian according to the cluster mean and variance (we don't know these (!) - we are trying to infer them). So we have $\textbf{$\mu$} = [\mu_1, ..., \mu_K]$ and $\textbf{$\Sigma$} = [\Sigma_1, ..., \Sigma_K]$

<blockquote class="tip">
<strong>Objective:</strong> we wish to learn $\textbf{$\pi$}, \textbf{$\mu$}, \textbf{$\Sigma$}$ based on the observed data.
</blockquote>

Remember we don't have the cluster labels $c_i$ like we do in a K-class Bayes classifier setting - our assumption is on how the data is generated and we will work back to the cluster labels based on this.

Below are some notes on using EM for a GMM example. In this setting we have the following data generating process (if all images don't display please refresh page):

<p align="center">
    <img src="/assets/img/emgmm1.png" alt="Image" width="600" height="800" />
</p>

<em class="figure">Fig. 1: page 1 of EM GMM example</em>

<p align="center">
    <img src="/assets/img/emgmm2.png" alt="Image" width="600" height="800" />
</p>

<em class="figure">Fig. 2: page 2 of EM GMM example</em>

<p align="center">
    <img src="/assets/img/emgmm3.png" alt="Image" width="600" height="800" />
</p>

<em class="figure">Fig. 3: page 3 of EM GMM example</em>


<hr class="with-margin">
<h4 class="header" id="final">Some final important comments on the EM algorithm</h4>

A few points we haven't made much of above but are worth emphasising:

##### Introducing latent variables can help

In the GMM example above we don't know the cluster labels $c_i$ and so we treat them as auxiliary data that we integrate out (i.e. sum over all possible values for cluster $K$). i.e we maximize the log-likelihood:


$$ \sum_{i=1}^{n} \ln p(x_i \mid \pi, \textbf{$\mu$}, \textbf{$\Sigma$}) $$

by using

$$ \sum_{i=1}^{n} \ln \sum_{k=1}^{K} p(x_i, c_i = k \mid \pi, \textbf{$\mu$}, \textbf{$\Sigma$}) $$

and note we will sum over all $K$ for each $c_i$ as these $c_i$ are unobserved so we 'marginalise' them out.

##### Easier form to optimize

It turns out having the log-likelihood in the above format is hard to work with due to the 'log-sum' nature of it. Once we work through the calculations for EM we end up in the M-step with something of the form 'sum-log' which is much easier to optimize (we can now differentiate and solve). In this sense EM has made things easier for us.

##### We are still using maximum likelihood to learn our parameters

Despite all its sophistication EM still uses MLE to find its parameter values and this means it can over-fit. It turns out we can use a Bayesian Gaussian Mixture Model whereby we can put a Dirichlet process prior on the number of clusters to regularize the model. See the [scikit-learn](https://scikit-learn.org/stable/modules/mixture.html) docs for a brief discussion.

<hr class="with-margin">
<h4 class="header" id="further">Further reading</h4>
Fantastic discussion on the EM algorithm [here](http://bjlkeng.github.io/posts/the-expectation-maximization-algorithm/)
and some lecture slides  [here](https://davidrosenberg.github.io/mlcourse/Archive/2017/Lectures/14a.EM-algorithm.pdf).

<hr class="with-margin">
<h4 class="header" id="sec3">Things I'm unclear on (or outstanding questions)</h4>

TBC

<hr class="with-margin">
<h4 class="header" id="sec4">What did the textbooks say?</h4>


To be updated.
