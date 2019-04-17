---
layout: post
title: ColumbiaX - ML - week 11
date: 2018-11-28
use_math: true
tags: ['hidden_markov_models', 'kalman_filtering']
image: "hmm.jpg"
comments: true
---
In week 11 we extend the Markov model by considering the hidden Markov model which is a latent variable model we can learn via the EM algorithm. We then take this further by looking at the Kalman filter and finally draw a link to probabilistic PCA which was covered in week 10.

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
<h4 class="header" id="intro">Week 11 (lectures 21 and 22): overview</h4>

We extend the Markov model which assumed we were observing a sequence of states to the hidden Markov model by now assuming that instead of observing the actual state, we now observe some noisy realisation of it. This observation sequence can be either discrete or continuous but we assume the underlying latent states (which we do not observe) are discrete.

Extending the model to assume the underlying states themselves are continuous will lead us to the Kalman filter.

* **Hidden Markov models:** here we assume a hidden (i.e. latent) sequence of discrete states that we don't get to observe. Instead what we see are a sequence of observations which can be either discrete in the case of the discrete HMM or continuous for the continuous HMM - learning the HMM uses the EM algorithm. There are many real-world applications of HMMs from speech recognition to financial modelling.
* **Kalman filter:** if we change the assumption from a discrete set of hidden states to allowing the states to live in continuous space (with continuous observations) then we end up with the Kalman filter. Here we have a Gaussian distribution for both the observations and the hidden continuous states. Kalman filters have many applications in navigational systems.

<hr class="with-margin">
<h4 class="header" id="hmm">Hidden Markov models (HMMs)</h4>

We will start by giving an overview of HMMs and discuss their set-up and possible estimations we might wish to make. We will not cover all the details involved in the underlying algorithms as this is beyond the scope of both the course and this post. More details are in Bishop's book, Chapter 13.

##### Defining a HMM (with example)

Let's use an example where we have two dice, one fair and one unfair. All we observe are a sequence of roll outcomes $(x_1, ..., x_T)$ but we don't know which die was used for any of the rolls. Thus we can think of our underlying two hidden states as whether we are using the fair or unfair die.

With this in mind, a HMM consists of:

* An $S$ by $S$ transition matrix $A$ for transitioning between the hidden, discrete underlying states $S$.
  * In this example $A$ is a 2 by 2 matrix corresponding to the transition probabilities. There are 4 of these transition probabilities:
    * Using fair die at time $t$, use fair die at time $t+1$.
    * Using fair die at time $t$, use unfair die at time $t+1$.
    * Using unfair die at time $t$, use unfair die at time $t+1$.
    * Using unfair die at time $t$, use fair die at time $t+1$.
* An initial state distribution for selecting which state we start in, we'll call this $\pi$, the initial state distribution.
  * This might be something like $\pi = [0.5, 0.5]$.
  * This means we have an equal chance of starting off with either the fair or unfair die.
* A state dependent 'emission distribution' which is a fancy way of saying the probability of the observed values given we are in a particular state. We call this matrix $B$ and write it as $p(x_i \mid s_i = k) = p(x_i \mid \theta_{s_i})$.
  * So for our example the probability of each of the outcomes from our die are different depending on which die we are using. These probabilities can be represented in the matrix $B$ which will have dimensions (2, 6) corresponding to the 2 states, each with 6 possible outcomes.
  * For the fair die in this example each entry would be $1/6$ though in general we won't know these entries for $B$.
  <hr class="with-margin">

Fig 1. illustrates this example:

<p align="center">
    <img src="/assets/img/discrete_hmm.png" alt="Image" width="600" height="400" />
</p>

<em class="figure">Fig. 1: Example set-up of a HMM, image from [course](https://courses.edx.org/courses/course-v1:ColumbiaX+CSMM.102x+3T2018/course/).</em>

##### What might be wish to estimate in the context of an HMM?

<blockquote class="tip">
<strong>Recall:</strong> if $p(x \mid \theta_{s})$ is continuous we have a continuous HMM, if it's discrete we have a discrete HMM. In both cases the underlying hidden states are discrete.
</blockquote>

Wait! Before we can estimate anything with our HMM we first have to learn it. That is, given an observation sequence $(x_1, ..., x_T)$ we wish to learn the parameters/entries for $\pi, A, B$. This is the problem of learning a HMM and we find the maximum likelihood (ML) estimates of the parameters using the EM algorithm. This will be discussed in the next section.

Now, once we've learned the HMM there are at least 2 things we might be interested in estimating:

* **State estimation:** this is the business of estimating the probability of being in a particular state at any point in time given a HMM and observation sequence.
  * In other words, which state were we in at time $i$?
  * Write as: $p(s_i = k \mid x_1, ..., x_T, \pi, A, B)$
  * This is a conditional posterior probability.
  * Note: we obviously don't get to know which state we were actually in (unless we use synthetic data), we get a probability distribution.
  * This estimation uses the ['forward-backward' algorithm](https://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm) which we won't cover here.
* **State sequence:** this is the business of estimating the most probable state sequence over the whole sequence.
  * Write as: $s_1, ...,s_T = \underset{\vec{s}}{\operatorname{argmax}}  \, p(s_1, ...,s_T \mid x_1, ..., x_T, \pi, A, B)$
  * This estimation uses the ['Viterbi' algorithm](https://en.wikipedia.org/wiki/Viterbi_algorithm) which we won't cover here.
  * It's important to note that the state sequence is not the same as taking the most probable state from the state estimation. In the state sequence we are considering the global view which has the sequential dependence of the data in whereas the state estimate is a point estimate for a single time point $i$.

##### Learning the HMM

This was covered as a sketch in lectures and is something I'm not going to cover now but will make a few comments. Further reading is found in the purported classic tutorial [here](https://www.ece.ucsb.edu/Faculty/Rabiner/ece259/Reprints/tutorial%20on%20hmm%20and%20applications.pdf).

Our goal is to maximize $p(\vec{x} \mid \pi, A, B)$. We can write this as follows:

$$ p(\vec{x} \mid \pi, A, B) =  \sum_{s_1=1}^S ... \sum_{s_T=1}^S p(\vec{x}, s_1, ..., s_T \mid \pi, A, B) $$

In the above we note the LHS doesn't have the hidden states in which means they've been integrated out somehow. This means in the above we can write the LHS as dependent on the hidden states, summed over all possible values of them. This is a common trick in Bayesian analysis. Then using the Markov property we can factorise the RHS to obtain:

$$ p(\vec{x} \mid \pi, A, B) =  \sum_{s_1=1}^S ... \sum_{s_T=1}^S \, \prod_{i=1}^T p(x_i \mid s_i, B) \, p(s_i \mid s_{i-1}, \pi, A)$$

Note it is the same thing to maximize the log-likelihood, i.e.

$$\ln p(\vec{x} \mid \pi, A, B) = \ln \sum_{s_1=1}^S ... \sum_{s_T=1}^S \, \prod_{i=1}^T p(x_i \mid s_i, B) \, p(s_i \mid s_{i-1}, \pi, A)$$

A few comments:

* This is hard to maximize as it's in log-sum form.
* If we could learn $\vec{s}$ somehow we could remove the summations.
* We can actually calculate $p(\vec{s} \mid \vec{x}, \pi, A, B)$ though it's quite complicated.

All this (yes, really!) means we can use the EM algorithm to maximize $p(\vec{x} \mid \pi, A, B)$. We will not cover any more here but just say that this gives us ML estimates for $\pi, A$ and $B$.

<hr class="with-margin">
<h4 class="header" id="kalman">The Kalman filter</h4>

##### Introduction

The Kalman filter is an example of a continuous-state Markov model i.e. the states themselves live in a continuous space.

As per a HMM we observe a sequence $(x_1, x_2, x_3, ...)$ where each $x \in \mathbb{R}^d$ and our goal is to learn the state sequence $(s_1, s_2, s_3, ...)$. We assume all distributions are Gaussian:

$$p(s_{t+1} = s \mid s_t) = N(C s_t, Q)$$

and

$$p(x_{t} = x \mid s_t) = N(D s_t, V) $$

for some parameters $C, Q, D, V$.

Note that for the Kalman filter we assume that we know the parameters $C, Q, D, V$ in advance or they can be estimated in advance. For example they might be the means and variances of stocks (and the underlying latent factors) we are trying to model.

##### A quick contrast to HMM

Recall that for the HMM we wanted to learn parameters $\pi, A, B$. Recall that $A$ was a transition matrix among the discrete set of states and $B$ contained the state dependent probability distributions.

By design for the Kalman filter there is no need to learn $A$ and $B$.

We don't have to learn $B$ because each state will be unique since the state is in a continuous space with a continuous transition distribution. Thus the distribution on each $x_t$ is also going to be different as it depends on  $s_t$ which is also continuous valued and unique.

The same goes for $A$, each transition is to a brand new state and so every state at every time point is going to be something different, a new continuous valued random vector.

A final point on the differences to HMM: both $x_t$ and $s_t$ are constrained by the distributional assumptions which is what is going to make the problem of learning the underlying state possible.

##### What can we learn with the Kalman filter?

We are commonly interested in two separate posterior distributions:

* **Kalman filtering problem:** $p(s_t \mid x_1, ..., x_t)$
  * This is the distribution on the current state given a sequence of data that we've observed up until time $t$.
  * This problem is learning the continuously evolving distributions on the states in a real time setting.
* **Kalman smoothing problem:** $p(s_t \mid x_1, ..., x_T)$
  * This is the distribution on each latent state given all the data, including future data.
  * This is for problems where we have all the data in advance and would like to do some sort of post processing.

We will focus on the Kalman filtering problem.

##### The Kalman filter set-up

<blockquote class="tip">
<strong>Goal:</strong> learn the sequence of distributions $p(s_t \mid x_1, ..., x_t)$ given a sequence of data $(x_1, x_2, x_3, ...)$.
</blockquote>

The assumed model (as stated above) is:

$$p(s_{t+1} = s \mid s_t) = N(C s_t, Q)$$

and

$$p(x_{t} = x \mid s_t) = N(D s_t, V) $$

Further recall we assume we know in advance $C, Q, D$ and $V$.

###### Q: how do we even get started with this?
###### A: Bayes' rule ... as always!

We can write the sequence we wish to learn as:

$$p(s_t \mid x_1, ..., x_t) \propto \underbrace{p(x_t \mid s_t)}_\text{likelihood} \, \underbrace{p(s_t \mid x_1, ..., x_{t-1})}_\text{prior} $$

where we have used Bayes' rule to break up the term we wish to learn. Notice that what we have called the prior is also the conditional posterior distribution of the state at time $t$ given data up to time $t-1$. This prior is for $s_t$ with data up until $t-1$, we will now show how to rewrite this prior using a little bit of manoeuvring as:

$$ p(s_t \mid x_1, ..., x_{t-1}) = \int p(s_t, s_{t-1} \mid x_1, ..., x_{t-1}) \, ds_{t-1} $$

where we are simply adding in $s_{t-1}$ and then integrating it out in the above RHS. We can now write the RHS as:

$$ p(s_t \mid x_1, ..., x_{t-1}) = \int p(s_t \mid s_{t-1}) \, p(s_{t-1} \mid x_1, ..., x_{t-1}) \, ds_{t-1}$$

where we have used the fact that $p(a,b \mid c) = p(a \mid b, c) \, p(b \mid c)$ and also the fact that given $s_{t-1}$ we are conditionally independent of the sequence up to $x_{t-1}$.

Notice that the posterior term in the integral is now of the form we'd expect, $s_{t-1}$ with data up to $t-1$ whereas before we had $s_{t}$ with data up to $t-1$.

Going back to the original equation we now have:

$$p(s_t \mid x_1, ..., x_t) \propto \underbrace{p(x_t \mid s_t)}_\text{likelihood} \, \int \underbrace{p(s_t \mid s_{t-1})}_\text{prior} \, \underbrace{p(s_{t-1} \mid x_1, ..., x_{t-1})}_\text{posterior = unknown distn} \, ds_{t-1} $$

A few comments:
* The LHS is the posterior on $s_t$ and the RHS contains the posterior on $s_{t+1}$
* The likelihood is: $p(x_{t} = x \mid s_t) = N(D s_t, V) $
* The prior is: $p(s_{t} = s \mid s_{t-1}) = N(C s_{t-1}, Q)$
* We want the integral to be in closed form and evaluate to a known distribution.
* We want the prior and likelihood to lead to a known distribution.
* We want future calculations for $s_{t+1}$ to be easy.

<blockquote class="tip">
<strong>Spoiler alert:</strong> it turns out that we only need to define a Gaussian prior on the first state $s_0$ to keep everything nice (and Gaussian) at future time-steps.
</blockquote>

We don't cover the details here but basically once this is done all future calculations are in closed form. This pretty much wraps up our discussion on the Kalman filter, for more reading of applications or explanations see [here](https://math.stackexchange.com/questions/840662/an-explanation-of-the-kalman-filter), [here](https://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/) and interestingly [here](https://www.mathworks.com/videos/series/understanding-kalman-filters.html).

<hr class="with-margin">
<h4 class="header" id="model_comp">Some model comparisons</h4>

Over the last few weeks we have discussed GMMs, HMMs, probabilistic PCA and the Kalman filter. Aside from some dubious acronyms, is there anything we can say to link them?

<p align="center">
    <img src="/assets/img/model_comp.jpg" alt="Image" width="500" height="700" />
</p>

<em class="figure">Fig. 2: Linking the latent variable models.</em>

More simply we can summarise as:

* Sequential, discrete latent states: HMM
* Sequential, continuous latent states: Kalman filter
* Non-sequential, discrete latent states: GMMs
* Non-sequential, continuous latent states: Probabilistic PCA

For all the above models any continuous distributions are Gaussian.

<hr class="with-margin">
<h4 class="header" id="unclear">Things I'm unclear on (or outstanding questions)</h4>

To be updated

<hr class="with-margin">
<h4 class="header" id="textbooks">What did the textbooks say?</h4>

To be updated
