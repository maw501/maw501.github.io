---
layout: post
title: ColumbiaX - ML - week 12
date: 2018-12-01
use_math: true
tags: ['association_analysis', 'model_selection']
image: "assoc_analysis.jpg"
comments: true
---
In week 12 we round off the course by looking at association analysis which can be used to understand the co-occurrence of items in a shopping basket in an efficient manner. We also look at model selection through the AIC and BIC metrics.

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
<h4 class="header" id="intro">Week 12 (lectures 23 and 24): overview</h4>

This is the final week of the course and with the final exam next week I won't be writing anything too lengthy - this is further compounded by the fact that I didn't find this week's topics particularly exciting.

A summary of the two main topics

* **Association Analysis:** this is simply working out which items co-occur together in, say, grocery store purchases. Whilst this is a simple idea (which doesn't really feel like machine learning, more data mining) and can be done exactly for small datasets we need a smarter algorithm to understand these patterns in big datasets. The algorithm used is called the Apriori algorithm.
* **Model selection:** in model selection we wish to consider the complexity of the model used in order to better select the correct complexity, assuming the appropriate model is chosen originally. The AIC and BIC metrics essentially add in penalty terms to the objective function based on the number of degrees of freedom (i.e. parameters) for a given model.

<hr class="with-margin">
<h4 class="header" id="assoc">Association Analysis</h4>

##### Problem description

Association analysis is a simple idea often used to find purchasing patterns in transaction data. For example, analysis of goods bought together may reveal that consumers who buy bread also buy milk with it and so the store may wish to construct some offer involving those two items.

Actually calculating this is as simple as counting all the transactions where someone bought bread and milk together out of all $N$ transactions for the store.

Let's introduce some terminology.

Call $p$ the number of different store items and $N$ the number of transactions for a given store. We call $X_n$ the items purchased by a customer in transaction $n$. We wish to find common subsets of goods, $K$, that occur together often. For example, if bread and milk occur together 50 times out of 500 transactions then we can say $p(K) = 0.1$ for $K$ equal to the set containing bread and milk.

<blockquote class="tip">
<strong>Goal:</strong> We would like to find all such subsets $K$ such that $p(K)$ is large.
</blockquote>

##### The combinatorial problem

Unfortunately it's not as simple as being able to count all the subsets in a naive manner. The number of sets of size $k$ out of $p$ items is:

$${p \choose k} = \dfrac{p!}{k!\,(p-k)!} $$

So if a store has 10,000 different items and wishes to look for all subsets up to size 5 this evaluates to roughly $10^{18}$ sets to check for in each of our $N$ transactions.

So whilst this is just a counting problem, we need a way to know which of the $K$ subsets to check for and which we can ignore (as we can't check for them all).

##### Some metrics we might care about

Let's quickly detail some metrics we might be interested in using.

* $p(K) = p(A, B)$ is the **prevalence** of items $A$ and $B$ co-occuring together.
* $p(B \mid A) = \dfrac{p(K)}{p(A)}$ is the **confidence** that $B$ occurs in a basket given $A$ is in the basket. We can use this information to design new offers.
* Call $L(A, B) = \dfrac{(B \mid A)}{p(B)}$ the **lift**. It is how much more confident we are in seeing $B$ in a basket given we have seen $A$ in the basket.

##### The Apriori algorithm - introduction

The goal of the Apriori algorithm is to quickly find all subsets $K$ that have a probability of occurring greater than some threshold $t$. i.e. a given $K$ will occur at least $tN$ times in the $N$ baskets/transactions.

The algorithm hinges on two key properties of set counting that are as follows, for a threshold $0 < t < 1$:

* If an subset doesn't occur frequently enough then bigger subsets containing that subset also won't occur often enough and so we don't need to bother checking them.
  * For example: if no-one has bought cottage cheese then we don't need to check the subsets of cottage cheese with all other items.
* On the flip side, if a big subset occurs often we don't need to bother checking all the smaller subsets.
  * For example: if lots of people bought bread and milk together (i.e. greater than $t$ people) then we don't also need to check individually for bread purchases or milk purchases.
    * Note a basket containing bread and milk counts as a hit for the 3 subsets of \{bread\}, \{milk\} and the joint set of \{bread, milk\}. So clearly the subset \{bread\} will occur more often than \{bread, milk\} as for bread sometimes people may have bought it without milk.

##### The Apriori algorithm

We can give a basic version of the algorithm as follows:

<hr class="with-margin">
Start with a threshold $0 < t < 1$ that is small:

* Check all sets of size 1:
  * Keep all items that appear more than $tN$ times.
* Check all pairs from the survivors of previous step and keep the pairs that appear more than $tN$ times
* ...
* $k$th step: check all sets of size $k$ using surviving sets of size $k-1$ from step $k-1$

Note we keep all sets of any size that appear in more than $tN$ baskets. As $k$ increases the number of survivors at each step will decrease and at some point no sets will remain and we can stop.

<hr class="with-margin">
<h4 class="header" id="model">Model Selection</h4>

##### The model selection problem

The general problem associated with model selection is how to decide how complex we wish to make the model. This is usually done via cross-validation but here we look at an alternative.

Recall that by adding more parameters into a model we increase the risk of over-fitting and that we can loosely say the number of degrees of freedom is equal to the number of model parameters, $K$.

The two approaches we discuss both add a penalty term to our objective function based on $K$. We define them as follows, for a function $L$ we wish to maximize we will minimize $-L$:

* **Akaike information criterion (AIC):** $-L + K$
* **Bayesian information criterion (BIC):** $-L + \dfrac{1}{2}K \ln N$

Note that when $\dfrac{1}{2} \ln N > 1$ BIC encourages a simpler model than AIC (this happens when $N \geq 8$).

We can stop there...there is nothing more algorithmically to do with these two methods, we simply add the penalty term on.

The rest of the lecture is spent deriving BIC and showing it actually has a sound theoretical basis. We will not regurgitate this derivation here.

<hr class="with-margin">
<h4 class="header" id="unclear">Things I'm unclear on (or outstanding questions)</h4>

To be updated

<hr class="with-margin">
<h4 class="header" id="textbooks">What did the textbooks say?</h4>

To be updated
