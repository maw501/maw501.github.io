---
layout: post
title: ColumbiaX - ML - week 10
date: 2018-11-22
use_math: true
tags: ['pca', 'markov_models']
image: "pca.png"
comments: true
---
In week 10 we look at a stalwart in the statistical learning tool-kit, PCA. We do however add a few tricks to our arsenal by considering extensions in the form of both probabilistic and kernel PCA before switching to look at sequential data and Markov chains.

<!--more-->
<hr class="with-margin">
This page is a summary of my notes for the above course, link [here](https://www.edx.org/course/machine-learning-columbiax-csmm-102x-4).

This is my first attempt at publishing course notes and I have no intention to make it comprehensive but rather to highlight the bits I feel are important and maybe explain some of the things I found a little trickier (or weren't explained to my taste). Understanding is deeply personal though if you spot any errors or have any questions please feel free to drop me an email.

PCA cover image from [here](https://chrisalbon.com/machine_learning/feature_engineering/dimensionality_reduction_with_pca/).

<hr class="with-margin">
<div class="list-of-contents">
  <h4>Contents</h4>
  <ul></ul>
</div>

<hr class="with-margin">
<h4 class="header" id="intro">Week 10 (lectures 19 and 20): overview</h4>

We started the week by looking at PCA which is something I've used extensively in the past but we go beyond 'classical' PCA to look at probabilistic and kernel PCA which are generalizations of classical PCA. We will thus not spend a great deal of time on PCA as it's a well-explained concept on various other sites that I'm not going to attempt to do a better job of explaining.

Quick warning: if you aren't comfortable with the kernel trick it's advisable to read about it before we start. A starting point is [here](/../posts/2018/10/19/The-kernel-trick).

* **Principal Components Analysis (PCA):**
Classical PCA is an unsupervised dimensionality reduction technique. Given the popularity of PCA there are some great explanations online, see [here](https://stats.stackexchange.com/questions/2691/making-sense-of-principal-component-analysis-eigenvectors-eigenvalues) and [here](https://georgemdallas.wordpress.com/2013/10/30/principal-component-analysis-4-dummies-eigenvectors-eigenvalues-and-dimension-reduction/).
  * **Probabilistic PCA:** is a reformulation of classical PCA as the maximum likelihood (ML) solution of a probabilistic latent variable model. What this means is that we can think of probabilistic PCA in terms of specifying a generative process for our data and then using ML to solve this probabilistic model. It essentially boils down to solving a matrix factorization problem for our data $X$ of the form $X \approx WZ$ and we use the EM algorithm to solve it. The nice thing about reformulating the problem in a probabilistic manner is that we get extra benefits such as being able to handle missing data, having distributions that can characterize uncertainty in predictions amongst other things.
  * **Kernel PCA:** is another generalization of PCA whereby we take an algorithm that uses dot products and then generalize this by specifying a non-linear kernel instead of the original dot product. This is essentially akin to performing the kernel trick to project each data vector into a higher dimensional space and then performing classical PCA there.
* **Markov chains:** are an approach to dealing with sequential data where we make the simplifying assumption that the distribution of possible outcomes for the next time step only depends on the current position - this is called a first-order Markov chain. In this way we can think of our time-series process transitioning from state to state governed by a probability transition matrix. There is nice background reading [here](https://brilliant.org/wiki/markov-chains/) and [here](http://setosa.io/ev/markov-chains/).

<hr class="with-margin">
<h4 class="header" id="pca">PCA</h4>

PCA is a well-covered technique so I will just state one definition (there are many equivalent definitions).

##### General form of PCA

The general form of PCA is to solve for $x_i \in \mathbb{R}^d$:

$$q = \underset{q}{\operatorname{argmin}} \sum_{i=1}^n \| x_i - \underbrace{\sum_{k=1}^K (x_i^T q_k)q_k}_\text{this approximates $x$} \|^2 $$

such that $q_k^T q_{k'} = 1$ if $k=k'$ and 0 otherwise.

A few comments on the above formulation:

* $q_k^T x_i = x_i^Tq_k$ is a scalar which represents projecting each $x_i$ data onto the line spanned by the vector $q_k$. In other words it's the extent to which a particular $x_i$ points in the direction of the vector $q_k.$ Refresher [here](https://math.oregonstate.edu/home/programs/undergrad/CalculusQuestStudyGuides/vcalc/dotprod/dotprod.html).
* Thus $(x_i^T q_k)q_k$ is a vector 'stretched' in the direction of $q_k$.
* We want to minimize the difference between our original data $x_i$ and the vectors $q_k$ that we are going to use to approximate each $x_i$ with.
* We will have $K$ such vectors $q_k$ which we sum over.
* These vectors $q_k$ are the eigenvectors of the data matrix $XX^T$ which has dimensions $d$ x $d$.
  * Note: in this example we have our data matrix $X$ with dimensions $d$ by $n$ and not $n$ by $d$ as usual.
* Essentially what we are doing is projecting our $d$ dimensional data down to a $K$ dimensional subspace which approximates the data. We do this with the condition that each $q_k$ is orthogonal to every other $q_k$ and has unit length. The above definition means that the first $q_k$ corresponds to the first eigenvalue and is the vector that explains most of the variance in the data.

<hr class="with-margin">
<h4 class="header" id="probpca">Probabilistic PCA</h4>

Probabilistic PCA feels like somewhat of a rabbit-hole that contains several technical details which we are not going to have time to cover here (and were not fully covered in the lecture) except to give a sketch overview. One of the authors of the original probabilistic PCA paper is Chris Bishop (author of one of the [texts](https://www.amazon.co.uk/Pattern-Recognition-Learning-Information-Statistics/dp/0387310738) for this course) - which might give some hint as to why this crept into the syllabus. If you want to read more about it you can see Chapter 12 of Bishop's book or the original paper [here](http://www.robots.ox.ac.uk/~cvrg/hilary2006/ppca.pdf).

Probabilistic PCA is closely tied to matrix factorization and we will quickly present this view first before stating the generative process.

Recall that with the SVD we can decompose any matrix $X$ into $X = USV^T$. In probabilistic PCA we will approximate $X \approx WZ$ with the following observations:

* $W$ is a $d$ x $K$ matrix which is sometimes called a factor loadings matrix which is similar to the eigenvectors but isn't orthonormal.
* The $i$th column of $Z$ is called $z_i \in \mathbb{R}^k$. This is essentially a low-dimensional representation of $x_i$ - see Fig. 1.

<p align="center">
    <img src="/assets/img/prob-pca.jpg" alt="Image" width="700" height="400" />
</p>

<em class="figure">Fig. 1: The matrix factorization view of probabilistic PCA.</em>

##### The generative process for probabilistic PCA

The generative process of probabilistic PCA is:

$$x_i \sim N(Wz_i, \sigma^2 I)$$

with $z_i \sim N(0, I)$.

Here we don't know what $W$ or $z_i$ are and our goal will be to learn them. Some comments on the above generative model:

* $z_i$ are latent variables which we don't observe and when we perform ML to find $W$ we will try to 'integrate them out'. Each $z_i$ is a $K$ dimensional Gaussian.
* It turns out we can't actually solve the above generative model using ML as it's intractable analytically.
* Instead we set-up the EM algorithm where we re-introduce the $z_i$ latent vectors in order to make the problem easier. This is a quirk of EM we discussed in week 8, [here](../../../2018/11/08/columbiaX-ML-week8), whereby introducing latent variables can actually make the calculations easier.

##### EM for probabilistic PCA

We will cover this very briefly. The joint marginal log-likelihood is:

$$ \sum_{i=1}^n \ln \int p(x_i, z_i \mid W) \, dz_i = \underbrace{\sum_{i=1}^n \int q(z_i) \ln \dfrac{p( x_i, z_i \mid W)}{q(z_i)} \, dz_i \,}_\text{call this term $\mathcal{L}$}  + \, \underbrace{ \sum_{i=1}^n \int q(z_i) \ln \dfrac{q(z_i)}{p( z_i \mid x_i, W)} \, dz_i}_\text{this is a KL divergence term} $$

A few comments:

* This is the same general form we had for the EM algorithm in week 8.
* We sum over all data-points $i$ to get the joint likelihood.
* On the LHS we have re-introduced $z_i$ and are then integrating it out - we include on the LHS to make clearer what we wish to do.
* We need to be able to calculate the posterior distribution term, $p(z_i \mid x_i, W)$. In this case we are able to do this.
* We will output a point estimate of $W$ and the above posterior distribution on each $z_i$.
* In this framework we can actually learn $K$ and $\sigma^2$ as well.
* Maximizing the $\mathcal{L}$ term with respect to $W$ needs to be easy otherwise we are still in a pickle. It turns out there is an analytic formula for $W$ to maximize $\mathcal{L}$ as part of the M-step which we couldn't obtain before we set up the problem as part of the EM algorithm.

<hr class="with-margin">
<h4 class="header" id="kernpca">Kernel PCA</h4>

Again, we will just give an overview. Recall that in classical PCA we find the eigenvectors of the matrix $X X^T$, this can equivalently be written in terms of outer products of our data vectors $x_i$ as:

$$ \sum_{i=1}^n x_i x_i^T = X X^T $$

And we wanted to solve the eigendecomposition:

$$X X^T q_k = \lambda_k q_k$$

to find the eigenvectors $q_k$. Now we wish to solve:

$$\sum_{i=1}^n \phi(x_i) \phi(x_i)^T =\lambda_k q_k$$

where $\phi(x)$ is some feature mapping (non-linear transformation) from $d$ dimensions to $D$ dimensions where $D \gg d$.

How we actually go about this involves a few sleights of hand to rewrite things and we will omit the details - you can refer to Bishop's book in Chapter 12 if you want them.

##### Comments on kernel PCA
* Using kernel PCA allows us to find linear projections onto the principal components of the transformed data that corresponds to non-linear projections in the original feature space.
* In terms of clustering this means that points that cannot be linearly separated in $d$ dimensions can almost always be linearly separated in the much higher dimensional space we will project to.
* By the kernel trick it turns out we can compute the kernel matrix of our data points instead of actually having to visit this high (potentially infinite) dimensional space.
* The price we pay for this is having to find the eigenvectors of an $n$ by $n$ matrix instead of a $d$ by $d$ matrix for classical PCA. Thus for large data sets we cannot solve this exactly.
* [Wikipedia](https://en.wikipedia.org/wiki/Kernel_principal_component_analysis) is actually fine for further reading.

<hr class="with-margin">
<h4 class="header" id="kernpca">Markov Chains</h4>

We switch to think about sequential data now.

A first order Markov chain is a sequence where we make the assumption that the distribution of possible (discrete) states at the next time step only depends on the current time step's state. In this way if we have $S$ different states overall we can define a matrix $M$ which is $S$ x $S$ such that:

$$M_{ij} = p(s_t = j \mid s_{t-1} = i)$$

That is, the $(ij)$th entry of the matrix $M$ gives the probability of transitioning from state $i$ at time $t-1$ to state $t$ at time $j$. Each row of this matrix sums to 1 and represents a discrete probability distribution.

We can generate a Markov sequence by sampling from the matrix $M$ given a starting state $s_0$. The starting state can be modelled separately with its own distribution.

##### Approximating the transition matrix $M$

If we have an observed sequence we can approximate the transition matrix $M$ using maximum likelihood (ML). This is as simple as counting the empirical number of times in the data we have witnessed a transition from state $i$ to state $j$ and dividing by the number of transitions from state $i$. We do this for all $i$ and $j$ to get the matrix $M$.

##### Estimating which state we'll be in at time $t+1$

It's interesting to ask at the beginning of the sequence if we can say which state we'll be in at step $t+1$.

We can approach this as follows:

* At time step $t$ we will have a probability distribution on which state we're in which we can call $p(s_t = i)$. Then the distribution on $s_{t+1}$ is:

$$p(s_{t+1} = j) = \sum_{i=1}^S p (s_{t+1} = j \mid s_t = i ) p(s_t = i) $$

* $p(s_t = i)$ is a row vector in $M$ (the $i$ th row) and we can call this the state distribution $w_t$. We can thus write the above as:

$$ w_{t+1}(j) = \sum_{i=1}^S M_{ij} w_t(i)$$

* This can be written as $w_{t+1} = w_tM$. This is just a probability weighted calculation summed over all states. See Fig 2 for a simple example.

<p align="center">
    <img src="/assets/img/markov_chain.jpg" alt="Image" width="500" height="700" />
</p>

<em class="figure">Fig. 2: Example of calculating the state distribution for a Markov chain with 2 states.</em>

So given the current state distribution $w_t$ we can calculate the distribution on the next state as $w_{t+1} = w_tM$.

##### Stationary distribution

What if we keep going and project out an infinite number of steps? Can we say anything about $w_{\infty}$ which is the state distribution after an infinite number of steps?

It turns out that if our state transitions obey two simple rules then $w_{\infty}$ will be the same distribution vector for every $w_0$. The properties are:

* We can eventually reach any state starting from any other state (i.e. there are no closed or one way lanes in our graph)
* The is no deterministic looping on the graph (i.e. we don't enter a cycle of state transitions where we end up at a pre-defined end point with certainty)

##### Semi-supervised learning

I was going to write more about this but it turns our someone has beaten me to it, [here](https://www.datasciencecentral.com/profiles/blogs/a-semi-supervised-classification-algorithm-using-markov-chain-and), using the same example from a previous running of this course.

I'll just comment on the general idea which is that in a situation with little labelled data and lots of unlabelled data we would like to use the structure in unlabelled data to help classify the unlabelled data. We can use a Markov chain to help with this process.

##### Note
The Markov property is not to be confused with the iid assumption we have made about our data in the past. The probability of a sequence is:

$$ p(s_1, ..., s_t) = p(s_1) \prod_{u=2}^t p(s_u \mid s_{u-1})$$

which is different from our iid assumption of:

$$ p(s_1, ..., s_t) = \prod_{u=1}^t p(s_u )$$

<hr class="with-margin">
<h4 class="header" id="unclear">Things I'm unclear on (or outstanding questions)</h4>

I definitely felt the topics of probabilistic and kernel PCA have more to them technically but given they are pretty niche in terms of application I haven't dwelt on them. That said they are perhaps worth a post on there own at some point to dive into some of the more technical details. In particular I would say I haven't fully had a chance to get my head around probabilistic PCA yet.

<hr class="with-margin">
<h4 class="header" id="textbooks">What did the textbooks say?</h4>

Most of the stuff we covered on PCA was straight out of Bishop Chapter 12 and this should be consulted for more details if they are sought.
