---
layout: post
title: Gaussian Processes
date: 2018-10-22
use_math: true
['gaussian_processes', 'kernels']
image: "gp.png"
---

**Note: still a work in progress**

In a loose sense Gaussian Processes can be thought of as a probabilistic non-parametric form of non-linear regression.

<!--more-->
<hr class="with-margin">

<div class="list-of-contents">
  <h4>Contents</h4>
  <ul></ul>
</div>

<hr class="with-margin">
<h4 class="header" id="intro">Introducing Gaussian Processes (GPs)</h4>

Something I don't usually do (but in line with the fact I'm not wanting to explain everything from scratch but just give my take on things): preliminary reading [here](http://katbailey.github.io/post/gaussian-processes-for-dummies/), [here](http://platypusinnovation.blogspot.com/2016/05/a-simple-intro-to-gaussian-processes.html) and [here](http://keyonvafa.com/gp-tutorial/). There are also good lectures [here](https://www.youtube.com/watch?v=4vGiHC35j9s&list=PLE6Wd9FR--EdyJ5lbFl8UuGjecvVw66F6&t=0s&index=9) from none other than Nando de Freitas that cover Gaussian Processes a lot more thoroughly than we did in lectures.

<hr class="with-margin">
<h4 class="header" id="intro">First attempt at an explanation</h4>

<blockquote class="tip">
<strong>Warning:</strong> This requires the above reading to have been done.
</blockquote>

We specify some prior on our data by defining a kernel function which tells us how similar two vectors in our input space are (i.e. how similar are two rows of our design matrix $X$ which are in $\mathbb{R}^d)$.
  * **Point of confusion 1:** all of the beginner tutorials for Gaussian Processes usually use a 1d example and non-linear regression and so it's easy to see when two points are near each other in $x$ space. But the $x$-axis could equally represent how close together our points are after they've come out of the kernel function. i.e. the $x$-axis now tells us how similar these points are in their $d$-dimensional space. We can have $n$ of these points (the rows in our dataset).

So our kernel (which we haven't yet defined) computes the similarity between each of our $n$ data points and thus returns a $n$ by $n$ matrix. We then define an $n$ dimensional Gaussian as $f \sim N(\mu_X, K_{XX})$ which is the prior probabilistic model of how our data is generated. $\mu_X$ is the mean function of each data point and is often assumed to be 0 for reasons not discussed here. Note $K_{XX}$ is a $n$ by $n$ matrix.

<blockquote class="tip">
<strong>Observation:</strong> we can now simulate functions from $f$
</blockquote>



At this point you probably hear phrases like "Gaussian Processes are simply an infinite-dimensional distribution over functions" which do nothing to aid understanding. "Ah yes" you say, "a distribution over functions, why I do that all the time".

The reason this phrase crops up is because theoretically we could have an infinite about of data points (we don't, we have a finite subset $n$) and we can sample from this prior $f \sim N(\mu_X, K_{XX})$ which turns out to be defining a distribution over the possible functions which define our data. We could start thinking of the range of output values of $K_{XX}$ as the domain of the $x$-axis of the function $f$. In other words if we pretended we were in a simple 1D regression setting the $x$-axis just corresponds to how close points are in input space and the height on the $y$-axis is $f$. We of course are interested in output points $f_{s}$ which are the outputs of the function at new data points $x_{s}$.

If you've read the above links this hopefully should be making sense. Either way we are going to walk through an example, with code and maths as I find examples with only one of them not clear.

<hr class="with-margin">
<h4 class="header" id="example">GP example warm-up</h4>

We are going to walk through a classic example presented for Gaussian Processes and make a few observations along the way. We will be loosely following the excellent example [here](http://katbailey.github.io/post/gaussian-processes-for-dummies/) which is also linked above - please ensure this has been read first before continuing.

We start with a formulation that specifies the posterior (joint probability of the outcomes) of our function $f$ of interest where entries marked with a subscript $s$ denote new data.

$$

\begin{pmatrix}
f \\
f_{s}
\end{pmatrix}

\sim \mathcal{N}{\left(
\begin{pmatrix}
\mu \\
\mu_{s}
\end{pmatrix}
,
\begin{pmatrix}
K & K_{s}\\
K_{s}^T & K_{ss}\\
\end{pmatrix}
\right)}

$$

<hr class="with-margin">
* **Side note:** There is an important result for multivariate Gaussians that says that when you have two sets of variables that are jointly Gaussian (here $f$ and $f_s$) then the conditional distribution of one on the other is again Gaussian, here, $p(f_s \mid f)$.
<hr class="with-margin">

<blockquote class="tip">
<strong>Key point:</strong> after some work we can go from $p(f_s, f)$ to $p(f_s \mid f)$ which means we can specify our posterior as a Gaussian $f_s \sim N(\mu_s, \Sigma_s)$ and the expressions for $\mu_s$ and $\Sigma_s$ we are able to calculate (we will see this in the code).
</blockquote>
<hr class="with-margin">
<h4 class="header" id="small_example">Small GP example</h4>
We will now walk through a small example detailing both the maths and the code:

<hr class="with-margin">
Recipe for GP example (before we have observed any data):
* Define the kernel function $\textbf{k}$ which will output matrices we call $K$, $K_s$ or $K_{ss}$ depending on which data points we are computing similarity between (e.g. train to train, train to test, test to test).
* Pick the domain we are interested in $X_{test}$, here $-5 \leq x \leq 5$ and compute the kernel between all evenly spaced points in this region.
  * Note as $X_{test}$ is evenly spaced the kernel will only have strong values down the diagonal as we are saying further points are less related.
* Sample from $f_s \sim N(\mu_s, \Sigma_s)$ to see what the functions look like. To actually sample we actually use $f_s \sim \mu_{s} + L \, N(0, I)$ with $LL^T = K_{ss}$ which involves using a Cholesky decomposion on $K_{ss}$.
  * Note this is really just using the fact that $x \sim N(\mu, \sigma^2)$ can be written as $x \sim \mu + \sigma N(0, 1)$ except we have a matrix instead of $\sigma$ and so taking the 'square root of a matrix' is what the Cholesky decomposition does, loosely.
  * We assume in this example that $\mu_s$ = 0.

<hr class="with-margin">

Example python code of the small GP example:
<pre><code class="language-python">import numpy as np
import matplotlib.pyplot as plt

def kernel(a, b):
    """ GP squared exponential kernel i.e. our distance/similarity function"""
    kernelParameter = 0.1
    sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    return np.exp(-.5 * (1/kernelParameter) * sqdist)

n = 50  # number of test points
Xtest = np.linspace(-5, 5, n).reshape(-1, 1)  # test points
K_ss = kernel(Xtest, Xtest)  # kernel at test points
# Draw samples from the prior at our test points:
L_prior = np.linalg.cholesky(K_ss + 1e-6*np.eye(n))   # K = LL^T and L is (n, n)
f_prior = np.dot(L_prior, np.random.normal(size=(n, 3)))  # Sample 3 points per f_s ~ LN(0, I)
plt.plot(Xtest, f_prior) # these are now samples from the GP prior
</code></pre>


Plotting this we get a look at our prior $p(f_s)$:

<p align="center">
    <img src="/assets/img/gprior_simple.png" alt="Image" width="350" height="250" />
</p>

<em class="figure">Fig. 1: Gaussian Prior samples</em>
<hr class="with-margin">

##### Comments on what we just did

* We say we are 'sampling functions' as we could technically compute each sample at more and more points in the $x$ domain instead of the 50 we did here.
* All the functions are different but they all come from the same distribution whose covariance (dictated by our choice of kernel) controls the smoothness of these functions (i.e. how far apart is the function value for two input points that are close by allowed to be).
* We call this the prior even though it's over what we termed the $X_{test}$ domain as it's our domain of interest which we wish to predict for. We thus view the sample of functions over this domain before seeing any data to get a sense of what they look like.
* When we then see some data this will reduce the set of functions from $p(f_s)$ to the posterior $p(f_s \mid f) = p(f_s \mid X_{test}, X, y)$ .

<hr class="with-margin">
<h4 class="header" id="extended_example">Extended GP example</h4>

Let's see some data now and see what happens.


<hr class="with-margin">
Recipe for GP example (after observing data):

* As for the smaller example we can calculate our prior and sample from it
* We now create some training data from an underlying function and add some noise to it (as we are pretending we observe a noisy version of it)
* We can then calculate the relevant values in order to obtain our posterior distribution which we can sample from. In particular we have that:

$$p(f_s \mid X_s, X, y) = N(f_s \mid \mu_s, \Sigma_s)$$

$$\mu_s = K_s^T K_y^{-1}y$$

$$\Sigma_s = K_{ss} - K_s^T K_y^{-1}K_s$$

where we are calling $K_y = K + noise$ (i.e. the kernel of the noisy data we see).

<hr class="with-margin">

Example python code of the extended GP example (excluding plotting code):

<pre><code class="language-python">import numpy as np
# This is the true unknown function we are trying to approximate
f = lambda x: np.sin(0.9*x).flatten()

# Inputs
N = 10         # number of observed training points
n = 50         # number of test points to predict for
s = 0.0005     # noise variance to create data with

# 1. As before create data for domain we're going to make predictions at (i.e. evenly spaced) and sample:
Xtest = np.linspace(-5, 5, n).reshape(-1,1)
K_ss = kernel(Xtest, Xtest)
L_prior = np.linalg.cholesky(K_ss + 1e-6*np.eye(n))
f_prior = np.dot(L_prior, np.random.normal(size=(n, 3)))

# 2. Observe some data get the noisy versions of the function evaluated at these points (this is our training data)
X = np.random.uniform(-5, 5, size=(N, 1))  # (10, 1)
y = f(X) + s*np.random.randn(N) # (10, 1)
K = kernel(X, X)  # N by N matrix of distances for training data
L = np.linalg.cholesky(K + s*np.eye(N))  # K = LL^T

# 3. We also need to compute the distance between the train and test data
alpha = np.linalg.solve(np.dot(L, L.T), y)
K_s = kernel(X, Xtest)

# 4. Get the mean and variance over whole test domain
mu = np.dot(K_s.T, alpha)
Lk = np.linalg.solve(L, K_s)
s_var = np.diag(K_ss) - np.sum(Lk**2, axis=0)
stdev = np.sqrt(s_var)

# 5. If we want, draw samples from the posterior within test domain
L_post = np.linalg.cholesky(K_ss + 1e-6*np.eye(n) - np.dot(Lk.T, Lk))
f_post = mu.reshape(-1,1) + np.dot(L_post, np.random.normal(size=(n, 3)))
</code></pre>

Plotting this we get the following:

<p align="center">
    <img src="/assets/img/predict.png" alt="Image" width="350" height="250" />
</p>

<em class="figure">Fig. 2: True function (blue), fitted function (red dash), noisy observed data (red cross) and 3*standard deviation (grey)</em>

<p align="center">
    <img src="/assets/img/post.png" alt="Image" width="350" height="250" />
</p>

<em class="figure">Fig. 3: Gaussian Posterior samples</em>
<hr class="with-margin">

##### A few comments on the above charts

* The uncertainty bands are wide when we have no observed data. In fact they are so wide even when we have data and actually look a bit silly. This is because the prior we specified was very flexible i.e. we chose a kernel that allowed functions which are not very smooth (relative to the underlying data generating process, a sine wave). We can change the kernel parameter to fix this, see figure 4 below.

<p align="center">
    <img src="/assets/img/predict_smooth.png" alt="Image" width="350" height="250" />
</p>

<em class="figure">Fig. 4: Gaussian Posterior with smoother kernel parameter (different data)</em>

* This looks much better as we are now more confident in regions where we have seen more data. Note that even if we specified a poor prior as we kept seeing more data it would eventually overwhelm our poor assumption. That said, GPs are powerful when we are able to make strong prior choices about the data generating process and thus can perform well with data paucity if we get this assumption correct.
* In figure 3 above when we sampled from the posterior what we are essentially doing is now drawing samples from the updated distribution in light of the data we have observed. As we can the samples now look a little closer the data we observed.

<hr class="with-margin">
<h4 class="header" id="example">Further thoughts</h4>

##### What is the kernel?

The kernel is actually the crucial thing that determines what sort of functions we end up with. It basically controls the smoothness and type of functions we can get from our prior. We are not going to discuss kernels in this post as it's already too long.

How do I choose the kernel? Read [this](http://www.cs.toronto.edu/~duvenaud/cookbook/index.html).

<hr class="with-margin">
<h4 class="header" id="further">Further Reading</h4>

The main source I used was [Machine Learning: A Probabilistic Perspective](https://www.amazon.co.uk/Machine-Learning-Probabilistic-Perspective-Computation/dp/0262018020) by Kevin Murphy which I found more accessible than the authority on the subject: [Rasmussen and Williams](http://www.gaussianprocess.org/gpml/).


For more random internet links see [here](http://people.ee.duke.edu/~lcarin/David1.27.06.pdf), [here](https://www.linkedin.com/pulse/machine-learning-intuition-gaussian-processing-chen-yang) and [here](https://www.eurandom.tue.nl/events/workshops/2010/YESIV/Prog-Abstr_files/Ghahramani-lecture2.pdf).
