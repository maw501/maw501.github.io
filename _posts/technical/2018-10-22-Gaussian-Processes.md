---
layout: post
title: Gaussian Processes
date: 2018-10-22
use_math: true
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
<h4 class="header" id="intro">Attempt at an explanation 1</h4>

<blockquote class="tip">
<strong>Warning:</strong> This requires the above reading to have been done.
</blockquote>

We specify some prior on our data by defining a kernel function which tells us how similar two vectors in our input space are (i.e. how similar are two rows of our design matrix $X$ which are in $\mathbb{R}^d)$.
  * Point of confusion 1: all of the beginner tutorials for Gaussian Processes usually use a 1d example and non-linear regression and so it's easy to see when two points are near each other in $x$ space. But the $x$-axis could equally represent how close together our points are after they've come out of the kernel function. i.e. the $x$-axis now tells us how similar these points are in their $d$-dimensional space. We can have $n$ of these points (the rows in our dataset).

So our kernel (which we haven't yet defined) computes the similarity between each of our $n$ data points and thus returns a $n$ by $n$ matrix. We then define an $n$ dimensional Gaussian as $f \sim N(\mu_X, K_{XX})$ which is the prior probabilistic model of how our data is generated. $\mu_X$ is the mean function of each data point and is often assumed to be 0 for reasons not discussed here. Note $K_{XX}$ is a $n$ by $n$ matrix.

**Observation: we can now simulate functions from $f$**



At this point you probably hear phrases like "Gaussian Processes are simply an infinite-dimensional distribution over functions" which do nothing to aid understanding. "Ah yes" you say, "a distribution over functions, why I do that all the time".

The reason this phrase crops up is because theoretically we could have an infinite about of data points (we don't, we have a finite subset $n$) and we can sample from this prior $f \sim N(\mu_X, K_{XX})$ which turns out to be defining a distribution over the possible functions which define our data. We could start thinking of the range of output values of $K_{XX}$ as the domain of the $x$-axis of the function $f$. In other words if we pretended we were in a simple 1D regression setting the $x$-axis just corresponds to how close points are in input space and the height on the $y$-axis is $f$. If you've read the above links this hopefully should be making sense.

Each sample from our prior resides in $n$ dimensional space...

**TBC**

##### What is the kernel?

<blockquote class="tip">
<strong>Warning:</strong> Incomplete section
</blockquote>

The kernel is actually the crucial thing that determines what sort of functions we end up with. It basically controls the smoothness and type of functions we can get from our prior. How do I choose the kernel? Read [this](http://www.cs.toronto.edu/~duvenaud/cookbook/index.html).

Example python code of a kernel:
```python
import numpy as np

def kernel(a, b):
    # GP squared exponential kernel i.e. our distance function
    kernelParameter = 0.1
    sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    return np.exp(-.5 * (1/kernelParameter) * sqdist)
```

##### Right, so I think I get the kernel/distribution over function thing, now what?

Well the point is that after we have specified our prior and then have seen some data we can sample from our posterior and see what our functions values look like probabilistically. Furthermore, when we see new data we can define a probability distribution over the set of possible values we think the output will take. This is like forming a predictive distribution which we have seen before.

**Gaussian Processes section TBC**

<hr class="with-margin">
<h4 class="header" id="further">Further Reading</h4>

More links [here](http://people.ee.duke.edu/~lcarin/David1.27.06.pdf), [here](https://www.linkedin.com/pulse/machine-learning-intuition-gaussian-processing-chen-yang) and [here](https://www.eurandom.tue.nl/events/workshops/2010/YESIV/Prog-Abstr_files/Ghahramani-lecture2.pdf).
