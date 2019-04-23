---
layout: post
title: The kernel trick
date: 2018-10-19
use_math: true
image: "kernel.png"
comments: true
tags: ['kernels', 'SVMs']
---
This post only covers the concept of the kernel trick (not kernels in general). From my experience kernels often get a bad reputation as they are usually poorly explained, usually in the context of SVMs. In this article our goal will be to explain them via a simple motivating example which we will follow through in detail to get to the heart of the trick.
<!--more-->
<hr class="with-margin">

<div class="list-of-contents">
  <h4>Contents</h4>
  <ul></ul>
</div>

<hr class="with-margin">
<h4 class="header" id="intro">Motivating example</h4>

Let us start by imagining we have a simple dataset with two columns which represent the height and weight of a set of people. For notational simplicity let's call the column representing height $x_1$ and the column representing weight $x_2.$

Further, imagine we are trying to model some target variable (e.g. binary classification as an adult or not) as a function of this data and that we can't solve this problem well with linear combinations of our columns. In other words our data might not be well separated by a linear decision boundary and so we either need a more powerful model or better features.

Now, our friend Bob pops over and suggests we might be able to solve the problem if we take some non-linear transformations of our data. In particular he suggests that we enhance our original dataset to create the following new columns:
  * A column with all 1s in it
  * A column equal to original $x_1$ height feature multiplied by $\sqrt{2}$, i.e. $\sqrt{2}x_1$
  * A column equal to original $x_2$ weight feature multiplied by $\sqrt{2}$, i.e. $\sqrt{2}x_2$
  * A column equal to $\sqrt{2}x_1x_2$
  * A column equal to $x_1^2$
  * A column equal to $x_2^2$

<blockquote class="tip">
<strong>Recall:</strong> SVMs rely on the notion of similarity between each data point in order to locate the most similar points that are in different classes - the support vectors. The goal is then to fit a hyperplane between these support vectors.
</blockquote>

Bob also tells us we can assess how similar two of our data points are (i.e. two rows of this new matrix) by element-wise multiplying the row vectors of our data together and summing the resultant numbers - this is computing the dot product between two points.

We decide to try this by hand.

<p align="center">
    <img src="/assets/img/svm_calc.jpg" alt="Image" width="600" height="400" />
</p>

<em class="figure">Fig. 0: Expanding our original dataset with new features (note it isn't clear yet why we use these new features)</em>

The above is:

$\textbf{a}^T = (a_1, a_2) = (1, 2)$ giving entries:

$$(1, \sqrt{2}a_1, \sqrt{2}a_2, \sqrt{2}a_1a_2, a_1^2, a_2^2) = (1, \sqrt{2}, 2\sqrt{2}, 2\sqrt{2}, 1, 4)$$

$\textbf{b}^T = (b_1, b_2) = (3, 4)$ giving entries:

$$(1, \sqrt{2}b_1, \sqrt{2}b_2, \sqrt{2}b_1b_2, b_1^2, b_2^2) = (1, 3\sqrt{2}, 4\sqrt{2}, 12\sqrt{2}, 9, 16)$$

If we compute the dot product of these two new rows we get 144.

Another friend Alice pops over and sees what we are doing. She laughs that we bothered creating all those new columns and tells us there is a simpler way to get the same answer. She says we should try just computing $(1 + \textbf{a}^T\textbf{b})^2 \,$ directly instead with our original data and not to bother with Bob's idea.

We are a bit sceptical about what Alice means (how can we possibly get the same answer?!) but we decide to try this anyway. Sure enough...

$$(1 + \textbf{a}^T\textbf{b})^2 = \bigg(1 + \left( \begin{array}{c} 1 & 2 \end{array} \right) \left( \begin{array}{c} 3 \\ 4 \end{array} \right) \bigg)^2 = (1 + 11)^2 = 144$$

##### What just happened?!

The kernel trick just happened, that's what.

<hr class="with-margin">
<h4 class="header" id="expl"> Explanation</h4>

<blockquote class="tip">
<strong>Spoiler:</strong> The kernel trick is when we can take a shortcut such as Alice suggested. This means we can perform a calculation between rows of our original data that is equivalent to having created a bunch of new features and performed a calculation with the newly created data. Such kernels are of interest to us when the equivalent long-winded expansion (as suggested by Bob) is a basis for a more complex transformation.
</blockquote>

Alice sees our confusion and starts as follows:

Call

$$\phi(\textbf{a}) = (1, \sqrt{2}a_1, \sqrt{2}a_2, \sqrt{2}a_1a_2, a_1^2, a_2^2)$$

the transformation of row $\textbf{a}$  and

$$\phi(\textbf{b}) = (1, \sqrt{2}b_1, \sqrt{2}b_2, \sqrt{2}b_1b_2, b_1^2, b_2^2)$$

the transformation of row $\textbf{b}$.

We have that:

$$\phi(\textbf{a})^T\phi(\textbf{b}) = 1 + 2a_1b_1 + 2a_2b_2 + 2a_1a_2b_1b_2 + a_1^2b_1^2 + a_2^2b_2^2$$

This is mathematically identical to

$$(1 + \textbf{a}^T\textbf{b})^2 = \bigg(1 + \left( \begin{array}{c} a_1 & a_2 \end{array} \right) \left( \begin{array}{c} b_1 \\ b_2 \end{array} \right) \bigg)^2 = (1 + a_1b_1 + a_2b_2)^2$$

and expanding the bracket gives

$$(1 + \textbf{a}^T\textbf{b})^2  = 1 + 2a_1b_1 + 2a_2b_2 + 2a_1a_2b_1b_2 + a_1^2b_1^2 + a_2^2b_2^2$$

We have thus shown that

$$\phi(\textbf{a})^T\phi(\textbf{b}) = (1 + \textbf{a}^T\textbf{b})^2 $$

for some transformation $\phi$. This transformation $\phi$ above is actually the basis for a second-order polynomial expansion. The reason it looks a little more complicated is because our dataset has 2 columns and so we get some cross terms as well.

So if we want to fit a second-order polynomial mapping for our data we have two choices:

1. Start creating loads of new columns as we did per Bob's suggestion to get $\phi(\textbf{a})$ and  $\phi(\textbf{b})$.
  * Note this will create a massive amount of columns if we start with more than 2 initially and want a polynomial of higher order.
2. Compute $(1 + \textbf{a}^T\textbf{b})^2 = \textbf{k}(\textbf{a},\textbf{b})$ instead. We will call this the kernel between two rows/data points.

It's clear that it's computationally much easier to work with our original data and not start creating new columns.

##### Further explanation

Both approaches are the same as 'mapping' our data to a 'higher dimensional space' (in this case that of 2nd order polynomials) and doing a calculation there. The **kernel trick** is that we are able to actually just use our original data and compute $\textbf{k}(\textbf{a},\textbf{b}) = (1 + \textbf{a}^T\textbf{b})^2$ without having to have the complexity of 'transforming' it to this higher dimensional space by creating new columns and so we save the computational overhead it would bring.

<blockquote class="tip">
<strong>Remember:</strong> The special functions for which it turns out allow us to stay in our original lower dimensional space but are equivalent to operating in a higher dimensional space are called kernel functions.
</blockquote>

For this to work we must be able to write the calculation in the new higher dimensional feature space as dot/inner products e.g. $\phi(\textbf{a})^T\phi(\textbf{b}).$

Thus we can stick with our original data, use a kernel function and know that it corresponds to taking the dot product of the transformed vectors in a higher dimensional space - **without even visiting it or sometimes even knowing what $\phi$ is (see section on below)!**  This allows us to find complex non linear boundaries that are able to better separate the classes in our dataset.

Thus a kernel function is a function where it happens to turn out that computing the kernel functions in lower dimensions is the same as computing the inner product in a higher dimensional feature space. This feature space is implicit, and often infinite dimensional.

<hr class="with-margin">
<h4 class="header" id="mercer"> Creating kernels</h4>

The discussion above doesn't tell us how to obtain functions $\textbf{k}(\textbf{a},\textbf{b})$ which are valid kernels. Luckily there is some simple arithmetic that allows us to create new kernels. Let $\textbf{k}_1$ and $\textbf{k}_2$ be any kernels, then constructing $\textbf{k}$ in the following (non-exhaustive) ways results in a new kernel:

* $\textbf{k}(\textbf{a},\textbf{b}) = \textbf{k}_1(\textbf{a},\textbf{b}) \, \textbf{k}_2(\textbf{a},\textbf{b})$
* $\textbf{k}(\textbf{a},\textbf{b}) = \textbf{k}_1(\textbf{a},\textbf{b}) \, + \, \textbf{k}_2(\textbf{a},\textbf{b})$
* $\textbf{k}(\textbf{a},\textbf{b}) = \exp\\{\textbf{k}_1(\textbf{a},\textbf{b})\\}$

In other words we can multiply, add and take the exponential of any existing kernels and still end up with a valid kernel. In these cases we may end up not knowing what the equivalent $\phi$ is, but this doesn't matter as we don't actually need it.

<hr class="with-margin">
<h4 class="header" id="intro">Further reading/videos</h4>

Good video [here](https://www.youtube.com/watch?v=XUj5JbQihlU&hd=1), reading [here](https://stats.stackexchange.com/questions/80398/how-can-svm-find-an-infinite-feature-space-where-linear-separation-is-always-p) and [here](https://stats.stackexchange.com/questions/152897/how-to-intuitively-explain-what-a-kernel-is).
<hr class="with-margin">
