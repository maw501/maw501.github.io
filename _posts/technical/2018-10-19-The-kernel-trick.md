---
layout: post
title: The kernel trick
date: 2018-10-19
use_math: true
image: "kernel.png"
---

**Note: a work in progress**

This post only covers the concept of the 'kernel trick' and not kernels in general which often get a bad rep as they are usually poorly explained - usually in the context of SVMs. We will start by considering an example.
<!--more-->
##### 1. Start by imagining we have a dataset with two columns which represent the height and weight of a bunch of people

For notational simplicity let's call the column representing height $x_1$ and the column representing weight $x_2$.

##### 2. Now let's pretend we are trying to model some target variable (e.g. classification as an adult or not) as a function of this data and it turns out we can't solve this problem well with linear combinations of our columns

In other words our data might not be linearly separable. Now, our friend Bob pops over and suggests we might be able to solve the problem if we take some non-linear transformations of our data. In particular he suggests that we scrap our original dataset and create the following new columns (which we will see will be equivalent to a second order polynomial basis):
  * A column with all 1s in it
  * A column equal to original $x_1$ feature multiplied by $\sqrt{2}$, i.e. $\sqrt{2}x_1$
  * A column equal to original $x_2$ feature multiplied by $\sqrt{2}$, i.e. $\sqrt{2}x_2$
  * A column equal to $\sqrt{2}x_1x_2$
  * A column equal to $x_1^2$
  * A column equal to $x_2^2$

##### 3. Bob now tells us we can assess how similar two of our data points are (i.e. two rows of this new matrix) by element-wise multiplying the vectors together

We decide to try this by hand. We have the first row of our original dataset $\textbf{a}$ which had $(x_1, x_2)$ entries $(a_1, a_2) = (1, 2)$ (yes, these are very small people!) and the second row $\textbf{b}$ which had $(x_1, x_2)$ entries $(b_1, b_2) = (3, 4)$. We now look at the new dataset we created and see:
 * $\textbf{a}$ has $(a_1, a_2) = (1, 2)$ giving $(1, \sqrt{2}a_1, \sqrt{2}a_2, \sqrt{2}a_1a_2, a_1^2, a_2^2) = (1, \sqrt{2}, 2\sqrt{2}, 2\sqrt{2}, 1, 4)$ after plugging the numbers in.
 * $\textbf{b}$ has $(b_1, b_2) = (3, 4)$ giving $(1, \sqrt{2}b_1, \sqrt{2}b_2, \sqrt{2}b_1b_2, b_1^2, b_2^2) = (1, 3\sqrt{2}, 4\sqrt{2}, 12\sqrt{2}, 9, 16)$ after plugging the numbers in.
 * We element wise multiply these two rows to get an answer of 144.
  * Note: $a_1$ and $a_2$ are the first and second column entries of row one (which we've called $\textbf{a}$). Similarly $b_1$ and $b_2$ are the first and second column entries of row two (which we've called $\textbf{b}$)

##### 4. Another friend Dave pops over and sees what we are doing. He laughs that we bothered creating all those new columns and tells us there is a simpler way to get the same answer. He says we should try just computing $(1 + \textbf{a}^T\textbf{b})^2 \,$ directly instead with our original data and not to bother with Bob's idea.

We are a bit sceptical about what Dave means (how can we possibly get the same answer?!) but we decide to try this anyway.
  * $$(1 + \textbf{a}^T\textbf{b})^2 = \bigg(1 + \left( \begin{array}{c} 1 & 2 \end{array} \right) \left( \begin{array}{c} 3 \\ 4 \end{array} \right) \bigg)^2$$ which is equal to $(1 + 11)^2 = 144$

**What just happened?!**

##### 5. Dave sees our confusion and writes the following down

* Call $\phi(\textbf{a})$ = $(1, \sqrt{2}a_1, \sqrt{2}a_2, \sqrt{2}a_1a_2, a_1^2, a_2^2)$ the transformation of row $\textbf{a}$ for both its columns.
* Call $\phi(\textbf{b})$ = $(1, \sqrt{2}b_1, \sqrt{2}b_2, \sqrt{2}b_1b_2, b_1^2, b_2^2)$ the transformation of row $\textbf{b}$ for both its columns.
* When we element-wise multiplied the two rows we really did $\phi(\textbf{a})^T\phi(\textbf{b})$ i.e. took the dot product.
* This gives: $\phi(\textbf{a})^T\phi(\textbf{b}) = 1 + 2a_1b_1 + 2a_2b_2 + 2a_1a_2b_1b_2 + a_1^2b_1^2 + a_2^2b_2^2$
* This is mathematically identical to $$(1 + \textbf{a}^T\textbf{b})^2 = \bigg(1 + \left( \begin{array}{c} a_1 & a_2 \end{array} \right) \left( \begin{array}{c} b_1 \\ b_2 \end{array} \right) \bigg)^2 = (1 + a_1b_1 + a_2b_2)^2$$ which if you expand out will match the expanded $\phi(\textbf{a})^T\phi(\textbf{b})$ above.

##### Explanation

The transformation $\phi$ above is actually the basis for a second-order polynomial expansion. The reason it looks a little more complicated is because our dataset has 2 columns and so we get some cross terms as well. It turns out as we showed that this is also the same as directly computing $(1 + \textbf{a}^T\textbf{b})^2$.

So if we want to fit a second-order polynomial mapping for our data we have two choices:

1. Start creating loads of new columns as we did for $\phi(\textbf{a})$.
  * Note this will create a massive amount of columns if we start with more than 2 initially and want a polynomial of higher order.
2. Compute $\textbf{k}(\textbf{a},\textbf{b}) = (1 + \textbf{a}^T\textbf{b})^2$ instead.

**Which do you think is easier???**

##### We are now ready to understand the kernel trick

Both are the same as 'mapping' our data to a 'higher dimensional space' (in this case that of 2nd order polynomials) and doing a calculation there. The **kernel trick** is that we are able to actually just use our original data and compute $\textbf{k}(\textbf{a},\textbf{b}) = (1 + \textbf{a}^T\textbf{b})^2$ without having to faff about with 'transforming' it to this higher dimensional space by creating new columns and we save all the computational overhead it would bring.

**The special functions for which it turns out allow us to stay in our original lower dimensional space but are equivalent to operating in a higher dimensional space are called kernel functions.**

For this to work we must be able to write the calculation in the new higher dimensional feature space as dot/inner products e.g. $\phi(\textbf{a})^T\phi(\textbf{b})$

So it turns out we can stick with our original data, use a kernel function and know that it corresponds to taking the dot product of the transformed vectors in a higher dimensional space - **without even visiting it or knowing what $\phi$ is!!!**  This allows us to find complex non linear boundaries that are able to better separate the classes in our dataset.

Thus a kernel function is a function where it happens to turn out that computing the kernel functions in lower dimensions is the same as computing the inner product in the higher dimensional feature space. The feature space is implicit, and often infinite dimensional.

##### Further reading/videos

Good video [here](https://www.youtube.com/watch?v=XUj5JbQihlU&hd=1) and reading [here](https://stats.stackexchange.com/questions/80398/how-can-svm-find-an-infinite-feature-space-where-linear-separation-is-always-p) and [here](https://stats.stackexchange.com/questions/152897/how-to-intuitively-explain-what-a-kernel-is)
