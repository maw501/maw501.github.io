---
layout: post
title: ColumbiaX - ML - week 6
date: 2018-10-25
use_math: true
---

This page is a summary of my notes for the above course, link [here](https://www.edx.org/course/machine-learning-columbiax-csmm-102x-4).

This is my first attempt at publishing course notes and I have no intention to make it comprehensive but rather to highlight the bits I feel are important and maybe explain some of the things I found a little trickier (or weren't explained to my taste). Understanding is deeply personal though if you spot any errors or have any questions please feel free to drop me an email.

## Week 6 (lectures 11 and 12): overview

Week 6 in some ways introduced the old guard before handing over to the (relatively) new kids on the block. SVMs were all the rage over a decade ago whilst more modern data mining techniques (on non-huge datasets) now tend to lean heavily on decision trees. Whilst I think one is reasonably unlikely to encounter SVMs too much in the wild (they are outperformed by other methods and don't scale well) I guess that are taught as they have a strong theoretical background and allow a course to add lots of maths in order to look serious. They are also some nice links to logistic regression and once understood, the idea of kernels is actually quite powerful. I will not be dwelling on decision trees, bagging or boosting this week as this is something I have reasonable knowledge of and the meat of the week lies with SVMs. Onwards.

* **Support Vector Machines (SVMs)**
  * SVMs extend our notions of fitting a hyperplane to linearly separable by addressing some key issues: we fit the biggest 'wedge' between the classes we can, we allow some violations in order to not over-fit and we use kernels to achieve non-linear decision boundaries.
* **Decision Trees**
  * Decision trees partition our feature space up by making 'cuts' at various points of each feature. We can use them to perform either classification or regression and modern implementations have many bells and whistles that need tuning. Extensions of basic decision trees are Random Forests which ensemble many over-fit but less correlated trees.
* **Boosting and bagging**
  * Statistical techniques to increase the performance and robustness of our decision trees. Bagging ultimately leads us from a decision tree to Random Forest (with the addition of some randomness).

## Week 6 (lectures 11 and 12): overview with more details

We are going to focus solely on SVMs this week.

#### SVMs take advantage of several powerful ideas

Thus it can be confusing to see the big picture. In the context of two linearly separable classes:

* **Maximal Margin Classifier:** why fit any old hyperplane when you can fit the one that maximizes the distance between your two classes?

* **Primal problem:** formulate the optimization problem to solve for the maximal margin classifier. Realise from an optimization stand point it looks tricky to solve.

* **Dual problem:** use Lagrange multipliers to reformulate this problem into something more friendly called the dual problem.
  * Note: primal and dual problems are from optimization theory. Great discussion [here](https://www.svm-tutorial.com/2016/09/duality-lagrange-multipliers/).

* **Bonus of the dual problem:** we realise that the dual problem is calculated more efficiently than the primal (if few support vectors) and further, in the dual problem we note we are taking the dot product between two data points...**enter the kernel trick** which allows us to still do the computation with our original data points but is equivalent to operating in a higher dimensional space.
  * If the kernel bit didn't make sense see [here](../19/The-kernel-trick).

#### Walking through the primal to dual formulation

**Note:** if you haven't read the lecture notes discussion on convex sets and why we wish to minimize the distance between them you should do this now as I'm not going to cover this.

##### Step 1: set up the primal problem

To find the hyperplane with $n$ linearly separable points $x_1, y_1), ... ,(x_n, y_n)$ with $y_i \in \\{-1, +1\\}$ we wish to solve the following optimization problem:

* Minimize: $\dfrac{1}{2} \|\|w\|\|^2$
* Subject to: $y_i(x_i^Tw + w_0) \geq 1$ for $i = 1,...,n$

**Q: why does this make sense?**

**A:** the second condition is that we get each point on the correct side of the hyperplane (refresh this if you are unsure why) and the main objective function is there because (of a technicality with linearly separable data such that) if the data is linearly separable we could keep increasing our weights to infinity in order to be more confident about the predictions even though the hyperplane wouldn't move. We thus add this as a regularizing term to some extent. See [here](https://homes.cs.washington.edu/~marcotcr/blog/linear-classifiers/) for a nice example of this.

##### Step 2: reformulate the primal problem using Lagrange multipliers

This technique means we can take the constraint above and subtract from our objective function by introducing $n$ Lagrange multipliers $\alpha_i > 0$.

This leads to:

$$\mathcal{L} = \dfrac{1}{2} \|w\|^2 - \sum_{i=1}^{n}\alpha_i y_i(x_i^Tw + w_0) + \sum_{i=1}^{n}\alpha_i$$

##### Step 2: differentiate the Lagrangian with respect to both $w$ and $w_0$ and set to 0

This is actually just one line of working and will find the minimum over $w$ and $w_0$:

$$\nabla_{w}\mathcal{L} = 0 $$

$$\Rightarrow w = \sum_{i=1}^{n}\alpha_i y_ix_i$$

Similarly for $w_0$:

$$\nabla_{w_0}\mathcal{L} = 0$$

$$\Rightarrow \sum_{i=1}^{n}\alpha_i y_i = 0$$

##### Step 3: plug the values we just found for $w$ and $w_0$ back into the statement of the Lagrangian

Whilst I won't add all the gory details hopefully it isn't so hard to convince yourself by plugging $w$ and $w_0$ into the Lagrangian we get something of the form (don't worry about small details, just get happy broadly with what went on):

$$\mathcal{L} = \sum_{i=1}^{n}\alpha_i - \dfrac{1}{2}\sum_{i=1}^{n}\sum_{j=1}^{n} \alpha_i \alpha_j y_i y_j(x_i^T x_j)$$

subject to:

$$ \sum_{i=1}^{n}\alpha_i y_i = 0$$

with $\alpha_i \geq 0$ for $i = 1, ..., n$

**This is now the statement of the dual problem.**

Some comments:

* We have two summations (literally just think of $i$ and $j$ as counters) as we need to sum twice over our data now, once when finding $w$ and then again in the equation for $\mathcal{L}$. We thus introduce $j$ as a different variable to not confuse it with $i$.
* We used the fact that $ \sum_{i=1}^{n}\alpha_i y_i = 0$ which got rid of the term involving $w_0$.
* The dual formulation is a convex optimization problem and so we can find an exact solution.

#### Understanding the dual formulation (minimal equations)

**Note:** the convex hull can be defined as taking linear combinations of all our points (weights must sum to 1 and be non-negative) and realising this can get us to all points inside this space but nowhere outside of it. Or just look [here](http://www.idav.ucdavis.edu/education/GraphicsNotes/Convex-Combinations/Convex-Combinations.html) at the convex set bit.

Recall that intuitively what we were doing when talking about convex sets was that we wanted to find the point in each class (set) that means the sets are closest to each other - we will then drive our hyperplane wedge into that gap. The lecture notes have nice images of this.

This idea of weighting all the points in each class means what we are really looking for is something the form:

$$ \| (\sum_{x_i \in S_1} \alpha_{1i}x_i)  -(\sum_{x_i \in S_0} \alpha_{0i}x_i) \|_2 $$

where $S_1$ is the set of $x$ points in class +1 and $S_0$ are those $x$ in class -1. We can thus think of $\alpha_{1}$ and $\alpha_{0}$ as probability vectors weighting each point in the set.

A few comments:

* This is calculating the distance between two points, one from each set/class
* The left term is a point in the convex hull of set $S_1$
* The right term is a point in the convex hull of set $S_0$
* We minimize the distance between these two points

It turns out with a bit of work we can rewrite the dual formulation to show it's actually doing the same thing as above - finding the closest points in the convex hulls constructed from data in class +1 and −1.

#### Extending the SVM: non-linearly separable data and using a kernel

These are the two big extensions that really make the SVM the powerful technique it is.

* **Slack variables:** allow data to be on the wrong side of the hyperplane but add a penalty for this into the objective function.
  * We can think of the slack variables telling us where the $i$th observation is located relative to the hyperplane and the margin.
  * We typically set some hyperparameter when fitting SVMs which controls the 'budget' of how many examples we are willing to misclassify - this partly controls the complexity of the decision boundary.
* **Kernels:** notice that in the dual formulation we have the term $x_i^T x_j$ which appears. We can replace this with $\phi(x_i)^T \phi(x_j)$ and note this is the same as calculating $\textbf{k}(x_i, x_j)$
  * This relies upon the 'kernel trick' and allows us to fit complex decision boundaries whilst still using our original data.

#### Further reading on SVMs
* [This](https://www.svm-tutorial.com/) is a nice site with a free e-book.
* A more technical treatment from Stanford CS229 by Andrew Ng [here](http://cs229.stanford.edu/notes/cs229-notes3.pdf)

## Comments on some of the mathematics used

* **Lagrange multipliers**
  * These have been used in another week and I refer you [here](https://www.khanacademy.org/math/multivariable-calculus/applications-of-multivariable-derivatives/constrained-optimization/a/lagrange-multipliers-single-constraint) to refresh.
* **Primal and dual problems**
  * This is actually something from optimization theory. You can read more [here](https://www.svm-tutorial.com/2016/09/duality-lagrange-multipliers/).
* **Summations**
  * You definitely have to be comfortable with the summation notation - it's generally a good idea to write things out yourself from one step to the next in order to ensure you are happy.
* **Matrix calculus**
  * This cropped up again when differentiating the Lagrangian, a solid reference is [here](https://en.wikipedia.org/wiki/Matrix_calculus).
    * Mental hack: when you are trying to follow a differentiation of something involving matrices and vectors I find it useful to just pretend initially that everything is just a normal variable and see where that gets me. Often the matrix calculus bit is just adjusting for a transpose/slight reordering and the notational details aren't so crucial first time around - you can (and should) always check dimensions to help you out at the end.

## Things I'm unclear on (or outstanding questions)

* TBC

## What did the textbooks say?

To be updated.
