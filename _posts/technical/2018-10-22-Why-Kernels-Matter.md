---
layout: post
title: Why do kernels matter?
date: 2018-10-22
use_math: true
---

**Note: a work in progress**

#### Kernels

A (loose but) simple way to think of a kernel is some function that takes in two vectors of equal length and outputs a similarity score between them which is always positive. It also doesn't matter which way we input the vectors so $\textbf{k}(a,b) = \textbf{k}(b,a)$. Our input vectors are usually the rows of our design matrix $X$ and thus reside in $\mathbb{R}^d$.

**But why do kernels matter?**

Kernels matter because for a lot of approaches (including linear regression!) we can actually reformulate the prediction of a new point in terms of dot products of the original data points after undergoing some transformation $\phi$. Except due to the kernel trick we don't actually need $\phi$ but can just work with the kernel function $\textbf{k}$ itself. This is somewhat of an about shift.
