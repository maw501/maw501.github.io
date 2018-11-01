---
layout: post
title: Why do kernels matter?
date: 2018-10-22
use_math: true
image: "kernel.png"
---

**Note: still a work in progress**

Here we try to understand all the fuss about kernels and why people keep mentioning the phrase 'high dimensional mapping'.

<!--more-->
<hr class="with-margin">

<div class="list-of-contents">
  <h4>Contents</h4>
  <ul></ul>
</div>

<hr class="with-margin">
<h4 class="header" id="intro">Introducing Kernels</h4>

A (loose but) simple way to think of a kernel is some function that takes in two vectors of equal length and outputs a similarity score between them which is always positive.

It also doesn't matter which way we input the vectors so $\textbf{k}(a,b) = \textbf{k}(b,a)$. Our input vectors are usually the rows of our design matrix $X$ and thus reside in $\mathbb{R}^d$.

<hr class="with-margin">
<h4 class="header" id="why">Why do kernels matter?</h4>

Kernels matter because for a lot of approaches (including linear regression!) we can actually reformulate the prediction of a new point in terms of dot products of the original data points after undergoing some transformation $\phi$. Except due to the kernel trick we don't actually need $\phi$ but can just work with the kernel function $\textbf{k}$ itself. This is somewhat of an about shift.
