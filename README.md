# scalaz-ml

[![Gitter](https://badges.gitter.im/scalaz/scalaz-ml.svg)](https://gitter.im/scalaz/scalaz-ml?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

## Goal

Scalaz ML provides a high-performance, purely functional library for doing machine learning in a safe, principled manner.

## Introduction & Highlights

Scalaz ML aims to provide the best end to end data science and machine learning experience for data scientists and engineers.

- Focus on usability
- Pure Type-safe, purely-functional interface that integrates with other scalaz projects
- Full data science cycle enabled with first class support for data analysis and visualisation via scalaz-analytics and scalaz-viz
- Built in model metrics and diagnostics
- High performance model training and scoring


## Other Libraries

Below is a selection of Machine Learning Libraries that we are being used as inspiration. Some of these metrics are somewhat subjective but they give an idea for what we are looking at from each library.

| Library | Comprehensive Data Science Support | Fault Tolerant | Good Debugging | Visualization | Model Metrics | FP  |
| ------- | ---------------------------------- | -------------- | -------------- | ------------- | ------------- | --- |
| [Scikit Learn](http://scikit-learn.org/) | ✔ | ✘ | ✘ | ✔ | ✔ | ✘ |
| [SparkML](https://spark.apache.org/docs/latest/ml-guide.html) | ✘ | ✔ | ✘ | ✘ | ✘ | ✘ |
| [SMILE](https://haifengl.github.io/smile/) | ✔ | ? | ? | ✔ | ✔ | ✘ |
| [R](https://www.r-project.org/) | ✔ | ? | ? | ✔ | ✔ | ✘ |
| [H2O](https://www.h2o.ai/) | ✔ | ✘ | ✔ | ✔ | ✔ | ✘ |
| [HLearn](https://github.com/mikeizbicki/HLearn) | ✘ | ✘ | ? | ✔ | ✔ | ✔ |


## Background

* Scikit API design [paper](https://arxiv.org/pdf/1309.0238.pdf) by the authors
* Scikit learn [algorithm cheat sheet](http://scikit-learn.org/stable/_static/ml_map.png)
* Fantastic hub of ML knowledge curated by DL4J [here](https://deeplearning4j.org/documentation)
* Fundamental Algebraic Structures - [Algebird](https://twitter.github.io/algebird/)
* [Elements of Statistical Learning](https://web.stanford.edu/~hastie/ElemStatLearn/) - Fantastic reference for many ML techniques
* H Learn - [paper](https://izbicki.me/public/papers/tfp2013-hlearn-a-machine-learning-library-for-haskell.pdf) and [code](https://github.com/mikeizbicki/HLearn)
* [subhask](https://github.com/mikeizbicki/subhask/blob/master/README.md) - Fast numerical computing abstractions in Haskell
* [Grenade](https://github.com/HuwCampbell/grenade) - Dependently Typed NNs in Haskell (and [talk](https://www.youtube.com/watch?v=sPjA6lS0GlQ))
* [Algebraic Classifiers](https://izbicki.me/public/papers/icml2013-algebraic-classifiers.pdf)
* [Gaussian Distributions as Monoids](https://izbicki.me/blog/gausian-distributions-are-monoids.html)
* [Fast Principled Cross Validation](https://izbicki.me/blog/hlearn-cross-validates-400x-faster-than-weka.html)
* Distributed Spark ML Performance [paper](https://arxiv.org/pdf/1612.01437.pdf)
* Mining Massive Data Sets [free online book](http://www.mmds.org/) - several useful chapters
* An MIT lecture on the algebraic geometry of algorithms [part 1](https://www.youtube.com/watch?v=VBwRZOnqs-I) and [part 2](https://www.youtube.com/watch?v=sR9XkFoyskA)
* XGBoost [overview](https://machinelearningmastery.com/gentle-introduction-xgboost-applied-machine-learning/)
* Examples of a basic end to end ML flow in R [datacamp](https://www.datacamp.com/community/tutorials/machine-learning-in-r), [machine learning mastery](https://machinelearningmastery.com/machine-learning-in-r-step-by-step/) and [kaggle](https://www.kaggle.com/camnugent/introduction-to-machine-learning-in-r-tutorial)
* [Caret](https://topepo.github.io/caret/index.html) - R package for machine learning. Similar interface to Python's Scikit-Learn. R's attempt to have one package containing most ML algos instead of having them spread through various packages.
* Visually explaining ML predictions with [LIME](https://github.com/marcotcr/lime)
* [Neurocat](https://github.com/mandubian/neurocat) - Reserach combining category theory and ML in Scala