# Smooth Variational Graph Embeddings for Efficient Neural Architecture Search

===============================================================================

Abstract
-----
In this paper, we propose an approach to neural architecture search (NAS) based on graph embeddings. NAS has been addressed previously using discrete, sampling based methods, which are computationally expensive as well as differentiable approaches, which come at lower costs but enforce stronger constraints on the search space. The proposed approach leverages advantages from both sides by building a smooth variational neural architecture embedding space in which we evaluate a structural subset of architectures at training time using the predicted performance while it allows to extrapolate from this subspace at inference time. We evaluate the proposed approach in the context of two common search spaces, the graph structure defined by the ENAS approach and the NAS-Bench-101 search space, and improve over the state of the art in both. 

Reference
---------
This repository containts code for the paper: ["Smooth Variational Graph Embeddings for Efficient Neural Architecture Search"](https://arxiv.org/abs/2010.04683).
*Authors: Jovita Lukasik, David Friede, Arber Zela, Frank Hutter, Margret Keuper*
