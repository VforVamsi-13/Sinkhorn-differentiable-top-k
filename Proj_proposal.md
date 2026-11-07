# Differentiable Top-k Patch Selection for Visual Recognition Using Sinkhorn-Based Relaxations

Shubhankar Atre (26M2120) and Vamsi Devarapalli(26M0759)
**Course:** CS740 — Mathematics for Visual Computing  
**Project Proposal**

## 1. Motivation and Problem Statement

Deep learning models for image recognition often process an entire image, even when only a subset of its regions may be particularly informative for the classification task. Selecting relevant image patches could provide a way to focus computation on these regions. However, selecting exactly the top-k patches using conventional sorting or top-k operations involves discrete decisions that do not generally provide useful gradients to the network responsible for predicting patch importance.

This creates a challenge for end-to-end training. Although a classification loss can be differentiated through a neural network, a discrete patch-selection operation can interrupt the gradient path to the selection module. Consequently, the network may have difficulty learning which patches to prioritize based on the downstream classification objective.

The proposed project will investigate **differentiable top-k selection for image recognition using optimal-transport-based relaxations and Sinkhorn normalization**. The objective is to implement the relevant mathematical operations, integrate them into a small computer vision pipeline, and study their numerical properties and effect on classification performance.

The project is motivated by two complementary research directions: differentiable top-k operators based on optimal transport and end-to-end differentiable patch selection for image recognition. Rather than attempting to develop an entirely new architecture, we intend to implement a manageable version of the selection mechanism and evaluate its behavior through controlled experiments.

## 2. Papers and Proposed Scope

We intend to study and implement the following papers:

**Paper 1 — Foundational Sinkhorn algorithm**

Richard Sinkhorn and Paul Knopp, _Concerning Nonnegative Matrices and Doubly Stochastic Matrices_, Pacific Journal of Mathematics, 1967.[Paper link](https://doi.org/10.2140/pjm.1967.21.343)

This paper studies the scaling of nonnegative matrices to obtain doubly stochastic matrices. We will use it as the mathematical foundation for implementing alternating row and column normalization. We will examine the conditions under which the procedure converges and validate its behavior on suitable nonnegative matrices.

Our implementation will focus on the core matrix-scaling procedure. This is distinct from the entropy-regularized optimal transport formulation, which uses Sinkhorn scaling as a computational component.


**Paper 2— Core mathematical implementation**

Yujia Xie et al., _Differentiable Top-k with Optimal Transport_, NeurIPS 2020.[Paper link](https://papers.nips.cc/paper_files/paper/2020/hash/ec24a54d62ce57ba93a531b460fa8d18-Abstract.html)

This paper introduces SOFT top-k, a differentiable approximation based on entropy-regularized optimal transport. We intend to implement the relevant Sinkhorn-based computation using tensor operations and investigate its behavior on small synthetic inputs before integrating it into an image-recognition pipeline.

**Paper 3 — Computer vision application**

Jean-Baptiste Cordonnier et al., _Differentiable Patch Selection for Image Recognition_, CVPR 2021.[Paper link](https://openaccess.thecvf.com/content/CVPR2021/html/Cordonnier_Differentiable_Patch_Selection_for_Image_Recognition_CVPR_2021_paper.html)

This paper demonstrates how a differentiable top-k operator can be incorporated into an end-to-end trainable patch-selection architecture. Its proposed method uses a perturbed-maximum approach, rather than the Sinkhorn-based SOFT top-k method in Paper 1. We will study its architecture and use it as a reference for the vision pipeline, while evaluating a Sinkhorn-based selection mechanism separately.

The intended scope is to implement the core selection operator, construct a simplified patch-selection pipeline, and compare its behavior with simpler alternatives. Reproducing the full architectures and all experiments in either paper is outside the initial scope.

## 3. Dataset and Experimental Setup

We plan to use **CIFAR-10** as the primary dataset. It contains 60,000 colour images of size $32 \times 32$, divided into ten classes, making it suitable for experiments with a small classifier and limited computational resources.

Images will be divided into a regular grid of patches. A lightweight scoring network will estimate patch importance, and a differentiable top-k module will select or weight a subset of patches before classification. A small CNN will serve as the classifier.

We will initially use a reduced training configuration to validate the pipeline, then increase the training budget if time and computational resources permit. CIFAR-100 may be considered as an additional evaluation dataset if the primary experiments are completed successfully; it is not a prerequisite for the initial implementation.

The proposed experiments will compare:

- **Random selection:** A baseline that selects patches without learned importance scores.
- **Hard top-****k****:** A conventional discrete selection baseline.
- **Sinkhorn-based differentiable top-****k****:** The proposed selection mechanism.

Where feasible, we will also compare against a published differentiable top-k alternative. The backbone architecture, data splits, and training budgets will be kept comparable across experiments to support a fair comparison.

## 4. Implementation, Evaluation, and Validation

The implementation will be developed in Python using PyTorch. We will proceed in stages, beginning with synthetic vectors and small matrices before integrating the selection operator into the image-classification model.

**Mathematical and numerical validation.** We will verify that Sinkhorn normalization produces approximately doubly stochastic matrices, with row and column sums approaching one as the number of iterations increases under suitable conditions. We will examine numerical stability and the effect of temperature and iteration count. Small synthetic examples will be used to compare the relaxed selection against the corresponding hard top-k result.

**Gradient validation.** We will inspect whether gradients propagate from the classification loss to the patch-scoring network. On small test cases, automatic differentiation will be checked against finite-difference estimates wherever the comparison is mathematically appropriate. We will also record gradient norms and investigate how they change with temperature and selection size.

**Task-level evaluation.** We will measure classification accuracy on held-out test data and examine how accuracy changes with the number or fraction of patches retained. We will compare the proposed method with random and hard-selection baselines. Runtime and sensitivity to the Sinkhorn iteration count will also be measured where practical.

**Ablation studies and visual analysis.** We will vary the number of selected patches, temperature, and number of Sinkhorn iterations. Visualizations will show the original images, patch importance scores, and selected regions. These will help us inspect whether the selection mechanism behaves sensibly and identify failure cases.

We will use fixed experimental configurations and random seeds where possible, and repeat important runs when resources permit. Results will be reported with appropriate caveats, including cases where the differentiable method does not outperform the baselines. We will not assume in advance that Sinkhorn-based selection necessarily improves classification accuracy.

%% ## 5. Expected Outcomes and Timeline

The expected outcome is a working implementation of a Sinkhorn-based differentiable top-k operator, integrated into a simplified image-recognition pipeline and accompanied by mathematical derivations, validation tests, comparative experiments, and visualizations.

 The project is planned over approximately three weeks:

- **Week 1:** Study the selected papers, implement and validate Sinkhorn normalization, and establish a working CIFAR-10 classifier.
- **Week 2:** Implement differentiable top-k, integrate it into the patch-selection pipeline, and run baseline comparisons.
- **Week 3 and 4:** Conduct focused ablation studies, analyze gradient behavior and failure cases, and prepare the report and demonstration. 

The primary emphasis will be on mathematical correctness, reproducibility, and understanding how differentiable selection behaves in an end-to-end learning pipeline. Additional datasets or architectural extensions will be attempted only if the core implementation and evaluation are completed within the available time. %%