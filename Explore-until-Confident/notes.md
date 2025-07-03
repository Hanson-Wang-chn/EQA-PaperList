# Explore until Confident: Efficient Exploration for Embodied Question Answering

## Problem

The paper addresses **Embodied Question Answering (EQA)**, where a robot navigates an unknown indoor environment to gather visual information and answer questions about the scene. The primary problem is to enable efficient exploration and accurate question answering using large vision-language models (VLMs), despite their limitations in scene memory and miscalibrated confidence, which can lead to premature stopping or over-exploration.

## Pipeline

![alt text](<截屏2025-07-03 18.05.28.png>)

The proposed framework leverages VLMs for semantic reasoning, builds an external semantic map for exploration, and uses conformal prediction (CP) to calibrate stopping criteria. The pipeline consists of the following modules:

### 1. Semantic Map Construction
- **Input**: RGB image $I_c^t$, depth image $I_d^t$, question $q$, robot pose $g^t$ at time $t$.
- **Output**: A 3D voxel-based semantic map $M$ storing occupancy, exploration status, and semantic values.
- **Submodules**:
  - **Voxel Map Update**:
    - **Input**: Depth image $I_d^t$, camera intrinsics.
    - **Process**: Uses Volumetric Truncated Signed Distance Function (TSDF) Fusion to update voxel occupancy and exploration status (voxels within a smaller field of view marked as explored).
    - **Output**: Updated 3D voxel map with occupancy and exploration flags.
  - **2D Projection**:
    - **Input**: 3D voxel map.
    - **Process**: Projects 3D map into a 2D point map, marking points as free if voxels up to 1.5m height are free, and explored if all voxels along the height are explored.
    - **Output**: 2D map $M$ with occupancy and exploration information.

### 2. VLM Visual Prompting for Semantic Values
![alt text](<截屏2025-07-03 18.06.36.png>)
- **Input**: RGB image $I_c^t$, question $q$, 2D map $M$.
- **Output**: Local Semantic Value (LSV) and Global Semantic Value (GSV) for sampled points $P$.
- **Submodules**:
  - **Point Sampling**:
    - **Input**: RGB image $I_c^t$, 2D map $M$.
    - **Process**: Projects $I_c^t$ onto $M$, keeps free points, and samples 3 points $P$ using farthest point sampling.
    - **Output**: Sampled points $P = \{A, B, C\}$.
  - **Local Semantic Value (LSV)**:
    - **Input**: Annotated image $I_{c,\mathcal{Y}_P}^t$ with points labeled $A, B, C$, question $q$.
    - **Process**: Prompts VLM with “Consider the question: $\{q\}$. Which direction (black letters on the image) would you explore then? Answer with a single letter.” Computes normalized probability $\hat{f}_{y_P}(I_c^t, s_{\text{LSV},q})$ for each point $p \in P$.
    - **Output**: $\text{LSV}_p(x^t) \in [0,1]$ for each point $p$.
  - **Global Semantic Value (GSV)**:
    - **Input**: RGB image $I_c^t$, question $q$.
    - **Process**: Prompts VLM with “Consider the question: $\{q\}$. Is there any direction shown in the image worth exploring? Answer with Yes or No.” Computes probability of ‘Yes’ as $\hat{f}_{\text{Yes}}(I_c^t, s_{\text{GSV},q})$.
    - **Output**: $\text{GSV}_p(x^t) \in [0,1]$ for each point $p$.
  - **Semantic Value (SV) Computation**:
    - **Input**: $\text{LSV}_p(x^t)$, $\text{GSV}_p(x^t)$, temperature scalars $\tau_{\text{LSV}}$, $\tau_{\text{GSV}}$.
    - **Process**: Computes $\text{SV}_p(x^t) = \exp(\tau_{\text{LSV}} \cdot \text{LSV}_p(x^t) + \tau_{\text{GSV}} \cdot \text{GSV}_p(x^t))$ with Gaussian smoothing.
    - **Output**: $\text{SV}_p(x^t)$ for each point $p$, stored in the semantic map.

### 3. Semantic-Value-Weighted Frontier Exploration
- **Input**: 2D map $M$, semantic values $\text{SV}_p$, normal direction values $\text{SV}_{p,\text{normal}}$.
- **Output**: Next robot pose $g^{t+1}$.
- **Submodules**:
  - **Frontier Identification**:
    - **Input**: 2D map $M$.
    - **Process**: Identifies frontiers (boundaries between explored and unexplored regions) using Frontier-Based Exploration (FBE).
    - **Output**: Set of frontier points.
  - **Frontier Sampling**:
    - **Input**: Frontier points, $\text{SV}_p$, $\text{SV}_{p,\text{normal}}$, temperature scalars $\tau_{\text{SV}}$, $\tau_{\text{SV,normal}}$.
    - **Process**: Computes weight $w_p = \exp(\tau_{\text{SV}} \cdot \text{SV}_p + \tau_{\text{SV,normal}} \cdot \text{SV}_{p,\text{normal}})$ and samples a frontier point.
    - **Output**: Selected frontier point and orientation.
  - **Path Planning**:
    - **Input**: Selected frontier point, current pose $g^t$.
    - **Process**: Uses a collision-free planner $\pi$ to determine the next pose within 3m.
    - **Output**: Next pose $g^{t+1}$.

### 4. Stopping Criterion with Multi-Step Conformal Prediction
- **Input**: RGB image $I_c^t$, question $q$, calibration dataset $Z = \{(\bar{x}_i, y_i)\}_{i=1}^N$.
- **Output**: Prediction set $C^t(x^t)$, final answer $y$.
- **Submodules**:
  - **Relevance-Weighted Confidence**:
    - **Input**: RGB image $I_c^t$, question $q$.
    - **Process**: Computes question-image relevance score $\text{Rel}(x^t) = \hat{f}_{\text{Yes}}(I_c^t, (q, s_{\text{Rel},q}))$ and answer confidence $\hat{f}_y(x^t)$. Defines $\rho_y^t(x^t) = \text{Rel}(x^t)(\hat{f}_y(x^t) - 1)$.
    - **Output**: Relevance-weighted confidence $\rho_y^t(x^t)$ for each answer $y$.
  - **Episode-Level Confidence**:
    - **Input**: Sequence $\bar{x} = (x^0, x^1, \ldots)$, confidence scores $\rho_y^t(x^t)$.
    - **Process**: Computes $\hat{\rho}_y(\bar{x}) = \min_{t \in [T]} \rho_y^t(x^t)$.
    - **Output**: Episode-level confidence $\hat{\rho}_y(\bar{x})$.
  - **Calibration**:
    - **Input**: Calibration dataset $Z$, user-defined confidence $1-\epsilon$.
    - **Process**: Computes non-conformity scores $\kappa_i = 1 - \hat{\rho}_{y_i}(\bar{x}_i)$ and sets threshold $\hat{q}$ as the $\frac{(N+1)(1-\epsilon)}{N}$ quantile.
    - **Output**: Confidence threshold $\hat{q}$.
  - **Prediction Set Construction**:
    - **Input**: Current input $x^t$, threshold $\hat{q}$.
    - **Process**: Constructs $C^t(x^t) = \{y \in \mathcal{Y} \mid \rho_y^t(x^t) \geq 1-\hat{q}\}$ and maintains intersection $\cap_{k=0}^t C^k(x^k)$.
    - **Output**: Prediction set $C^t(x^t)$.
  - **Stopping Decision**:
    - **Input**: Intersection of prediction sets $\cap_{k=0}^t C^k(x^k)$.
    - **Process**: Stops if the intersection contains one answer or time limit $T$ is reached. If multiple answers remain, selects $y$ with highest $\hat{f}_y(x^t)$ at the time step with highest $\text{Rel}(x^t)$.
    - **Output**: Final answer $y$.

## Performance

### Metrics
- **Success Rate**: Percentage of scenarios where the correct answer is predicted.
- **Normalized Time Steps**: Number of steps taken divided by the maximum allowable steps $T_e$.

### Experimental Setup
- **Simulation**: Uses the HM-EQA dataset (500 scenarios, 267 scenes from Habitat-Matterport 3D) with questions in five categories (Identification, Counting, Existence, State, Location). Compares against baselines FBE and CLIP-FBE.
- **Hardware**: Tests on a Fetch robot in home/office-like environments with 6 scenarios.

### Results
- **Simulation**:
  - Achieves ~60% success rate with maximum time steps, outperforming FBE and CLIP-FBE (Fig. 7a).
  - With fine-tuned VLM, success rate improves to 68.1% from 56.2%.
  - Shows significant improvement for Count and Existence questions (Fig. A6).
  - CP-based stopping reduces time steps (e.g., 71% vs. 85% for Relevance at 58% success rate, Fig. 9).
- **Hardware**:
  - Achieves 4/6 correct answers, matching Relevance but using fewer steps (e.g., 6 vs. 17 in Scenario 2, Fig. 10).
  - Failures in Scenarios 5 and 6 due to VLM’s incorrect predictions despite relevant views.

## My Thoughts

**I haven't fully understand the Conformal Prediction part. Many maths details are included. I intend to read more papers and then go back to it.**

This is solid work and has been cited by many other papers. This pipeline has become a commonly used baseline and actually the result is still rather good. The project provides codes and dataset, and I'm going to run it myself.

I can find some limitations in this paper. The VLMs are required to provide a **probability score** aside from the predicted result. That means most state-of-the-art VLMs (like GPT-4v) cannot be applied to this pipeline.
