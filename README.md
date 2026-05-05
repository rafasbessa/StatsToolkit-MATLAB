# StatsToolkit for MATLAB

**StatsToolkit** is an automated, highly structured framework for Linear and Generalized Linear Mixed Models (LMM/GLMM) and Compositional Data Analysis (CoDA) in MATLAB. 

Designed for rigorous experimental designs (such as behavioral neuroscience data), this toolkit streamlines the statistical pipeline from raw data transformation to optimized model fitting, fixed-effect extraction, and complex post-hoc inverse mappings.

## 🚀 Key Features
* **Compositional Data Analysis (CoDA):** Robust transformation of bounded data (simplex) into unbounded real space using ILR, CLR, or ALR transformations, preventing the geometric distortion of constant-sum constraints.
* **Automated Model Optimization:** Algorithmic selection of optimal probability distributions and random slope structures via Information Criteria (AICc, BIC).
* **Standardized Effect Sizes:** Automatic computation of partial Eta-squared ($\eta_p^2$) and Omega-squared ($\omega_p^2$) for fixed effects in mixed-effects designs.
* **Rigorous Post-Hoc Comparisons:** N-way simple slopes and pairwise comparisons with False Discovery Rate (FDR) and Holm-Bonferroni corrections.
* **Topological Inverse Mapping:** Specialized inverse suites to map LMM/GLMM marginal predictions back to the original proportional space seamlessly.

## 🗂️ Module Architecture

The framework is organized into hierarchical levels, establishing a clear pipeline from data ingestion to output formatting.

### Level 0: Pre-processing & Transformations
* `StatsToolkit_Level0_Transform.m`: Projects raw proportional/compositional data into continuous space using Isomectric (ILR), Centered (CLR), or Additive (ALR) Log-Ratio transformations.

### Level 1: Model Selection & Fitting
* `StatsToolkit_Level1_SelectDistribution.m`: Evaluates and fits the optimal theoretical distribution for GLMM applications.
* `StatsToolkit_Level1_GlobalMetrics.m`: Extracts baseline model metrics and goodness-of-fit parameters.
* `StatsToolkit_Level1_SelectRandomSlopes.m`: Iteratively builds and compares random effect structures (intercepts vs. slopes) to prevent singular fits and optimize parsimony.

### Level 2: Main Effects Inference
* `StatsToolkit_Level2_FixedEffects.m`: Computes ANOVA tables and standardized effect sizes for the optimized models.

### Level 3: Post-Hoc Analysis
* `StatsToolkit_Level3_PostHoc.m`: Executes high-resolution pairwise comparisons and simple main effects for complex interaction terms (e.g., Treatment * Time).

### Level 4: Inverse Mapping & Results
* `StatsToolkit_Level4_CompositionalInverse.m`: Extracts conditional predictions from multiple orthogonal LME coordinates and applies exact matrix inversions (e.g., inverse Helmert matrices) and closure operations to return data to the geometric simplex.
* `StatsToolkit_Level4_InverseLinkSuite.m`: Handles standard GLMM inverse link functions with safety mechanisms for conditional random-effect predictions.

## ⚙️ Requirements
* **MATLAB** (Developed and tested on version R2025a)
* **Statistics and Machine Learning Toolbox**

## 💻 Quick Start
1. Clone the repository to your local machine:
   ```bash
   git clone [https://github.com/rafasbessa/StatsToolkit-MATLAB.git](https://github.com/rafasbessa/StatsToolkit-MATLAB.git)
   
   
2. Add the folder to your MATLAB path:

   Matlab

   addpath(genpath('/path/to/StatsToolkit-MATLAB'));
   savepath;

3. Call the functions hierarchically following your experimental design needs. See individual function headers for specific syntax and argument details.

## 📄 License
This project is licensed under the **GNU General Public License v3.0** - see the `LICENSE.md` file for details.
