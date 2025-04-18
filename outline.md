# Project Title

*Summarizes the main idea of your project.*

Reimplementation of *SOFTS: Efficient Multivariate Time Series Forecasting with Series-Core Fusion* (NeurIPS 2024).

## Who
<!-- - Letian Shen -->
|   Name  |  GitHub  |  Login |
|:-------:|:--------:|:-------:|
| Letian Shen | syd-mbv | lshen33 |
| Tianxi Lu   | degeneratorL | tlu44 |
| Yinan Zhai   | Kyle-zhai  |  |
## Introduction

- **Problem Definition:**  
  - *Implementing an existing paper: describe the paper's objectives and reasons for choosing this paper.*
  - Objectives: Effectively integrating the robustness of channel independence and utilizing the correlation between channels in a simpler and more efficient manner is crucial for building better time series forecasting models.
  - The paper makes the following 3 contributions:
      1.  Presents Series-cOre Fused Time Series (SOFTS) forecaster, a a simple MLP-based model that demonstrates state-of-the-art performance with lower complexity.
      2.  Introduces the  STar Aggregate-Redistribute (STAR) module, which serves as the foundation of SOFTS. STAR is designed as a centralized structure that uses a core to aggregate and exchange information from the channels. Compared to distributed structures like attention, the STAR not only reduces the complexity but also improves robustness against anomalies in channels.
      3.  Through extensive experiments, the effectiveness and scalability of SOFTS are validated. The universality of STAR is also validated on various attention-based time series forecasters.

  - *Clearly state the type of problem (Classification, Regression, Structured prediction, Reinforcement Learning, Unsupervised Learning, etc.).*
  - This problem, time series forecasting, is primarily a supervised learning problem, typically formulated as a regression task.

## Related Work

- **Literature Review:**
  - *Briefly summarize (one paragraph) at least one relevant paper/article/blog beyond the paper you are implementing or the novel idea you're researching.*
  - The researchers used this method as it is adapted from ITRANSFORMER paper, to consider normalization as a hyperparameter. Reversible Instance Normalization in iTransformer simply applies attention and feed-forward networks on the inverted dimensions, enabling the model to capture multivariate correlations and learn nonlinear representations effectively.

- **References and Implementations:**
  - Include URLs to any public implementations you find relevant.
  - *(Treat this as a "living list" and update it as you find new resources.)*
  - ITRANSFORMER:
    - Paper: https://arxiv.org/abs/2310.06625
    - Code repository: https://github.com/thuml/iTransformer

## Data

- **Dataset Description:**
  - *Brief description (if standard dataset like MNIST, briefly mention; otherwise, explain source and collection method).*
  - *Size of the dataset.*
  - *Discuss preprocessing requirements (if any).*
  - Datasets used in the paper: https://drive.google.com/drive/folders/1QPM7MMKlzVffdzbGGkzARDuIqiYRed_f
    - Electricity
    - ETT-small
    - PEMS
    - Solar
    - Traffic
    - Weather
  - ETT(Electricity Transformer Temperature) comprises two hourly-level datasets (ETTh) and two 15-minute-level datasets (ETTm). Each dataset contains seven oil and load features of electricity transformers from July 2016 to July 2018. https://github.com/zhouhaoyi/ETDataset
  - Traffic describes the road occupancy rates. It contains the hourly data recorded by the sensors of San Francisco freeways from 2015 to 2016. https://pems.dot.ca.gov/
  - Electricity collects the hourly electricity consumption of 321 clients from 2012 to 2014. https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014
  - Weather r includes 21 indicators of weather, such as air temperature, and humidity. Its data is recorded every 10 min for 2020 in Germany.
  - Solar-Energy records the solar power production of 137 PV plants in 2006, which is sampled every 10 minutes.
  - PEMS contains public traffic network data in California collected by 5-minute windows. https://pems.dot.ca.gov/
- Size of the dataset:
  |   Dataset  |  Channels | Dataset Split | Size(MB) |
  |:-------:|:--------:|:--------:|:--------:|
  | ETTh1, ETTh2 | 7 | (8545, 2881, 2881) | 2.5, 2.3 |
  | ETTm1, ETTm2 | 7 | (34465, 11521, 11521) | 9.9, 9.2 |
  | Weather  | 21 | (36792, 5271, 10540) | 6.9 |
  | ECL  | 321 | (18317, 2633, 5261) | 91.1 |
  | Traffic  | 862 | (12185, 1757, 3509) | 130 |
  | Solar-Energy | 137 | (36601, 5161, 10417) | 171 |
  | PEMS03 | 358 | (15617,5135,5135) | 15.0 |
  | PEMS04 | 307 | (10172,3375,3375) | 31.4 |
  | PEMS07 | 883 | (16911,5622,5622) | 41.6 |
  | PEMS08 | 170 | (10690,3548,3548) | 17.6 |

## Metrics

- **Definition of Success:**
  - Success will be defined by achieving comparable or better performance metrics than those reported in the original SOFTS paper.
  - The implementation should demonstrate linear complexity with respect to the number of channels and time steps as claimed in the paper.
  - The STAR module should effectively capture channel correlations while maintaining robustness against anomalies.

- **Experiments Planned:**
  - Implement the SOFTS model with the STAR module using TensorFlow.
  - Reproduce the multivariate time series forecasting experiments on the datasets used in the original paper (ETT, Traffic, Electricity, Weather, Solar-Energy, and PEMS).
  - Compare performance against baseline models mentioned in the paper (DLinear, TSMixer, TiDE, FEDformer, Stationary, PatchTST, Crossformer, iTransformer, SCINet, TimesNet).
  - Perform ablation studies on different pooling methods in the STAR module (mean pooling, max pooling, weighted average, stochastic pooling).
  - Test the universality of STAR by replacing attention mechanisms in other transformer-based models.
  - Evaluate model performance with varying lookback window lengths.
  - Analyze the impact of hyperparameters (hidden dimension, core dimension, number of encoder layers).
  - Test model robustness against channel noise.

- **Metrics:**
  - Primary evaluation metrics: Mean Squared Error (MSE) and Mean Absolute Error (MAE).
  - Model efficiency metrics: memory usage and inference time.
  - The original paper reported significant improvements over state-of-the-art methods (e.g., 4.4% reduction in average MSE on Traffic dataset and 13.9% reduction on PEMS07 dataset).

- **Goals:**
  - Base goal: Successfully implement the SOFTS model in TensorFlow that can train and generate predictions on the benchmark datasets.
  - Target goal: Achieve performance metrics within 5% of those reported in the original paper across all datasets.
  - Stretch goal: Optimize the implementation to achieve better efficiency than reported in the paper while maintaining comparable accuracy, and potentially extend the model to additional datasets or applications.

## Ethics

- **Broader societal issues related to your chosen problem space:**
  - Time series forecasting has significant applications in critical domains like traffic management, energy consumption, healthcare, and financial markets. Improved forecasting models can lead to better resource allocation, reduced energy waste, and more efficient systems.
  - However, over-reliance on automated forecasting systems without human oversight could lead to issues if predictions fail during critical situations or unusual circumstances.
  - The deployment of such models in sensitive domains like healthcare or financial systems requires careful consideration of potential downstream impacts on human lives.

- **Dataset collection and labeling concerns, representativeness, and possible biases:**
  - The datasets used in this study are collected from specific regions and time periods, which may limit their generalizability to other contexts. For example, traffic patterns in San Francisco may not represent traffic behaviors in other cities or countries.
  - Temporal biases may exist in these datasets, such as seasonal patterns, economic cycles, or unique events during the collection period that might not be representative of future data.
  - The model performance may vary across different channels within the datasets, potentially leading to uneven quality of predictions across different sensors or variables.
  - Data from real-world sensors often contains noise, missing values, or anomalies that might not be uniformly distributed, affecting the model's performance on certain subgroups of data.

- **Major stakeholders and implications of algorithmic errors:**
  - Stakeholders include utility companies, traffic management authorities, healthcare systems, investors, and the general public affected by decisions made based on these forecasts.
  - Errors in energy load forecasting could lead to grid instability, blackouts, or unnecessary energy production with environmental consequences.
  - Inaccurate traffic predictions might result in congestion, longer commute times, and increased pollution.
  - In healthcare applications, errors could potentially impact patient care decisions or resource allocation.
  - The STAR module's ability to handle anomalous channels is particularly important, as the paper claims improved robustness against abnormal data points compared to other models. This feature should be carefully validated in the implementation.

## Division of Labor

- **Letian Shen:**
  - Lead the overall implementation of the SOFTS model in TensorFlow
  - Implement the core STAR module functionality
  - Coordinate integration of components
  - Design and run experiments for model performance comparison

- **Tianxi Lu:**
  - Implement data preprocessing pipeline for all datasets
  - Set up training and evaluation framework
  - Conduct ablation studies on pooling methods
  - Analyze and visualize results

- **Yinan Zhai:**
  - Implement the series embedding and linear predictor components
  - Test model robustness against channel noise
  - Design experiments for varying lookback window lengths
  - Create documentation and prepare final presentation
