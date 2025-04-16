# Project Title

*Summarizes the main idea of your project.*

Reimplementation of *SOFTS: Efficient Multivariate Time Series Forecasting with Series-Core Fusion* (NeurIPS 2024).

## Who
<!-- - Letian Shen -->
|   Name  |  Login   |
|:-------:|:--------:|
| Letian Shen | syd-mbv |
| Tianxi Lu   | degeneratorL |
| Yinan Zhai   |  |
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
  - Clearly define what constitutes "success."

- **Experiments Planned:**
  - Specify experiments to be conducted.
  - Justify the choice of metrics (accuracy or alternative metrics).
  - For existing paper implementations, mention original metrics and expectations.
  - If novel, explain how performance will be assessed.

- **Goals:**
  - Base, target, and stretch goals.

## Ethics

*(Select and discuss at least two of the following points relevant to your project.)*

- Broader societal issues related to your chosen problem space.
- Suitability of deep learning approach for your problem.
- Dataset collection and labeling concerns, representativeness, and possible biases.
- Major stakeholders and implications of algorithmic errors.
- Methods to quantify or measure error/success and related implications.
- *(Optional)* Any additional ethical issue related to your algorithm.

## Division of Labor

- Brief outline of each group member’s responsibilities.