# Project Title


## Introduction
  - Objectives: Effectively integrating the robustness of channel independence and utilizing the correlation between channels in a simpler and more efficient manner is crucial for building better time series forecasting models.
  - The paper makes the following 3 contributions:
      1.  Presents Series-cOre Fused Time Series (SOFTS) forecaster, a a simple MLP-based model that demonstrates state-of-the-art performance with lower complexity.
      2.  Introduces the  STar Aggregate-Redistribute (STAR) module, which serves as the foundation of SOFTS. STAR is designed as a centralized structure that uses a core to aggregate and exchange information from the channels. Compared to distributed structures like attention, the STAR not only reduces the complexity but also improves robustness against anomalies in channels.
      3.  Through extensive experiments, the effectiveness and scalability of SOFTS are validated. The universality of STAR is also validated on various attention-based time series forecasters.


## Challenges
  - Re-implement the system in TensorFlow framework instead of PyTorch. 
  - Simple ```Dataloader``` in PyTorch must be implemented manually using TensorFlow.
  - 
## Insights
Are there any concrete results you can show at this point?
How is your model performing compared with expectations?
 - We have built the entire system in the tensorflow framework.
 - Successfully completed most of the experiments in the paper, completed training and testing on the datasets mentioned in the paper and collected results.
 - 
 - The Data pipeline is tricky to build.
 - Some OS operations need to consider different scenarios for Linux and Windows.
## Plan
Are you on track with your project?
What do you need to dedicate more time to?
What are you thinking of changing, if anything?
 - Basically, we are on track with out project.
 - We need to invest more time in experimenting on new datasets and processing the results.
 - Maybe it could, hopefully, reduce the model prediction error a little bit.
