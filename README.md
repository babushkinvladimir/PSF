### PFEM-based Ensemble Model for electricity load forecasting
For winter internship in Wolfram Lab I consider a robust Pattern Forecasting Ensemble Model (PFEM) for electricity load forecasting [1]. It extends the idea of Label-Based Forecasting (LBF) [2] and Pattern Sequence-based Forecasting (PSF) [3] algorithms, that analyzes similarity of pattern sequences to predicts the future dynamics in a time series and is capable to overperform several conventional methods as Naive Bayes, Neural Networks, ARIMA, Weighted Nearest Neighbors and the Mixed Models. The original PFEM algorithms deploys five independent experts, namely the K-means model (PSF itself), Self- Organizing Map model, Hierarchical Clustering model, K- medoids model, and Fuzzy C-means model. The predictions are obtained from each expert separately and combined into a weighted linear combination, where weights are determined iteratively. This iterative process results into minimization the forecasting error rates.


### References
[1] W. Shen, V. Babushkin, Z. Aung and W. Woon, An ensemble model for
day-ahead electricity demand time series forecasting, In Proceedings of
the 4th ACM Conference on Future Energy Systems (e-Energy), 2013.

[2] F. Martınez-Alvarez, A. Troncoso, J. Riquelme and J. S. Aguilar-Ruiz,
LBF: A labeled- based forecasting algorithm and its application to
electricity price time series,, Proceedings of the 8th IEEE
International Conference on Data Mining (ICDM’08), 2008.

[3] F. Martınez-Alvarez, A. Troncoso, J. C. Riquelme and J. S. A. Ruiz,
Energy time series forecasting based on pattern sequence similarity,
IEEE Transactions on Knowledge and Data Engineering, 2011.
