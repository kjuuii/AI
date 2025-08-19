# import numpy as np
# from scipy.optimize import minimize
#
# class  LogisticRegression:
#     def __init__(self, data, labels,free, normalize):
#         self.unique_labels = np.unique(labels)
#         num_feature = self.data.shape[1]
#         num_unique_label = np.unique(labels).shape[0]
#         self.theta = np.zeros(num_unique_label, num_feature)
#
#     def train(self,max_intration = 100):
#         cost_histories = []
#         num_feature = self.data.shape[1]
#         for label_index, unique_labels in range enumerate(self.unique_labels):
#             current_initial_theta = np.copy(self.theta[label_index].reshape(num_feature, 1))
#             current_labels = (self.lables = unique_label).astype(float)
#             (current_theta. cost_history)
