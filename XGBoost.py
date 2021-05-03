import os
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import xgboost as xgb
import scipy
from sklearn.metrics import fbeta_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report
from PIL import Image

random_seed = 1
random.seed(random_seed)
np.random.seed(random_seed)


# Function to extract the image features
def extract_features(df, data_path):
    im_features = df.copy()

    N = len(im_features.image_name.values)

    r_mean = np.zeros(N)
    g_mean = np.zeros(N)
    b_mean = np.zeros(N)

    r_std = np.zeros(N)
    g_std = np.zeros(N)
    b_std = np.zeros(N)

    r_max = np.zeros(N)
    g_max = np.zeros(N)
    b_max = np.zeros(N)

    r_min = np.zeros(N)
    g_min = np.zeros(N)
    b_min = np.zeros(N)

    r_kurtosis = np.zeros(N)
    g_kurtosis = np.zeros(N)
    b_kurtosis = np.zeros(N)

    r_skewness = np.zeros(N)
    g_skewness = np.zeros(N)
    b_skewness = np.zeros(N)

    for i, image_name in enumerate(tqdm(im_features.image_name.values, miniters=1000)):
        im = Image.open(data_path + image_name)
        im = np.array(im)[:, :, :3]

        r = im[:, :, 0].ravel()
        g = im[:, :, 1].ravel()
        b = im[:, :, 2].ravel()

        r_mean[i] = np.mean(r)
        g_mean[i] = np.mean(g)
        b_mean[i] = np.mean(b)

        r_std[i] = np.std(r)
        g_std[i] = np.std(g)
        b_std[i] = np.std(b)

        r_max[i] = np.max(r)
        g_max[i] = np.max(g)
        b_max[i] = np.max(b)

        r_min[i] = np.min(r)
        g_min[i] = np.min(g)
        b_min[i] = np.min(b)

        r_kurtosis[i] = scipy.stats.kurtosis(r)
        g_kurtosis[i] = scipy.stats.kurtosis(g)
        b_kurtosis[i] = scipy.stats.kurtosis(b)

        r_skewness[i] = scipy.stats.skew(r)
        g_skewness[i] = scipy.stats.skew(g)
        b_skewness[i] = scipy.stats.skew(b)

    im_features['r_mean'] = r_mean
    im_features['g_mean'] = g_mean
    im_features['b_mean'] = b_mean

    im_features['rgb_mean_mean'] = (r_mean + g_mean + b_mean) / 3.0

    im_features['r_std'] = r_std
    im_features['g_std'] = g_std
    im_features['b_std'] = b_std

    im_features['rgb_mean_std'] = (r_std + g_std + b_std) / 3.0

    im_features['r_max'] = r_max
    im_features['g_max'] = g_max
    im_features['b_max'] = b_max

    im_features['rgb_mean_max'] = (r_max + r_max + b_max) / 3.0

    im_features['r_min'] = r_min
    im_features['g_min'] = g_min
    im_features['b_min'] = b_min

    im_features['rgb_mean_min'] = (r_min + g_min + b_min) / 3.0

    im_features['r_range'] = r_max - r_min
    im_features['g_range'] = g_max - g_min
    im_features['b_range'] = b_max - b_min

    im_features['r_kurtosis'] = r_kurtosis
    im_features['g_kurtosis'] = g_kurtosis
    im_features['b_kurtosis'] = b_kurtosis

    im_features['r_skewness'] = r_skewness
    im_features['g_skewness'] = g_skewness
    im_features['b_skewness'] = b_skewness

    return im_features


class XGBoostClassifier:

    def __init__(self, max_depth, learning_rate, n_estimators, objective, n_jobs, gamma, min_child_weight, max_delta_step,
                 subsample, colsample_bytree, colsample_bylevel, reg_alpha, reg_lambda, scale_pos_weight, base_score, seed,
                 use_label_encoder, eval_metric):
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.objective = objective
        self.n_jobs = n_jobs
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.max_delta_step = max_delta_step
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.scale_pos_weight = scale_pos_weight
        self.base_score = base_score
        self.seed = seed
        self.use_label_encoder = use_label_encoder
        self.eval_metric = eval_metric
        self.models = []
        self.n_classes = None

    def create_model(self):
        model = xgb.XGBClassifier(max_depth=self.max_depth, learning_rate=self.learning_rate,
                                  n_estimators=self.n_estimators, objective=self.objective, n_jobs=self.n_jobs, gamma=self.gamma,
                                  min_child_weight=self.min_child_weight, max_delta_step=self.max_delta_step,
                                  subsample=self.subsample, colsample_bytree=self.colsample_bytree,
                                  colsample_bylevel=self.colsample_bylevel, reg_alpha=self.reg_alpha,
                                  reg_lambda=self.reg_lambda, scale_pos_weight=self.scale_pos_weight,
                                  base_score=self.base_score, seed=self.seed, use_label_encoder=self.use_label_encoder, eval_metric=self.eval_metric)

    def fit(self, X_train, y_train):
        self.n_classes = X_train.shape[1]
        for i in tqdm(range(self.n_classes), miniters=1):
            model = self.create_model()
            model.fit(X_train, y_train[:, i])
            self.models.append(model)

    def predict(self, X):
        y_pred = np.zeros((X.shape[0], self.n_classes))  # (num_of_test_images, n_classes=17)
        for i in tqdm(range(self.n_classes), miniters=1):
            model = self.models[i]
            y_pred[:, i] = model.predict_proba(X)[:, 1]

        return y_pred
