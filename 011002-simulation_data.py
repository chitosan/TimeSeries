import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns; sns.set_theme()
import requests
import io
import yfinance as yf
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn import linear_model
from scipy.stats import norm
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf, pacf
from scipy.special import comb


def sim_dataset(AR, MA, t=100, trend = None, polynomial_root = -0.5, MA_weight = 0.5):
    """
    Simulates a dataset given AR or MA order.
    Selects AR terms so that the polynomial root is the given value;
    as long as the root is within (-1, 1), the series will be stationary.
    """
    if trend is None:
        trend = lambda x: 0
    arparams = np.array([comb(AR, i)*(polynomial_root)**(i) for i in range(1, AR + 1)])
    maparams = np.array([MA_weight] * MA)
    ar = np.r_[1, arparams] # add zero-lag
    ma = np.r_[1, maparams] # add zero-lag
    arma_process = sm.tsa.ArmaProcess(ar, ma)
    print("ARMA process is stationary: ", arma_process.isstationary)
    print("ARMA process is invertible: ", arma_process.isinvertible)
    y = arma_process.generate_sample(t)
    y = np.array([_y + trend(j) for j, _y in enumerate(y)])
    return y


# White Noise
np.random.seed(1234)
white_noise = sim_dataset(0, 0, 100)
plt.plot(white_noise)
plt.show()
sm.graphics.tsa.plot_acf(white_noise, lags=15)
plt.show()
sm.graphics.tsa.plot_pacf(white_noise, lags=15)
plt.show()

np.random.seed(1234)
AR_series = sim_dataset(2, 0, 250, polynomial_root = -0.8)
plt.plot(AR_series)
plt.show()
sm.graphics.tsa.plot_acf(AR_series, lags=15)
plt.show()
sm.graphics.tsa.plot_pacf(AR_series, lags=15)
plt.show()