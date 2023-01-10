import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from pandas_datareader import data as pdr
import yfinance as yfin
from scipy.stats import norm
import statistics

yfin.pdr_override()


def get_data(stocks, start, end):
    data = pdr.get_data_yahoo(stocks, start, end)
    data = data['Close']
    returns = data.pct_change()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    return mean_returns, cov_matrix


def monte_carlo_simulation(weights, mean_returns, cov_matrix, initial_portfolio,
                           NUM_SIMULATIONS, TIME_FRAME):
    """Uses formula r = mu + Cz to calculate expected return,
    where C is cholseky transformation of covariance matrix and
    z is vector of random returns from normal distribution

    Args:
        weights (numpy array): Array of randomly generated weights (weights sum to 1)
        mean_returns (pandas series): mean percent return over specified period for each stock
        cov_matrix (pandas df): covariance matrix of returns
        initial_portfolio (float): starting value of portfolio
        NUM_SIMULATIONS (int): number of simulations to be computed
        TIME_FRAME (int): graph x axis value

    Returns:
        _type_: _description_
    """
    mean_matrix = np.full(shape=(TIME_FRAME, len(weights)),
                          fill_value=mean_returns)
    mean_matrix = mean_matrix.T  # transpose of the array
    portfolio_sims = np.full(shape=(TIME_FRAME, NUM_SIMULATIONS),
                             fill_value=0.0)
    for i in range(0, NUM_SIMULATIONS):
        Z = np.random.normal(size=(TIME_FRAME,
                                   len(weights)))  # vector of random returns from normal distribution
        lower_triangle = np.linalg.cholesky(
            cov_matrix)  # use Cholesky transformation of cov matrix to get lower trianglular matrix
        daily_returns = mean_matrix + np.inner(lower_triangle, Z)
        portfolio_sims[:, i] = np.cumprod(
            np.inner(weights, daily_returns.T) + 1) * initial_portfolio
    return portfolio_sims


def plot_line_graph(portfolio_sims):
    plt.plot(portfolio_sims)
    plt.ylabel('Portfolio Value ($)')
    plt.xlabel('Days')
    plt.title('Monte Carlo Simulation of a Stock Portfolio')


def plot_histogram(portfolio_sims, NUM_SIMULATIONS):
    expected_vals = portfolio_sims[-1:].ravel()
    filtered = expected_vals[~is_outlier(expected_vals)]
    bin_count = int(np.ceil(np.log2(
        NUM_SIMULATIONS) + 1))  # sturge's rule for estimating number of bins in histogram
    plt.hist(filtered, bins=bin_count)  # need to update to auto bins

    mu, std = norm.fit(expected_vals)

    title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
    plt.title(title)

    plt.ylabel('Portfolio Value ($)')
    plt.xlabel('Count')


def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.

        Taken from Stack Overflow.
    """
    if len(points.shape) == 1:
        points = points[:, None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median) ** 2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


if __name__ == "__main__":
    INITIAL_PORTFOLIO = 100000.
    NUM_SIMULATIONS = 50000
    TIME_FRAME = 300
    DURATION = TIME_FRAME
    STOCK_LIST = ['AMZN', 'MSFT', 'AAPL', 'TSLA']
    stocks = [stock for stock in STOCK_LIST]
    end_date = dt.datetime.now()
    # end_date = dt.datetime.strptime("28/12/22 12:30", "%d/%m/%y %H:%M")
    start_date = end_date - dt.timedelta(days=DURATION)
    mean_returns, cov_matrix = get_data(stocks, start_date, end_date)

    weights = np.random.random(len(mean_returns))
    weights /= np.sum(weights)

    portfolio_sims = monte_carlo_simulation(weights, mean_returns, cov_matrix,
                                            INITIAL_PORTFOLIO, NUM_SIMULATIONS,
                                            TIME_FRAME)
    # print(portfolio_sims)
    # print(portfolio_sims[-1,:])
    plt.subplot(2, 1, 1)
    plot_line_graph(portfolio_sims)
    plt.subplot(2, 1, 2)
    plot_histogram(portfolio_sims, NUM_SIMULATIONS)
    plt.tight_layout()
    plt.show()