import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from pandas_datareader import data as pdr
import yfinance as yfin
from scipy.stats import norm
import streamlit as st
from st_aggrid import AgGrid, DataReturnMode, GridUpdateMode, GridOptionsBuilder
import csv
import re
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


yfin.pdr_override()

DEBUG_MODE = 0
# DEBUG_MODE == 0 IS RANDOM WEIGHTS, 1 IS NO USER INPUT, 2 IS SET WEIGHTS

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
    # print(expected_vals)
    filtered = expected_vals[~is_outlier(expected_vals)]
    try:
        bin_count = int(np.ceil(np.log2(
            NUM_SIMULATIONS) + 1))  # sturge's rule for estimating number of bins in histogram
        plt.hist(filtered, bins=bin_count)  # need to update to auto bins
    except:
        plt.hist(filtered)
        st.markdown(":red[Error: Please set Time Frame and Number of "
                    "Simulations to Run to be greater than 0]")
    mu, std = norm.fit(expected_vals)

    title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
    plt.title(title)

    plt.ylabel('Count')
    plt.xlabel('Portfolio Value ($)')


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

def random_weights_user_inputs():
    st.slider("Starting Portfolio Value", 0, 1000000, key="initial_val")
    st.slider("Time Frame (number of days)", 0, 1095, key="num_days")
    st.slider("Number of Simulations to Run", 0, 10000, key="num_simuls")
    options = st.multiselect(
        'What stocks make up your portfolio?', tickers_list, key="stock_list")

def user_inputs():
    st.slider("Time Frame (number of days)", 0, 1095, key="num_days")
    st.slider("Number of Simulations to Run", 0, 10000, key="num_simuls")
    num_row = st.number_input("Number of Rows", min_value=2, max_value=50)
    st.session_state.num_row = num_row
    if 'num_row' not in st.session_state:
        st.session_state.num_row = 2
    df_portfolio = pd.DataFrame(
        '',
        index=range(st.session_state.num_row),
        columns=["Tickers (ex. AAPL)", "Value (ex. 5000)"]
    )
    with st.form('Portfolio'):
        st.subheader("Current Portfolio")
        response = AgGrid(df_portfolio, editable=True, fit_columns_on_grid_load=True)
        st.form_submit_button()
        # st.write(response['data'])
    return response['data']

def header():
    st.header("iPortfolio")
    st.write("Estimate risk and uncertainty of a portfolio of "
                 "multiple assets through Correlated Monte "
                 "Carlo Simulation using "
                 "Cholesky "
                 "Decomposition.")
    st.write("Begin by toggling the initial portfolio value worth, time frame "
             "of historical prices, list of tickers within the portfolio, "
             "and number of simulations to run.")

def front_end():
    st.pyplot(plt)

def clean_val(x):
    special_string = "spe@#$ci87al*&"
    x = re.sub(special_string, "", x)
    x = x.upper()
    return x

if __name__ == "__main__":
    header()
    weights = []
    try:
        if DEBUG_MODE == 0:
            INITIAL_PORTFOLIO = 50000
            # STOCK_LIST = ['AMZN', 'MSFT', 'AAPL', 'TSLA', 'GOOGL', 'V']
            with open('sp500_companies.csv') as f:
                reader = csv.reader(f)
                data = pd.DataFrame(reader)
                tickers_list = data[1].tolist()
                tickers_list = tickers_list[1:]
            STOCK_LIST = [tickers_list[i] for i in random.sample(range(0,
                                                                       495),
                                                                 random.randint(2, 50))]
            print(STOCK_LIST)
            TIME_FRAME = random.randint(100, 500)
            print(TIME_FRAME)
            NUM_SIMULATIONS = random.randint(1, 10000)
            print(NUM_SIMULATIONS)
        elif DEBUG_MODE == 1:
            random_weights_user_inputs()
            INITIAL_PORTFOLIO = st.session_state.initial_val
            STOCK_LIST = st.session_state.stock_list
            TIME_FRAME = st.session_state.num_days
            NUM_SIMULATIONS = st.session_state.num_simuls
        elif DEBUG_MODE == 2:
            col1 = "Tickers (ex. AAPL)"
            col2 = "Value (ex. 5000)"
            df_portfolio = user_inputs()
            TIME_FRAME = st.session_state.num_days
            NUM_SIMULATIONS = st.session_state.num_simuls
            try:
                df_portfolio[col2] = df_portfolio[col2].astype('float')
                INITIAL_PORTFOLIO = df_portfolio[col2].sum()
                df_portfolio[col1] = df_portfolio[col1].apply(lambda x :
                                                             clean_val(x))
                STOCK_LIST = df_portfolio[col1].tolist()
                weights = df_portfolio[col2].to_numpy()
            except:
                st.markdown(":red[Error: Please ensure all weights are properly "
                            "entered and there are no duplicates]")
        DURATION = TIME_FRAME
        stocks = [stock for stock in STOCK_LIST]
        end_date = dt.datetime.now()
        # end_date = dt.datetime.strptime("28/12/22 12:30", "%d/%m/%y %H:%M")
        start_date = end_date - dt.timedelta(days=DURATION)
        mean_returns, cov_matrix = get_data(stocks, start_date, end_date)


        if DEBUG_MODE != 2:
            weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)

        portfolio_sims = monte_carlo_simulation(weights, mean_returns, cov_matrix,
                                                INITIAL_PORTFOLIO, NUM_SIMULATIONS,
                                                TIME_FRAME)
        plt.subplot(2, 1, 1)
        plot_line_graph(portfolio_sims)
        plt.subplot(2, 1, 2)
        plot_histogram(portfolio_sims, NUM_SIMULATIONS)
        plt.tight_layout()
        plt.show()
        # st.write('You selected:', options)
    except:
        INITIAL_PORTFOLIO = 0
        STOCK_LIST = []
        TIME_FRAME = 0
    front_end()
