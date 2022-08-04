import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.graphics.tsaplots import plot_acf
import numpy as np
import scipy.optimize as opt
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')
np.random.seed(3)


class OU:
    """
    This file holds the code for modeling volatility using the Ornstein-Uhlenbeck model

    Sometimes, an additional drift term is added - this is known as the Vasicek model
    """

    def mle_norm(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the mean and variance of a numpy array
        :param x: ndarray
        :return: mean and variance of ndarray
        """
        mu_hat = np.mean(x)
        sigma2_hat = np.var(x)
        return mu_hat, sigma2_hat

    def _log_likelihood(self, theta: list[int, int], x: np.ndarray) -> np.ndarray:

        mu = theta[0]
        sigma = theta[1]

        l_theta = np.sum(np.log(stats.norm.pdf(x, loc=mu, scale=sigma)))

        return -l_theta

    def _sigma_pos(self, theta):
        sigma = theta[1]
        return sigma

    def optimize(self, x):
        constraint = {'type': 'ineq', 'fun': self._sigma_pos}
        theta0 = [2, 3]
        optimizer = opt.minimize(fun=self._log_likelihood, x0=theta0, args=(x,), constraints=constraint)
        return optimizer

    def mu(self, x, dt, kappa, theta):
        ekt = np.exp(-kappa*dt)
        return x*ekt + theta*(1-ekt)

    def std(self, dt, kappa, sigma):
        e2kt = np.exp(-2*kappa*dt)
        return sigma*np.sqrt((1-e2kt)/(2*kappa))

    def log_likelihood_ou(self, theta_hat, x):
        kappa = theta_hat[0]
        theta = theta_hat[1]
        sigma = theta_hat[2]

        x_dt = x[1:]
        x_t = x[:-1]

        dt = 1 / 252

        mu_OU = self.mu(x_t, dt, kappa, theta)
        sigma_OU = self.std(dt, kappa, sigma)

        l_theta_hat = np.sum(np.log(stats.norm.pdf(x_dt, loc=mu_OU, scale=sigma_OU)))

        return -l_theta_hat

    def kappa_pos(self, theta_hat):
        kappa = theta_hat[0]
        return kappa

    def sigma_pos(self, theta_hat):
        sigma = theta_hat[2]
        return sigma

    def optimize_ou(self, x: np.ndarray):
        theta0 = [1, 1, 1]
        cons_set = [{'type': 'ineq', 'fun': self.kappa_pos},
                    {'type': 'ineq', 'fun': self.sigma_pos}]
        optimizer = opt.minimize(fun=self.log_likelihood_ou, x0=theta0, args=(x,), constraints=cons_set)
        return optimizer


def get_data(stocks: str | list[str], start="2002-01-01") -> pd.DataFrame:
    data = yf.download(stocks, start=start)
    return data


stock = "^GSPC"
s_p = get_data(stock)
log_returns = np.log(s_p["Adj Close"] / s_p["Adj Close"].shift(1)).dropna()

log_returns.plot()
plt.title("S&P Daily Log Returns")
plot_acf(log_returns)  # little to no correlation shown in log returns
plt.show()

# square of log returns
log_returns_sq = np.square(log_returns)
log_returns_sq.plot()
plt.title("S&P Squared Daily Log Returns")
plot_acf(log_returns_sq)  # very strong correlation of the absolute magnitude
# It shows Volatility Clustering - periods with high vol will continue to see high vol and vice versa
# shown by the auto-correlation of abs magnitude of log rets
plt.show()

TRADING_DAYS = 40

vol = (log_returns.rolling(window=TRADING_DAYS).std() * np.sqrt(252)).dropna()

vol.plot()
plt.title("S&P Vol")
plt.show()

vol = np.array(vol)
returns = s_p["Adj Close"] / s_p["Adj Close"].shift(1).dropna()
stock_rets = np.array(np.log(s_p["Adj Close"].pct_change()))
ou = OU()
opt_ou = ou.optimize_ou(vol)
print(
    f"Kappa (1 = 1 year): {round(opt_ou.x[0], 3)}, Theta (mean of Kappa): {round(opt_ou.x[1], 3)}, "
    f"Sigma (standard deviation of Kappa): {round(opt_ou.x[2], 3)}"
)

