import numpy as np
from scipy.stats import norm

def plot_with_quantiles(ax, x, data, c, q=[0.1,0.9], **kwargs):
    mean = data.mean(axis=0)
    quantiles = np.quantile(data, axis=0, q=q)
    ax.plot(x, mean, c=c, **kwargs)
    ax.fill_between(x, quantiles[0], quantiles[1], color=c, alpha=0.25)
    
    
def plot_with_confidence(ax, x, data, c, alpha, **kwargs):
    mean = data.mean(axis=0)
    ci_width = norm.ppf(1-alpha/2)*data.std(axis=0)/np.sqrt(len(mean))
    ax.plot(x, mean, c=c, **kwargs)
    ax.fill_between(x, mean-ci_width, mean+ci_width, color=c, alpha=0.25)
    
def plot_with_agresti_coull(ax, x, data, c, q=[0.05,0.95], **kwargs):
    n = len(data) + 4
    n_s = np.sum(data, axis=0) + 2
    p_adj = n_s / n
    
    p_hat = np.mean(data, axis=0)
    
    ci_width = 1.96 * np.sqrt(p_adj * (1-p_adj)/n)
    ci_lb = np.clip(p_hat - ci_width, 0,1)
    ci_ub = np.clip(p_hat + ci_width, 0,1)
    
    ax.plot(x, p_hat, c=c, **kwargs)
    ax.fill_between(x, ci_lb, ci_ub, color=c, alpha=0.25)
    
def indicate_point(ax, x, y, **kwargs):
    xmin, xmax = ax.get_xlim() 
    ymin, ymax = ax.get_ylim() 
    
    ax.hlines(y=y, xmin=xmin, xmax=x, **kwargs)
    ax.vlines(x=x, ymin=ymin, ymax=y,  **kwargs)
    
    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))