import pandas as pd
import numpy as np
from scipy.stats import norm, uniform, expon, poisson, binom, geom, gamma, beta, lognorm, chi2, t, f
from scipy.stats import randint, multinomial, dirichlet
from scipy.stats import describe, kstest
from statsmodels.stats.multitest import multipletests

def test_col_distributions(data, alpha=0.05):
    """
    Test the distribution of each column in a pandas DataFrame against specified distributions.
    
    Parameters:
    - data: pandas DataFrame
    - alpha: significance level for hypothesis tests
    
    Returns:
    - results: DataFrame with distribution summary for each column
    """

    distributions = {
        'Gaussian': norm,
        'Uniform': uniform,
        'Exponential': expon,
        'Poisson': poisson,
        'Binomial': binom,
        'Geometric': geom,
        'Gamma': gamma,
        'Beta': beta,
        'Log-Normal': lognorm,
        'Chi-Squared': chi2,
        "Student's t": t,
        'F': f,
        'Discrete Uniform': randint,
        'Multinomial': multinomial,
        'Dirichlet': dirichlet
    }

    results_columns = ['Column', 'Distribution', 'p-value', 'Summary Statistics']
    results_data = []

    for col in data.columns:
        col_data = data[col].dropna()

        best_fit_dist = None
        best_fit_pvalue = np.inf
        best_fit_params = None

        for dist_name, dist_func in distributions.items():

            try:
                # Fit the distribution
                params = dist_func.fit(col_data)

                # Perform a Kolmogorov-Smirnov test
                _, p_value = kstest(col_data, dist_name, args=params)
                
                # Update best fit if current distribution has a lower p-value
                if p_value < best_fit_pvalue:
                    best_fit_dist = dist_name
                    best_fit_pvalue = p_value
                    best_fit_params = params
            except (RuntimeError, ValueError) as e:
                # Handle runtime and value errors during distribution fitting
                print(f"Warning: {e} occurred during fitting {dist_name} to column {col}. Skipping this distribution.")
                continue

        # Summary statistics for the best-fitted distribution
        if best_fit_dist == 'Dirichlet':
            summary_stats = describe(col_data)
            summary_stats_str = f'Mean Vector: {summary_stats.mean}'
        else:
            summary_stats_str = 'N/A'

        results_data.append([col, best_fit_dist, best_fit_pvalue, summary_stats_str])

    results_df = pd.DataFrame(results_data, columns=results_columns)

    return results_df
