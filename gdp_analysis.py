# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
from scipy import stats
from scipy.optimize import curve_fit

# Define functions for reading and transposing data
def read_data_excel(excel_url, sheet_name, new_cols, countries):
    """
    Read and transpose data from an Excel file.

    Parameters:
    - excel_url: URL or local path of the Excel file.
    - sheet_name: Name of the sheet containing the data.
    - new_cols: List of column names to include in the result.
    - countries: List of countries to include in the result.

    Returns:
    - data_read: DataFrame with selected columns and countries.
    - data_transpose: Transposed DataFrame for further analysis.
    """
    data_read = pd.read_excel(excel_url, sheet_name=sheet_name, skiprows=3)
    data_read = data_read[new_cols]
    data_read.set_index('Country Name', inplace=True)
    data_read = data_read.loc[countries]
    return data_read, data_read.T

# Function to perform clustering using k-means
def perform_clustering(data, num_clusters):
    """
    Perform clustering on numeric data using k-means algorithm.

    Parameters:
    - data: Input DataFrame with numeric columns.
    - num_clusters: Number of clusters for k-means.

    Returns:
    - clustered_data: DataFrame with additional 'Cluster' column.
    - kmeans: KMeans model fitted to the data.
    """
    numeric_data = data.select_dtypes(include=[np.number])
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clustered_data = data.copy()
    clustered_data['Cluster'] = kmeans.fit_predict(numeric_data)
    return clustered_data, kmeans

# Function to fit a curve using curve_fit
def perform_curve_fit(x, y, func):
    """
    Fit a curve to given x and y data using the curve_fit function.

    Parameters:
    - x: Independent variable (x-axis) data.
    - y: Dependent variable (y-axis) data.
    - func: Function to fit the curve.

    Returns:
    - popt: Optimal values for the parameters so that the sum of the squared residuals is minimized.
    - pcov: Estimated covariance of popt.
    """
    x_numeric = pd.to_numeric(x, errors='coerce')
    y_numeric = pd.to_numeric(y, errors='coerce')
    xy_numeric = pd.DataFrame({'x': x_numeric, 'y': y_numeric}).dropna()
    if not xy_numeric.empty:
        popt, pcov = curve_fit(func, xy_numeric['x'], xy_numeric['y'])
        return popt, pcov
    else:
        print("No numeric values available for curve fitting.")
        return None, None

# Simple exponential growth function for curve fitting
def exponential_growth(x, a, b):
    """
    Simple exponential growth function.

    Parameters:
    - x: Independent variable.
    - a: Amplitude.
    - b: Growth rate.

    Returns:
    - y: Calculated exponential growth values.
    """
    return a * np.exp(b * x)

# Function to plot clustering results and curve fitting
def plot_results(data, x_col, y_col, cluster_col, func, title, kmeans):
    """
    Plot clustering results and curve fitting.

    Parameters:
    - data: DataFrame containing data to be plotted.
    - x_col: Column representing the x-axis values.
    - y_col: Column representing the y-axis values.
    - cluster_col: Column representing the clusters.
    - func: Function used for curve fitting.
    - title: Title of the plot.
    - kmeans: Fitted KMeans model.
    """
    plt.figure(figsize=(20, 8))
    
    # Plot clustering results
    plt.scatter(data.index, data[y_col], c=data[cluster_col], cmap='viridis', s=50, alpha=0.8, edgecolors='w')
    
    # Plot cluster centers
    cluster_centers = kmeans.cluster_centers_
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='X', s=200, label='Cluster Centers')
    
    # Plot curve fitting
    x_fit_values = pd.to_numeric(data.index, errors='coerce').dropna()
    
    if not x_fit_values.empty:
        x_fit = np.linspace(min(x_fit_values), max(x_fit_values), 100)
        y_fit = func(x_fit, *popt)
        plt.plot(x_fit, y_fit, '--', color='black', label='Curve Fit')
    else:
        print("No numeric values available for curve fitting.")
    
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend()
    plt.show()


# The Excel URL below indicates GDP growth (annual %)
excel_url_GDP = 'https://api.worldbank.org/v2/en/indicator/NY.GDP.MKTP.KD.ZG?downloadformat=excel'

# Parameters for reading and transposing data
sheet_name = 'Data'
new_cols = ['Country Name', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022']
countries = ['Germany', 'United States', 'United Kingdom', 'Pakistan', 'China', 'India', 'Norway']

# Read and transpose GDP growth data
data_GDP, data_GDP_transpose = read_data_excel(excel_url_GDP, sheet_name, new_cols, countries)

# Perform clustering on GDP growth data
num_clusters = 3  # You can adjust the number of clusters
data_GDP_clustered, kmeans = perform_clustering(data_GDP, num_clusters)

# Fit an exponential growth curve to GDP growth data

x_col = 'Country Name'
y_col = '2010'  # You can choose any year
# Fit an exponential growth curve to GDP growth data for each country
for country in data_GDP.index:
    popt, pcov = perform_curve_fit(new_cols[1:], data_GDP.loc[country, new_cols[1:]], exponential_growth)
    if popt is not None:
        print(f"Curve parameters for {country}: {popt}")

# Plot clustering results and curve fitting for GDP growth
plot_results(data_GDP_clustered, x_col, y_col, 'Cluster', exponential_growth, 'Clustering and Curve Fitting for GDP Growth', kmeans)

# Visualize GDP growth data for each country and print values
for country in data_GDP.index:
    x_values = new_cols[1:]
    y_values = data_GDP.loc[country, new_cols[1:]]

    print(f"\nGDP Growth Data for {country}:")
    print(pd.DataFrame({'Year': x_values, 'GDP Growth (%)': y_values}))

    plt.figure(figsize=(8, 5))
    plt.plot(x_values, y_values, marker='o', label=country)
    plt.title(f'GDP Growth for {country}')
    plt.xlabel('Year')
    plt.ylabel('GDP Growth (%)')
    plt.legend()
    plt.show()

# Define logistic function for curve fitting
def logistic_function(x, a, b, c):
    """
    Logistic growth function for curve fitting.

    Parameters:
    - x: Independent variable.
    - a: Maximum growth value (asymptote).
    - b: Growth rate.
    - c: Time at which maximum growth occurs (inflection point).

    Returns:
    - y: Calculated logistic growth values.
    """
    return a / (1 + np.exp(-b * (x - c)))


# Fit exponential and logistic growth curves for each country
for country in data_GDP.index:
    x_values = pd.to_numeric(new_cols[1:], errors='coerce')
    y_values = data_GDP.loc[country, new_cols[1:]]

    print(f"\nFitting Curves for GDP Growth in {country}:")

    # Exponential growth curve fitting
    popt_exp, pcov_exp = perform_curve_fit(x_values, y_values, exponential_growth)
    if popt_exp is not None:
        print(f"Exponential Growth Parameters for {country}: {popt_exp}")

    # Logistic growth curve fitting
    popt_log, pcov_log = perform_curve_fit(x_values, y_values, logistic_function)
    if popt_log is not None:
        print(f"Logistic Growth Parameters for {country}: {popt_log}")

    # Visualize the data and fitted curves
    plt.figure(figsize=(8, 5))
    plt.plot(x_values, y_values, marker='o', label='Actual Data')
    
    if popt_exp is not None:
        y_fit_exp = exponential_growth(x_values, *popt_exp)
        plt.plot(x_values, y_fit_exp, '--', label='Exponential Fit')
    
    if popt_log is not None:
        y_fit_log = logistic_function(x_values, *popt_log)
        plt.plot(x_values, y_fit_log, '--', label='Logistic Fit')

    plt.title(f'Curve Fitting for GDP Growth in {country}')
    plt.xlabel('Year')
    plt.ylabel('GDP Growth (%)')
    plt.legend()
    plt.show()
# Perform curve fitting
def perform_curve_fit_updated(x, y, func, initial_params, maxfev=None):
    """
    Perform curve fitting with adjusted initial parameters.

    Parameters:
    - x: Independent variable.
    - y: Dependent variable.
    - func: Function used for curve fitting.
    - initial_params: Initial parameters for the curve fitting.
    - maxfev: Maximum number of function evaluations for curve fitting.

    Returns:
    - popt: Optimal values for the parameters so that the sum of the squared residuals is minimized.
    - pcov: Covariance matrix of the fitted parameters.
    """
    x_numeric = pd.to_numeric(x, errors='coerce')
    y_numeric = pd.to_numeric(y, errors='coerce')
    
    # Remove NaN and infinite values
    mask_finite = np.isfinite(x_numeric) & np.isfinite(y_numeric)
    x_numeric = x_numeric[mask_finite]
    y_numeric = y_numeric[mask_finite]

    # Perform curve fitting with adjusted initial parameters
    popt, pcov = curve_fit(func, x_numeric, y_numeric, p0=initial_params, maxfev=maxfev)
    return popt, pcov

def err_ranges(func, p, pcov, x, alpha=0.05):
    """
    Estimate the confidence range for the fitted curve.

    Parameters:
    - func: The function used for curve fitting.
    - p: Fitted parameters.
    - pcov: Covariance matrix of the fitted parameters.
    - x: x values for prediction.
    - alpha: Significance level for confidence interval.

    Returns:
    - y_lower: Lower bound of the confidence interval.
    - y_upper: Upper bound of the confidence interval.
    """
    n = len(x)
    dof = max(0, n - len(p))  # Degrees of freedom
    t_val = stats.t.ppf(1 - alpha / 2, dof)
    delta = np.sqrt(np.diag(pcov))
    y_lower = func(x, *(p - t_val * delta))
    y_upper = func(x, *(p + t_val * delta))
    return y_lower, y_upper

# Function to make predictions and plot results
def make_predictions_and_plot(data, func, title, initial_params):
    """
    Make predictions and plot results using a specified function.

    Parameters:
    - data: DataFrame containing GDP growth data.
    - func: Function used for curve fitting.
    - title: Title for the plot.
    - initial_params: Initial parameters for the curve fitting.

    Returns:
    - None (Displays the plot).
    """
    plt.figure(figsize=(20, 8))

    for country in data.index:
        x_values = pd.to_numeric(new_cols[1:], errors='coerce')
        y_values = data.loc[country, new_cols[1:]]

        # Scale x-axis values
        x_scaled = (x_values - min(x_values)) / (max(x_values) - min(x_values))

        # Fit the logistic growth curve with adjusted initial parameters
        popt_log, pcov_log = perform_curve_fit_updated(x_scaled, y_values, logistic_function_updated, initial_params_logistic, maxfev=5000)

        if popt_log is not None:
            # Predict future values for the next 20 years
            x_future = np.arange(2023, 2043)
            y_pred = logistic_function_updated(x_future, *popt_log)

            # Estimate confidence range
            y_lower, y_upper = err_ranges(func, popt_log, pcov_log, x_future)

            # Plot the fitted curve, predictions, and confidence range
            plt.plot(x_values, y_values, 'o', label=f'{country} - Actual Data')
            plt.plot(x_future, y_pred, label=f'{country} - Predictions', linestyle='--')
            plt.fill_between(x_future, y_lower, y_upper, alpha=0.2, label=f'{country} - Confidence Range')

    plt.title(title)
    plt.xlabel('Year')
    plt.ylabel('GDP Growth (%)')
    plt.legend()
    plt.show()

# Updated logistic growth function
def logistic_function_updated(x, a, b, c, d):
    """
    Updated logistic growth function with additional parameters.

    Parameters:
    - x: Input values.
    - a: Parameter controlling the curve's maximum value.
    - b: Parameter controlling the steepness of the curve.
    - c: Parameter representing the x-value of the sigmoid's midpoint.
    - d: Parameter controlling the curve's minimum value.

    Returns:
    - Output values based on the logistic growth model.
    """
    return a / (1 + np.exp(-b * (x - c))) + d


# Initial parameters for the logistic growth function
initial_params_logistic = [1.0, 1.0, 1.0, 0.0]

# Make predictions and plot results for GDP growth
make_predictions_and_plot(data_GDP, logistic_function_updated, 'Predictions and Confidence Range for GDP Growth', initial_params_logistic)

# Print the content and structure of the DataFrame
print(data_GDP)

