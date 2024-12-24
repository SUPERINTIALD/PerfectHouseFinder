import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
import re


# Load and merge ZHVI datasets
def load_and_merge_data(bottom_path, top_path):
    # Load datasets
    bottom_data = pd.read_csv(bottom_path)
    top_data = pd.read_csv(top_path)

    # Merge datasets on specified columns
    merge_columns = ['RegionID', 'RegionName', 'StateName', 'Metro', 'CountyName', 'RegionType']
    merged_data = pd.merge(
        bottom_data,
        top_data,
        on=merge_columns,
        suffixes=('_bottom', '_top'),
        how='outer'
    )

    # Combine date columns by averaging
    date_columns_bottom = [col for col in merged_data.columns if '_bottom' in col]
    date_columns_top = [col.replace('_bottom', '_top') for col in date_columns_bottom]

    for b_col, t_col in zip(date_columns_bottom, date_columns_top):
        merged_data[b_col] = pd.to_numeric(merged_data[b_col], errors='coerce')
        merged_data[t_col] = pd.to_numeric(merged_data[t_col], errors='coerce')
        merged_data[b_col] = merged_data[[b_col, t_col]].mean(axis=1, skipna=True)
        merged_data.rename(columns={b_col: b_col.replace('_bottom', '')}, inplace=True)

    # Drop 'top' columns
    merged_data.drop(columns=date_columns_top, inplace=True)

    return merged_data

# for b_col, t_col in zip(date_columns_bottom, date_columns_top):
#     # Ensure numeric values and handle errors
#     merged_data[b_col] = pd.to_numeric(merged_data[b_col], errors='coerce')
#     merged_data[t_col] = pd.to_numeric(merged_data[t_col], errors='coerce')

#     # Take the mean while skipping NaN values
#     merged_data[b_col] = merged_data[[b_col, t_col]].mean(axis=1, skipna=True)
#     merged_data.rename(columns={b_col: b_col.replace('_bottom', '')}, inplace=True)

# # Drop 'top' columns after processing
# merged_data.drop(columns=date_columns_top, inplace=True)

# NLP Function to Parse Queries
def parse_query(query):
    params = {}
    # Model type
    if "sarima" in query.lower():
        params['model'] = 'SARIMA'
    elif "arima" in query.lower():
        params['model'] = 'ARIMA'
    else:
        raise ValueError("Unsupported model. Use ARIMA or SARIMA.")

    # Forecast horizon
    match = re.search(r'next (\d+) (days|months|years)', query.lower())
    params['forecast_horizon'] = int(match.group(1)) if match else 30

    # Monte Carlo simulations
    match = re.search(r'(\d+) monte carlo', query.lower())
    params['monte_carlo'] = int(match.group(1)) if match else 1000

    # ARIMA parameters p, d, q
    match = re.search(r'p\s*=\s*(\d+),\s*d\s*=\s*(\d+),\s*q\s*=\s*(\d+)', query.lower())
    if match:
        params['p'], params['d'], params['q'] = map(int, match.groups())
    else:
        params['p'], params['d'], params['q'] = 1, 1, 1

    # Seasonal component for SARIMA
    match = re.search(r'seasonal\s*\((\d+)\)', query.lower())
    params['seasonal'] = int(match.group(1)) if match else None

    return params


# ARIMA/SARIMA Forecast with Monte Carlo Simulation
def forecast_with_monte_carlo(data, query, start_date):
    from matplotlib.ticker import FuncFormatter

    # Format Y-axis labels as 'k'
    def thousands_formatter(x, pos):
        return f'{int(x/1000)}k'

    # Parse query
    params = parse_query(query)
    model_type = params['model']
    p, d, q = params['p'], params['d'], params['q']
    seasonal = params['seasonal']
    forecast_horizon = params['forecast_horizon']
    simulations = params['monte_carlo']

    # Ensure datetime index starts from first date
    data.index = pd.date_range(start=start_date, periods=len(data), freq='M')

    # Fit ARIMA/SARIMA Model
    if model_type == 'SARIMA' and seasonal:
        model = SARIMAX(data, order=(p, d, q), seasonal_order=(p, d, q, 12))
    else:
        model = ARIMA(data, order=(p, d, q))

    results = model.fit()

    # Forecast Mean
    forecast = results.get_forecast(steps=forecast_horizon)
    mean_forecast = forecast.predicted_mean
    residuals = results.resid

    # Monte Carlo Simulations
    simulated_paths = []
    for _ in range(simulations):
        noise = np.random.normal(0, np.std(residuals), size=forecast_horizon)
        simulated_path = mean_forecast + noise
        simulated_paths.append(simulated_path)

    simulated_paths = np.array(simulated_paths)

    # Plot Results
    forecast_dates = pd.date_range(data.index[-1] + pd.DateOffset(months=1), periods=forecast_horizon, freq='M')
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data.values, label="Historical Data", color='blue')
    plt.plot(forecast_dates, mean_forecast, label="Forecast", color='green')

    # Confidence Interval
    plt.fill_between(forecast_dates,
                     np.percentile(simulated_paths, 2.5, axis=0),
                     np.percentile(simulated_paths, 97.5, axis=0),
                     color='gray', alpha=0.3, label="95% CI")

    plt.gca().yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.title(f'{model_type} Forecast with Monte Carlo Simulations')
    plt.legend()
    plt.grid(True)
    plt.show()


# Main Workflow
if __name__ == "__main__":
    bottom_path = "../datasets/ZHVI/City_ZHVI_All_Homes_Bottom_tier_time_series.csv"
    top_path = "../datasets/ZHVI/City_ZHVI_All_Homes_Top_tier_time_series.csv"

    # Load and merge data
    housing_data = load_and_merge_data(bottom_path, top_path)

    # Filter Boulder data
    region_data = housing_data.loc[housing_data['RegionName'] == 'Seattle']
    date_columns = [col for col in region_data.columns if re.match(r'\d{4}-\d{2}-\d{2}', col)]

    data = region_data[date_columns].iloc[0]
    data.index = pd.to_datetime([col.split('_')[0] for col in data.index])

    # Fill missing values and smooth data
    # data = data.interpolate(method='linear').rolling(3).mean().dropna()
    data = data.interpolate(method='linear').fillna(method='ffill')


    # # Assume Denver data exists for early years
    # denver_data = housing_data.loc[housing_data['RegionName'] == 'Denver']
    # denver_series = denver_data[date_columns].iloc[0]
    # denver_series.index = pd.to_datetime([col.split('_')[0] for col in denver_series.index])

    # # Scale Boulder data based on Denver trend
    # scale_factor = data.iloc[0] / denver_series.iloc[-1]
    # boulder_filled = denver_series * scale_factor
    # boulder_filled = boulder_filled.append(data)

    query = "Forecast prices for the next 120 months using SARIMA with p=1, d=1, q=1 and seasonal(12) and run 500 Monte Carlo simulations."
    forecast_with_monte_carlo(data, query, start_date=data.index[0])
