
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
import re
import io
import base64

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

    # Process date columns
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

def parse_query(query):
    params = {}
    if "sarima" in query.lower():
        params['model'] = 'SARIMA'
    elif "arima" in query.lower():
        params['model'] = 'ARIMA'
    else:
        raise ValueError("Unsupported model. Use ARIMA or SARIMA.")

    match = re.search(r'next (\d+) (days|months|years)', query.lower())
    params['forecast_horizon'] = int(match.group(1)) if match else 30

    match = re.search(r'(\d+) monte carlo', query.lower())
    params['monte_carlo'] = int(match.group(1)) if match else 0

    match = re.search(r'p\s*=\s*(\d+),\s*d\s*=\s*(\d+),\s*q\s*=\s*(\d+)', query.lower())
    if match:
        params['p'], params['d'], params['q'] = map(int, match.groups())
    else:
        params['p'], params['d'], params['q'] = 1, 1, 1

    match = re.search(r'seasonal\s*\((\d+)\)', query.lower())
    params['seasonal'] = int(match.group(1)) if match else None

    return params

def forecast_with_monte_carlo(data, query, start_date):
    params = parse_query(query)
    model_type = params['model']
    p, d, q = params['p'], params['d'], params['q']
    seasonal = params['seasonal']
    forecast_horizon = params['forecast_horizon']
    simulations = params['monte_carlo']

    data.index = pd.date_range(start=start_date, periods=len(data), freq='ME')
    data = data.sort_index()

    if model_type == 'SARIMA' and seasonal:
        model = SARIMAX(data, order=(p, d, q), seasonal_order=(p, d, q, 12))
    else:
        model = ARIMA(data, order=(p, d, q))

    results = model.fit()
    forecast = results.get_forecast(steps=forecast_horizon)
    mean_forecast = forecast.predicted_mean
    residuals = results.resid

    if simulations > 0:
        simulated_paths = []
        for _ in range(simulations):
            noise = np.random.choice(residuals, size=forecast_horizon, replace=True)
            simulated_path = mean_forecast + noise
            simulated_paths.append(simulated_path)
        simulated_paths = np.array(simulated_paths)
    else:
        simulated_paths = None

    return mean_forecast, simulated_paths, forecast.conf_int()

def process_forecast_query(query, merged_data, location):
    location_data = merged_data[merged_data['RegionName'].str.contains(location, case=False)]
    if location_data.empty:
        return f"No data available for {location}."

    date_columns = [col for col in location_data.columns if re.match(r'\d{4}-\d{2}-\d{2}', col)]
    data = location_data[date_columns].iloc[0]
    data.index = pd.to_datetime([col.split('_')[0] for col in data.index])
    data = data.interpolate(method='linear').ffill()

    mean_forecast, simulated_paths, ci = forecast_with_monte_carlo(data, query, start_date=data.index[0])

    forecast_dates = pd.date_range(
        start=data.index[-1] + pd.DateOffset(months=1),
        periods=len(mean_forecast), freq='ME'
    )

    plot_image = generate_forecast_plot(
        data, mean_forecast, simulated_paths, forecast_dates, location, "SARIMA", ci
    )

    return f"Forecast for {location}: {mean_forecast.iloc[-1]:.2f}", plot_image

def generate_forecast_plot(data, mean_forecast, simulated_paths, forecast_dates, location, model_type, ci):
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data.values, label="Historical Data", color='blue')
    plt.plot(forecast_dates, mean_forecast, label="Forecast", color='green')

    # Plot ARIMA/SARIMA CI
    plt.fill_between(forecast_dates, ci.iloc[:, 0], ci.iloc[:, 1], color='orange', alpha=0.3, label="Model CI")

    if simulated_paths is not None:
        # Plot Monte Carlo CI
        lower_bound = simulated_paths.min(axis=0)
        upper_bound = simulated_paths.max(axis=0)
        plt.fill_between(forecast_dates, lower_bound, upper_bound, color='gray', alpha=0.3, label="Monte Carlo CI")

    plt.xlabel('Time')
    plt.ylabel('Price ($)')
    plt.title(f'{model_type} Forecast for {location}')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    buffer.close()
    plt.close()

    return encoded_image


# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from statsmodels.tsa.statespace.sarimax import SARIMAX
# from statsmodels.tsa.arima.model import ARIMA
# import re

# import io
# import base64
# import matplotlib.pyplot as plt
# def load_and_merge_data(bottom_path, top_path):
#     # Load datasets
#     bottom_data = pd.read_csv(bottom_path)
#     top_data = pd.read_csv(top_path)

#     # Merge datasets on specified columns
#     merge_columns = ['RegionID', 'RegionName', 'StateName', 'Metro', 'CountyName', 'RegionType']
#     merged_data = pd.merge(
#         bottom_data,
#         top_data,
#         on=merge_columns,
#         suffixes=('_bottom', '_top'),
#         how='outer'
#     )

#     # Get date columns
#     date_columns_bottom = [col for col in merged_data.columns if '_bottom' in col]
#     date_columns_top = [col.replace('_bottom', '_top') for col in date_columns_bottom]

#     # Process date columns
#     for b_col, t_col in zip(date_columns_bottom, date_columns_top):
#         # Ensure numeric values and handle errors
#         merged_data[b_col] = pd.to_numeric(merged_data[b_col], errors='coerce')
#         merged_data[t_col] = pd.to_numeric(merged_data[t_col], errors='coerce')

#         # Take the mean while skipping NaN values
#         merged_data[b_col] = merged_data[[b_col, t_col]].mean(axis=1, skipna=True)
#         merged_data.rename(columns={b_col: b_col.replace('_bottom', '')}, inplace=True)

#     # Drop 'top' columns after processing
#     merged_data.drop(columns=date_columns_top, inplace=True)

#     return merged_data

# # Load and merge ZHVI datasets
# # def load_and_merge_data(bottom_path, top_path):
# #     # Load datasets
# #     bottom_data = pd.read_csv(bottom_path)
# #     top_data = pd.read_csv(top_path)

# #     # Merge datasets on specified columns with outer join
# #     merge_columns = ['RegionID', 'RegionName', 'StateName', 'Metro', 'CountyName', 'RegionType']
# #     merged_data = pd.merge(
# #         bottom_data,
# #         top_data,
# #         on=merge_columns,
# #         suffixes=('_bottom', '_top'),
# #         how='outer'
# #     )

# #     for b_col, t_col in zip(date_columns_bottom, date_columns_top):
# #         # Ensure numeric values and handle errors
# #         merged_data[b_col] = pd.to_numeric(merged_data[b_col], errors='coerce')
# #         merged_data[t_col] = pd.to_numeric(merged_data[t_col], errors='coerce')

# #         # Take the mean while skipping NaN values
# #         merged_data[b_col] = merged_data[[b_col, t_col]].mean(axis=1, skipna=True)
# #         merged_data.rename(columns={b_col: b_col.replace('_bottom', '')}, inplace=True)

# #     # Drop 'top' columns after processing
# #     merged_data.drop(columns=date_columns_top, inplace=True)


# #     return merged_data

# # NLP Function to Parse Queries
# def parse_query(query):
#     params = {}
#     # Model type
#     if "sarima" in query.lower():
#         params['model'] = 'SARIMA'
#     elif "arima" in query.lower():
#         params['model'] = 'ARIMA'
#     else:
#         raise ValueError("Unsupported model. Use ARIMA or SARIMA.")

#     # Forecast horizon
#     match = re.search(r'next (\d+) (days|months|years)', query.lower())
#     params['forecast_horizon'] = int(match.group(1)) if match else 30

#     # Monte Carlo simulations
#     match = re.search(r'(\d+) monte carlo', query.lower())
#     params['monte_carlo'] = int(match.group(1)) if match else 1000

#     # ARIMA parameters p, d, q
#     match = re.search(r'p\s*=\s*(\d+),\s*d\s*=\s*(\d+),\s*q\s*=\s*(\d+)', query.lower())
#     if match:
#         params['p'], params['d'], params['q'] = map(int, match.groups())
#     else:
#         params['p'], params['d'], params['q'] = 1, 1, 1

#     # Seasonal component for SARIMA
#     match = re.search(r'seasonal\s*\((\d+)\)', query.lower())
#     params['seasonal'] = int(match.group(1)) if match else None

#     return params

# # ARIMA/SARIMA Forecast with Monte Carlo Simulation
# def forecast_with_monte_carlo(data, query, start_date):
#     # Parse query
#     params = parse_query(query)
#     model_type = params['model']
#     p, d, q = params['p'], params['d'], params['q']
#     seasonal = params['seasonal']
#     forecast_horizon = params['forecast_horizon']
#     simulations = params['monte_carlo']

#     # Ensure datetime index starts from the first available date in the dataset
#     data.index = pd.date_range(start=start_date, periods=len(data), freq='ME')
#     data = data.sort_index()

#     # Fit ARIMA/SARIMA Model
#     if model_type == 'SARIMA' and seasonal:
#         model = SARIMAX(data, order=(p, d, q), seasonal_order=(p, d, q, 12))
#     else:
#         model = ARIMA(data, order=(p, d, q))

#     results = model.fit()

#     # Forecast Mean
#     forecast = results.get_forecast(steps=forecast_horizon)
#     mean_forecast = forecast.predicted_mean
#     residuals = results.resid

#     # Monte Carlo Simulations only on prediction
#     simulated_paths = []
#     for _ in range(simulations):
#         noise = np.random.choice(residuals, size=forecast_horizon, replace=True)
#         simulated_path = mean_forecast + noise
#         simulated_paths.append(simulated_path)

#     simulated_paths = np.array(simulated_paths)

#     # Forecast Dates
#     forecast_dates = pd.date_range(
#         start=data.index[-1] + pd.DateOffset(months=1), periods=forecast_horizon, freq='ME'
#     )

#     # Plot Results
#     plt.figure(figsize=(10, 6))
#     plt.plot(data.index, data.values, label="Historical Data", color='blue')
#     plt.plot(forecast_dates, mean_forecast, label="Forecast", color='green')

#     # Confidence Intervals only on prediction
#     lower_bound = np.percentile(simulated_paths, 2.5, axis=0)
#     upper_bound = np.percentile(simulated_paths, 97.5, axis=0)
#     plt.fill_between(forecast_dates, lower_bound, upper_bound, color='gray', alpha=0.3, label="95% CI")

#     plt.xlabel('Time')
#     plt.ylabel('Values')
#     plt.title(f'{model_type} Forecast with Monte Carlo Simulations')
#     plt.legend()
#     plt.grid(True)
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     # plt.show()

#     return mean_forecast, simulated_paths



# def generate_forecast_plot(data, mean_forecast, simulated_paths, forecast_dates, location, model_type):
#     plt.figure(figsize=(10, 6))
#     plt.plot(data.index, data.values, label="Historical Data", color='blue')
#     plt.plot(forecast_dates, mean_forecast, label="Forecast", color='green')

#     # Confidence Intervals
#     lower_bound = simulated_paths.min(axis=0)
#     upper_bound = simulated_paths.max(axis=0)
#     plt.fill_between(forecast_dates, lower_bound, upper_bound, color='gray', alpha=0.3, label="95% CI")

#     plt.xlabel('Time')
#     plt.ylabel('Price ($)')
#     plt.title(f'{model_type} Forecast for {location}')
#     plt.legend()
#     plt.grid(True)
#     plt.xticks(rotation=45)
#     plt.tight_layout()

#     # Encode the plot as base64
#     buffer = io.BytesIO()
#     plt.savefig(buffer, format='png')
#     buffer.seek(0)
#     encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
#     buffer.close()
#     plt.close()

#     return encoded_image


# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from statsmodels.tsa.statespace.sarimax import SARIMAX
# from statsmodels.tsa.arima.model import ARIMA
# import re
# import io
# import base64

# def load_and_merge_data(bottom_path, top_path):
#     # Load datasets
#     bottom_data = pd.read_csv(bottom_path)
#     top_data = pd.read_csv(top_path)

#     # Merge datasets on specified columns
#     merge_columns = ['RegionID', 'RegionName', 'StateName', 'Metro', 'CountyName', 'RegionType']
#     merged_data = pd.merge(
#         bottom_data,
#         top_data,
#         on=merge_columns,
#         suffixes=('_bottom', '_top'),
#         how='outer'
#     )

#     # Process date columns
#     date_columns_bottom = [col for col in merged_data.columns if '_bottom' in col]
#     date_columns_top = [col.replace('_bottom', '_top') for col in date_columns_bottom]

#     for b_col, t_col in zip(date_columns_bottom, date_columns_top):
#         merged_data[b_col] = pd.to_numeric(merged_data[b_col], errors='coerce')
#         merged_data[t_col] = pd.to_numeric(merged_data[t_col], errors='coerce')
#         merged_data[b_col] = merged_data[[b_col, t_col]].mean(axis=1, skipna=True)
#         merged_data.rename(columns={b_col: b_col.replace('_bottom', '')}, inplace=True)

#     # Drop 'top' columns
#     merged_data.drop(columns=date_columns_top, inplace=True)

#     return merged_data

# def parse_query(query):
#     params = {}
#     if "sarima" in query.lower():
#         params['model'] = 'SARIMA'
#     elif "arima" in query.lower():
#         params['model'] = 'ARIMA'
#     else:
#         raise ValueError("Unsupported model. Use ARIMA or SARIMA.")

#     match = re.search(r'next (\d+) (days|months|years)', query.lower())
#     params['forecast_horizon'] = int(match.group(1)) if match else 30

#     match = re.search(r'(\d+) monte carlo', query.lower())
#     params['monte_carlo'] = int(match.group(1)) if match else 1000

#     match = re.search(r'p\s*=\s*(\d+),\s*d\s*=\s*(\d+),\s*q\s*=\s*(\d+)', query.lower())
#     if match:
#         params['p'], params['d'], params['q'] = map(int, match.groups())
#     else:
#         params['p'], params['d'], params['q'] = 1, 1, 1

#     match = re.search(r'seasonal\s*\((\d+)\)', query.lower())
#     params['seasonal'] = int(match.group(1)) if match else None

#     return params

# def forecast_with_monte_carlo(data, query, start_date):
#     params = parse_query(query)
#     model_type = params['model']
#     p, d, q = params['p'], params['d'], params['q']
#     seasonal = params['seasonal']
#     forecast_horizon = params['forecast_horizon']
#     simulations = params['monte_carlo']

#     data.index = pd.date_range(start=start_date, periods=len(data), freq='ME')
#     data = data.sort_index()

#     if model_type == 'SARIMA' and seasonal:
#         model = SARIMAX(data, order=(p, d, q), seasonal_order=(p, d, q, 12))
#     else:
#         model = ARIMA(data, order=(p, d, q))

#     results = model.fit()
#     forecast = results.get_forecast(steps=forecast_horizon)
#     mean_forecast = forecast.predicted_mean
#     residuals = results.resid

#     simulated_paths = []
#     for _ in range(simulations):
#         noise = np.random.choice(residuals, size=forecast_horizon, replace=True)
#         simulated_paths.append(mean_forecast + noise)

#     simulated_paths = np.array(simulated_paths)
#     return mean_forecast, simulated_paths

# def process_forecast_query(query, merged_data, location):
#     location_data = merged_data[merged_data['RegionName'].str.contains(location, case=False)]
#     if location_data.empty:
#         return f"No data available for {location}."

#     date_columns = [col for col in location_data.columns if re.match(r'\d{4}-\d{2}-\d{2}', col)]
#     data = location_data[date_columns].iloc[0]
#     data.index = pd.to_datetime([col.split('_')[0] for col in data.index])
#     data = data.interpolate(method='linear').ffill()

#     mean_forecast, simulated_paths = forecast_with_monte_carlo(data, query, start_date=data.index[0])

#     forecast_dates = pd.date_range(
#         start=data.index[-1] + pd.DateOffset(months=1),
#         periods=len(mean_forecast), freq='ME'
#     )

#     plot_image = generate_forecast_plot(
#         data, mean_forecast, simulated_paths, forecast_dates, location, "SARIMA"
#     )

#     return f"Forecast for {location}: {mean_forecast.iloc[-1]:.2f}", plot_image

# def generate_forecast_plot(data, mean_forecast, simulated_paths, forecast_dates, location, model_type):
#     plt.figure(figsize=(10, 6))
#     plt.plot(data.index, data.values, label="Historical Data", color='blue')
#     plt.plot(forecast_dates, mean_forecast, label="Forecast", color='green')

#     lower_bound = simulated_paths.min(axis=0)
#     upper_bound = simulated_paths.max(axis=0)
#     plt.fill_between(forecast_dates, lower_bound, upper_bound, color='gray', alpha=0.3, label="95% CI")

#     plt.xlabel('Time')
#     plt.ylabel('Price ($)')
#     plt.title(f'{model_type} Forecast for {location}')
#     plt.legend()
#     plt.grid(True)
#     plt.xticks(rotation=45)
#     plt.tight_layout()

#     buffer = io.BytesIO()
#     plt.savefig(buffer, format='png')
#     buffer.seek(0)
#     encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
#     buffer.close()
#     plt.close()

#     return encoded_image