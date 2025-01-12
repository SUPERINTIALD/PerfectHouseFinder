
import pandas as pd
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import io
import base64

# Load and merge ZHVI datasets
def load_and_merge_housing_data(bottom_path, top_path):
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
        merged_data[b_col] = merged_data[[b_col, t_col]].mean(axis=1)
        merged_data.rename(columns={b_col: b_col.replace('_bottom', '')}, inplace=True)

    # Drop 'top' columns
    merged_data.drop(columns=date_columns_top, inplace=True)

    return merged_data

# Generate price trend chart
def generate_price_trend_chart(merged_data, region):
    # Filter data for the selected region
    region_data = merged_data[merged_data['RegionName'].str.contains(region, case=False, na=False)]
    if region_data.empty:
        return None

    # Extract and process monthly prices
    date_columns = [col for col in merged_data.columns if re.match(r'\d{4}-\d{2}-\d{2}', col)]
    avg_prices = region_data[date_columns].mean()

    # Convert columns into datetime index
    dates = pd.to_datetime(date_columns)

    # Interpolate only if missing values exist
    if avg_prices.isnull().any():
        avg_prices = avg_prices.interpolate(method='linear').ffill().bfill()

    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(dates, avg_prices.values, marker='o', markersize=3, linestyle='-', linewidth=1, color='blue', label='Historical Prices')

    # Add labels and grid
    plt.title(f'Price Trends in {region}')
    plt.xlabel('Date')
    plt.ylabel('Average Price ($)')
    plt.grid(visible=True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=45)

    # Add legend
    plt.legend()

    # Save plot as base64 string
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')  # Prevent cutoff
    buffer.seek(0)
    encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    buffer.close()
    plt.close()

    return encoded_image

# Process price query
def process_price_query(query, merged_data):
    # Use regex to extract numeric values and keywords
    price = None
    region = None

    # Match region
    region_match = re.search(r'around ([A-Za-z\s]+)', query)
    if region_match:
        region = region_match.group(1).strip()

    # Handle trend query
    if region:
        chart = generate_price_trend_chart(merged_data, region)
        if chart:
            response = f"Here is the price trend in {region}.\n"
            response += f'<img src="data:image/png;base64,{chart}"/>'
        else:
            response = f"No data found for {region}."

    else:
        response = "Sorry, I couldn't process your query. Please specify a valid location."

    return response


# import pandas as pd
# from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
# import re
# import matplotlib.pyplot as plt
# import io
# import base64

# # Load the NLP model
# tokenizer = AutoTokenizer.from_pretrained('deepset/roberta-base-squad2')
# model = AutoModelForQuestionAnswering.from_pretrained('deepset/roberta-base-squad2')
# nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)

# # Load and merge ZHVI datasets
# # def load_and_merge_housing_data(bottom_path, top_path):
# #     bottom_data = pd.read_csv(bottom_path)
# #     top_data = pd.read_csv(top_path)

# #     merge_columns = ['RegionID', 'RegionName', 'StateName', 'Metro', 'CountyName', 'RegionType']
# #     merged_data = pd.merge(
# #         bottom_data,
# #         top_data,
# #         on=merge_columns,
# #         suffixes=('_bottom', '_top'),
# #         how='outer'
# #     )

# #     return merged_data.copy()  # Ensures a new memory allocation for improved performance

# def load_and_merge_housing_data(bottom_path, top_path):
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

#     # Combine date columns by averaging
#     date_columns_bottom = [col for col in merged_data.columns if '_bottom' in col]
#     date_columns_top = [col.replace('_bottom', '_top') for col in date_columns_bottom]

#     for b_col, t_col in zip(date_columns_bottom, date_columns_top):
#         merged_data[b_col] = pd.to_numeric(merged_data[b_col], errors='coerce')
#         merged_data[t_col] = pd.to_numeric(merged_data[t_col], errors='coerce')
#         merged_data[b_col] = merged_data[[b_col, t_col]].mean(axis=1)
#         merged_data.rename(columns={b_col: b_col.replace('_bottom', '')}, inplace=True)

#     # Drop 'top' columns
#     merged_data.drop(columns=date_columns_top, inplace=True)

#     return merged_data


# # Categorize homes by price
# def categorize_price(price):
#     if price < 300000:
#         return "Low Tier"
#     elif 300000 <= price <= 700000:
#         return "Middle Tier"
#     else:
#         return "High Tier"

# # Query processing for price-based suggestions
# def filter_houses_by_price(merged_data, price):
#     price_column = merged_data.columns[-1]  # Use the last column for prices
#     filtered_data = merged_data[(merged_data[price_column] >= price - 50000) & 
#                                 (merged_data[price_column] <= price + 50000)]
#     return filtered_data

# # Generate price trend chart
# def generate_price_trend_chart(merged_data, region):
#     # Filter data for the selected region
#     region_data = merged_data[merged_data['RegionName'].str.contains(region, case=False, na=False)]
#     if region_data.empty:
#         return None

#     # Extract monthly prices
#     date_columns = [col for col in merged_data.columns if re.match(r'\d{4}-\d{2}-\d{2}', col)]
#     avg_prices = region_data[date_columns].mean()

#     # Plot the trend
#     plt.figure(figsize=(10, 5))
#     plt.plot(avg_prices.index, avg_prices.values, marker='o')
#     plt.xticks(rotation=45)
#     plt.title(f'Price Trends in {region}')
#     plt.xlabel('Date')
#     plt.ylabel('Average Price ($)')
#     plt.grid(True)

#     # Save plot to a base64 string
#     buffer = io.BytesIO()
#     plt.savefig(buffer, format='png')
#     buffer.seek(0)
#     encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
#     buffer.close()
#     plt.close()

#     return encoded_image

# # NLP query handling
# def process_price_query(query, merged_data):
#     # Use regex to extract numeric values and keywords
#     price = None
#     region = None

#     # Match price
#     price_match = re.search(r'(\d+(?:,\d{3})*(?:\.\d+)?)', query)
#     if price_match:
#         price_str = price_match.group(1).replace(",", "")
#         try:
#             price = int(float(price_str))
#         except ValueError:
#             pass

#     # Match region
#     region_match = re.search(r'around ([A-Za-z\s]+)', query)
#     if region_match:
#         region = region_match.group(1).strip()

#     # Handle price-based query
#     if price and not region:
#         filtered_houses = filter_houses_by_price(merged_data, price)
#         if not filtered_houses.empty:
#             neighborhood_counts = filtered_houses['RegionName'].value_counts().head(10)
#             response = f"We found {len(filtered_houses)} houses around ${price}. Top neighborhoods include: "
#             response += ", ".join([f"{name} ({count})" for name, count in neighborhood_counts.items()]) + '.'
#         else:
#             response = f"No houses found around ${price}. Try adjusting your range to ${price - 100000} - ${price + 100000}."

#     # Handle trend query
#     elif region:
#         chart = generate_price_trend_chart(merged_data, region)
#         if chart:
#             response = f"Here is the price trend in {region}.\n"
#             response += f'<img src="data:image/png;base64,{chart}"/>'
#         else:
#             response = f"No data found for {region}."

#     else:
#         response = "Sorry, I couldn't process your query. Please specify price or location."

#     return response

