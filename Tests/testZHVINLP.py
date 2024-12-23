import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import re

# Load the NLP model
tokenizer = AutoTokenizer.from_pretrained('deepset/roberta-base-squad2')
model = AutoModelForQuestionAnswering.from_pretrained('deepset/roberta-base-squad2')
nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)

# Load and merge ZHVI datasets
def load_and_merge_data(bottom_path, top_path):
    # bottom_data = pd.read_csv("../datasets/ZHVI/City_ZHVI_All_Homes_Bottom_tier_time_series.csv",)
    # top_data = pd.read_csv("../datasets/ZHVI/City_ZHVI_All_Homes_Top_tier_time_series.csv")
    bottom_data = pd.read_csv(bottom_path)
    top_data = pd.read_csv(top_path)
    # merge_columns = ['RegionID', 'RegionName', 'StateName', 'Metro', 'CountyName', 'RegionType']
    # merged_data = pd.merge(
    #     bottom_data,
    #     top_data,
    #     on=merge_columns,
    #     suffixes=('_bottom', '_top'),
    #     how='outer'
    # )
    # return merged_data
     # Merge datasets on specified columns with outer join
    merge_columns = ['RegionID', 'RegionName', 'StateName', 'Metro', 'CountyName', 'RegionType']
    merged_data = pd.merge(
        bottom_data,
        top_data,
        on=merge_columns,
        suffixes=('_bottom', '_top'),
        how='outer'
    )

    # Consolidate columns to avoid fragmentation
    all_columns = set(bottom_data.columns).union(set(top_data.columns))
    # merged_data = pd.concat(
    #     [merged_data] + [pd.DataFrame({col: None}, index=merged_data.index) for col in all_columns if col not in merged_data.columns],
    #     axis=1
    # )

    # Return consolidated and de-fragmented DataFrame
    return merged_data.copy()  # Ensures a new memory allocation for improved performance


# Categorize homes by price
def categorize_price(price):
    if price < 300000:
        return "Low Tier"
    elif 300000 <= price <= 700000:
        return "Middle Tier"
    else:
        return "High Tier"

# Query processing for price-based suggestions
def filter_houses_by_price(merged_data, price):
    # price_column = '2024-01-31'  # Example column for the latest ZHVI values
    price_column = merged_data.columns[-1]  # Use the last column for prices

    filtered_data = merged_data[(merged_data[price_column] >= price - 50000) & 
                                (merged_data[price_column] <= price + 50000)]
    return filtered_data

# # NLP query handling
# def process_query(query, merged_data):
#     # Extract numerical value for price from the query
#     price = None
#     if "$" in query:
#         price_str = query.split("$")[-1].split()[0].replace(",", "")
#         if price_str.isdigit():
#             price = int(price_str)

#     if price:
#         filtered_houses = filter_houses_by_price(merged_data, price)
#         if not filtered_houses.empty:
#             response = f"We found {len(filtered_houses)} houses around ${price}. Popular neighborhoods include {', '.join(filtered_houses['RegionName'].unique()[:5])}."
#         else:
#             response = f"No houses found around ${price}."
#     else:
#         response = "Sorry, I couldn't extract a price from your query."

#     return response
# NLP query handling
def process_query(query, merged_data):
    # Use regex to extract the first numeric value after a "$" sign
    price = None
    # match = re.search(r'\$(\d+(?:,\d{3})*(?:\.\d+)?)', query)
    match = re.search(r'(\d+(?:,\d{3})*(?:\.\d+)?)', query)

    if match:
        price_str = match.group(1).replace(",", "")  # Remove commas
        try:
            price = int(float(price_str))  # Handle decimals if present
        except ValueError:
            pass

    if price:
        filtered_houses = filter_houses_by_price(merged_data, price)
        if not filtered_houses.empty:
            response = f"We found {len(filtered_houses)} houses around ${price}. Popular neighborhoods include {', '.join(filtered_houses['RegionName'].dropna().unique()[::])}."
        else:
            response = f"No houses found around ${price}. Try adjusting your range to ${price - 100000} - ${price + 100000}."
    else:
        response = "Sorry, I couldn't extract a valid price from your query."

    return response
if __name__ == "__main__":
    # Example usage
    bottom_tier_path = "../datasets/ZHVI/City_ZHVI_All_Homes_Bottom_tier_time_series.csv"
    top_tier_path = "../datasets/ZHVI/City_ZHVI_All_Homes_Top_tier_time_series.csv"
    merged_data = load_and_merge_data(bottom_tier_path, top_tier_path)

    # Example query
    user_query = "What houses are available around $500,000?"
    print(process_query(user_query, merged_data))
