import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import TapexTokenizer, BartForConditionalGeneration
import re
from sklearn.preprocessing import MinMaxScaler
from fuzzywuzzy import fuzz
import nltk
from nltk.stem import PorterStemmer
import os
nltk.download('punkt')
ps = PorterStemmer()
# Load the processed dataset
tables = pd.read_csv("sample_data(5).csv")
print("First few rows of the dataset:")
print(tables.head())
num_rows = tables.shape[0]
device = "cuda" if torch.cuda.is_available() else "cpu"
# Load the sentence transformer model
model_dir="model_files"
retriver = SentenceTransformer(os.path.join(model_dir, "sentence_transformer"))
#retriver = SentenceTransformer("deepset/all-mpnet-base-v2-table", device=device)

def stem_words(text):
    words = nltk.word_tokenize(text.lower())
    return ' '.join([ps.stem(word) for word in words])

def preprocess_tables(tables):
    processed = []
    for _, row in tables.iterrows():
        processed_row = "\n".join(stem_words(str(value)) for value in row)
        processed.append(processed_row)
    return processed
processed_tables = preprocess_tables(tables)

# Create a faiss index
index = faiss.IndexFlatIP(retriver.get_sentence_embedding_dimension())
embeddings = []
ids = []

for i, table in enumerate(processed_tables):
    embedding = retriver.encode([table]).tolist()[0]
    embeddings.append(embedding)
    ids.append(i)

print(f"\nNumber of embeddings generated: {len(embeddings)}")
print(f"Number of rows in the original dataset: {num_rows}")

if len(embeddings) == num_rows:
    print("The entire dataset has been embedded successfully.")
else:
    print("The number of embeddings does not match the number of rows in the original dataset.")

index.add(np.array(embeddings))

def normalize_query(query):
    # Convert to lowercase and remove extra spaces
    query = re.sub(r'\s+', ' ', query.lower().strip())

    # Replace common variations
    replacements = {
        'vendor': ['vendor', 'sellers', 'brands'],
        'handbag': ['handbag', 'bag', 'purse'],
        'review': ['review', 'reviewed', 'rating'],
        'price': ['price', 'cost', 'value'],
        'delivery': ['delivery', 'shipping'],
    }

    for standard, variations in replacements.items():
        for var in variations:
            query = query.replace(var, standard)
    query = stem_words(query)

    return query

def fuzzy_match(word, choices, threshold=80):
    for choice in choices:
        if fuzz.ratio(word, choice) > threshold:
            return choice
    return word

def analyze_query(query):
    query = normalize_query(query)
    weights = {
        'price': 0.2,
        'delivery cost': 0.2,
        'delivery_time': 0.2,
        'rating': 0.2,
        'review': 0.2
    }

    keywords = {
        'price': ['price', 'cost', 'expensive', 'cheap', 'affordable', 'budget'],
        'delivery cost': ['shipping', 'delivery fee', 'postage','delivery price'],
        'delivery_time': ['delivery', 'shipping time', 'arrive', 'fast', 'quick'],
        'rating': ['rating', 'stars', 'score'],
        'review': ['review', 'feedback', 'comments', 'popular']
    }

    for feature, feature_keywords in keywords.items():
        if any(fuzzy_match(word, feature_keywords) in query for word in query.split()):
            weights[feature] += 0.5

    # Detect price range
    price_range = re.findall(r'(\d+)\s*to\s*(\d+)', query)
    if price_range:
        weights['price'] += 0.5

    # Normalize weights
    total = sum(weights.values())
    normalized_weights = {k: v / total for k, v in weights.items()}

    print("Query weights:")
    for k, v in normalized_weights.items():
        print(f"{k}: {v:.4f}")

    return normalized_weights, price_range

def calculate_dynamic_score(df, weights, price_range=None):
    features = ['price', 'delivery cost', 'delivery_time', 'rating', 'review']

    scaler = MinMaxScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df[features]), columns=features, index=df.index)

    # Invert the scales for price, delivery cost, and delivery time
    for feature in ['price', 'delivery cost', 'delivery_time']:
        df_normalized[feature] = 1 - df_normalized[feature]

    # Apply sigmoid function to smooth out extreme values
    df_normalized = 1 / (1 + np.exp(-5 * (df_normalized - 0.5)))

    # Apply price range filter if specified
    if price_range:
        min_price, max_price = map(float, price_range[0])
        price_mask = (df['price'] >= min_price) & (df['price'] <= max_price)
        df_normalized = df_normalized[price_mask]

    # Calculate the weighted score
    for feature in features:
        df_normalized[feature] *= weights[feature]

    return df_normalized.sum(axis=1)

def extract_top_k(query):
    word_to_num = {
        'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
        'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90,
        'hundred': 100
    }
    match = re.search(r'top\s+(\d+|[a-zA-Z]+)\s+vendors?', query.lower())
    if match:
        num = match.group(1)
        if num.isdigit():
            return int(num)
        elif num in word_to_num:
            return word_to_num[num]
    return None

#Original
def query_faiss(query, use_dynamic_scoring=True, top_k=10):
    xq = retriver.encode([query]).tolist()[0]
    distances, indices = index.search(np.array([xq]), top_k)
    relevant_rows = tables.iloc[indices[0]].copy()

    if use_dynamic_scoring:
        weights, price_range = analyze_query(query)
        relevant_rows['dynamic_score'] = calculate_dynamic_score(relevant_rows, weights, price_range)
        relevant_rows = relevant_rows.sort_values('dynamic_score', ascending=False)

    return relevant_rows

model_name = "microsoft/tapex-large-finetuned-wtq"
tokenizer = TapexTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name).to(device)

import pandas as pd
import numpy as np
import re
import textwrap

def format_currency(value):
    try:
        return f"₹{float(value):,.2f}"
    except ValueError:
        return value

def wrap_text(text, width=50):
    return '\n'.join(textwrap.wrap(text, width))

def normalize_query(query):
    return query.lower()

def format_product_details(product, feature=None):
    details = f"Title: {wrap_text(product['title'])}\n"
    details += f"Vendor: {product['Vendor']}\n"

    if feature:
        if feature in ['price', 'delivery cost']:
            details += f"{feature.capitalize()}: {format_currency(product[feature])}"
        else:
            details += f"{feature.capitalize()}: {product[feature]}"

    return details.strip()

def get_answer_from_table(rows, query):
    if not isinstance(rows, pd.DataFrame):
        rows = pd.DataFrame([rows])
    rows = rows.reset_index(drop=True)
    for col in rows.columns:
        rows[col] = rows[col].astype(str)
    query_lower = normalize_query(query)
    numeric_features = ['price', 'delivery cost', 'delivery_time', 'rating', 'review']
    operations = ['highest', 'lowest', 'fastest', 'slowest', 'best', 'worst']

    # Handle range queries for numeric features
    for feature in numeric_features:
        range_pattern = rf'({feature})\s*(between|from)\s*(\d+(?:\.\d+)?)\s*(to|and)\s*(\d+(?:\.\d+)?)'
        range_match = re.search(range_pattern, query_lower)
        if range_match:
            feature_name = range_match.group(1)
            min_val, max_val = float(range_match.group(3)), float(range_match.group(5))
            filtered_rows = rows[(rows[feature_name].astype(float) >= min_val) & (rows[feature_name].astype(float) <= max_val)]
            if not filtered_rows.empty:
                if feature_name in ['price', 'delivery cost']:
                    content = f"{feature_name.capitalize()} between {format_currency(min_val)} and {format_currency(max_val)}:\n\n"
                else:
                    content = f"{feature_name.capitalize()} between {min_val} and {max_val}:\n\n"
                content += "\n\n".join([format_product_details(row, feature_name) for _, row in filtered_rows.iterrows()])
                return content, "keyword"

    # Handle "within" queries for numeric features
    for feature in numeric_features:
        within_pattern = rf'({feature})\s*(within|under|below)\s*(\d+(?:\.\d+)?)'
        within_match = re.search(within_pattern, query_lower)
        if within_match:
            max_val = float(within_match.group(3))
            filtered_rows = rows[rows[feature].astype(float) <= max_val]
            if not filtered_rows.empty:
                content = f"{feature.capitalize()} within {format_currency(max_val) if feature in ['price', 'delivery cost'] else max_val}:\n\n"
                content += "\n\n".join([format_product_details(row, feature) for _, row in filtered_rows.iterrows()])
                return content, "keyword"

    # Handle "above" or "greater than" queries for numeric features
    for feature in numeric_features:
        above_pattern = rf'({feature})\s*(above|greater than|more than)\s*(\d+(?:\.\d+)?)'
        above_match = re.search(above_pattern, query_lower)
        if above_match:
            min_val = float(above_match.group(3))
            filtered_rows = rows[rows[feature].astype(float) > min_val]
            if not filtered_rows.empty:
                content = f"{feature.capitalize()} above {format_currency(min_val) if feature in ['price', 'delivery cost'] else min_val}:\n\n"
                content += "\n\n".join([format_product_details(row, feature) for _, row in filtered_rows.iterrows()])
                return content, "keyword"

    # Handle "top N" queries
    top_n_match = re.search(r'top\s+(\d+)', query_lower)
    if top_n_match:
        top_n = int(top_n_match.group(1))
        for feature in numeric_features:
            if f'lowest {feature}' in query_lower or f'cheapest {feature}' in query_lower:
                rows[feature] = pd.to_numeric(rows[feature], errors='coerce')
                top_n_rows = rows.nsmallest(top_n, feature)
                result = "\n\n".join([
                    f"Title: {row['title']}\n"
                    f"Vendor: {row['Vendor']}\n"
                    f"{feature.capitalize()}: {format_currency(row[feature]) if feature in ['price', 'delivery cost'] else row[feature]}"
                    for _, row in top_n_rows.iterrows()
                ])
                return f"Top {top_n} items with lowest {feature}:\n\n{result}", "keyword"
            elif f'highest {feature}' in query_lower or f'most expensive {feature}' in query_lower:
                rows[feature] = pd.to_numeric(rows[feature], errors='coerce')
                top_n_rows = rows.nlargest(top_n, feature)
                result = "\n\n".join([
                    f"Title: {row['title']}\n"
                    f"Vendor: {row['Vendor']}\n"
                    f"{feature.capitalize()}: {format_currency(row[feature]) if feature in ['price', 'delivery cost'] else row[feature]}"
                    for _, row in top_n_rows.iterrows()
                ])
                return f"Top {top_n} items with highest {feature}:\n\n{result}", "keyword"

    # Handle highest/lowest/best/worst queries for numeric features
    for feature in numeric_features:
        for operation in operations:
            if feature in query_lower and operation in query_lower:
                rows[feature] = pd.to_numeric(rows[feature], errors='coerce')
                if operation in ['highest', 'fastest', 'best']:
                    best_option = rows.loc[rows[feature].idxmax()]
                else:
                    best_option = rows.loc[rows[feature].idxmin()]
                return f"The {operation} {feature} option is '{best_option['title']}' from vendor '{best_option['Vendor']}' with a {feature} of {best_option[feature]}.", "keyword"

    # Handle best vendor queries
    if 'best vendor' in query_lower or 'top vendor' in query_lower:
        best_vendor = rows.loc[rows['vendor_score'].astype(float).idxmax()]
        return f"The best vendor is '{best_vendor['Vendor']}' with a vendor score of {best_vendor['vendor_score']}.", "keyword"

    # Use the model for unhandled queries
    encoding = tokenizer(table=rows, query=query, return_tensors="pt").to(device)
    if encoding.input_ids.shape[1] > tokenizer.model_max_length:
        print("Warning: Input is too long. Truncating...")
        encoding = tokenizer(table=rows, query=query, return_tensors="pt",
                             max_length=tokenizer.model_max_length, truncation=True).to(device)
    outputs = model.generate(**encoding)
    answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    # Fallback if model doesn't provide a meaningful answer
    if answer.strip() == '' or answer.lower() == 'none':
        return f"Based on the query, the best match is '{rows.iloc[0]['title']}' from vendor '{rows.iloc[0]['Vendor']}' with a price of {rows.iloc[0]['price']}.", "fallback"

    return answer, "model"

def query_system(query):
    print(f"\nQuery: {query}")

    # Extract top_k from the query
    extracted_top_k = extract_top_k(query)
    top_k = extracted_top_k + 10 if extracted_top_k else 10

    #print(f"Searching for top {top_k} results")

    # Get relevant rows using dynamic scoring
    relevant_rows = query_faiss(query, use_dynamic_scoring=True, top_k=top_k)

    # Get answer using TAPEX model
    answer, source = get_answer_from_table(relevant_rows, query)

    print(f"\nAnswer: {answer}")
    print(f"Answer source: {source}")

    print(f"\nTop {top_k} rows after dynamic scoring:")
    #print(relevant_rows.head(min(10, top_k)))
    print(relevant_rows.iloc[:top_k+1])

    return answer, source

#question:give me the top <number>vendors/vendor name having <feature> <condition>
query = "give me the top 10 vendors having rating above 4"
result, source = query_system(query)
print(f"The answer was generated using: {source}")