import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
from openai import OpenAI
import utils as ut

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ['GROQ_API_KEY'],
)
#change

def load_model(filename):
  with open(filename, 'rb') as file:
    return pickle.load(file)


xgboost_model = load_model('xgb_model1.pkl')

naive_baye_model = load_model('mb_model.pkl')

decision_tree_model = load_model('dt_model.pkl')

lr_model = load_model('lr_model.pkl')

hgb_model = load_model('hgb_model.pkl')


def prepare_input(amount, zip_code, latitude, longitude, merchant_latitude,
                  merchant_longitude, category):

    input_dict = {
        'amt': amount,
        'zip': zip_code,
        'lat': latitude,
        'long': longitude,
        'merch_lat': merchant_latitude,
        'merch_long': merchant_longitude,
        'category_entertainment': 1 if category == 'entertainment' else 0,
        'category_food_dining': 1 if category == 'food_dining' else 0,
        'category_gas_transport': 1 if category == 'gas_transport' else 0,
        'category_grocery_net': 1 if category == 'grocery_net' else 0,
        'category_grocery_pos': 1 if category == 'grocery_pos' else 0,
        'category_health_fitness': 1 if category == 'health_fitness' else 0,
        'category_home': 1 if category == 'home' else 0,
        'category_kids_pets': 1 if category == 'kids_pets' else 0,
        'category_misc_net': 1 if category == 'misc_net' else 0,
        'category_misc_pos': 1 if category == 'misc_pos' else 0,
        'category_personal_care': 1 if category == 'personal_care' else 0,
        'category_shopping_net': 1 if category == 'shopping_net' else 0,
        'category_shopping_pos': 1 if category == 'shopping_pos' else 0,
        'category_travel': 1 if category == 'travel' else 0
    }
    # # Feature engineering
    # input_dict['CLV'] = tenure * num_of_products * is_active_member
    # input_dict['TenureAgeRatio'] = tenure / age if age != 0 else 0
    # input_dict[
    #     'ProductTimeRatio'] = num_of_products / tenure if tenure != 0 else 0
    # input_dict[
    #     'BalanceStabilityRatio'] = balance / estimated_salary if estimated_salary != 0 else 0

    # # Age group features
    # input_dict['AgeGroup_Middle Age'] = 1 if 30 <= age < 50 else 0
    # input_dict['AgeGroup_Senior'] = 1 if 50 <= age < 65 else 0
    # input_dict['AgeGroup_Elderly'] = 1 if age >= 65 else 0
    input_df = pd.DataFrame([input_dict])
    return input_df, input_dict


def make_predictions(input_df, input_dict):
    # Reorder input_df to match the feature order the model expects
    #expected_feature_order = xgboost_model.get_booster().feature_names
    #input_df = input_df[expected_feature_order]
    input_df1 = input_df.apply(pd.to_numeric, errors='coerce')
    probabilities = {
        'Decision Tree': decision_tree_model.predict_proba(input_df)[0][1],
        'XGBOOST ': xgboost_model.predict_proba(input_df1)[0][1],
    }

    avg_probability = np.mean(list(probabilities.values()))

    col1, col2 = st.columns(2)

    with col1:
        fig = ut.create_gauge_chart(avg_probability)
        st.plotly_chart(fig, use_container_width=True)
        st.write(
            f"The customer has a {avg_probability * 100:.2f}% chance of churning."
        )

    with col2:
        fig = ut.create_model_probability_chart(probabilities)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Model Probabilities")
    for model, prob in probabilities.items():
        st.write(f"{model} {prob}")
    st.write(f"Average Probability: {avg_probability}")
    return avg_probability


def explain_prediction(probability, input_dict, first):
    prompt = f"""
You are an expert data scientist at a bank, specializing in interpreting and explaining predictions of machine learning models related to credit card fraud detection.

Your machine learning model has predicted that a transaction made by a customer with name {first} has a {round(probability * 100, 1)}% probability of being fraudulent, based on the information provided below.

Here is the transaction’s information: {input_dict}

Below are the top 10 most important features contributing to the fraud prediction:

Feature	Importance
category_grocery_pos	0.624211
category_gas_transport	0.167259
amt	0.043478
category_misc_net	0.035939
category_shopping_pos	0.028643
category_grocery_net	0.015969
category_travel	0.016247
category_entertainment	0.012873
category_food_dining	0.007079
category_health_fitness	0.006233
{pd.set_option('display.max.columns', None)}

Below are the summary statistics for transactions flagged as fraudulent: {df[df['is_fraud'] == 1].describe()}

Below are the summary statistics for non-fraudulent transactions: {df[df['is_fraud'] == 0].describe()}

Instructions:
For high-risk transactions (more than 50% probability of fraud), provide a concise 3-paragraph explanation:

In the first paragraph, state the transaction’s fraud risk (high or low) and the probability of being fraudulent.
In the second paragraph, describe the most important factors contributing to the fraud risk.
In the third paragraph, provide recommended actions, such as flagging, manual review, or transaction decline.
For low-risk transactions (less than 50% probability of fraud), follow the same structure:

Paragraph 1: State the transaction’s fraud risk and probability.
Paragraph 2: Describe the factors supporting the transaction being genuine.
Paragraph 3: Provide suggestions for ensuring continued safe transactions or monitoring practices.
Formatting Guidelines:
Use simple, clear sentences and avoid unnecessary complexity.
Keep each paragraph concise and focused on actionable insights.
Your tone should be objective, concise, and actionable, directly addressing the fraud prevention team or risk management department.
Avoid using "I" or "you" and speak only in the third person.
"""

    print("EXPLANATION PROMPT", prompt)

    raw_response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{
            "role": "user",
            "content": prompt
        }],
    )
    return raw_response.choices[0].message.content


def generate_email(probability, input_dict, explanation, first):
    prompt = f"""You are a manager at HS Bank. You are responsible for
ensuring customers stay with the bank and are incentivized with
various offers.

You noticed a customer named {first} has a {round(probability * 
100, 1)}% probability of churning.

Here is the customer's information:
{input_dict}

Here is some explanation as to why the customer might be at risk 
of churning:
{explanation}

Generate an email to the customer based on their information, 
asking them to stay if they are at risk of churning, or offering them 
incentives so that they become more loyal to the bank.

Make sure to list out a set of incentives to stay based on their 
information, in bullet point format. Don't ever mention the 
probability of churning, or the machine learning model to the 
customer.
"""
    raw_response = client.chat.completions.create(model="Llama-3.1-8b-instant",
                                                  messages=[{
                                                      "role": "user",
                                                      "content": prompt
                                                  }])

    print("\\n\\nEMAIL PROMPT", prompt)

    return raw_response.choices[0].message.content


st.title("Credit Card Fraud Detection")

df = pd.read_csv("FraudTest.csv")

customers = [
    f"{row['trans_num']} - {row['first']}" for _, row in df.iterrows()
]

selected_transaction_option = st.selectbox("Select a Transaction", customers)

if selected_transaction_option:

    selected_transaction_id = selected_transaction_option.split(" - ")[0]
    print("Selected Transaction ID: ", selected_transaction_id)

selected_name = selected_transaction_option.split(" - ")[1]

print("Selected CC name: ", selected_name)

selected_transaction = df.loc[df['trans_num'] ==
                              selected_transaction_id].iloc[0]

print("Selected Transaction: ", selected_transaction)

col1, col2 = st.columns(2)

# Create two columns in the Streamlit app
col1, col2 = st.columns(2)

with col1:
    # Input for transaction amount
    amount = st.number_input("Transaction Amount",
                             min_value=0.0,
                             value=float(selected_transaction['amt']))

    # Dropdown to select the transaction category
    category = st.selectbox("Transaction Category", [
        "entertainment", "food_dining", "gas_transport", "grocery_net",
        "grocery_pos", "health_fitness", "home", "kids_pets", "misc_net",
        "misc_pos", "personal_care", "shopping_net", "shopping_pos", "travel"
    ],
                            index=[
                                "entertainment", "food_dining",
                                "gas_transport", "grocery_net", "grocery_pos",
                                "health_fitness", "home", "kids_pets",
                                "misc_net", "misc_pos", "personal_care",
                                "shopping_net", "shopping_pos", "travel"
                            ].index(selected_transaction['category']))

    # Input for the latitude of the transaction
    latitude = st.number_input("Transaction Latitude",
                               min_value=-90.0,
                               max_value=90.0,
                               value=float(selected_transaction['lat']))

    # Input for the longitude of the transaction
    longitude = st.number_input("Transaction Longitude",
                                min_value=-180.0,
                                max_value=180.0,
                                value=float(selected_transaction['long']))

    # Input for the transaction location's zip code
    zip_code = st.text_input("ZIP Code",
                             value=str(selected_transaction['zip']))

with col2:
    # Input for the merchant's latitude
    merchant_latitude = st.number_input("Merchant Latitude",
                                        min_value=-90.0,
                                        max_value=90.0,
                                        value=float(
                                            selected_transaction['merch_lat']))

    # Input for the merchant's longitude
    merchant_longitude = st.number_input(
        "Merchant Longitude",
        min_value=-180.0,
        max_value=180.0,
        value=float(selected_transaction['merch_long']))

    # Input for whether the transaction was made with a credit card

    # Input for whether the customer is an active member

    # Input for the estimated salary (if relevant for the transaction)

input_df, input_dict = prepare_input(amount, zip_code, latitude, longitude,
                                     merchant_latitude, merchant_longitude,
                                     category)

avg_probability = make_predictions(input_df, input_dict)

explanation = explain_prediction(avg_probability, input_dict,
                                 selected_transaction['first'])

st.markdown("---")

st.subheader("Explanation of Prediction")

st.markdown(explanation)

email = generate_email(avg_probability, input_dict, explanation,
                       selected_transaction['first'])

st.markdown("---")

st.subheader("Personalized Email")

st.markdown(email)
