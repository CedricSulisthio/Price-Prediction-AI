import pandas as pd
from fuzzywuzzy import fuzz
from sklearn.ensemble import RandomForestRegressor
import joblib
import os
from scipy.optimize import linprog
import tkinter as tk
from tkinter import filedialog
import requests
from bs4 import BeautifulSoup  # Correct import for BeautifulSoup
import re
import random

# Hide tkinter main window for file dialog usage
root = tk.Tk()
root.withdraw()

def convert_currency(amount, to_currency):
    """Converts given amount to the specified currency."""
    conversion_rates = {
        'IDR': 1,
        'USD': 16736,
        'SGD': 12974,
        'Ringgit': 3876,
        'Baht': 500.57,
        'Yuan': 2294,
        'Yen': 117.64,
        'Euro': 19059,
        'Won': 11.68
    }
    if to_currency not in conversion_rates:
        raise ValueError(f"Currency {to_currency} not supported.")
    if to_currency == 'IDR':
        return amount
    return amount / conversion_rates[to_currency]

def train_model():
    """Trains the AI model using user-selected Excel/CSV file."""
    print("Please select your Excel or CSV file to train the model.")
    file_path = filedialog.askopenfilename(title="Select Excel or CSV file", filetypes=[("Excel Files", ".xlsx"), ("CSV Files", ".csv")])
    if not file_path:
        print("No file selected. Exiting training.")
        return
    try:
        if file_path.lower().endswith('.csv'):
            data = pd.read_csv(file_path)
        else:
            data = pd.read_excel(file_path)
    except Exception as e:
        print("Error reading file:", e)
        return

    required_cols = ['Product Name', 'Demand', "Competitor's Price", 'Review Score', 'Optimal Price']
    if not all(col in data.columns for col in required_cols):
        print("Excel/CSV missing required columns. Expected:", required_cols)
        return
    
    X = data[['Demand', "Competitor's Price", 'Review Score']]
    y = data['Optimal Price']

    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)
    joblib.dump(model, 'pricing_model.pkl')
    print("Model trained and saved to 'pricing_model.pkl'")

def predict_price(demand, competitor_price, review_score):
    """Predicts the price using the trained model."""
    if not os.path.exists('pricing_model.pkl'):
        print("Model file not found. Please train the model first.")
        return None
    model = joblib.load('pricing_model.pkl')
    X_new = pd.DataFrame({'Demand': [demand], "Competitor's Price": [competitor_price], 'Review Score': [review_score]})
    pred = model.predict(X_new)[0]
    if pred > competitor_price * 1.2:
        pred = competitor_price * 1.2
    return pred

def extract_price(text):
    """Extracts numeric price from string using regex."""
    if not text:
        return None
    text = text.replace(',', '').replace(' ', '')
    matches = re.findall(r'\d+\.?\d*', text)
    if matches:
        # Take the largest number as price guess
        numbers = [float(m) for m in matches]
        return max(numbers)
    return None

def scrape_product_details(url):
    """Scrapes product details (name and price) from the given URL."""
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        res = requests.get(url, headers=headers)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, 'html.parser')

        # Try common title selectors
        title = (soup.find('h1') or soup.find('title') or soup.find('span', {'class': 'product-title'}))
        if title:
            name = title.get_text(strip=True)
        else:
            name = "Unknown Product"

        # Try several selectors for price
        price_text_candidates = []
        selectors = [
            {'name': 'span', 'attrs': {'class': re.compile('price', re.I)}},
            {'name': 'div', 'attrs': {'class': re.compile('price', re.I)}},
            {'name': 'span', 'attrs': {'id': re.compile('price', re.I)}},
            {'name': 'div', 'attrs': {'id': re.compile('price', re.I)}},
        ]
        for sel in selectors:
            tag = soup.find(sel['name'], sel['attrs'])
            if tag and tag.get_text(strip=True):
                price_text_candidates.append(tag.get_text(strip=True))
        # If no candidates, try meta tags (some sites have price in meta)
        if not price_text_candidates:
            meta_price = soup.find('meta', {'itemprop': 'price'})
            if meta_price and meta_price.get('content'):
                price_text_candidates.append(meta_price['content'])

        price = None
        for candidate in price_text_candidates:
            price = extract_price(candidate)
            if price is not None:
                break

        if price is None:
            return None
        return {'name': name, 'price': price}
    except Exception as e:
        print("Error scraping website:", e)
        return None

def scrape_from_excel():
    """Scrapes data from selected Excel or CSV file."""
    print("Please select your Excel or CSV file to search in.")
    file_path = filedialog.askopenfilename(title="Select Excel or CSV file", filetypes=[("Excel Files", ".xlsx"), ("CSV Files", ".csv")])
    if not file_path:
        print("No file selected.")
        return None
    keyword = input("Enter product keyword to search: ").strip()
    try:
        if file_path.lower().endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
    except Exception as e:
        print("Error reading file:", e)
        return None
    col_name = df.columns[0]
    col_price = df.columns[1]
    results = []
    for idx, product in enumerate(df[col_name]):
        score = fuzz.ratio(keyword.lower(), str(product).lower())
        if score > 70:
            results.append({'name': product, 'price': df.at[idx, col_price]})
    return results

def optimize_price(demand, competitor_price, review_score):
    """Uses linear programming to optimize price."""
    c = [-1]  # maximize price => minimize negative price
    A = [[1]]  # price <= competitor_price * 1.2
    b = [competitor_price * 1.2]
    bounds = [(0, None)]  # price >= 0
    res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
    if not res.success:
        print("Optimization failed, returning competitor price.")
        return competitor_price
    base_price = -res.fun
    adjusted_price = base_price * (1 + review_score / 5)
    return max(competitor_price, adjusted_price)

def response2content(response):
    """Extracts and processes the response from the AI, removing <think> tags."""
    try:
        content = response['choices'][0]['message']['content']
        # Remove <think> and </think> tags and anything inside
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        return content
    except Exception:
        return "Failed to parse AI response."

def query_AI(prompt, system_prompt="", chat_history=[], max_tokens=1000):
    """Queries the AI and ensures it responds only to relevant pricing questions."""
    url = "http://localhost:11434/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    model = "deepseek-r1:8b"
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages += chat_history + [{"role": "user", "content": prompt}]
    data = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": max_tokens
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            return response2content(response.json())
        else:
            return f"API error: {response.status_code} {response.text}"
    except Exception as e:
        return f"Error connecting to AI server: {e}"

def run_chatbot(system_prompt, initial_context=None):
    """Runs the AI chatbot, asking and answering relevant pricing questions."""
    max_length = 100
    chat_history = []
    
    if initial_context:
        chat_history.append({"role": "system", "content": system_prompt})
        chat_history.append({"role": "user", "content": initial_context})
    else:
        chat_history.append({"role": "system", "content": system_prompt})

    print("\nStart chatting with pricing assistant (type 'exit' or 'bye' to quit):")

    while True:
        userInput = input("You: ")
        if userInput.lower() in ["exit", "bye"]:
            print("Chatbot session ended.")
            break
        
        chat_history.append({"role": "user", "content": userInput})
        
        response = query_AI(userInput, system_prompt=system_prompt, chat_history=chat_history)
        print("Bot:", response)
        
        chat_history.append({"role": "assistant", "content": response})
        
        # Ask follow-up questions based on user input
        follow_up_question = ask_follow_up_question(userInput)
        if follow_up_question:
            print("Bot (follow-up):", follow_up_question)
            chat_history.append({"role": "assistant", "content": follow_up_question})
        
        if len(chat_history) > max_length:
            chat_history = chat_history[-max_length:]

def ask_follow_up_question(user_input):
    """Generates dynamic follow-up questions based on the user input."""
    # Example dynamic questions based on user inputs
    follow_up_questions = [
        "Would you like to compare your price with additional competitors?",
        "How much profit margin are you targeting?",
        "Do you want me to adjust based on demand forecast?"
    ]
    
    # Randomly choose a follow-up question to ask (can be more sophisticated in production)
    if random.random() > 0.5:  # Arbitrary chance of asking a follow-up question
        return random.choice(follow_up_questions)
    return None

def main():
    print("Welcome to Pricing Optimizer AI!\n")
    if not os.path.exists('pricing_model.pkl'):
        train_now = input("Model not found, train model now? (y/n): ").strip().lower()
        if train_now == 'y':
            train_model()
        else:
            print("Model required for price prediction. Exiting.")
            return
    
    while True:
        print("\nChoose data source:")
        print("1. Search from Local Excel/CSV")
        print("2. Quit")
        choice = input("Your choice (1/2): ").strip()
        if choice == '2':
            print("Goodbye!")
            break
        elif choice == '1':
            products = scrape_from_excel()
            if not products:
                print("No products found or no file selected.")
                continue
            print(f"Found {len(products)} matching products:")
            for i, p in enumerate(products, 1):
                print(f"{i}. {p['name']} - Price: {p['price']}")
            if len(products) > 1:
                sel = input(f"Select product number (1-{len(products)}): ").strip()
                try:
                    idx = int(sel) - 1
                    product = products[idx]
                except:
                    print("Invalid selection.")
                    continue
            else:
                product = products[0]
        else:
            print("Invalid choice, try again.")
            continue
        
        print(f"\nProceeding with product: {product['name']} at price {product['price']}")
        try:
            demand = float(input("Enter demand: ").strip())
            competitor_price = float(input("Enter competitor price: ").strip())
            review_score = float(input("Enter review score (0-5): ").strip())
            if not (0 <= review_score <= 5):
                print("Review score must be between 0 and 5.")
                continue
        except ValueError:
            print("Invalid numerical input, please try again.")
            continue

        predicted_price = predict_price(demand, competitor_price, review_score)
        if predicted_price is None:
            print("Price prediction failed.")
            continue
        optimized_price = optimize_price(demand, competitor_price, review_score)

        print(f"\nPredicted Price (AI): {predicted_price:.2f} IDR")
        print(f"Optimized Price (with OR): {optimized_price:.2f} IDR")
        currency = input("Enter currency code to convert price (IDR, USD, SGD, Ringgit, Baht, Yuan, Yen, Euro, Won): ").strip()
        try:
            converted_price = convert_currency(optimized_price, currency)
            print(f"Converted price: {converted_price:.2f} {currency}")
        except Exception as e:
            print(e)

        explain = input("\nDo you want to start a chatbot to discuss this optimized price? (y/n): ").strip().lower()
        if explain == 'y':
            system_prompt = (
                "You are a pricing expert assistant helping to explain optimized prices. "
                "Please only answer questions related to price prediction, calculation methods, "
                "and the purpose of this program. "
                "If asked questions unrelated to pricing or this program, respond politely: "
                "'I'm sorry, but I can only assist with pricing-related questions.'"
            )
            initial_context = (
                f"Product: {product['name']}\n"
                f"Demand: {demand}\n"
                f"Competitor Price: {competitor_price}\n"
                f"Review Score: {review_score}\n"
                f"Optimized Price: {optimized_price:.2f} IDR\n"
                "Please help explain why this optimized price is reasonable."
            )
            run_chatbot(system_prompt, initial_context)

if __name__ == "__main__":
    main()