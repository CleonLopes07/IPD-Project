from fastapi import FastAPI, HTTPException, File, UploadFile
import requests
import os
import re

from sklearn.preprocessing import StandardScaler, LabelEncoder
from dotenv import load_dotenv # type: ignore
from fastapi.middleware.cors import CORSMiddleware
import cv2 # type: ignore
import numpy as np
from typing import List, Dict
import json
import logging
import pandas as pd # type: ignore
from sklearn.metrics import accuracy_score
import pickle

import joblib

model = joblib.load("random_forest_model.pkl")  # ✅ Use joblib, not pickle

LOCAL_CSV_FILE = "final_merged_resolved_data.csv"
df_sample = pd.read_csv(LOCAL_CSV_FILE, nrows=10)
df_sample.columns = df_sample.columns.str.strip()
print(df_sample.columns.tolist())
X_train_example = pd.read_csv("final_merged_resolved_data.csv")[
    ['salt_100g', 'sugars_100g', 'saturated-fat_100g', 'fiber_100g', 'proteins_100g', 'additives_n']
]
scaler = StandardScaler()
scaler.fit(X_train_example)
label_encoder = LabelEncoder()
label_encoder.fit(['Healthy', 'Moderate', 'Unhealthy']) 


use_cols = ['salt_100g', 'sugars_100g', 'saturated-fat_100g', 'fiber_100g', 'proteins_100g', 'additives_n', 'label']


if all(col in df_sample.columns for col in use_cols):
    # Now, load the full dataset, ensuring column names are stripped
    chunk_size = 100_000
    chunks = pd.read_csv(
        LOCAL_CSV_FILE,
        sep=',',
        low_memory=False,
        on_bad_lines='skip',
        usecols=use_cols,
        chunksize=chunk_size
    )

df_test = pd.concat(chunks, ignore_index=True)

# Ensure the feature columns match those used in training
features = ['salt_100g', 'sugars_100g', 'saturated-fat_100g', 'fiber_100g', 'proteins_100g', 'additives_n']

df_test.columns = df_test.columns.str.strip()

X_test = df_test[features]
y_test = df_test['label']  # replace 'label' with your actual label column name

# Convert to numeric if needed
X_test = X_test.apply(pd.to_numeric, errors='coerce').fillna(0)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
accuracy_percent = f"{accuracy * 100:.2f}%"


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load configuration from .env
load_dotenv()
GROQ_API_TOKEN = os.getenv("GROQ_API_TOKEN")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants and thresholds
DAILY_VALUES = {
    'salt': 2.3,       # mg (2.3g)
    'sugars': 36,       # g (WHO recommendation)
    'saturated-fat': 20, # g
    'fiber': 28,        # g
    'proteins': 50      # g
}

NUTRIENT_WEIGHTS = {
    'salt': 0.3,
    'sugars': 0.4,      # Increased weight for sugar
    'saturated-fat': 0.3,
    'fiber': 0.1,
    'proteins': 0.1
}

PRESERVATIVE_INFO = {
    # Artificial Colors (Common in Indian snacks, sweets, and beverages)
    "e102": {"name": "Tartrazine", "description": "Yellow synthetic dye", "risk": "High", "harmful_effects": "Linked to ADHD, banned in EU"},
    "e110": {"name": "Sunset Yellow", "description": "Orange-yellow dye", "risk": "High", "harmful_effects": "Hyperactivity, allergic reactions"},
    "e122": {"name": "Carmoisine", "description": "Red synthetic dye", "risk": "High", "harmful_effects": "Banned in US, carcinogenic in high doses"},
    "e124": {"name": "Ponceau 4R", "description": "Red food color", "risk": "High", "harmful_effects": "Asthma triggers, banned in Norway"},
    "e129": {"name": "Allura Red", "description": "Red dye in packaged foods", "risk": "High", "harmful_effects": "Hyperactivity in children"},

    # Preservatives (Widely used in Indian pickles, sauces, and processed foods)
    "e200": {"name": "Sorbic Acid", "description": "Mold inhibitor", "risk": "Low", "harmful_effects": "Moderate skin irritation"},
    "e202": {"name": "Potassium Sorbate", "description": "Cheese/bakery preservative", "risk": "Low", "harmful_effects": "Allergies in sensitive people"},
    "e211": {"name": "Sodium Benzoate", "description": "Soft drink preservative", "risk": "Moderate", "harmful_effects": "Forms benzene with Vitamin C"},
    "e220": {"name": "Sulfur Dioxide", "description": "Dried fruit preservative", "risk": "High", "harmful_effects": "Asthma attacks, throat irritation"},
    "e250": {"name": "Sodium Nitrite", "description": "Processed meat preservative", "risk": "High", "harmful_effects": "Linked to stomach cancer"},

    # Flavor Enhancers (Common in Indian instant noodles, chips, and street food)
    "e621": {"name": "MSG", "description": "Taste enhancer", "risk": "Moderate", "harmful_effects": "Headaches, 'Chinese Restaurant Syndrome'"},
    "e627": {"name": "Disodium Guanylate", "description": "Savory flavor booster", "risk": "Low", "harmful_effects": "Avoid for gout patients"},
    "e631": {"name": "Disodium Inosinate", "description": "Synergistic with MSG", "risk": "Low", "harmful_effects": "May cause overeating"},

    # Emulsifiers/Stabilizers (Used in Indian dairy, ice creams, and breads)
    "e407": {"name": "Carrageenan", "description": "Dairy stabilizer", "risk": "Moderate", "harmful_effects": "Gut inflammation risks"},
    "e412": {"name": "Guar Gum", "description": "Thickener in curd/yogurt", "risk": "Low", "harmful_effects": "Bloating in excess"},
    "e471": {"name": "Mono-Diglycerides", "description": "Vanaspati/processed fat", "risk": "Moderate", "harmful_effects": "Trans fat source"},

    # Antioxidants (Oily snacks and packaged foods)
    "e319": {"name": "TBHQ", "description": "Oil/fried snack preservative", "risk": "Moderate", "harmful_effects": "Liver damage at high doses"},
    "e320": {"name": "BHA", "description": "Cereal/potato chip preservative", "risk": "High", "harmful_effects": "Possible carcinogen"},
    "e321": {"name": "BHT", "description": "Chewing gum preservative", "risk": "Moderate", "harmful_effects": "Endocrine disruptor"},

    # Artificial Sweeteners (Diet products and sugar-free claims)
    "e951": {"name": "Aspartame", "description": "Diet drinks/sugar-free", "risk": "Moderate", "harmful_effects": "Migraines, controversial"},
    "e954": {"name": "Saccharin", "description": "Tabletop sweetener", "risk": "Low", "harmful_effects": "Bitter aftertaste"},
    "e955": {"name": "Sucralose", "description": "Heat-stable sweetener", "risk": "Low", "harmful_effects": "Alters gut microbiome"},

    # Acidity Regulators (Soft drinks, packaged juices)
    "e330": {"name": "Citric Acid", "description": "Natural preservative", "risk": "Low", "harmful_effects": "Safe in moderation"},
    "e338": {"name": "Phosphoric Acid", "description": "Cola drinks", "risk": "Moderate", "harmful_effects": "Bone density loss"},

    # Anti-Caking Agents (Powdered spices, salt)
    "e551": {"name": "Silicon Dioxide", "description": "Prevents clumping", "risk": "Low", "harmful_effects": "Inert but avoid inhalation"},
    "e554": {"name": "Sodium Aluminosilicate", "description": "Salt additive", "risk": "Moderate", "harmful_effects": "Aluminum exposure risk"},

    # Raising Agents (Bakery products)
    "e500": {"name": "Sodium Bicarbonate", "description": "Baking soda", "risk": "Low", "harmful_effects": "Safe in small quantities"},
    "e503": {"name": "Ammonium Carbonate", "description": "Traditional bakery agent", "risk": "Moderate", "harmful_effects": "Ammonia smell"},

    # Miscellaneous
    "e904": {"name": "Shellac", "description": "Confectionery glaze", "risk": "Low", "harmful_effects": "Vegan concern (insect-derived)"},
    "e920": {"name": "L-Cysteine", "description": "Bread softener", "risk": "Low", "harmful_effects": "Often sourced from hair/feathers"},
    "e1414": {"name": "Acetylated Starch", "description": "Instant noodle additive", "risk": "Low", "harmful_effects": "Digestive issues"},
    "e1442": {"name": "Hydroxypropyl Starch", "description": "Sauce thickener", "risk": "Low", "harmful_effects": "Bloating"},
    
    # Indian-Specific Additives
    "e160b": {"name": "Annatto", "description": "Natural cheese color", "risk": "Low", "harmful_effects": "Rare allergies"},
    "e296": {"name": "Malic Acid", "description": "Tamarind-based souring", "risk": "Low", "harmful_effects": "Tooth enamel erosion"},
    "e416": {"name": "Karaya Gum", "description": "Used in pan masala", "risk": "Moderate", "harmful_effects": "Gastrointestinal issues"},
    "e999": {"name": "Quillaia Extract", "description": "Foaming agent in sharbats", "risk": "Low", "harmful_effects": "Moderate toxicity in excess"},
    "e124": { "name": "Ponceau 4R", "description": "Synthetic red dye; banned in some countries", "risk": "Moderate" ,"harmful_effects": "May cause allergic reactions"},
    "e322": { "name": "Lecithins", "description": "Natural emulsifier from soy/eggs", "risk": "Generally safe for most people" },
    "e171": { "name": "Titanium Dioxide", "description": "White pigment; banned in EU", "risk": "High","harmful_effects":"Possible genotoxic effects" },
    "e330": { "name": "Citric Acid", "description": "Natural acid from citrus fruits", "risk": "Low" },
    "e250": { "name": "Sodium Nitrite", "description": "Preservative in meats; potential carcinogen", "risk":"High" },
    
    "e150d": { "name": "Sulphite Ammonia Caramel", "description": "Brown coloring in soft drinks; may affect digestion", "risk":"Moderate" ,"harmful_effects":"Possible allergen, linked to hyperactivity" },
    "e451": { "name": "Triphosphates", "description": "Stabilizer and emulsifier in processed foods", "risk":"Moderate" ,"harmful_effects":"May affect kidney function, calcium balance" },
    "e501": { "name": "Potassium Carbonates", "description": "Raising agent in baked goods, cocoa", "risk":"Low","harmful_effects":"Generally low risk; excessive intake may upset balance" },
    "e508": { "name": "Potassium Chloride", "description": "Flavor enhancer and salt substitute", "risk": "Low","harmful_effects":"High doses may affect heart function" },
    "e635": { "name": "Disodium 5'-Ribonucleotides", "description": "Flavor enhancer in snacks, soups", "risk":"High","harmful_effects": "May cause allergic reactions or hyperactivity" },
   
    "e460": { 
    "name": "Cellulose", 
    "description": "Bulking agent, anti-caking, dietary fiber", 
     "risk": "Low",
    "harmful_effects":"Generally safe; excessive intake may cause bloating" 
   
  },
  "e460i": { 
    "name": "Microcrystalline Cellulose", 
    "description": "Refined cellulose used as thickener, stabilizer", 
    "risk": "Low",
    "harmful_effects": "Generally safe; excessive intake may cause digestive discomfort" 
  }


 
    
     
    
}
INGREDIENT_ALIASES = {
    "sodium chloride": "Salt",
    "sucrose": "Sugar",
    "monosodium glutamate": "MSG",
    "ascorbic acid": "Vitamin C",
    "tocopherols": "Vitamin E"
}

ECO_SCORE_MAPPING = {
    'a': 'A',
    'b': 'B',
    'c': 'C',
    'd': 'D',
    'e': 'E',
    'unknown': 'Not Available'
}

def get_eco_score(grade: str) -> str:
    return ECO_SCORE_MAPPING.get(grade.lower(), 'Not Available')

def fetch_product_data(barcode: str) -> dict:
    try:
        if not re.match(r'^\d{8,14}$', barcode):
            raise HTTPException(status_code=400, detail="Invalid barcode format")

        url = f"https://world.openfoodfacts.org/api/v0/product/{barcode}.json"
        resp = requests.get(url, timeout=10)
        
        if resp.status_code != 200:
            raise HTTPException(status_code=503, detail="Food database unavailable")
        
        data = resp.json()
        if data.get("status") == 0:
            raise HTTPException(status_code=404, detail="Product not found")
        
        product = data.get("product", {})
        


        
        return {
            "product_name": product.get("product_name", "Unnamed Product"),
            "ingredients_text": product.get("ingredients_text", "Ingredients not available"),
            "nutriments": product.get("nutriments", {}),
            "additives_tags": product.get("additives_tags", []),
            "nova_group": product.get("nova_group", 0),
            "ecoscore_grade": get_eco_score(product.get("ecoscore_grade", "unknown")),
            "allergens": product.get("allergens", "Not specified"),
            "categories": product.get("categories", "")  # Added to get category data
        }
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Database connection timeout")
    

    #new changes 
def bayesian_health_score(nutriments):
    prior = 0.5
    odds = prior / (1 - prior)
    nutrient_tests = [
        {
            "name": "sugars",
            "value": nutriments.get("sugars", 0),
            "threshold": 15,
            "sensitivity": 0.8,
            "specificity": 0.7,
        },
        {
            "name": "salt",
            "value": nutriments.get("salt", 0),
            "threshold": 1.2,
            "sensitivity": 0.75,
            "specificity": 0.7,
        },
        {
            "name": "saturated-fat",
            "value": nutriments.get("saturated-fat", 0),
            "threshold": 5,
            "sensitivity": 0.7,
            "specificity": 0.7,
        },
        {
            "name": "fiber",
            "value": nutriments.get("fiber", 0),
            "threshold": 3,
            "sensitivity": 0.7,
            "specificity": 0.6,
            "reverse": True,
        },
        {
            "name": "proteins",
            "value": nutriments.get("proteins", 0),
            "threshold": 3,
            "sensitivity": 0.7,
            "specificity": 0.6,
            "reverse": True,
        },
    ]
    for test in nutrient_tests:
        value = test["value"]
        threshold = test["threshold"]
        sensitivity = test["sensitivity"]
        specificity = test["specificity"]
        reverse = test.get("reverse", False)
        if reverse:
            test_positive = value >= threshold
        else:
            test_positive = value <= threshold
        if test_positive:
            lr = sensitivity / (1 - specificity)
        else:
            lr = (1 - sensitivity) / specificity
        odds *= lr
    prob_healthy = odds / (1 + odds)
    health_score = round(prob_healthy * 100, 1)
    return health_score

    #new changes

def compute_health_score(nutriments: dict, additives: list) -> dict:
    score = 70
    components = {}

    nutrient_values = {
        'salt': nutriments.get('salt', 0),
        'sugars': nutriments.get('sugars', 0),
        'saturated-fat': nutriments.get('saturated-fat', 0),
        'fiber': nutriments.get('fiber', 0),
        'proteins': nutriments.get('proteins', 0)
    }

    for nutrient in ['salt', 'sugars', 'saturated-fat']:
        dv = DAILY_VALUES[nutrient]
        value = nutrient_values[nutrient]
        percentage = (value / dv) * 100
        if percentage > 100:
            excess = percentage - 100
            deduction = (excess ** 1.5) * NUTRIENT_WEIGHTS[nutrient]
            score -= deduction
            components[nutrient] = f"-{deduction:.1f}% (dangerous)"
        elif percentage > 50:
            deduction = (percentage - 50) * NUTRIENT_WEIGHTS[nutrient]
            score -= deduction
            components[nutrient] = f"-{deduction:.1f}% (high)"

    for nutrient in ['fiber', 'proteins']:
        dv = DAILY_VALUES[nutrient]
        value = nutrient_values[nutrient]
        percentage = (value / dv) * 100
        if percentage < 50:
            deduction = (50 - percentage) * NUTRIENT_WEIGHTS[nutrient]
            score -= deduction
            components[nutrient] = f"-{deduction:.1f}% (low)"
        else:
            addition = (percentage - 50) * NUTRIENT_WEIGHTS[nutrient]
            score += addition
            components[nutrient] = f"+{addition:.1f}% (good)"

    additive_penalty = len(additives) * 3
    score -= additive_penalty
    components['additives'] = f"-{additive_penalty} ({len(additives)} additives)"

    if nutrient_values['sugars'] > 20 and nutrient_values['proteins'] < 5:
        score -= 15
        components['empty_calories'] = "-15% (high sugar, low nutrition)"

    final_score = max(0, min(100, round(score, 1)))
    bayesian_score = bayesian_health_score(nutriments)


    range_bracket = int(final_score // 10) * 10
    assessment_map = {
        90: "Exceptional (90-100)",
        80: "Excellent (80-89)",
        70: "Good (70-79)",
        60: "Moderate (60-69)",
        50: "Fair (50-59)",
        40: "Poor (40-49)",
        30: "Unhealthy (30-39)",
        20: "Harmful (20-29)",
        10: "Dangerous (10-19)",
        0: "Hazardous (0-9)"
    }
    health_assessment = assessment_map.get(range_bracket, "Hazardous (0-9)")

    # Prepare input for prediction
    input_df = pd.DataFrame([{
        'salt_100g': nutrient_values['salt'],
        'sugars_100g': nutrient_values['sugars'],
        'saturated-fat_100g': nutrient_values['saturated-fat'],
        'fiber_100g': nutrient_values['fiber'],
        'proteins_100g': nutrient_values['proteins'],
        'additives_n': len(additives)
    }])
    input_scaled = scaler.transform(input_df)

    predicted_class = model.predict(input_scaled)[0]
    predicted_proba = model.predict_proba(input_scaled)[0]
    confidence_score = max(predicted_proba) * 100

    predicted_label = (
        label_encoder.inverse_transform([predicted_class])[0]
        if label_encoder else
        {0: 'Healthy', 1: 'Moderate', 2: 'Unhealthy'}.get(predicted_class, 'Unknown')
    )

    return {
        "score": final_score,
        "bayesian_score": bayesian_score,
        "components": components,
        # "assessment": health_assessment,
        "classification_model": {
            "prediction": predicted_label,
            "confidence": f"{confidence_score:.2f}%"
        }
    }
def analyze_additives(additives: list) -> list:
    results = []
    for tag in additives:
        code = tag.split(":")[-1].lower()
        if code in PRESERVATIVE_INFO:
            info = PRESERVATIVE_INFO[code].copy()
            info["code"] = code.upper()
            results.append(info)
        else:
            results.append({
                "code": code.upper(),
                "name": "Unknown Additive",
                "risk": "Uncertain",
                "description": "Requires further research",
                "harmful_effects": "Potential unknown health risks"
            })
    return results

def categorize_product_with_llm(product_data: dict) -> str:
    """Use LLM to determine product category"""
    if not GROQ_API_TOKEN:
        logger.warning("No GROQ API token found. Using basic category detection.")
        # Fallback to basic detection if no API token
        product_lower = product_data['product_name'].lower()
        if any(keyword in product_lower for keyword in ['chips', 'crisps', 'snack']):
            return 'chips'
        return 'general'
        
    try:
        prompt = f"""
        Categorize this food product into a specific food category.
        Product: {product_data['product_name']}
        Ingredients: {product_data['ingredients_text']}
        Categories from database: {product_data.get('categories', 'Not specified')}
        
        Return ONLY a single word category that best describes this product.
        If it's chips or crisps, simply return "chips".
        Other example categories: beverage, snack, dairy, bakery, cereal, pasta, sauce, etc.
        
        Your response should be a single word only.
        """
        
        headers = {"Authorization": f"Bearer {GROQ_API_TOKEN}"}
        
        logger.info(f"Sending category request to LLM for product: {product_data['product_name']}")
        
        response = requests.post(
            GROQ_API_URL,
            json={
                "model": "llama3-70b-8192",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 50
            },
            headers=headers,
            timeout=15
        )
        
        # Debug print
        print(f"LLM category response status: {response.status_code}")
        
        if response.status_code == 200:
            response_data = response.json()
            print(f"LLM category raw response: {json.dumps(response_data, indent=2)}")
            
            category = response_data['choices'][0]['message']['content'].strip().lower()
            print(f"Extracted category: {category}")
            
            # Clean up response to ensure a single word
            if '\n' in category:
                category = category.split('\n')[0]
            if ' ' in category:
                category = category.split(' ')[0]
                
            return category
        else:
            print(f"LLM category error response: {response.text}")
            return "general"
    except Exception as e:
        logger.error(f"Error in LLM categorization: {str(e)}")
        print(f"LLM categorization error: {str(e)}")
        return "general"

def generate_healthier_alternatives(product_data: dict, category: str) -> dict:
    """Generate healthier alternatives using LLM"""
    if not GROQ_API_TOKEN:
        logger.warning("No GROQ API token found. Cannot generate alternatives.")
        return {"alternatives": [
            {
                "alternative": "Alternative options unavailable",
                "brand": "LLM API not configured",
                "benefits": "Please configure LLM API",
                "improvements": "N/A",
                "additive_reduction": "N/A"
            }
        ]}
            
    try:
        # Enhanced prompt with more specific guidance
        prompt = f"""
        Analyze this food product and suggest 3 specific healthier Indian alternatives:
        Product: {product_data['product_name']}
        Category: {category}
        Nutrition per 100g: {{
            Salt: {product_data['nutriments'].get('salt', 0)}g,
            Sugars: {product_data['nutriments'].get('sugars', 0)}g,
            Sat.Fat: {product_data['nutriments'].get('saturated-fat', 0)}g,
            Fiber: {product_data['nutriments'].get('fiber', 0)}g,
            Protein: {product_data['nutriments'].get('proteins', 0)}g
        }}
        Additives count: {len(product_data['additives_tags'])}
        
        Suggest 3 real, healthier alternatives with specific brand names. Include:
        1. A lower fat/sodium option in the same category
        2. A higher fiber/protein option in the same category
        3. A completely natural alternative with minimal processing
        
        For example, if it's chips, suggest specific healthier chip options with real brand names.
        
        Format your response exactly as a markdown table:
        | Alternative | Brand | Benefits | Improvements | Additive Reduction |
        | --- | --- | --- | --- | --- |
        | Product Name | Brand Name | Key health benefits | Quantified improvements | Reduction in additives |
        
        Make sure to only output the table in the specified format. 
        If it's chips, include something like banana chips which are among the healthiest options.
        """
        
        logger.info(f"Sending alternatives request to LLM for product: {product_data['product_name']}")
        print(f"LLM alternatives prompt: {prompt}")
        
        headers = {"Authorization": f"Bearer {GROQ_API_TOKEN}"}
        response = requests.post(
            GROQ_API_URL,
            json={
                "model": "gemma2-9b-it", #different model for recommendations
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 800
            },
            headers=headers,
            timeout=20
        )
        
        # Debug print
        print(f"LLM alternatives response status: {response.status_code}")
        
        if response.status_code == 200:
            response_data = response.json()
            print(f"LLM alternatives raw response: {json.dumps(response_data, indent=2)}")
            return parse_llm_table(response_data)
        else:
            print(f"LLM alternatives error response: {response.text}")
            return {"alternatives": [{"alternative": "Error retrieving alternatives", "brand": "API Error", "benefits": "N/A", "improvements": "N/A", "additive_reduction": "N/A"}]}
            
    except Exception as e:
        logger.error(f"Error generating alternatives: {str(e)}")
        print(f"LLM alternatives error: {str(e)}")
        return {"alternatives": [{"alternative": "Error retrieving alternatives", "brand": "Exception", "benefits": "N/A", "improvements": "N/A", "additive_reduction": "N/A"}]}

def parse_llm_table(response: dict) -> dict:
    try:
        table_data = []
        raw_text = response['choices'][0]['message']['content']
        print(f"LLM table parsing - raw text: {raw_text}")
        
        # Improved parsing to handle various table formats
        lines = raw_text.split('\n')
        header_pattern = re.compile(r'^\s*\|\s*Alternative\s*\|\s*Brand\s*\|', re.IGNORECASE)
        separator_pattern = re.compile(r'^\s*\|\s*-+\s*\|\s*-+\s*\|')
        data_pattern = re.compile(r'^\s*\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|')
        
        in_table = False
        for i, line in enumerate(lines):
            # Skip empty lines
            if not line.strip():
                continue
                
            # Skip header and separator lines
            if header_pattern.match(line):
                in_table = True
                print(f"Found table header at line {i}: {line}")
                continue
                
            if separator_pattern.match(line):
                continue
                
            # Extract data rows
            if in_table and '|' in line:
                print(f"Processing potential data row at line {i}: {line}")
                match = data_pattern.match(line)
                if match:
                    alternative, brand, benefits, improvements, additive_reduction = [g.strip() for g in match.groups()]
                    table_data.append({
                        "alternative": alternative,
                        "brand": brand,
                        "benefits": benefits,
                        "improvements": improvements,
                        "additive_reduction": additive_reduction
                    })
                    print(f"Extracted data row: {alternative} | {brand}")
                elif line.count('|') >= 6:  # Fallback if regex doesn't match but has enough columns
                    parts = [p.strip() for p in line.split('|')]
                    if len(parts) >= 7:  # Account for empty parts at start/end
                        # Skip empty parts at beginning/end
                        parts = [p for p in parts if p]
                        if len(parts) >= 5:
                            table_data.append({
                                "alternative": parts[0],
                                "brand": parts[1],
                                "benefits": parts[2],
                                "improvements": parts[3],
                                "additive_reduction": parts[4]
                            })
                            print(f"Extracted data row using fallback: {parts[0]} | {parts[1]}")
        
        print(f"Table parsing result: found {len(table_data)} alternatives")
        
        # If no data was found in table format, try to extract key information from text
        if not table_data:
            print("No table format found, trying text extraction")
            
            # Try to find product mentions and extract information
            lines = raw_text.split('\n')
            current_item = {}
            current_products = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    # Save current item if we have an alternative name
                    if current_item.get("alternative"):
                        current_products.append(current_item)
                        current_item = {}
                    continue
                    
                # Try to identify product names
                if re.match(r'^[1-3]\.\s', line):  # Lines starting with 1., 2., 3.
                    product_info = line[3:].strip()
                    # Try to extract brand
                    brand_match = re.search(r'by\s+([^-:]+)', product_info)
                    if brand_match:
                        brand = brand_match.group(1).strip()
                        product_name = product_info.split('by')[0].strip()
                        current_item = {
                            "alternative": product_name,
                            "brand": brand,
                            "benefits": "Not specified",
                            "improvements": "Not specified",
                            "additive_reduction": "Not specified"
                        }
                    else:
                        current_item = {
                            "alternative": product_info,
                            "brand": "Not specified",
                            "benefits": "Not specified",
                            "improvements": "Not specified", 
                            "additive_reduction": "Not specified"
                        }
                
                # Try to identify key attributes
                if current_item.get("alternative"):
                    # Look for benefits
                    if "benefit" in line.lower():
                        current_item["benefits"] = line.split(":", 1)[1].strip() if ":" in line else line
                    # Look for improvements
                    elif "improvement" in line.lower():
                        current_item["improvements"] = line.split(":", 1)[1].strip() if ":" in line else line
                    # Look for additive reduction
                    elif "additive" in line.lower() or "reduction" in line.lower():
                        current_item["additive_reduction"] = line.split(":", 1)[1].strip() if ":" in line else line
            
            # Add final item if not empty
            if current_item.get("alternative"):
                current_products.append(current_item)
                
            if current_products:
                print(f"Text extraction found {len(current_products)} products")
                table_data = current_products
        
        if not table_data:
            # As a last resort, split by numbered items and make best guess
            print("Using last resort extraction method")
            products = []
            text_blocks = re.split(r'(?:\r?\n){2,}', raw_text)
            
            for block in text_blocks:
                if len(block.strip()) > 10:  # Non-empty block
                    lines = block.strip().split('\n')
                    # Try to get product name from first line
                    product_name = lines[0].strip()
                    if re.match(r'^[1-3]\.\s', product_name):
                        product_name = product_name[3:].strip()
                        
                    # Extract any keyword that might be a brand
                    brand = "Unknown"
                    for line in lines[1:]:
                        if "brand" in line.lower() and ":" in line:
                            brand = line.split(":", 1)[1].strip()
                            break
                    
                    benefits = "Healthier alternative"
                    # Try to find a line that might describe benefits
                    for line in lines[1:]:
                        if len(line) > 15 and not line.startswith('|'):
                            benefits = line.strip()
                            break
                            
                    products.append({
                        "alternative": product_name,
                        "brand": brand,
                        "benefits": benefits,
                        "improvements": "See benefits",
                        "additive_reduction": "Reduced additives"
                    })
            
            if products:
                print(f"Last resort extraction found {len(products)} products")
                table_data = products[:3]
        
        # If we still have no data, create a basic error message
        if not table_data:
            print("No alternatives could be extracted, returning error")
            return {"alternatives": [{
                "alternative": "LLM did not return properly formatted alternatives",
                "brand": "Error",
                "benefits": "Check LLM configuration",
                "improvements": "N/A",
                "additive_reduction": "N/A"
            }]}
        
        # Return up to 3 alternatives
        return {"alternatives": table_data[:3]}
    except Exception as e:
        logger.error(f"Error parsing LLM table: {str(e)}")
        print(f"Table parsing error: {str(e)}")
        return {"alternatives": [{
            "alternative": "Error parsing LLM response",
            "brand": "Parser Error",
            "benefits": str(e),
            "improvements": "N/A",
            "additive_reduction": "N/A"
        }]}

def simplify_ingredients(ingredients_list: list) -> list:
    """Simplifies food ingredients, removes unnecessary details, and translates from French to English."""

    if not GROQ_API_TOKEN:
        print("⚠️ Warning: No LLM API token found. Using basic alias mapping.")
        simplified = []
        for ingredient in ingredients_list:
            clean_ingredient = ingredient.lower().strip()
            simplified_name = INGREDIENT_ALIASES.get(clean_ingredient, ingredient)
            simplified.append(simplified_name)
        return simplified

    # 🔹 Improved System Prompt
    system_prompt = f"""
    You are a food ingredient expert. Process this list of food ingredients:
    
    - Remove unnecessary details like percentages (e.g., "13%", "4.7%").
    - Remove words like "may contain", "sans gluten", and other allergy warnings.
    - Convert complex scientific names into common consumer-friendly terms.
    - Translate French ingredients into English naturally.
    
    ### Important:  
    - **Return ONLY a JSON array** of ingredient names.  
    - **Do not include explanations, text, or formatting.**  
    - Example Output: `["Wheat flour", "Sugar", "Palm oil"]`
    
    Process this ingredient list:  
    ```json
    {json.dumps(ingredients_list)}
    ```
    """

    headers = {"Authorization": f"Bearer {GROQ_API_TOKEN}"}

    response = requests.post(
        GROQ_API_URL,
        json={
            "model": "llama3-70b-8192",
            "messages": [
                {"role": "system", "content": system_prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 250
        },
        headers=headers
    )

    if response.status_code == 200:
        try:
            response_data = response.json()
            response_text = response_data["choices"][0]["message"]["content"].strip()

            # 🔹 Ensure it's valid JSON before parsing
            if response_text.startswith("[") and response_text.endswith("]"):
                simplified_list = json.loads(response_text)
                return simplified_list
            else:
                print(f"⚠️ Unexpected Response Format: {response_text}")
                return ["Error: Unexpected response format"]
        
        except json.JSONDecodeError:
            return ["Error: Failed to parse JSON response"]
    else:
        return [f"Error: API call failed with status {response.status_code}"]

@app.get("/product/{barcode}", response_model=Dict)
def get_product(barcode: str):
    try:
        product_data = fetch_product_data(barcode)
        
        # Parse ingredients into a list
        ingredients_list = [ing.strip() for ing in product_data['ingredients_text'].split(',')]
        
        # Use LLM to categorize the product
        logger.info(f"Categorizing product: {product_data['product_name']}")
        category = categorize_product_with_llm(product_data)
        logger.info(f"Product {product_data['product_name']} categorized as: {category}")
        
        # Generate healthier alternatives
        logger.info(f"Generating alternatives for {product_data['product_name']}")
        alternatives = generate_healthier_alternatives(product_data, category)
        logger.info(f"Generated {len(alternatives['alternatives'])} alternatives")
        
        return {
            "product_info": {
                "name": product_data['product_name'],
                "nova_group": product_data['nova_group'],
                "ecoscore": product_data['ecoscore_grade'],
                "category": category
            },
            "ingredients": {
                "full_list": ingredients_list,
                "simplified_list": simplify_ingredients(ingredients_list),
                "additives": analyze_additives(product_data['additives_tags']),
                "preservatives": [a for a in analyze_additives(product_data['additives_tags']) 
                               if 'preservative' in a['description'].lower()]
            },
            "nutrition_analysis": compute_health_score(
                product_data['nutriments'],
                product_data['additives_tags']
            ),
            "healthier_alternatives": alternatives
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in get_product: {str(e)}")
        raise HTTPException(500, detail=str(e))

@app.post("/product/image")
async def process_image(file: UploadFile = File(...)):
    """Process image containing barcode with improved error handling"""
    try:
        # Validate image format
        if not file.content_type.startswith('image/'):
            raise HTTPException(400, "Invalid file type: Image required")

        # Read and decode image
        img_data = await file.read()
        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(400, "Invalid image format")

        # Try OpenCV barcode detector first
        detector = cv2.barcode.BarcodeDetector()
        detected, decoded_info, _ = detector.detectAndDecode(img)
        
        # If not detected, try with preprocessed image
        if not detected or not decoded_info:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 11, 2)
            detected, decoded_info, _ = detector.detectAndDecode(binary)
        
        # If still not detected, try pyzbar
        barcode = None
        if not detected or not decoded_info:
            import pyzbar.pyzbar as pyzbar
            decoded_objects = pyzbar.decode(img)
            logger.info(f"Pyzbar detected: {decoded_objects}")
            if decoded_objects:
                barcode = decoded_objects[0].data.decode('utf-8')
                logger.info(f"Using barcode from pyzbar: {barcode}")
                detected = True
            else:
                # Try pyzbar with preprocessed image as last resort
                decoded_objects = pyzbar.decode(binary)
                logger.info(f"Pyzbar (on binary) detected: {decoded_objects}")
                if decoded_objects:
                    barcode = decoded_objects[0].data.decode('utf-8')
                    logger.info(f"Using barcode from pyzbar (binary): {barcode}")
                    detected = True
        
        # If OpenCV detected something, use that (only if pyzbar didn't find anything)
        if barcode is None and detected and decoded_info:
            barcode = decoded_info[0] if isinstance(decoded_info, (list, tuple)) else decoded_info
            logger.info(f"Using barcode from OpenCV: {barcode}")
        
        # Final check - do we have a barcode?
        if not detected or barcode is None:
            img_shape = img.shape if img is not None else "Unknown"
            return {
                "status": "error",
                "message": "No barcode detected",
                "suggestion": "Ensure barcode is clear and centered in the image",
                "debug_info": {
                    "image_dimensions": f"{img_shape[1]}x{img_shape[0]}" if isinstance(img_shape, tuple) else "Unknown",
                    "opencv_result": f"detected={detected}, info={decoded_info}"
                }
            }
        
        # Validate barcode format
        if not re.match(r'^\d{8,14}$', barcode):
            raise HTTPException(400, f"Invalid barcode format detected: {barcode}")

        return get_product(barcode)

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Image processing failed: {str(e)}")
        raise HTTPException(500, f"Image processing error: {str(e)}")
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)