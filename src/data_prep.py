import json
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR/'data/raw/online_shoppers_intention.csv'
OUTPUT_DIR = BASE_DIR/'data/processed'

def create_target_json(row):
    """Derives the structured JSON labels from the raw row data."""
    revenue =bool(row['Revenue'])

    if revenue:
        intent = "high"
        action = "trigger_checkout_incentive"
    elif row['PageValues'] > 0 or row['ProductRelated_Duration'] > 1000:
        intent = "medium"
        action = "show_trending_products"
    else:
        intent = "low"
        action = "no_action"
        
    return {
        "revenue_prediction": revenue,
        "intent_level": intent,
        "recommended_action": action
    }

def format_chatml(row):
    """Maps the row into Qwen's expected ChatML message format."""
    features = row.drop('Revenue').to_dict()
    target =create_target_json(row)

    return {
        "messages": [
            {
                "role": "system", 
                "content": "You are an E-Commerce Session Intent Interpreter. Analyze the session analytics and output strictly valid JSON containing 'revenue_prediction' (boolean), 'intent_level' ('high', 'medium', 'low'), and 'recommended_action'."
            },
            {
                "role": "user", 
                "content": json.dumps(features)
            },
            {
                "role": "assistant", 
                "content": json.dumps(target)
            }
        ]
    }


def main():
    print(f"Loading raw data from {INPUT_FILE}...")
    df=pd.read_csv(INPUT_FILE)

    print("Converting rows to ChatML Schema...")
    formatted_data= df.apply(format_chatml,axis=1).tolist()

    print("Splitting into train data(80%), Val(10%),Test(10%)...")
    train_data,temp_data=train_test_split(formatted_data,test_size=0.2,random_state=42)
    val_data, test_data= train_test_split(temp_data,test_size=0.5,random_state=42)

    os.makedirs(OUTPUT_DIR,exist_ok=True)
    print("Writing .jsonl files...")
    
    for split_name, dataset in [("train", train_data), ("val", val_data), ("test", test_data)]:
        output_path = os.path.join(OUTPUT_DIR, f'{split_name}.jsonl')
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in dataset:
                f.writelines(json.dumps(item) + '\n')
                
    print(f"Pipeline complete! Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
if __name__=='__main__':
    main()