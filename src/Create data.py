import os
import json
import pandas as pd
import re
from pathlib import Path

# Функція для очистки тексту
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

base_dir = Path('../data/fakenewsnet_dataset/gossipcop')

data = []

for label in ['fake', 'real']:
    label_dir = base_dir / label
    for news_folder in label_dir.iterdir():
        news_json_path = news_folder / 'news content.json'
        if news_json_path.exists():
            try:
                with open(news_json_path, 'r', encoding='utf-8') as f:
                    news_data = json.load(f)
                # Витягуємо title і текст або summary
                title = news_data.get('title', '').strip()
                text = news_data.get('text', '').strip()
                summary = news_data.get('summary', '').strip()
                main_content = text if text else summary

                # Об’єднуємо title + текст
                combined_text = f"{title} {main_content}".strip()

                # Очищення тексту
                cleaned_combined_text = clean_text(combined_text)

                data.append({
                    'id': news_folder.name,
                    'binary_label': 0 if label == 'fake' else 1,
                    'statement': cleaned_combined_text
                })
            except Exception as e:
                print(f"❗ Error reading {news_json_path}: {e}")

df = pd.DataFrame(data)

print(df.head())

df.to_csv('../data/gossipcop_clean.csv', index=False)
print("✅ Saved to ../data/gossipcop_clean.csv")
