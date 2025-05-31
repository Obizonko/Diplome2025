import pandas as pd
import numpy as np
import wikipedia
from rank_bm25 import BM25Okapi
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import re
import spacy
import time

try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("Downloading en_core_web_sm model...")
    spacy.cli.download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')
print("spaCy NER model loaded successfully.")

WIKI_LANG = 'en'
try:
    wikipedia.set_lang(WIKI_LANG)
    print(f"Wikipedia language set to: {WIKI_LANG}")
except Exception as e:
    print(f"Error setting Wikipedia language: {e}. Defaulting to 'en'.")
    wikipedia.set_lang('en')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device for NLI: {device}")
nli_model_name = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
try:
    nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
    nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name).to(device)
    print(f"NLI model '{nli_model_name}' loaded successfully.")
    nli_pipeline_instance = None
except Exception as e:
    print(f"Error loading NLI model directly: {e}")
    print("Using Hugging Face pipeline as fallback for NLI.")
    nli_pipeline_instance = pipeline("text-classification", model=nli_model_name, tokenizer=nli_model_name,
                                     device=0 if device.type == 'cuda' else -1)
    nli_model = None
    nli_tokenizer = None

try:
    news_df = pd.read_csv('../data/liar/test_filtered.csv')
    news_df.dropna(subset=['statement'], inplace=True)
    news_df = news_df[news_df['statement'].apply(lambda x: isinstance(x, str) and x.strip() != '')].copy()
    print(f"Loaded {len(news_df)} news statements.")
except FileNotFoundError:
    print("Error: News dataset file not found.")
    exit()
except KeyError:
    print("Error: 'statement' column not found.")
    exit()


def extract_key_phrases_for_search(text, nlp_model, max_phrases=3):
    doc = nlp_model(text)
    entities = [ent.text for ent in doc.ents if
                ent.label_ in ("PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "NORP", "FAC")]

    if len(entities) < max_phrases:
        noun_phrases = [chunk.text for chunk in doc.noun_chunks]
        entities.extend(noun_phrases)

    seen = set()
    unique_phrases = [x for x in entities if not (x in seen or seen.add(x))]

    if not unique_phrases:
        return [" ".join(text.split()[:7])]

    return unique_phrases[:max_phrases]


def search_wikipedia_pages_content(query_terms, max_pages_per_term=1, sentences_per_page=5):
    all_page_contents = []
    for term in query_terms:
        try:
            search_results = wikipedia.search(term, results=max_pages_per_term)
            for page_title in search_results:
                try:
                    page = wikipedia.page(page_title, auto_suggest=False,
                                          redirect=True)
                    content = page.summary
                    if len(content.split()) < 30 and hasattr(page, 'content'):
                        full_content = page.content
                        content = " ".join(full_content.split()[:1000])
                    all_page_contents.append({"title": page.title, "content": content, "url": page.url})
                    print(f" Fetched Wikipedia page: {page.title} for query '{term}'")
                except wikipedia.exceptions.PageError:
                    print(
                        f"Wikipedia PageError (page not found or ambiguous) for title: '{page_title}' from query '{term}'.")
                except wikipedia.exceptions.DisambiguationError as e:
                    print(f" Wikipedia DisambiguationError for query '{term}': {e.options[:3]}... Skipping.")
                except Exception as e_page:
                    print(f"Error fetching page content for '{page_title}': {e_page}")
                time.sleep(0.1)
        except Exception as e_search:
            print(f" Error during Wikipedia search for '{term}': {e_search}")
    return all_page_contents


def get_relevant_sentences_from_content(page_contents_list, claim_text, num_sentences_total=5):
    all_sentences_with_source = []
    for page_data in page_contents_list:
        content = page_data["content"]
        sentences = [s.strip() for s in re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', content) if
                     s.strip() and len(s.split()) > 3]
        for s in sentences:
            all_sentences_with_source.append(
                {"text": s, "source_title": page_data["title"], "source_url": page_data["url"]})

    if not all_sentences_with_source:
        return []

    tokenized_evidence_texts = [s_data["text"].lower().split() for s_data in all_sentences_with_source]

    try:
        if not any(tokenized_evidence_texts):
            print(" All tokenized evidence texts are empty for BM25.")
            return []
        bm25 = BM25Okapi(tokenized_evidence_texts)
        tokenized_claim = claim_text.lower().split()
        scores = bm25.get_scores(tokenized_claim)

        top_n_indices = np.argsort(scores)[::-1][:num_sentences_total]
        relevant_evidence_data = [all_sentences_with_source[i] for i in top_n_indices if
                                  scores[i] > 0.1]
        return relevant_evidence_data
    except ValueError as e:
        print(f" BM25 Error: {e}. Tokenized evidence texts sample: {tokenized_evidence_texts[:2]}")
        return []
    except Exception as e_bm:
        print(f"Unexpected error in BM25: {e_bm}")
        return []


def verify_claim_with_nli(claim, evidence_data_list, nli_model_instance, nli_tokenizer_instance, device_to_use,
                          nli_pipeline_instance=None):
    if not evidence_data_list:
        return {"label": "NOT ENOUGH INFO", "score": 0.0, "evidence_text": "No evidence found",
                "evidence_source": "N/A"}

    results = []
    for evidence_data in evidence_data_list:
        premise = evidence_data["text"]
        hypothesis = claim
        source_title = evidence_data["source_title"]
        source_url = evidence_data["source_url"]

        if nli_model_instance and nli_tokenizer_instance:
            inputs = nli_tokenizer_instance(premise, hypothesis, return_tensors='pt', truncation=True, padding=True,
                                            max_length=512).to(device_to_use)
            with torch.no_grad():
                logits = nli_model_instance(**inputs).logits

            probabilities = torch.softmax(logits, dim=1).squeeze()
            predicted_class_id = torch.argmax(logits, dim=1).item()
            predicted_label = nli_model_instance.config.id2label[predicted_class_id]
            score = probabilities[predicted_class_id].item()

            results.append({"label": predicted_label.upper(), "score": score,
                            "evidence_text": premise, "evidence_source_title": source_title,
                            "evidence_source_url": source_url})

        elif nli_pipeline_instance:
            sep_token = getattr(nli_tokenizer, 'sep_token', '</s></s>')  # Get sep_token or use default
            sequence_pair = f"{premise} {sep_token} {hypothesis}"
            nli_result_pipeline = nli_pipeline_instance(sequence_pair)
            main_result = nli_result_pipeline[0] if isinstance(nli_result_pipeline, list) else nli_result_pipeline
            results.append({"label": main_result['label'].upper(), "score": main_result['score'],
                            "evidence_text": premise, "evidence_source_title": source_title,
                            "evidence_source_url": source_url})
        else:
            return {"label": "NLI ERROR", "score": 0.0, "evidence_text": "NLI model/pipeline not available",
                    "evidence_source": "N/A"}

    if not results:
        return {"label": "NOT ENOUGH INFO", "score": 0.0, "evidence_text": "No NLI results", "evidence_source": "N/A"}

    contradictions = [res for res in results if res['label'] == 'CONTRADICTION']
    if contradictions:
        return max(contradictions, key=lambda x: x['score'])

    entailments = [res for res in results if res['label'] == 'ENTAILMENT']
    if entailments:
        return max(entailments, key=lambda x: x['score'])

    return max(results, key=lambda x: x['score'] if x['label'] == 'NEUTRAL' else -1) if any(
        r['label'] == 'NEUTRAL' for r in results) else results[0]


num_news_to_check = 10
results_fact_checking = []

for index, row in news_df.head(num_news_to_check).iterrows():
    news_statement = row['statement']
    actual_label = row['binary_label']

    print(f"\nNews Statement ({index + 1}/{num_news_to_check}): {news_statement}")
    print(f"True Label: {'REAL' if actual_label == 1 else 'FAKE'}")

    search_phrases = extract_key_phrases_for_search(news_statement, nlp, max_phrases=2)
    print(f"Search Phrases: {search_phrases}")

    wiki_page_contents = search_wikipedia_pages_content(search_phrases, max_pages_per_term=1)

    all_retrieved_evidence_data = []
    if wiki_page_contents:
        all_retrieved_evidence_data = get_relevant_sentences_from_content(wiki_page_contents, news_statement,
                                                                          num_sentences_total=5)
        if all_retrieved_evidence_data:
            print(f" Top relevant evidence sentences found: {len(all_retrieved_evidence_data)}")
            for i, ev_data in enumerate(all_retrieved_evidence_data):
                print(f"        Ev {i + 1} (from {ev_data['source_title']}): {ev_data['text'][:150]}...")
        else:
            print(" No relevant sentences extracted from Wikipedia content.")
    else:
        print("No Wikipedia content found for search phrases.")

    if all_retrieved_evidence_data:
        nli_final_result = verify_claim_with_nli(news_statement, all_retrieved_evidence_data, nli_model, nli_tokenizer,
                                                 device, nli_pipeline_instance=nli_pipeline_instance)
        print(f"NLI Final Decision: Label: {nli_final_result['label']}, Score: {nli_final_result['score']:.4f}")
        print(
            f"     Supporting Evidence (Title: {nli_final_result.get('evidence_source_title', 'N/A')}): {nli_final_result.get('evidence_text', 'N/A')[:200]}...")

        predicted_news_label = "UNCERTAIN"
        if nli_final_result['label'] == 'ENTAILMENT' and nli_final_result['score'] > 0.65:
            predicted_news_label = "LIKELY REAL (Supported by Wiki)"
        elif nli_final_result['label'] == 'CONTRADICTION' and nli_final_result['score'] > 0.65:
            predicted_news_label = "LIKELY FAKE (Contradicted by Wiki)"
        elif nli_final_result['label'] == 'NOT ENOUGH INFO':
            predicted_news_label = "CANNOT VERIFY (Not Enough Info from Wiki)"
    else:
        print("No evidence retrieved from Wikipedia to verify the claim.")
        nli_final_result = {"label": "NO WIKI EVIDENCE", "score": 0.0}
        predicted_news_label = "CANNOT VERIFY (No Wiki Evidence)"

    results_fact_checking.append({
        "statement": news_statement,
        "true_label_numeric": actual_label,
        "true_label_text": 'REAL' if actual_label == 1 else 'FAKE',
        "nli_decision": nli_final_result['label'],
        "nli_score": nli_final_result['score'],
        "predicted_news_status": predicted_news_label,
        "retrieved_evidence_count": len(all_retrieved_evidence_data)
    })
    time.sleep(0.5)

results_df = pd.DataFrame(results_fact_checking)
print("\n\nFact-Checking Results Summary:")
print(results_df[['statement', 'true_label_text', 'predicted_news_status', 'nli_decision', 'nli_score',
                  'retrieved_evidence_count']])

if not results_df.empty:
    def map_status_to_binary(status):
        if "REAL" in status: return 1
        if "FAKE" in status: return 0
        return -1


    results_df['predicted_binary_fc'] = results_df['predicted_news_status'].apply(map_status_to_binary)
    evaluable_fc_results = results_df[
        results_df['predicted_binary_fc'] != -1].copy()  # Use .copy() to avoid SettingWithCopyWarning

    if not evaluable_fc_results.empty:
        print("\nSimplified Fact-Checking System Performance (on evaluable predictions):")

        evaluable_fc_results['true_label_numeric'] = pd.to_numeric(evaluable_fc_results['true_label_numeric'],
                                                                   errors='coerce')
        evaluable_fc_results['predicted_binary_fc'] = pd.to_numeric(evaluable_fc_results['predicted_binary_fc'],
                                                                    errors='coerce')
        evaluable_fc_results.dropna(subset=['true_label_numeric', 'predicted_binary_fc'], inplace=True)

        if not evaluable_fc_results.empty:
            # Import classification_report if not already imported at the top
            from sklearn.metrics import classification_report

            print(classification_report(evaluable_fc_results['true_label_numeric'],
                                        evaluable_fc_results['predicted_binary_fc'], zero_division=0))
        else:
            print("No evaluable predictions after ensuring numeric types and dropping NaNs.")
    else:
        print("\nNo conclusive predictions made by the fact-checking system to evaluate performance.")