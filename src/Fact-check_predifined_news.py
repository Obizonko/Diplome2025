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
print("spaCy model loaded successfully.")

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
nli_tokenizer = None  # Initialize nli_tokenizer
try:
    nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
    nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name).to(device)
    print(f"NLI model '{nli_model_name}' loaded successfully.")
    nli_pipeline_instance = None
except Exception as e:
    print(f"Error loading NLI model directly: {e}")
    print("Using Hugging Face pipeline as fallback for NLI.")
    fallback_tokenizer_for_pipeline = AutoTokenizer.from_pretrained(
        nli_model_name) if nli_tokenizer is None else nli_tokenizer
    nli_pipeline_instance = pipeline("text-classification", model=nli_model_name,
                                     tokenizer=fallback_tokenizer_for_pipeline,
                                     device=0 if device.type == 'cuda' else -1)
    nli_model = None
    if nli_tokenizer is None:  # Ensure nli_tokenizer is set if pipeline is used
        nli_tokenizer = fallback_tokenizer_for_pipeline

predefined_news = [
    {
        "statement": "NASA confirmed the discovery of an Earth-like exoplanet with conditions suitable for life.",
        "true_label_text": "REAL",
        "true_label_numeric": 1
    },
    {
        "statement": "A new study found that consuming dark chocolate daily increases lifespan by 10 years.",
        "true_label_text": "FAKE",
        "true_label_numeric": 0
    },
    {
        "statement": "The Eiffel Tower was originally intended to be a temporary structure and was almost dismantled.",
        "true_label_text": "REAL",
        "true_label_numeric": 1
    },
    {
        "statement": "Scientists have successfully cloned a woolly mammoth using preserved DNA.",
        "true_label_text": "UNCERTAIN_OR_SPECULATIVE",
        "true_label_numeric": -1
    },
    {
        "statement": "Drinking coffee before bedtime improves sleep quality.",
        "true_label_text": "FAKE",
        "true_label_numeric": 0
    },
    {
        "statement": "The Amazon rainforest produces 20% of the world's oxygen supply.",
        "true_label_text": "REAL",
        "true_label_numeric": 1
    },
    {
        "statement": "A secret underground city was discovered beneath the streets of Paris.",
        "true_label_text": "UNCERTAIN_OR_SPECULATIVE",
        "true_label_numeric": -1
    },
    {
        "statement": "The Great Pyramid of Giza was originally covered in white limestone, making it shine like a jewel.",
        "true_label_text": "REAL",
        "true_label_numeric": 1
    },
    {
        "statement": "A new law requires all citizens to wear helmets while walking in public spaces.",
        "true_label_text": "FAKE",
        "true_label_numeric": 0
    },
    {
        "statement": "A recent breakthrough in medicine allows humans to regenerate lost limbs.",
        "true_label_text": "UNCERTAIN_OR_SPECULATIVE",
        "true_label_numeric": -1
    }
]
news_df = pd.DataFrame(predefined_news)
print(f"Loaded {len(news_df)} predefined news statements.")


def generate_refined_search_queries(text, nlp_model, max_queries=2):
    doc = nlp_model(text)
    queries = set()

    entities = sorted(
        list(set([ent.text.strip() for ent in doc.ents if ent.label_ in (
            "PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "NORP", "FAC") and len(
            ent.text.split()) >= 1])),
        key=len, reverse=True
    )
    for ent in entities:
        queries.add(ent)
        if len(queries) >= max_queries: break

    if len(queries) < max_queries:
        subjects = [tok.text for tok in doc if "subj" in tok.dep_]
        objects = [tok.text for tok in doc if "obj" in tok.dep_ or "attr" in tok.dep_]
        main_verb = next((tok.lemma_ for tok in doc if tok.pos_ == "VERB" and not tok.is_stop), None)

        if subjects and main_verb:
            query_parts = [subjects[0], main_verb]
            if objects:
                query_parts.append(objects[0])
            queries.add(" ".join(query_parts))

    if not queries or len(list(queries)[0].split()) < 2:
        cleaned_text = re.sub(r"^(Says|Claims|Asserts|Reports that|that)\s+", "", text, flags=re.IGNORECASE).strip()
        queries.add(" ".join(cleaned_text.split()[:10]))

    final_queries = sorted([q for q in list(queries) if q and len(q.strip().split()) > 1], key=len,
                           reverse=True)
    return final_queries[:max_queries] if final_queries else [text[:70]]


def fetch_wikipedia_content_robust(query_terms, max_pages_to_try=2):
    fetched_content = []
    processed_titles = set()

    for term in query_terms:
        try:
            print(f"    Searching Wikipedia for query: '{term}'")
            try:
                page = wikipedia.page(term, auto_suggest=False, redirect=True)
                if page.title not in processed_titles:
                    print(f"    Found direct page: {page.title}")
                    fetched_content.append({"title": page.title, "content": page.content, "url": page.url})
                    processed_titles.add(page.title)
                if len(fetched_content) >= max_pages_to_try: break
                continue
            except wikipedia.exceptions.DisambiguationError as e:
                print(f"    Disambiguation for '{term}'. Options: {e.options[:3]}")
                for option_title in e.options[:2]:
                    if option_title not in processed_titles:
                        try:
                            page = wikipedia.page(option_title, auto_suggest=False, redirect=True)
                            print(f"    Fetched (from disambiguation): {page.title}")
                            fetched_content.append({"title": page.title, "content": page.content, "url": page.url})
                            processed_titles.add(page.title)
                            if len(fetched_content) >= max_pages_to_try: break
                        except Exception:
                            continue
                if len(fetched_content) >= max_pages_to_try: break
            except wikipedia.exceptions.PageError:
                print(f"    No direct page found for '{term}'. Trying search.")
                search_results = wikipedia.search(term, results=max_pages_to_try)
                for page_title in search_results:
                    if page_title not in processed_titles:
                        try:
                            page = wikipedia.page(page_title, auto_suggest=False, redirect=True)
                            print(f"    Fetched (from search): {page.title}")
                            fetched_content.append({"title": page.title, "content": page.content, "url": page.url})
                            processed_titles.add(page.title)
                            if len(fetched_content) >= max_pages_to_try: break
                        except Exception:
                            continue
                if len(fetched_content) >= max_pages_to_try: break
        except Exception as e_outer:
            print(f"    Unexpected error searching for '{term}': {e_outer}")
        time.sleep(0.3)
    return fetched_content


def rank_and_select_evidence(page_contents_list, claim_text, num_top_sentences=5, bm25_threshold=0):
    all_sentences_data = []
    for page_data in page_contents_list:
        content = page_data.get("content", "")
        if not content: continue

        sentences = [s.strip() for s in re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', content) if
                     s.strip() and len(s.split()) >= 5]
        for s_text in sentences:
            all_sentences_data.append({
                "text": s_text,
                "source_title": page_data["title"],
                "source_url": page_data["url"]
            })

    if not all_sentences_data:
        return []

    tokenized_evidence_texts = [s_data["text"].lower().split() for s_data in all_sentences_data]
    valid_tokenized_evidence = [tokens for tokens in tokenized_evidence_texts if tokens]
    valid_indices = [i for i, tokens in enumerate(tokenized_evidence_texts) if tokens]

    if not valid_tokenized_evidence:
        print("      All tokenized evidence sentences are empty after filtering.")
        return []

    try:
        bm25 = BM25Okapi(valid_tokenized_evidence)
        tokenized_claim = claim_text.lower().split()
        scores = bm25.get_scores(tokenized_claim)

        scored_evidence = []
        for i, score_val in enumerate(scores):
            original_index = valid_indices[i]
            if score_val > bm25_threshold:
                evidence_item = all_sentences_data[original_index]
                evidence_item["bm25_score"] = score_val
                scored_evidence.append(evidence_item)

        scored_evidence.sort(key=lambda x: x["bm25_score"], reverse=True)
        return scored_evidence[:num_top_sentences]
    except Exception as e_bm25:
        print(f"      BM25 Error: {e_bm25}")
        return []


def verify_claim_with_nli(claim, evidence_data_list, nli_model_instance, nli_tokenizer_instance, device_to_use,
                          nli_pipeline_instance=None):
    if not evidence_data_list:
        return {"label": "NOT ENOUGH INFO", "score": 0.0, "evidence_text": "No evidence found",
                "evidence_source_title": "N/A", "evidence_source_url": "N/A"}

    results = []
    for evidence_data in evidence_data_list[:3]:
        premise = evidence_data["text"]
        hypothesis = claim
        source_title = evidence_data["source_title"]
        source_url = evidence_data["source_url"]

        current_tokenizer = nli_tokenizer_instance if nli_tokenizer_instance else (
            nli_pipeline_instance.tokenizer if nli_pipeline_instance and hasattr(nli_pipeline_instance,
                                                                                 'tokenizer') else None)

        if current_tokenizer is None and nli_pipeline_instance is not None and not hasattr(nli_pipeline_instance,
                                                                                           'tokenizer'):
            # Fallback if pipeline exists but tokenizer is not an attribute, try to get it via AutoTokenizer
            try:
                current_tokenizer = AutoTokenizer.from_pretrained(nli_pipeline_instance.model.name_or_path)
            except Exception:
                pass  # If this also fails, current_tokenizer remains None

        if nli_model_instance and current_tokenizer:
            inputs = current_tokenizer(premise, hypothesis, return_tensors='pt', truncation=True, padding=True,
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
        elif nli_pipeline_instance and current_tokenizer:
            sep_token = current_tokenizer.sep_token if hasattr(current_tokenizer,
                                                               'sep_token') and current_tokenizer.sep_token is not None else '</s></s>'
            sequence_pair = f"{premise} {sep_token} {hypothesis}"
            nli_result_pipeline_output = nli_pipeline_instance(sequence_pair)
            main_result = nli_result_pipeline_output[0] if isinstance(nli_result_pipeline_output,
                                                                      list) else nli_result_pipeline_output
            results.append({"label": main_result['label'].upper(), "score": main_result['score'],
                            "evidence_text": premise, "evidence_source_title": source_title,
                            "evidence_source_url": source_url})
        else:
            return {"label": "NLI_SETUP_ERROR", "score": 0.0, "evidence_text": "NLI model/tokenizer not available",
                    "evidence_source_title": "N/A", "evidence_source_url": "N/A"}

    if not results: return {"label": "NO_NLI_RESULTS", "score": 0.0, "evidence_text": "No NLI results after processing",
                            "evidence_source_title": "N/A", "evidence_source_url": "N/A"}

    contradictions = [res for res in results if res['label'] == 'CONTRADICTION']
    if contradictions: return max(contradictions, key=lambda x: x['score'])
    entailments = [res for res in results if res['label'] == 'ENTAILMENT']
    if entailments: return max(entailments, key=lambda x: x['score'])
    return max(results, key=lambda x: x['score'] if x['label'] == 'NEUTRAL' else -1) if any(
        r['label'] == 'NEUTRAL' for r in results) else results[0]


results_fact_checking_refined = []

for i, news_item in enumerate(predefined_news):
    news_statement = news_item['statement']
    true_label_text = news_item['true_label_text']

    print(f"\nNews Statement ({i + 1}/{len(predefined_news)}): {news_statement}")
    print(f"Predefined True Label: {true_label_text}")

    search_queries = generate_refined_search_queries(news_statement, nlp, max_queries=2)
    print(f"  Generated Search Queries: {search_queries}")

    wiki_pages_data = fetch_wikipedia_content_robust(search_queries, max_pages_to_try=2)

    top_evidence_data = []
    if wiki_pages_data:
        top_evidence_data = rank_and_select_evidence(wiki_pages_data, news_statement, num_top_sentences=5,
                                                     bm25_threshold=0.01)
        if top_evidence_data:
            print(f"      Top relevant evidence sentences found overall: {len(top_evidence_data)}")
            for j, ev_data in enumerate(top_evidence_data):
                print(
                    f"        Ev {j + 1} (from {ev_data['source_title']}, BM25: {ev_data['bm25_score']:.2f}): {ev_data['text'][:120]}...")
        else:
            print("      No relevant sentences extracted from Wikipedia content after BM25.")
    else:
        print("    No Wikipedia content found for any search query.")

    predicted_news_label_fc = "CANNOT VERIFY (No Actionable Evidence)"
    nli_final_result = {"label": "NO_NLI_RUN", "score": 0.0, "evidence_text": "N/A", "evidence_source_title": "N/A"}

    if top_evidence_data:
        # Pass nli_tokenizer if nli_model is used (not pipeline), otherwise it's handled within verify_claim_with_nli
        tokenizer_for_nli = nli_tokenizer if nli_model else None
        nli_final_result = verify_claim_with_nli(news_statement, top_evidence_data, nli_model,
                                                 tokenizer_for_nli, device,
                                                 nli_pipeline_instance=nli_pipeline_instance)
        print(f"  NLI Final Decision: Label: {nli_final_result['label']}, Score: {nli_final_result['score']:.4f}")
        best_evidence_text = nli_final_result.get('evidence_text', 'N/A')
        print(
            f"     Supporting Evidence (Title: {nli_final_result.get('evidence_source_title', 'N/A')}): {best_evidence_text[:200]}...")

        if nli_final_result['label'] == 'ENTAILMENT' and nli_final_result['score'] > 0.8:
            predicted_news_label_fc = "HIGHLY LIKELY REAL (Strong Wiki Support)"
        elif nli_final_result['label'] == 'CONTRADICTION' and nli_final_result['score'] > 0.8:
            predicted_news_label_fc = "HIGHLY LIKELY FAKE (Strong Wiki Contradiction)"
        elif nli_final_result['label'] == 'ENTAILMENT' and nli_final_result['score'] > 0.5:
            predicted_news_label_fc = "POSSIBLY REAL (Moderate Wiki Support)"
        elif nli_final_result['label'] == 'CONTRADICTION' and nli_final_result['score'] > 0.5:
            predicted_news_label_fc = "POSSIBLY FAKE (Moderate Wiki Contradiction)"
        elif nli_final_result['label'] == 'NOT ENOUGH INFO' or nli_final_result['label'] == 'NO_NLI_RESULTS':
            predicted_news_label_fc = "CANNOT VERIFY (NLI: Not Enough Info / No Results)"
        elif nli_final_result['label'] == 'NEUTRAL' and nli_final_result['score'] > 0.5:
            predicted_news_label_fc = "NEUTRAL (Wiki evidence not decisive)"
        else:
            predicted_news_label_fc = f"UNCERTAIN (NLI: {nli_final_result['label']}, Score: {nli_final_result['score']:.2f})"
    else:
        print("  No relevant evidence from Wikipedia to pass to NLI.")
        predicted_news_label_fc = "CANNOT VERIFY (No Wiki Evidence Found)"

    results_fact_checking_refined.append({
        "statement": news_statement,
        "true_label_text": true_label_text,
        "nli_decision": nli_final_result['label'],
        "nli_score": nli_final_result['score'],
        "predicted_news_status_fc": predicted_news_label_fc,
        "retrieved_evidence_count_after_bm25": len(top_evidence_data),
        "best_evidence_text_fc": nli_final_result.get('evidence_text', 'N/A'),
        "best_evidence_source_fc": nli_final_result.get('evidence_source_title', 'N/A')
    })
    time.sleep(1)

print("\n\nFact-Checking Results Summary (Refined Approach):")
results_fc_df_refined = pd.DataFrame(results_fact_checking_refined)
print(results_fc_df_refined[['statement', 'true_label_text', 'predicted_news_status_fc', 'nli_decision', 'nli_score',
                             'retrieved_evidence_count_after_bm25']])

results_fc_df_refined.to_csv('../results/predefined_news_fact_checking_results_refined.csv', index=False)
print("\nSaved refined fact-checking results to ../results/predefined_news_fact_checking_results_refined.csv")