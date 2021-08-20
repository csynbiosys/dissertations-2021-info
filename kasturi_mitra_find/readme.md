The scripts and Jupyter notebooks need to be run in the following order:
1 techcrunch_scrape/web_scraper_tc.py
2 techcrunch_scrape/ws_tc_article_content.py
3 techcrunch_scrape/ner_spacy_tc.py
4 techcrunch_scrape/ner_nltk_tc.py
5 techcrunch_scrape/ner_stanford_tc.py
6 techcrunch_scrape/combine_ner_output_org.py
7 techcrunch_scrape/combine_ner_output_person.py
8 techcrunch_scrape/corepo_lookup_selenium.py
9 techcrunch_scrape/corepo_extract_org_details.py
10 techcrunch_scrape/ner_analysis.ipynb
11 techcrunch_scrape/retrieve_empty_reposnses.ipynb
12 evaluation/crunchbase_data_tranform.ipynb
13 evaluation/corepo_orgs_preprocessing.ipynb
14 evaluation/evaluate_scraped_data.ipynb
15 classification/retrive_funding_type.ipynb
16 classification/preprocessing_for_classification.ipynb
17 classification/classifiers_rough.ipynb
18 classification/classifiers.ipynb

Final output is stored in classification/unprocessed_corepo_predicted_labels.csv