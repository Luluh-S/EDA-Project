# EDA-Project
Unit 3 project:  PhiUSIIL Phishing URL

# Phishing URL Detection

What makes a phishing link different from a real one? This project explores the PhiUSIIL Phishing URL
Dataset (235,795 links), finds a leak in the data, trains models, and puts it all in a Streamlit app
with a link checker.

## What's inside

| File | What it is |
|---|---|
| `PhiUSIIL_Phishing_URL_Dataset.csv` | The original data: 235,795 links and 56 columns |
| `eda+model.ipynb` | Cleaning, feature engineering, EDA, the leakage check, outliers, the models and a final summary |
| `clean_features.csv` | The cleaned data (235,370 links, 64 columns), saved by the notebook |
| `link_checker.py` | Reads a link and judges it. Used by the notebook and the app |
| `app.py` | The Streamlit app |
| `models/link_model.joblib` | The trained link checker and its test results |
| `models/popular_domains_top10k.txt` | The 10,000 most visited websites, from the Tranco list |
| `figures/eda/` | The charts from the notebook |
| `.streamlit/config.toml` | The app's colours and fonts |



## The app

- **Dashboard:** filters for the ending, trust score and class, univariate and bivariate charts, a data
  preview (raw data, cleaning steps, clean data, summary statistics and outliers), the model results
  and the main insights.
- **Signs of phishing:** 13 warning signs with examples, in English and Arabic.
- **Check a link:** paste a link and get a verdict with the reasons. It combines the model's score with
  red-flag rules, a short list of trusted Saudi sites and the list of popular websites. The link is
  never opened.

## What I found

- Phishing pages are tiny. The typical one has 12 lines of code, while a real one has 1,105.
- Trust signals matter most. Pages with none are all phishing, and pages with 6 or 7 almost never are.
- Cheap endings like `.top`, `.cf` and `.ml` are almost all phishing.
- The data has a leak: every real link is a bare `https://www.` homepage, so some columns (like
  `URLSimilarityIndex`) give the answer away. I left 21 of them out of the honest model.

## Models

Every model was tested on websites it never saw during training.

| Model | What it reads | Result |
|---|---|---|
| Model A | All columns, leaky ones included | 100%, because of the leak |
| Model B | 33 columns without the leak, but it needs the page itself | 99.84% right |
| Link checker (used in the app) | Only the website name and its ending | 84.6% right, catches 73% of phishing, 89% of its warnings are right |

The link checker's score is calibrated, so a score of 40% means roughly a 40% chance of phishing.

## Limitations

- Every real link in the data is a homepage, so the checker never learned from real links with a path
  like `/login`.
- Only 42 links in the data are Saudi, which is why the trusted Saudi list was written by hand.
- The checker only reads the link, so it can't spot a real website that was hacked.
- Ordinary names with hyphens, like `my-shop-riyadh.com`, can still get flagged.

## Data

- PhiUSIIL Phishing URL Dataset (Prasad and Chandra, 2024), UCI Machine Learning Repository:
  https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset
- Popular websites: the Tranco list (https://tranco-list.eu), 25 October 2026
