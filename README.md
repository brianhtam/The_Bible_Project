# The_Bible_Project

### Metis Project 4: NLP and Unsupervised Learning

# Bible NLP: by Brian Tam

## Context & Goal

This was my project 4 from my experiance at the Metis Data Science boot camp. I wanted to extract meaning and truth, and was curious to see if the advancements in Data science could help me draw insights that years of seminary have not discovered. Given the NLP and unsupervised learning requirements of this project, I believe this passion of mine to be a good fit for exploration.

To start off, I needed to choose a dataset. There are countless version of the bible and here were my considerations:
- Original Hebrew - Straight from the source but the Hebrew language would be unsupported by spaCy and NLTK.
- KJV - Version that spread to most of Christianity as we see today, but old English may not work well with modern NLP toolkits
- NRSV - Ultimately what I decided to use do to the scholarly backing behind it and use in Academia 

## Methodologies
- I used the [Bible Corpus](https://www.kaggle.com/oswinrh/bible)
- Imported and exported data as SQL tables using d6tstack
#### Translation_EDA.ipynb
#### 