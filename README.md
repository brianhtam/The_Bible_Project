- Watch the full presentation on [Youtube](https://www.youtube.com/watch?v=wm-76yHYsoc&feature=youtu.be)
- My barcharts are published on [Tableau Public](https://public.tableau.com/profile/bgood2me#!/vizhome/Biblebythebooks/BiblebytheBooks?publish=yes)
- You can interact with my scattertext and pyLDAvis visuals on my [Flask App](https://the-bible-app.herokuapp.com/) (takes about 30 secounds to render)

### Metis Project 4: NLP and Unsupervised Learning

# Bible NLP: by Brian Tam

## Context & Goal

This was my project 4 from my experiance at the Metis Data Science boot camp. I wanted to extract meaning and truth, and was curious to see if the advancements in Data science could help me draw insights that years of seminary have not discovered. Given the NLP and unsupervised learning requirements of this project, I believe this passion of mine to be a good fit for exploration.

To start off, I needed to choose a dataset. There are countless version of the bible and here were my considerations:
- Original Hebrew - Straight from the source but the Hebrew language would be unsupported by spaCy and NLTK.
- KJV - Version that spread to most of Christianity as we see today, but old English may not work well with modern NLP toolkits
- NRSV - Scholarly backing behind it and use in Academia 
- **BBE - Ulimately what I proceeded with.** Uses the Basic English language restrictions that limit the vocab down to 1000 base words with ~3000 unique pronouns. This causes immense dimensionality reduction for our analysis.

## Methodologies
- I used the [Bible Corpus](https://www.kaggle.com/oswinrh/bible)
- Imported and exported data as SQL tables using d6tstack
#### Translation_EDA.ipynb

#### 
