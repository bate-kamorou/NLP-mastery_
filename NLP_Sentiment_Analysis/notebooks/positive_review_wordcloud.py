import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud



# load the cleaned dataser
df = pd.read_csv("NLP_Sentiment_Analysis/data/processed/5000-cleaned-reviews.csv")

# create a filter for the positive reviews  
positive_filtre = df["sentiment"] == "positive"

# sort for only the positive reviews 
postive_review = df[positive_filtre]

# index the  cleaned positive reviews in the dataset
postive_review_cleaned = postive_review["cleaned_review"]

# concatenate all the positive reviews into of one big text (cloud)
positive_text = postive_review_cleaned.str.cat(sep=" ")

# create a figure
plt.rcParams["figure.figsize"]= [10, 10]
# plot the wordcloud of the 50 most used words in the poistive  reviews
pr_word_cloud = WordCloud(max_font_size=50, max_words=50, background_color="white", colormap="flag").generate(positive_text)
plt.plot()
plt.imshow(pr_word_cloud, interpolation="bilinear")
plt.axis("off")
plt.title("Most used words in the positive reviwes")
plt.show()