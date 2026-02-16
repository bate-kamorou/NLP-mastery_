import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud



# load the cleaned dataser
df = pd.read_csv("NLP_Sentiment_Analysis/data/processed/5000-cleaned-reviews.csv")

# creat a filter for the negative reviews
negative_filter = df["sentiment"] == "negative"

# sort for the negative reviews
negative_review = df[negative_filter]

# index the cleaned negative reviews
negative_review_cleaned = negative_review["cleaned_review"]

# groupe the negative reviews  from the cleaned_review column in one bloc of text  
negative_text = negative_review_cleaned.str.cat(sep=" ")

# create a figure for plotting
plt.rcParams["figure.figsize"]= [10, 10]

# plot the 50 most used words in the wordcloud of the poistive reviews
nr_word_cloud = WordCloud(max_font_size=50, max_words=50, background_color="white", colormap="flag").generate(negative_text)
plt.plot()
plt.imshow(nr_word_cloud, interpolation="bilinear")
plt.axis("off")
plt.title("Most used words in the negative reviews")
plt.show()