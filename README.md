# IMDB Movie Reviews Sentiment Analysis

## This project focuses on performing sentiment analysis on movie reviews from the IMDB dataset. The goal is to classify reviews as positive or negative based on their content

## Dataset

The dataset used in this project is the IMDB Movie Reviews dataset, which contains 50,000 movie reviews labeled as positive or negative. The dataset is split into 40,000 reviews for training and 10,000 reviews for testing.

## Methodology

1. **Data Preprocessing**: The reviews were cleaned by removing HTML tags, punctuation, and stop words. The text was then vectorized into sequences of integers using a vectorizer
2. **Models traned**: A liner regression model and a Naive Bayes model was trained on the vectorized data to classify the reviews as positive or negative then the best model was selected based on the evaluation metrics
3. **Evaluation**: The model's performance was evaluated using accuracy, precision, recall, and F1-score,metrics
4. **application**: A streamlit application was developed to allow users to input their own movie reviews and receive a sentiment classification and the features that lead to model make the prediction.

## Results

The model achieved an accuracy of ![Accuracy](https://img.shields.io/badge/Accuracy-88.94%25-green) on the test set, indicating that it is effective in classifying movie reviews based on their sentiment. The precision, recall, and F1-score also showed that the model performs well in distinguishing between positive and negative reviews

## Conclusion

The sentiment analysis model developed in this project demonstrates a strong ability to classify movie reviews as positive or negative. This can be useful for various applications, such as analyzing customer feedback, monitoring social media sentiment, and improving recommendation systems. Future work could involve exploring more advanced models, such as deep learning techniques, to further enhance the performance of the sentiment analysis.

## setup

To set up the environment for this project, follow these steps:

1. Clone the repository:

   ```bash
   git clone 
    ```

2. Navigate to the project directory:

    ```bash
    cd NLP-Sentiment-Analysis
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Run the main script to train and evaluate the model:

    ```bash
    python app.py
    ```
