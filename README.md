# Sentiment-Analysis


## IMDb Movie Reviews Sentiment Analysis

This project performs sentiment analysis on a large dataset of IMDb movie reviews using deep learning with Keras and GloVe word embeddings.

### Dataset

The dataset used is the IMDb Movie Reviews dataset, which contains 50,000 movie reviews labeled as either positive or negative sentiment. The dataset is automatically downloaded and extracted when running the code.
![image](https://github.com/27priyanshu/Sentiment-Analysis/assets/95427620/af96a7bc-c14c-4fae-bac9-46f1724b51bf)


### Pre-processing

The dataset is pre-processed with the following steps:

1. **Remove special characters, numbers, and other non-alphabetic characters** from the movie reviews
2. **Convert sentiment labels "positive" and "negative" to numeric values 1 and 0 respectively**

### Word Embeddings

1. **The GloVe pre-trained word embeddings are used to build an embedding dictionary**
2. **An embedding matrix is created for the corpus based on the GloVe embeddings**

### Model Training

1. **An LSTM model is built and trained using Keras**
2. **The model is trained on the pre-processed dataset**
3. **Model performance is evaluated and results are analyzed**
4. ![image](https://github.com/27priyanshu/Sentiment-Analysis/assets/95427620/b45a8b16-64d1-4691-8e81-6d30646c1b55)


### Prediction

1. **The trained model is used to perform sentiment predictions on real IMDb movie reviews**
2. **The model predicts whether each review has positive or negative sentiment**
   ![image](https://github.com/27priyanshu/Sentiment-Analysis/assets/95427620/2c6d3946-9672-4e87-b644-31ae4a51431e)
   ![image](https://github.com/27priyanshu/Sentiment-Analysis/assets/95427620/52ab808f-a153-45ab-b227-7fa89b0fa2bb)
    ![image](https://github.com/27priyanshu/Sentiment-Analysis/assets/95427620/a5b1d164-d06c-48ef-98a2-d8013220f565)



### Requirements

- Python 3.x
- Keras
- TensorFlow
- NumPy
- Pandas
- NLTK
- Scikit-learn

### Usage

1. **Clone the repository**
2. **Install required packages**
3. **Run the Jupyter notebook or Python script**

### Results

The trained LSTM model achieved an accuracy of 86% on the IMDb movie reviews dataset. The model was able to accurately predict the sentiment of real-world movie reviews.
![image](https://github.com/27priyanshu/Sentiment-Analysis/assets/95427620/27a89a50-31d6-4cd6-b793-6c817d25df3c)


### Conclusion

This project demonstrates how deep learning techniques like LSTMs and pre-trained word embeddings can be effectively applied to the task of sentiment analysis on large text datasets. The results show the potential of these methods for understanding human sentiment from text data.

### References

- [IMDb Movie Reviews Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
- [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)
- [Keras: The Python Deep Learning API](https://keras.io/)
