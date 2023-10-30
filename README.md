# Fake News Prediction
1.	Importing Libraries: The code starts by importing necessary Python libraries, including NumPy, Pandas, regular expression (re), NLTK (Natural Language Toolkit) for text processing, scikit-learn for machine learning, and the necessary NLTK stopwords and PorterStemmer.
2.	Data Loading: The news dataset is loaded from a CSV file named 'train.csv'.
3.	Handling Missing Values: The code checks for missing values in the dataset and replaces them with empty strings.
4.	Feature Engineering: The 'content' column is created by combining the 'author' and 'title' columns, which will be used as the input features for the classification model.
5.	Data Splitting: The data is split into input features (X) and labels (Y). X contains the 'content' column, while Y contains the 'label' column.
6.	Text Preprocessing: The code defines a function called stemming that processes the text content by:
    •	Removing non-alphabetic characters and converting the text to lowercase.
    •	Splitting the text into words.
    •	Applying stemming to reduce words to their root forms using the Porter Stemmer.
    •	Removing stopwords from the text.
    •	Joining the stemmed words back into a single string.
7.	Applying Text Preprocessing: The stemming function is applied to the 'content' column of the dataset to preprocess the text.
8.	Text Vectorization: The TfidfVectorizer is used to convert the preprocessed text into feature vectors (X) for machine learning. It calculates TF-IDF (Term Frequency-Inverse Document Frequency) values for each word in the corpus.
9.	Data Splitting (Training and Testing): The data is split into training and testing sets using the train_test_split function.
10.	Model Training: A logistic regression model is created and trained on the training data (X_train and Y_train).
11.	Training and Testing Accuracy: The code calculates and prints the accuracy of the model on both the training and testing data.
12.	Making Predictions: The code allows the user to input an index to choose a specific news article from the test data. It then makes a prediction on that news article and displays whether it is classified as "real" or "fake."
13.	Display Actual Value: The code also displays the actual label value for the chosen news article.

