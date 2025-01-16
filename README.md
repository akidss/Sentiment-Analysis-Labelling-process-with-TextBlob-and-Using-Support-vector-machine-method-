# Sentiment Analysis with TextBlob Labeling and SVM Classification

## Overview
This project demonstrates a sentiment analysis pipeline that integrates TextBlob for initial data labeling and Support Vector Machine (SVM) for classification. By combining the simplicity of TextBlob for sentiment scoring and the robustness of SVM for classification, this approach ensures an efficient and accurate sentiment analysis workflow.

## Features
- **TextBlob Labeling**:
  - Uses TextBlob to calculate the polarity of text.
  - Automatically assigns sentiment labels:
    - **Positive**: Polarity > 0
    - **Negative**: Polarity < 0


- **Support Vector Machine Analysis**:
  - Utilizes SVM for supervised classification of the labeled data.
  - Supports feature extraction techniques like TF-IDF or Word embeddings.
  - Provides metrics for evaluating model performance.

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   ```
2. Navigate to the project directory:
   ```bash
   cd your-repo-name
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. **Labeling with TextBlob**:
   - Load the dataset.
   - Apply TextBlob to calculate sentiment polarity and assign labels:
     ```python
     from textblob import TextBlob

     def label_sentiment(text):
         polarity = TextBlob(text).sentiment.polarity
         if polarity > 0:
             return 'Positive'
         elif polarity < 0:
             return 'Negative'
         else:
             return 'Neutral'

     data['sentiment'] = data['text'].apply(label_sentiment)
     ```

2. **Training with SVM**:
   - Preprocess the labeled data.
   - Extract features using TF-IDF:
     ```python
     from sklearn.feature_extraction.text import TfidfVectorizer

     vectorizer = TfidfVectorizer()
     X = vectorizer.fit_transform(data['text'])
     ```
   - Train the SVM model:
     ```python
     from sklearn.svm import SVC

     model = SVC(kernel='linear')
     model.fit(X_train, y_train)
     ```

3. **Evaluate the Model**:
   - Use metrics such as accuracy, precision, recall, and F1-score:
     ```python
     from sklearn.metrics import classification_report

     predictions = model.predict(X_test)
     print(classification_report(y_test, predictions))
     ```

## Example Workflow
1. Load raw textual data.
2. Apply TextBlob for sentiment labeling.
3. Preprocess and vectorize the text.
4. Train the SVM model.
5. Evaluate model performance.

## Dependencies
- Python 3.7+
- pandas
- numpy
- textblob
- scikit-learn
- nltk

## Project Structure
```
.
├── textblob_svm_pipeline.py  # Main script for labeling and classification
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── data/                     # Folder for datasets
└── examples/                 # Example scripts and notebooks
```

## Contributing
Contributions are welcome! If you have suggestions for improvements or new features, feel free to open an issue or submit a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
This project is inspired by the need for combining simple labeling techniques and robust machine learning models for effective sentiment analysis.
