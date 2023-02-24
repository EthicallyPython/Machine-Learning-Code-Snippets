# pipeline implementation for ML. Uses up less code because it can perform multiple steps before doing actual ML
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction import CountVectorizer

## variables
X = df['numeric_column'] # can use more than one column
y = df['column_to_predict']

## split data into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# pipeline setup
pipe = Pipeline([
        ('bow', CountVectorizer(analyzer='word')),
        ('tfidf', TfidfTransformer()),
        ('model', MultinomialNB())
    ])

pipe.fit(X_train, y_train)
predictions = pipe.predict(X_test)

# metrics to measure performance
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
