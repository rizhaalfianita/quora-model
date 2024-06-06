import sys
sys.path.append("../../..")
from sklearn.model_selection import train_test_split
from MultinomialWithEvidence import MultinomialNB
from TFIDFVectorizer import TFIDF

class KF:
    def __init__(self, X, y, split_size):
        self.X = X
        self.y = y 
        self.split_size = split_size
        self.scores = []

    def split(self):
        for i in range(28, 38):
            X_train_cv, X_val_cv, y_train_cv, y_val_cv = train_test_split(self.X, self.y, test_size=self.split_size, random_state=i)
            tfidf_vectorizer = TFIDF(X_train_cv)
            X_train_cv_tfidf = tfidf_vectorizer.transform_tfidf(X_train_cv)
            X_val_cv_tfidf = tfidf_vectorizer.transform_tfidf(X_val_cv)

            mnb = MultinomialNB()
            mnb.fit(X_train_cv_tfidf, y_train_cv, tfidf_vectorizer.idf)

            score = mnb.score(X_val_cv_tfidf, y_val_cv)
            self.scores.append(round(score, 3))
            print(self.scores)

