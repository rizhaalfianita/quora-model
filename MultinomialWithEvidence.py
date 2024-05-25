import numpy as np

class MultinomialNB:
    def __init__(self, alpha=1):
        self.alpha = alpha
        self.class_counts = {}
        self.total_feature_counts = {}
        self.total_each_word_per_class = {}
        self.priors = {}
        self.likelihood = {}
        self.posteriors = []
        self.predictions = {}

    def fit(self, X_train_tfidf, y_train, idf):
        classes = np.unique(y_train) 
        total_questions = len(y_train)
        X_train_tfidf = np.array(X_train_tfidf)

        # PRIORS
        for label in classes:
            self.class_counts[label] = 0
            for sentence_label in y_train:
                if sentence_label == label:
                    self.class_counts[label] += 1
            self.priors[label] = self.class_counts[label] / total_questions

        # compute self.total_feature_counts
        for label in classes:
            for tfidf_val, tfidf_class in zip(X_train_tfidf, y_train):
                if tfidf_class == label:
                    self.total_feature_counts[label] = np.sum(tfidf_val)

        total_each_word_per_class = np.zeros((len(classes), X_train_tfidf.shape[1]))
        # compute sum each word each class
        for label in classes:
            tfidf_per_class = np.where(y_train == label)
            total_each_word_per_class[label] = np.sum(X_train_tfidf[tfidf_per_class], axis=0)
            self.total_each_word_per_class[label] = total_each_word_per_class

        for label in classes:
            self.likelihood[label] = (total_each_word_per_class[label] + 1) / (self.total_feature_counts[label] + sum(idf.values()))
    
    def predict(self, X_test_tfidf):
        predictions = []
        for question_tfidf in X_test_tfidf:
            posteriors = {}
            for label, prior in self.priors.items():
                posterior = prior
                for i, tfidf in enumerate(question_tfidf):
                    if tfidf != 0.0:
                        # Multiply non-zero TF-IDF value with likelihood
                        posterior *= self.likelihood[label][i]
                posteriors[label] = posterior

            evidence = sum(posteriors.values())
            
            for label in posteriors:
                posteriors[label] = round(posteriors[label]/evidence, 3)
            print(posteriors)
            self.posteriors.append(posteriors)

            predicted_label = max(posteriors, key=posteriors.get)
            predictions.append(predicted_label)
        return predictions
    
    def predict_per_sentence(self, x_test_tfidf):
        predictions = 0
        posteriors = {}
        for label, prior in self.priors.items():
            posterior = prior
            for i, tfidf in enumerate(x_test_tfidf):
                if tfidf != 0.0:
                    # Multiply non-zero TF-IDF value with likelihood
                    posterior *= self.likelihood[label][i]
                posteriors[label] = posterior

        evidence = sum(posteriors.values())
            
        for label in posteriors:
            posteriors[label] = round(posteriors[label]/evidence, 3)

        self.posteriors = posteriors
        predicted_label = max(posteriors, key=posteriors.get)
        predictions = predicted_label
            
        return predictions
    
    def score(self, X, y):
        # Make predictions
        y_pred = self.predict(X)
        
        # Calculate the number of correct predictions
        correct_predictions = 0
        for pred, true in zip(y_pred, y):
            if pred == true:
                correct_predictions += 1
        
        # Calculate the total number of predictions
        total_predictions = len(y)
        
        # Calculate accuracy
        accuracy = correct_predictions / total_predictions
        return accuracy