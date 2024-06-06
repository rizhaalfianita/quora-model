import numpy as np

class TFIDF:
    def __init__(self, X):
        self.dataset = X
        self.vocabulary = self.create_vocabulary()
        self.document_freq = self.count_document_freq()
        self.idf = self.count_idf()

    def create_vocabulary(self):
        vocabulary = []
        for question in self.dataset:
            for term in question.split():
                if term not in vocabulary:
                    vocabulary.append(term)
        return vocabulary
    
    def create_term_count(self, question): # hasilnya vocab dan freq dalam 1 kalimat
        term_counts = {}
        for term in question.split():
            if term not in term_counts:
                term_counts[term] = 1
            else:
                term_counts[term] += 1
        return term_counts
    
    def count_tf(self, question): # hasilnya per kalimat
        question_len = len(question.split())
        tf = self.create_term_count(question)

        for vocab in tf:
            tf[vocab] /= question_len
        return tf
    
    def count_document_freq(self): # hasilnya sesuai panjang vocab
        document_frequency = {}
        for vocab in self.vocabulary:
            document_frequency[vocab] = 0
            for question in self.dataset:
                for term in question.split():
                    if vocab == term:
                        document_frequency[vocab] += 1
                        break
        return document_frequency
    
    def count_idf(self): # hasilnya sesuai panjang vocab
        idf = self.count_document_freq()
        total_dataset = len(self.dataset)

        for val in idf:
            idf[val] = np.log(total_dataset/idf[val])
        return idf
    
    def transform_tfidf(self, dataset):
        tf_matrix = []
        
        for question in dataset:
            tfidf_list = [0] * len(self.vocabulary)
            for term in question.split():
                for i, vocab in enumerate(self.vocabulary):
                    if term == vocab:
                        tfidf_list[i] = self.count_tf(question)[term] * self.idf[vocab]
            tf_matrix.append(tfidf_list)
        return tf_matrix
    
    def transform_single_tfidf(self, question):
        tfidf_list = [0] * len(self.vocabulary)
        for term in question.split():
            for i, vocab in enumerate(self.vocabulary):
                if term == vocab:
                    tfidf_list[i] = self.count_tf(question)[term] * self.idf[vocab]
        return tfidf_list

if __name__ == "__main__":
    questions = ['xi jinping took year kind dirty method kill enemy make dictator china', 'could israeli people ignore kill starve mile away', 'do create mobile app without write code', 'delete quora account use email address create another one']
    tfidf = TFIDF(questions)
    print(tfidf.transform_tfidf(['xi jinping took year kind dirty method kill enemy make dictator china', 'could israeli people ignore kill starve mile away', 'do create mobile app without write code', 'delete quora account use email address create another one']))
    print(tfidf.create_term_count())
    # print(tfidf.idf)
    #'could israeli people ignore kill starve mile away'
    # idf = tfidf.document_freq
    # for val in idf:
    #     idf[val] = np.log10(len(questions)/idf[val])
    #     print(idf[val])


