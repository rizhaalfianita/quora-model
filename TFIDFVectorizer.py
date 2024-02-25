import numpy as np 

class TFIDF:
    def __init__(self, dataset):
        self.dataset = dataset
        self.vocabulary = self.create_vocabulary()
        self.document_frequency = self.create_doc_freq()

    def create_vocabulary(self): # berisi list kata unik yang ada
        vocab = [] # pakai list karna ga pake key
        for sentence in self.dataset:
            for word in sentence.split():
                if word not in vocab:
                    vocab.append(word)
        return vocab

    def create_doc_freq(self): # menghitung freq tiap vocab di seluruh kalimat
        document_frequency = {} # pakai dictionary karna butuh key 'word' dan value 'freq'
        for term in self.vocabulary:
            document_frequency[term] = 0 
            for sentence in self.dataset:
                for word in sentence.split():
                    if term == word:
                        document_frequency[term] += 1
        return document_frequency
    
    def create_word_count(self, sentence): # menghitung freq tiap kata di satu kalimat
        word_count = {}
        for word in sentence.split():
            if word not in word_count:
                word_count[word] = 1
            else:
                word_count[word] += 1
        return word_count
    
    def create_tfidf_dict(self, sentence):
        sentence_length = len(sentence)
        dataset_length = len(self.dataset)
        tfidf_scores = {}

        for word in sentence.split():
            if word in self.vocabulary:
                tf = self.create_word_count(sentence)[word] / sentence_length
                idf = np.log(dataset_length / self.document_frequency[word])
                tfidf_scores[word] = tf * idf
        return tfidf_scores
    
    def create_tfidf_list(self, sentence):
        sentence_length = len(sentence) # jumlah kata di kalimat
        dataset_length = len(self.dataset) # jumlah kalimat di dataset
        tfidf_list = [0] * len(self.vocabulary)  # membuat list yang berisi 0 sebanyak vocabulary

        # for word in sentence.split():
        #     for i in range(len(self.vocabulary)):
        #         if word == self.vocabulary[i]: # buat nge cek kata itu ada ndak di kamus kata
        #             tf = self.create_word_count(sentence)[word] / sentence_length
        #             idf = np.log(dataset_length / self.document_frequency[word])
        #             tfidf = tf * idf
        #             tfidf_list[i] = tfidf
        # return tfidf_list 

        for word in sentence.split():
            for i, vocab in enumerate(self.vocabulary):
                if word == vocab: # buat nge cek kata itu ada ndak di kamus kata
                    tf = self.create_word_count(sentence)[word] / sentence_length
                    idf = np.log(dataset_length / self.document_frequency[word])
                    tfidf = tf * idf
                    tfidf_list[i] = tfidf
        return tfidf_list 
    
    def transform_tfidf(self, dataset):
        matrix_tfidf = []

        for sentence in dataset:
            matrix_tfidf.append(self.create_tfidf_list(sentence))
        return matrix_tfidf
    
    def transform_tfidf_with_feature(self, dataset):
        tfidf_with_feature = []

        for sentence in dataset:
            tfidf_with_feature.append(self.create_tfidf_dict(sentence))
        return tfidf_with_feature