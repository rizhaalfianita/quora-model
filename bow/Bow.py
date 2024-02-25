import numpy as np 

class BOW:
    def __init__(self, dataset):
        self.dataset = dataset
        self.vocabulary = self.create_vocabulary()

    def create_vocabulary(self): # berisi list kata unik yang ada
        vocab = [] # pakai list karna ga pake key
        for sentence in self.dataset:
            for word in sentence.split():
                if word not in vocab:
                    vocab.append(word)
        return vocab
    
    def create_word_count(self, sentence): # menghitung freq tiap kata di satu kalimat
        word_count = {}
        for word in sentence.split():
            if word not in word_count:
                word_count[word] = 1
            else:
                word_count[word] += 1
        return word_count
    
    def create_bow_list(self, sentence):
        bow_list = [0] * len(self.vocabulary)  # membuat list yang berisi 0 sebanyak vocabulary

        for word in sentence.split():
            for i in range(len(self.vocabulary)):
                if word == self.vocabulary[i]: # buat nge cek kata itu ada ndak di kamus kata
                    freq = self.create_word_count(sentence)[word] 
                    bow_list[i] = freq
        return bow_list 
    
    def transform(self, dataset):
        bow = []

        for sentence in dataset:
            bow.append(self.create_bow_list(sentence))
        return bow