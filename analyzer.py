# -*- coding: utf-8 -*-
import multiprocessing
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from gensim.models.word2vec import LineSentence, Word2Vec
from gensim import corpora, models
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn import manifold
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


mpl.rcParams['font.sans-serif'] = ['simhei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 显示负号


class Analyzer(object):
    """
    cut_result:分词结果
    authors: 作者列表
    tfidf_word_vector: 用tf-idf为标准得到的词向量
    w2v_word_vector: 用word2vector得到的词向量
    w2v_model: 用word2vector得到的model
    lda_model：lda 主题分析结果
    frequent_itemsets：
    rules：
    tfidf_word_vector_tsne: 降维后的词向量
    w2v_word_vector_tsne: 降维后的词向量
    """

    def __init__(self, cut_result, saved_dir):
        self.cut_result = cut_result
        self.authors = list(cut_result.author_poetry_dict.keys())
        print('begin analyzing cut result...')
        self.cut_result = cut_result
        print("calculating poets' tf-idf word vector...")
        self.tfidf_word_vector = self._author_word_vector(cut_result.author_poetry_dict)
        print("calculating poets' w2v word vector...")
        self.w2v_model, self.w2v_word_vector = self._word2vec(cut_result.author_poetry_dict)
        print("Calculating poets' LDA topics...")
        self.lda_model, self.lda_topics = self._lda_topics(cut_result.author_poetry_dict)
        print("Analyzing frequent patterns and association rules...")
        self.frequent_itemsets, self.rules = self.find_frequent_patterns()
        print("use t-sne for dimensionality reduction...")
        self.tfidf_word_vector_tsne = self._tsne(self.tfidf_word_vector)
        self.w2v_word_vector_tsne = self._tsne(self.w2v_word_vector)
        print("result saved.")

    def find_frequent_patterns(self, min_support=0.1):
        """Identify frequent patterns and association rules in the poetry corpus."""
        texts = [doc.split() for doc in self.cut_result.author_poetry_dict.values()]
        te = TransactionEncoder()
        te_ary = te.fit_transform(texts)
        df = pd.DataFrame(te_ary, columns=te.columns_)

        # Applying Apriori to find frequent itemsets
        frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)

        # Generating association rules
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
        return frequent_itemsets, rules

    @staticmethod
    def _lda_topics(author_poetry_dict):
        """Generate LDA topics from the poetry corpus."""
        # Prepare texts
        texts = [doc.split() for doc in author_poetry_dict.values()]
        # Create a dictionary representation of the documents.
        dictionary = corpora.Dictionary(texts)
        # Convert dictionary to a bag of words corpus
        corpus = [dictionary.doc2bow(text) for text in texts]

        # Generate LDA model
        lda = models.LdaModel(corpus, num_topics=10, id2word=dictionary, passes=15)
        topics = lda.print_topics(num_words=4)
        return lda, topics

    @staticmethod
    def _author_word_vector(author_poetry_dict):
        """用tf-idf为标准解析每个作者的词向量"""
        poetry = list(author_poetry_dict.values())
        vectorizer = CountVectorizer(min_df=15)
        word_matrix = vectorizer.fit_transform(poetry).toarray()
        transformer = TfidfTransformer()
        tfidf_word_vector = transformer.fit_transform(word_matrix).toarray()
        return tfidf_word_vector

    @staticmethod
    def _word2vec(author_poetry_dict):
        """用word2vector解析每个作者的词向量"""
        dimension = 600
        authors = list(author_poetry_dict.keys())
        poetry = list(author_poetry_dict.values())
        with open("cut_poetry", 'w') as f:
            f.write("\n".join(poetry))
        model = Word2Vec(LineSentence("cut_poetry"), vector_size=dimension, min_count=15,
                         workers=multiprocessing.cpu_count())
        word_vector = []
        for i, author in enumerate(authors):
            vec = np.zeros(dimension)
            words = poetry[i].split()
            count = 0
            for word in words:
                word = word.strip()
                try:
                    vec += model.wv[word]
                    count += 1
                except KeyError:  # 有的词语不满足min_count则不会被记录在词表中
                    pass
            word_vector.append(np.array([v / count for v in vec]))
        os.remove("cut_poetry")
        return model, word_vector

    @staticmethod
    def _tsne(word_vector):
        word_vector = np.array(word_vector)
        t_sne = manifold.TSNE(n_components=2, init='pca', random_state=0)
        word_vector_tsne = t_sne.fit_transform(word_vector)
        return word_vector_tsne

    def find_similar_poet(self, poet_name, use_w2v=False):
        """
        通过词向量寻找最相似的诗人
        :param: poet: 需要寻找的诗人名称
        :return:最匹配的诗人
        """
        word_vector = self.tfidf_word_vector if not use_w2v else self.w2v_word_vector
        poet_index = self.authors.index(poet_name)
        x = word_vector[poet_index]
        min_angle = np.pi
        min_index = 0
        for i, author in enumerate(self.authors):
            if i == poet_index:
                continue
            y = word_vector[i]
            cos = x.dot(y) / (np.sqrt(x.dot(x)) * np.sqrt(y.dot(y)))
            angle = np.arccos(cos)
            if min_angle > angle:
                min_angle = angle
                min_index = i
        return self.authors[min_index]

    def find_similar_word(self, word):
        return self.w2v_model.wv.most_similar(word)


def plot_vectors(X, target):
    """绘制结果"""
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], target[i],
                 # color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 4}
                 )
    plt.show()
