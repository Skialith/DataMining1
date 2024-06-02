import os

from analyzer import Analyzer, plot_vectors
from preprocessor import CutResult, cut_poetry
def example():
    saved_dir = os.path.join(os.curdir, "out")
    result = cut_poetry("全唐诗.txt", saved_dir)
    analyzer = Analyzer(result, saved_dir)

    print("LDA Topics:")
    for topic in analyzer.lda_topics:
        print(topic)

if __name__ == '__main__':
    example()