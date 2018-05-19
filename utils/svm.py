from sklearn.linear_model import LogisticRegression
from sklearn import svm
from utils.plot_learning_curve import plot_learning_curve


import nltk
# nltk.download('wordnet')
# nltk.download('reuters')
# nltk.download('punkt')
nltk.download('cess_esp')
from nltk.corpus import cess_esp as cess
from nltk.corpus import reuters
from nltk.corpus import wordnet as wn
from collections import Counter
import pyphen

class SVM(object):

    def __init__(self, language):
        self.language = language
        # from 'Multilingual and Cross-Lingual Complex Word Identification' (Yimam et. al, 2017)
        if language == 'english':
            self.avg_word_length = 5.3
            # from Beker, Henry; Piper, Fred. Cipher Systems: The Protection of Communications.
            self.char_frequency = {'a': 8.167, 'b': 1.492,'c': 2.782,'d': 4.253,'e': 12.702,
                                   'f': 2.228, 'g': 2.015,'h': 6.094,'i': 6.966,'j': 0.153,
                                   'k': 0.772, 'l': 4.025,'m': 2.406,'n': 6.749,'o': 7.507,
                                   'p': 1.929, 'q': 0.095,'r': 5.987,'s': 6.327,'t': 9.056,
                                   'u': 2.758, 'v': 0.978,'w': 2.360,'x': 0.150,'y': 1.974,
                                   'z': 0.074}
            self.dic = pyphen.Pyphen(lang='en')
            self.reuters = reuters.words()
            self.unigram_counts = Counter(self.reuters)
            bigrams = []
            for sent in reuters.sents():
                bigrams.extend(nltk.bigrams(sent, pad_left=True, pad_right=True))
            self.bigram_counts = Counter(bigrams)
        else:  # spanish
            self.avg_word_length = 6.2
            # self.char_frequency = {'a': 12.525,'b': 2.215,'c': 4.139,'d': 5.860,'e': 13.681,
            #                        'f': 0.692,'g': 1.768,'h': 0.703,'i': 6.247,'j': 0.443,
            #                        'k': 0.011,'l': 4.967,'m': 3.157,'n': 6.71,'o': 8.683,
            #                        'p': 2.510, 'q': 0.877,'r': 6.871,'s': 7.977,'t': 4.632,
            #                        'u': 3.927, 'v': 1.138,'w': 0.017,'x': 0.215,'y': 1.008,
            #                        'z': 0.517,'á': 0.502, 'é': 0.433, 'í': 0.725, 'ñ': 0.311,
            #                        'ó': 0.827, 'ú': 0.168, 'ü': 0.012}
            # self.dic = pyphen.Pyphen(lang='es')
            self.cess = cess.words()
            self.unigram_counts = Counter(self.cess)
            bigrams = []
            for sent in cess.sents():
                bigrams.extend(nltk.bigrams(sent, pad_left=True, pad_right=True))
            self.bigram_counts = Counter(bigrams)
        # self.clf = svm.SVC()
        # self.model = LogisticRegression()
        self.model = svm.SVC(gamma=5)


    def extract_features(self, word, preword, afterword):
        len_chars = len(word) / self.avg_word_length
        len_tokens = len(word.split(' '))
        term_frequency = self.unigram_counts[word]
        bigram_pre = self.bigram_counts[(preword, word)]
        bigram_after = self.bigram_counts[(word, afterword)]
        return [len_chars, len_tokens, term_frequency, bigram_pre, bigram_after]

    # extract features for english words
    def extract_more_features(self, word):
        letterts = list(word)
        char_freguency = 0.0
        for char in letterts:
            char_freguency += 0 if char not in self.char_frequency.keys() else self.char_frequency[char]
        char_freguency = char_freguency / len(letterts)
        synonyms = len(wn.synsets(word))
        syllables = len(self.dic.inserted(word).split('-'))
        return [char_freguency, synonyms, syllables]

    def train(self, trainset):
        X = []
        y = []
        for sent in trainset:
            words = sent['sentence'].split(" ")
            preword = None
            afterword = None
            if len(sent['target_word'].split(' ')) == 0 and words.index(sent['target_word']) > 0:
                preword = words[words.index(sent['target_word']) - 1].lower()
            if len(sent['target_word'].split(' ')) == 0 and words.index(sent['target_word']) < len(words) - 1:
                afterword = words[words.index(sent['target_word']) + 1].lower()
            if self.language == 'english':
                X.append(self.extract_features(sent['target_word'].lower(), preword, afterword) +
                         self.extract_more_features(sent['target_word'].lower()))
            else:
                X.append(self.extract_features(sent['target_word'].lower(), preword, afterword))
            y.append(sent['gold_label'])
        self.model.fit(X, y)
        plot_learning_curve(self.model, "learning curve", X, y, cv=10)

    def test(self, testset):
        X = []
        for sent in testset:
            words = sent['sentence'].split(" ")
            preword = None
            afterword = None
            if len(sent['target_word'].split(' ')) == 0 and words.index(sent['target_word']) > 0:
                preword = words[words.index(sent['target_word']) - 1].lower()
            if len(sent['target_word'].split(' ')) == 0 and words.index(sent['target_word']) < len(words) - 1:
                afterword = words[words.index(sent['target_word']) + 1].lower()
            if self.language == 'english':
                X.append(self.extract_features(sent['target_word'].lower(), preword, afterword) +
                         self.extract_more_features(sent['target_word'].lower()))
            else:
                X.append(self.extract_features(sent['target_word'].lower(), preword, afterword))
        return self.model.predict(X)
