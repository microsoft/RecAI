import re
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def _clean_string(string: str) -> str:
    """ Only keep alphanumerical characters """
    string = re.sub(r'[^A-Za-z0-9 ]+', '', string.lower())
    string = re.sub('\s+', ' ', string).strip()
    return string


class TFIDF_model:
    def _create_ngrams(self, string: str) -> List[str]:
        """ Create n_grams from a string

        Steps:
            * Extract character-level ngrams with `self.n_gram_range` (both ends inclusive)
            * Remove n-grams that have a whitespace in them
        """
        string = _clean_string(string)

        result = []
        for n in range(3, 4):
            ngrams = zip(*[string[i:] for i in range(n)])
            # ngrams = [''.join(ngram) for ngram in ngrams if ' ' not in ngram]
            ngrams = [''.join(ngram) for ngram in ngrams]
            result.extend(ngrams)

        return result
    
    def __init__(self, corpus):
        self.corpus = corpus
        self.vectorizer = TfidfVectorizer(min_df=1, analyzer=self._create_ngrams).fit(corpus)

    def match(self, from_docs, to_docs):
        froms = self.vectorizer.transform(from_docs)
        tos = self.vectorizer.transform(to_docs)
        cosine_sim = cosine_similarity(froms, tos)
        return cosine_sim

    def map_index(self, docs):
        score = self.match(docs, self.corpus)
        index_list = []
        for x in score:
            max_score = -999
            max_index = 0
            for index, y in enumerate(x):
                if y > max_score:
                    max_score = y
                    max_index = index
            index_list.append(max_index)
        return index_list