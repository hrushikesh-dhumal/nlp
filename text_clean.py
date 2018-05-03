# coding=utf-8
"""
Author:: Hrushikesh Dhumal <hrushikesh.dhumal@gmail.com>

Sample usage:
from text_clean import TextCleaner

text="This is a TeXt cleaning sample created in python 2´7¡ \
        It uses NLTK at backend to tokenize words and strings them together using space!\nYou can use it to remove \
        n-grams such as this ** good and bad**."

tc_basic = TextCleaner(stopwords)
print(tc_basic.clean(text))
>> This is TeXt cleaning sample created in python 2´7¡ It uses NLTK at backend to tokenize words and strings them together using space ! You can use it to remove n-grams such as this ** good and bad** .

To use more feature:

from text_clean import endcode_text
from text_clean import RE_NUMERIC
from text_clean import RE_SPECIAL_CHAR
from text_clean import RE_MULTIPLE_SPACE
from text_clean import RE_CONSECUTIVE_SENTENCE_END

stopwords = ['the', 'a', 'good and bad']
before_filters=[endcode_text,
                lambda text: RE_SPECIAL_CHAR.sub(u'', text),
                lambda text: RE_NUMERIC.sub(u"", text),
                lambda text: RE_MULTIPLE_SPACE.sub(u' ', text)]
after_filters=[lambda text: RE_MULTIPLE_SPACE.sub(u' ', text),
               lambda text: RE_CONSECUTIVE_SENTENCE_END.sub(u'', text)]
tc = TextCleaner(stopwords, before_filters, after_filters)
tc.clean(text)
>> u'This is TeXt cleaning sample created in python It uses NLTK at backend to tokenize words and strings them together using space ! You can use it to remove n-grams such as this .'
"""

import pandas as pd
from collections import defaultdict
import nltk
import re
from nltk import word_tokenize
nltk.download('punkt')


# Support functions
def endcode_text(text, encoding="ascii", errors='ignore'):
    """
    Encodes text to given encoding, if the text is a number, converts the text to number.

    :param text: string or number to be encoded
    :param str encoding: encoding format, by default it is utf-8
    :param str errors: method to handle errors. Errors are ignored by default
    :return: encoded text
    """
    op = str('').decode(encoding)
    try:
        op = text.decode(encoding, errors=errors)
    except AttributeError:
        op = str(text).decode(encoding, errors=errors)

    return op


RE_NUMERIC = re.compile(r"[0-9]+", re.UNICODE)
# everything except !?.; which acts as sentence delimiters
RE_SPECIAL_CHAR = re.compile(r"[@#&$*(\[]+\ *")
# remove apostrophe
RE_APOSTROPHE = re.compile(r"('\w)")
# remove consecutive dots, useful when words are removed and there are empty sentences
RE_CONSECUTIVE_SENTENCE_END = re.compile(r'(\.|\!|\?\s*){2,}')
# remove multiple spaces
RE_MULTIPLE_SPACE = re.compile(r'\s{2,}')


def strip_re_string(s, re, substitute_string=''):
    """
    Remove re complied pattern from a string

    :param str s: Unicode string to remove pattern
    :param str re: Re compiled pattern
    :param str substitute_string: Default to empty string
    :return: String with the pattern striped
    :rtype: str

    """
    return re.sub(substitute_string, s)


class TextCleaner(object):
    """
    Generic class to remove stop word preserving. Takes input as-
    1. Stopwords
    2. The list of functions that need to be applied to text before removing stop words
    3. The list of functions that need to be applied after removing stop words.

    Use the clean method to clean the text.
    """

    def _map_stop_words(self):
        """Divide stop words in list n-grams"""
        self.STOP_WORDS = {}
        for tokens in self.STOP_WORDS_LIST:
            len_tokens = len(word_tokenize(tokens))
            if len_tokens not in self.STOP_WORDS.keys():
                self.STOP_WORDS[len_tokens] = [tokens.lower()]
            else:
                self.STOP_WORDS[len_tokens].append(tokens.lower())

        #             self.n_grams = max(self.STOP_WORDS.keys())
        try:
            self.n_grams = max(self.STOP_WORDS, key=int)
        except ValueError:
            self.n_grams = 0

    def _create_hash_map(self):
        """Convert the n-grams lists to hashed maps for faster execution"""
        self._map_stop_words()
        self.STOP_WORDS_HASHED = defaultdict(set)
        for key in self.STOP_WORDS.keys():
            self.STOP_WORDS_HASHED[key] = set(self.STOP_WORDS[key])

    def is_a_stop_word(self, token, stop_words, cutoff=0.9, algo=None):
        """Check if a word is in hashed map stopwords"""
        token = token.lower()

        if algo == 'gestalt':
            # gestalt pattern matching- to find out similar words
            matches = difflib.get_close_matches(token.lower(), stop_words, 1, cutoff=cutoff)
            if len(matches) > 0:
                return True
            else:
                return False
        else:
            return token in stop_words

    def remove_stop_words(self, text):
        """Checks if a word or combination of word is in the stop words list and removes them"""
        token_list = word_tokenize(text)
        indices_to_remove = []

        for i, token in enumerate(token_list):
            for j in range(0, self.n_grams + 1):
                if self.is_a_stop_word(' '.join(token_list[i:i + j]), self.STOP_WORDS_HASHED[j]):
                    for k in range(i, i + j):
                        indices_to_remove.append(k)

        token_list = [i for j, i in enumerate(token_list) if j not in indices_to_remove]
        return ' '.join(token_list)

    def _apply_filters(self, io, pipeline, store_intermediate=False):
        """Execute the chained functions.
        Feeds the input starting from left and takes output from each function and feeds it to the next.
        Returns the output from last. Has the capability to return output from all intermediate
        functions."""

        if store_intermediate:
            intermediate_results = []
            for function in pipeline:
                io = function(io)
                intermediate_results.append(io)
            return intermediate_results
        else:
            for function in pipeline:
                io = function(io)
            return io

    def clean(self, text):
        """Function to apply filters before removing stop words, then remoev stop words and apply filters."""
        if text is None:
            return ''

        if self.before_filters is not None:
            text = self._apply_filters(text, self.before_filters, self.store_before)
        text = self.remove_stop_words(text)
        if self.after_filters is not None:
            text = self._apply_filters(text, self.after_filters, self.store_after)
        return text

    def __init__(self, stop_word_list, before_filters=None, after_filters=None, store_before=False, store_after=False):

        self.STOP_WORDS_LIST = stop_word_list
        self.before_filters = before_filters
        self.after_filters = after_filters
        self.store_before = store_before
        self.store_after = store_after
        self._create_hash_map()

