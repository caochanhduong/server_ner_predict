# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
import re
import unicodedata
import string
import tldextract
from functools import partial
def url_replace(match):
    return tldextract.extract(match.group(0)).domain

class PreprocessClass(object):
    '''Regex class - help convert regex pattern'''

    '''Init Regex pattern'''
    PATTERN_PHONENB = re.compile(
        r'(\(\+?0?84\))?(09|012|016|018|019)((\d(\s|\.|\,)*){8})')
    SUB_PHONENB = r' PHONEPATT '

    PATTERN_URL = re.compile(
        r'(((ftp|https?)\:\/\/)|(www\.))?[\d\w\.\-\_]+\.[\w]{2,6}(:[\d\w]+)?[\#\d\w\-\.\_\?\,\'\/\\\+\;\%\=\~\$\&]*(.html?)?')
    SUB_URL = r' URLPATT '

    PATTERN_EMAIL = re.compile(
        r'(^|\W)([^@\s]+@[a-zA-Z0-9\-][a-zA-Z0-9\-\.]{0,254})(\W|$)')
    SUB_EMAIL = r' EMAILPATT '

    PATTERN_NUMBER = re.compile(r'((\d+(\s|\.|\,|-){,2}\d*){3,})')
    SUB_NUMBER = r' NUMPATT '

    PATTERN_HTMLTAG = re.compile(r'<[^>]*>')

    PATTERN_PUNCTATION = re.compile(r'([%s]+)' % re.escape(string.punctuation))

    PATTERN_LINEBRK = re.compile(r'\t|\v|\f|(\s){2,}|\r\n|\r|\n')

    PATTERN_NOT_PUNC_WSPACE_ALNUM = re.compile(r'[^%s\w\d]' % re.escape(
                        string.punctuation + string.whitespace), re.UNICODE)

    @staticmethod
    def convert(mention, tokenize=True, keep_only_readable=False, 
                max_word_len=None, normalize=False, keep_url_host=True):
        if not isinstance(mention, str):
            mention = str(mention)
        if normalize:
            mention =  unicodedata.normalize('NFC', mention)

        '''replace pattern in mention by regex definex in onepattdict'''

        # Remove tag
        mention = PreprocessClass.PATTERN_HTMLTAG.sub(r'', mention)
        # Lower_case
        mention =  mention.lower()
        # Replace phone number
        mention = PreprocessClass.PATTERN_PHONENB.sub(PreprocessClass.SUB_PHONENB, mention)
        # Replace url
        if keep_url_host:
            mention = PreprocessClass.PATTERN_URL.sub(partial(url_replace), mention)
        else:
            mention = PreprocessClass.PATTERN_URL.sub(PreprocessClass.SUB_URL, mention)
        # Replace email
        mention = PreprocessClass.PATTERN_EMAIL.sub(PreprocessClass.SUB_EMAIL, mention)
        # Replace all remained number
        mention = PreprocessClass.PATTERN_NUMBER.sub(PreprocessClass.SUB_NUMBER, mention)

        if tokenize:
            mention = PreprocessClass.tokenize(mention)

        # Remove line break
        mention = PreprocessClass.PATTERN_LINEBRK.sub(r' ', mention)

        if keep_only_readable:
            mention = PreprocessClass.keep_punc_wspace_alnum_chars(mention)

        if max_word_len:
            mention = PreprocessClass.keep_max_word_len(mention, max_word_len)

        return mention

    @staticmethod
    def tokenize(mention, return_tokens=False, no_tokenize_chars=''):

        mention = PreprocessClass.PATTERN_PUNCTATION.sub(r' <punct> ', mention)

        return mention

    @staticmethod
    def keep_punc_wspace_alnum_chars(mention):
        mention = PreprocessClass.PATTERN_NOT_PUNC_WSPACE_ALNUM.sub('', mention)
        return mention

    @staticmethod
    def keep_max_word_len(mention, max_word_len):
        words = mention.split()
        mention = ' '.join(word for word in words if len(word) <= max_word_len)
        return mention


def build_dict_items(predict_items):
    '''Build dict items with keys is id and value is item from predict_items'''
    return_dict = {}
    for item in predict_items:
        return_dict[item.id] = item
    return return_dict

def preprocess_predict_items(dict_items):
    preprocess_obj = PreprocessClass()
    for item in dict_items:
        item.content = preprocess_obj.convert(item.content)
    return dict_items
