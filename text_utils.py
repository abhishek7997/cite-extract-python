# -*- coding: utf-8 -*-
"""
Created on Wed May 10 18:40:31 2023

@author: Abhi
"""

import re
import string
import csv
import unidecode

import nltk
import nltk.tag
import nltk.data
from nltk.sentiment import SentimentIntensityAnalyzer

from pdfminer.high_level import extract_text

from constants import CONSTANTS

from typing import List, Tuple


class TextUtils:
    '''Contains all methods related to operations on text'''
    @classmethod
    def unicode_to_ascii(cls, unicode_text: str) -> str:
        '''Replace all non-english characters from extracted text'''
        # new_text = unicode_text.replace("\\p{No}+", "")
        new_text = re.sub(r"[¼½¾⅐⅑⅒⅓⅔⅕⅖⅗⅘⅙⅚⅛⅜⅝⅞↉\x0c]+", ' ', unicode_text)
        converted_text = unidecode.unidecode(new_text).strip()
        return converted_text

    @classmethod
    def remove_unicode(cls, text: str) -> str:
        '''Remove all non-english characters from extracted text'''
        return ''.join([i if ord(i.lower()) < 128 else ' ' for i in text])

    @classmethod
    def restructure_text(cls, page_texts: List[str]) -> str:
        '''Properly format the raw text'''
        new_text = ''

        for line in page_texts:
            lines = line.strip().split('\n')
            new_line = ' '.join(list(lines)) + '\n'
            new_text += new_line + '\n'

        # newText = newText.replace('- ', '')
        new_text = new_text.replace('" ', '')
        new_text = re.sub(' +', ' ', new_text)
        new_text = re.sub(r'-\s+', '', new_text)
        return new_text

    @classmethod
    def get_paragraphs(cls, text: str) -> List[str]:
        '''Return list of paragraphs from raw text'''
        return text.split('\n\n')

    @classmethod
    def get_references(cls, text: str) -> str:
        '''Return just the text which contains all references'''
        pos = text.rfind("References")
        return text[pos::]

    @classmethod
    def get_references_list(cls, references_text: str) -> List[str]:
        '''Returns list of all references from the raw text containing all references'''
        return sorted(list(set(re.findall(CONSTANTS.REFERENCES_REGEX, references_text))))

    @classmethod
    def tokenize_text(cls, text):
        '''Tokenize the raw text using NLTK'''
        tokens = nltk.word_tokenize(text)
        return tokens

    @classmethod
    def get_nltk_text(cls, tokens):
        '''Returns text that can be operated upon by NLTK'''
        final_tokens = [token.lower() for token in tokens if (
            token.isalpha() and (token.lower() not in CONSTANTS.STOPWORDS))]
        # cleaned_tokens = [tok for tok in final_tokens if tok.lower() not in stopwords]
        parsed_text = nltk.Text(final_tokens)
        return parsed_text

    @classmethod
    def remove_punctuation(cls, text):
        '''Removes punctuations from a string'''
        return text.strip().lower().translate(str.maketrans('', '', string.punctuation))

    @classmethod
    def extract_paragraphs(cls, file):
        raw_text = extract_text(file)
        paragraphs = cls.merge_lines(raw_text)
        return paragraphs

    @classmethod
    def merge_lines(cls, raw_text):
        text = re.sub(r'-\n', '', raw_text)  # Merge hyphenated lines
        text = re.sub(r'\n\n+', '\n', text)  # Remove extra empty lines
        paragraphs = text.split('\n')
        return paragraphs


class AdjectiveUtils:
    '''Contains all methods related to operations on adjectives or list of adjectives'''
    @classmethod
    def get_adjectives(cls, tagged_results):
        '''Return list of adjectives from Part-Of-Speech tokens'''
        adjectives = [(re.sub(r'[^A-Za-z]', '', word), tag)
                      for (word, tag) in tagged_results if tag in
                          ('JJ', 'JJR', 'JJS') and len(re.sub(r'[^A-Za-z]', '', word)) > 2]
        return adjectives

    @classmethod
    def get_adjectives_words(cls, tagged_results):
        '''Return list of adjectives from Part-Of-Speech tokens (words only)'''
        return list({
            re.sub(r'[^A-Za-z]', '', word) for (word, token) in tagged_results
            if (token in ('JJ', 'JJS', 'JJR') and len(re.sub(r'[^A-Za-z]', '', word)) > 2)})

    @classmethod
    def get_positive_adjectives(cls, adjectives):
        '''Return list of adjectives that are in the list of positive words'''
        positive_adjectives = [
            adjective for adjective in adjectives if adjective in CONSTANTS.POSITIVE_WORDS]
        return len(positive_adjectives), positive_adjectives

    @classmethod
    def get_negative_adjectives(cls, adjectives):
        '''Return list of adjectives that are in the list of negative words'''
        negative_adjectives = [
            adjective for adjective in adjectives if adjective in CONSTANTS.NEGATIVE_WORDS]
        return len(negative_adjectives), negative_adjectives

    @classmethod
    def get_adjectives_from_citations_data(cls, citations_data: List[Tuple[str, str, List[str], Tuple[int, list], Tuple[int, list]]]) -> List[str]:
        '''Return list of adjectives from citations_data'''
        result = set()
        for (_, _, adjs, _, _) in citations_data:
            result.update(adjs)

        return sorted(list(result))

    @classmethod
    def get_polarity_of_adjectives(cls, adjectives):
        '''Return list of positive, negative and neutral adjectives
        based on their sentiment calculated by Sentiment Intensity Analyzer'''
        sia = SentimentIntensityAnalyzer()
        positive_words = []
        negative_words = []
        neutral_words = []

        for (adjective, _) in adjectives:
            if (sia.polarity_scores(adjective)['compound']) >= 0.5:
                positive_words.append(adjective)
            elif (sia.polarity_scores(adjective)['compound']) <= -0.5:
                negative_words.append(adjective)
            else:
                neutral_words.append(adjective)

        return (list(set(negative_words)), list(set(positive_words)), list(set(neutral_words)))

    @classmethod
    def write_adjectives_to_csv(cls, adjectives_list: List[str]):
        '''Save list of adjectives to a CSV file'''
        rows = zip(adjectives_list, [0]*len(adjectives_list))
        with open('adjectives.csv', 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
            writer.writerow(("adjective", "score"))
            for row in rows:
                writer.writerow(row)


class TaggingUtils:
    '''Contains all methods related to NLTK POS-tags'''
    @classmethod
    def pos_tagger(cls, parsed_text):
        '''Apply Part-Of-Speech tagger on words'''
        return nltk.pos_tag(parsed_text)

    @classmethod
    def create_unigram_tagger(cls, citations_list):
        '''Creates and returns a unigram tagger object'''
        #default_tagger = nltk.data.load(nltk.tag._POS_TAGGER)
        default_tagger = nltk.data.load(
            'taggers/maxent_treebank_pos_tagger/english.pickle')
        model = {}
        for cit in citations_list:
            model[cit] = 'NNP'
        tagger = nltk.tag.UnigramTagger(model=model, backoff=default_tagger)
        return tagger

    @classmethod
    def citations_tagger(cls, tagger, paragraphs):
        '''Adds taggs to citations present in text (paragraphs)'''
        return tagger.tag(paragraphs)
