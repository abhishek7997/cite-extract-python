# -*- coding: utf-8 -*-
"""
Created on Wed May 10 19:25:15 2023

@author: Abhi
"""

import re

import nltk
import nltk.tag
import nltk.data
import chardet


class CONSTANTS:
    @staticmethod
    def __get_words_from_file(file_name):
        '''Read the file and return list of words'''
        with open(file_name, 'rb') as file:
            file_encoding = chardet.detect(file.read())['encoding']
        with open(file_name, 'r', encoding=file_encoding) as file:
            return [word.strip() for word in file.readlines()]

    PAPERS = {
        # 'text0': "./papers/gpt-4.pdf",
        'text1': "./papers/Document-level sentiment classification An empirical comparison between SVM and ANN.pdf",
        'text2': "./papers/Lexicon-Based Methods for Sentiment Analysis.pdf",
        'text3': "./papers/Sentiment analysis A combined approach.pdf",
        'text4': "./papers/Sentiment Analysis and Opinion Mining A survey.pdf",
        'text5': "./papers/Sentiment Strength Detection in Short Informal Text.pdf",
        'text6': "./papers/Scopus as data source.pdf",
    }

    POSITIVE_WORDS = __get_words_from_file(file_name="positive-words.txt")
    NEGATIVE_WORDS = __get_words_from_file(file_name="negative-words.txt")

    OUTPUTFILE = "results_abhishek.txt"
    POSITIVE_WORDS_FILE = "positive-words.txt"
    NEGATIVE_WORDS_FILE = "negative-words.txt"
    CITATIONS_REGEX_V1 = r"[\w]+[ ][&][ ][\w]+[,][ ][\d]+"
    CITATIONS_REGEX_V3 = r"\b(?!(?:Although|Also)\b)(?:[A-Za-z][A-Za-z'`-]+)(?:,? (?:(?:and |& )?(?:[A-Z][A-Za-z'`-]+)|(?:et al.?)))*(?:,? *(?:19|20)[0-9][0-9](?:, p\.? [0-9]+)?| *\((?:19|20)[0-9][0-9](?:, p\.? [0-9]+)?\))"
    CITATIONS_REGEX_V4 = r"^(?:[A-Z](?:(?!$)[A-Za-z\s&.,'’])+)\((?:\d{4})\)\.?\s*(?:[^()]+?[?.!])\s*(?:(?:(?:(?:(?!^[A-Z])[^.]+?)),\s*(?:\d+)[^,.]*(?=,\s*\d+|.\s*Ret))|(?:In\s*(?:[^()]+))\(Eds?\.\),\s*(?:[^().]+)|(?:[^():]+:[^().]+\.)|(?:Retrieved|Paper presented))"
    CITATIONS_REGEX_V5 = r"\b(?!(?:Although|Also)\b)(?:[A-Z][A-Za-z'`-]+)(?:,? (?:(?:and |& )?(?:[A-Z][A-Za-z'`-]+)|(?:et al.?)))*(?:(?:,? |,*)*(?:19|20)[0-9][0-9](?:, p\.? [0-9]+)?| *\((?:19|20)[0-9][0-9](?:, p\.? [0-9]+)?\))"

    CITATIONS_REGEX = re.compile("(%s|%s|%s|%s)" % (
        CITATIONS_REGEX_V1, CITATIONS_REGEX_V3, CITATIONS_REGEX_V4, CITATIONS_REGEX_V5))

    REFERENCES_REGEX_V1 = r"^(?:[A-Za-z](?:(?:(?!$)[A-Za-z\s&.,'’-]+)\(?[0-9]{4}(?:,? ?[A-Za-z]*)?\)?))."
    REFERENCES_REGEX_V2 = r"^(?:[A-Za-z](?:(?:(?!$)[A-Za-z\s&.,'’-]+)\(?[0-9]{4}(?:,? ?(?:January|February|March|April|May|June|July|August|September|October|November|December))?\)?))."
    REFERENCES_REGEX = re.compile("(%s|%s)" % (
        REFERENCES_REGEX_V1, REFERENCES_REGEX_V2), re.MULTILINE)

    YEARS_REGEX = r"[\d]{4}"

    STOPWORDS = nltk.corpus.stopwords.words("english")
    STOPWORDS.remove('not')
