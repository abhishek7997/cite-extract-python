# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 19:23:21 2022

@author: Abhi
"""

from constants import CONSTANTS
from file_utils import FileUtils
from text_utils import TextUtils, AdjectiveUtils
from citation_utils import CitationUtils

ALL_CITATIONS = []
ALL_ADJECTIVES = []

open(CONSTANTS.OUTPUTFILE, 'w').close()
open("default_sentiments.txt", 'w').close()

for currentText in CONSTANTS.PAPERS.values():
    print(f"Processing text pdf: {currentText}")
    # pages = readPageTextIncremented(currentText)
    pages = FileUtils.extract_paragraphs(currentText)
    restructuredText = TextUtils.restructure_text(pages)
    citations = CitationUtils.get_citations(restructuredText)
    # print(citations)
    #locations = CitationUtils.get_citations_locations(restructuredText)
    # print(locations)
    #paragraphs = TextUtils.get_paragraphs(restructuredText)
    #references = TextUtils.get_references(restructuredText)
    #references_list = TextUtils.get_references_list(references)
    #cited_text = CitationUtils.cite_text(restructuredText, citations)

    citations_dictionary = CitationUtils.get_citations_paragraph_location(
        restructuredText, citations)

    # ========== CITATIONS DATA ===========
    citations_data = CitationUtils.get_citations_data(citations_dictionary)
    ALL_CITATIONS.append(citations_data)

    # ========== ADJECTIVES DATA ===========
    ALL_ADJECTIVES.extend(
        AdjectiveUtils.get_adjectives_from_citations_data(citations_data))
    ALL_ADJECTIVES = sorted(list(set(ALL_ADJECTIVES)))

    AdjectiveUtils.write_adjectives_to_csv(ALL_ADJECTIVES)

    # print("============ SENTIMENTS (NLTK SIA) =============")
    sentiments = CitationUtils.get_sentiments(citations_dictionary)
    # printSentiments(sentiments)
    CitationUtils.write_sentiments_to_file(sentiments)

    # print("============ WRITE TO FILE =============")
    CitationUtils.write_citations_data(citations_data, currentText)

CitationUtils.write_citations_to_csv(ALL_CITATIONS)
