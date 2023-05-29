# -*- coding: utf-8 -*-
"""
Created on Wed May 10 19:54:39 2023

@author: Abhi
"""

import re
import csv

import nltk
import nltk.tag
import nltk.data

from constants import CONSTANTS
from text_utils import TextUtils, AdjectiveUtils, TaggingUtils
from io import StringIO

from pdfminer.layout import LAParams, LTTextBox, LTTextLine, LTImage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.pdfpage import PDFPage

from typing import List, Tuple, Dict

from nltk.sentiment import SentimentIntensityAnalyzer


class CitationUtils:
    '''Contains methods related to operations on citations'''
    @classmethod
    def parenthesize_citations_years(cls, text: str):
        '''Place open and close brackets around year'''
        years = set(re.findall(CONSTANTS.YEARS_REGEX, text))

        for year in years:
            text = text.replace(year, '(' + year + ')')

        return text

    @classmethod
    def get_citations(cls, text: str) -> List[str]:
        '''Extract all citations from text (Regex)'''
        return sorted(list(set(re.findall(CONSTANTS.CITATIONS_REGEX, text))))

    @classmethod
    def get_citations_nltk(cls, text: str):
        '''Extract all citations from text (NLTK)'''
        return sorted(list(set(nltk.regexp_tokenize(text, CONSTANTS.CITATIONS_REGEX))))

    @classmethod
    def get_citations_paragraph_location(cls, text: str, citations_list: List[str]) -> List[Dict[str, str]]:
        '''Return list of citations and their enclosing paragraph'''
        result = []
        reference_pos = text.rfind("References")
        search_text = text[0:reference_pos]
        citation_paragraphs = TextUtils.get_paragraphs(search_text)
        for citation in citations_list:
            matches = ' '.join(
                [re.sub(' +', ' ', para) for para in citation_paragraphs
                 if citation in para and len(para) > 10]).strip()
            matches = matches.replace('- ', '')
            if len(matches) > 10:
                result.append({citation: matches})
        return result

    @classmethod
    def get_citations_locations(cls, text: str) -> List[Tuple[str, Tuple[int, int]]]:
        '''Extract the starting and ending indices of all citations'''
        citation_locations = []
        for citation in CONSTANTS.CITATIONS_REGEX.finditer(text):
            citation_locations.append((citation.group(), citation.span()))

        return citation_locations

    @classmethod
    def get_references(cls, text: str):
        '''Return just the text which contains all references'''
        pos = text.rfind("References")
        return text[pos::]

    @classmethod
    def get_references_list(cls, references_text):
        '''Returns list of all references from the raw text containing all references'''
        return sorted(list(set(re.findall(CONSTANTS.REFERENCES_REGEX, references_text))))

    @classmethod
    def cite_text(cls, text: str, citations_list: List[str]) -> str:
        '''Replace all citation text with a special token string <CIT${citation_index}$>'''
        for idx, citation in enumerate(citations_list):
            text = text.replace(citation, f"<CIT${idx}$>")
        return text

    @classmethod
    def get_citations_data(cls, dictionary: List[Dict[str, str]]) -> List[Tuple[str, str, List[str], Tuple[int, list], Tuple[int, list]]]:
        '''Return list of citations, its pargraph and adjectives'''
        results = []
        # adjectives = getAdjectives(blob.tags)
        for item in dictionary:
            for (citation, paragraph) in item.items():
                if len(paragraph) > 30:
                    # parsed_text = getNLTKText(paragraph)
                    tagged_result = TaggingUtils.pos_tagger(
                        paragraph.lower().split())
                    adjectives = AdjectiveUtils.get_adjectives_words(
                        tagged_result)
                    positive_adjectives = AdjectiveUtils.get_positive_adjectives(
                        adjectives)
                    negative_adjectives = AdjectiveUtils.get_negative_adjectives(
                        adjectives)
                    results.append((citation, paragraph, adjectives,
                                   positive_adjectives, negative_adjectives))
        return results

    @classmethod
    def write_citations_data(cls, sentiments: List[Tuple[str, str, List[str], Tuple[int, list], Tuple[int, list]]], file_name: str):
        '''Write all results to file'''
        # print("\tCitation\t\t\t\tParagraph\t\t\tPolarity\t\tSentiment")

        with open(f"{CONSTANTS.OUTPUTFILE}", "a", encoding='utf-8') as file:
            file.write("=================================\n")
            file.write(f"Research Paper: {file_name}\n")
            file.write("=================================\n")

            pos_count = 0
            neg_count = 0
            neu_count = 0

            for (citation, paragraph, adjectives,
                 (pos_adj, positive_adjectives), (neg_adj, negative_adjectives)) in sentiments:
               # print(f"{citation[:15]+'...'}\t\t| {paragraph[:15]+'...'} \
                #   \t\t| {scores.polarity:.3f} | \t\t {sentiment}")

                file.write("Citation: " + citation + "\n\n")
                file.write("Paragraph: " + paragraph + "\n\n")
                # file.write("Scores: " + f"{scores.polarity:.4f}" + "\n\n")
                file.write(f"Adjectives : {adjectives}" + '\n\n')
                file.write(
                    f"Positive Adjectives : {positive_adjectives}" + '\n\n')
                file.write(
                    f"Negative Adjectives : {negative_adjectives}" + '\n\n')

                neu_count = len(adjectives) - (pos_adj + neg_adj)
                pos_count += pos_adj
                neg_count += neg_adj

                # file.write("=================================\n")

            file.write("Positive Adjectives : " + str(pos_count) + '\n')
            file.write("Negative Adjectives : " + str(neg_count) + '\n')
            file.write("Neutral Adjectievs : " + str(neu_count) + '\n')
            file.write("===========================================\n\n")

    @classmethod
    def write_citations_to_csv(cls, citations):
        '''Write citation data to CSV file'''
        with open("citations-data.csv", "w", newline="", encoding='utf-8') as out:
            csv_out = csv.writer(out)
            csv_out.writerow(['citation', 'citation_text',
                             'adj', 'pos_adj', 'neg_adj'])
            for paper in citations:
                for (citation, citation_text, adjs, pos_adj, neg_adj) in paper:
                    #csv_out.writerow((citation, citation_text))
                    csv_out.writerow(
                        (citation, citation_text, adjs, pos_adj, neg_adj))

    @classmethod
    def get_sentiments(cls, dictionary: List[Dict[str, str]]) -> List[Tuple[str, str, Dict[str, float]]]:
        '''Return the sentiment of citations from their paragraphs using inbuilt nltk Sentiment Intensity Analyzer'''
        results = []
        sia = SentimentIntensityAnalyzer()
        for item in dictionary:
            for (citation, paragraph) in item.items():
                if len(paragraph) > 30:
                    sentiment = sia.polarity_scores(paragraph)
                    results.append((citation, paragraph, sentiment))
        return results

    @classmethod
    def print_sentiments(cls, sentiments):
        '''Print the citation, its paragraph and its sentiment values'''
        print("\tCitation\t\t\t\tParagraph\t\t\t\tNeg\t  Neu\t  Pos\t\t Sentiment")
        for (citation, paragraph, scores) in sentiments:
            if len(paragraph) > 10:
                sentiment = "Neutral"

                if scores['compound'] >= 0.5:
                    sentiment = "Positive"
                elif scores['compound'] <= -0.5:
                    sentiment = "Negative"

                print(
                    f"{citation[:15]+'...'}\t\t|{paragraph[:15]+'...'}\t\t| {scores['neg']:.2f} || {scores['neu']:.2f} || {scores['pos']:.2f}\t\t{sentiment}")

    @classmethod
    def write_sentiments_to_file(cls, sentiments: List[Tuple[str, str, Dict[str, float]]]):
        '''Print the citation, its paragraph and its sentiment values'''
        with open("default_sentiments.txt", "a") as file:
            for (citation, paragraph, scores) in sentiments:
                if len(paragraph) > 30:
                    sentiment = "Neutral"

                    if scores['compound'] >= 0.5:
                        sentiment = "Positive"
                    elif scores['compound'] <= -0.1:
                        sentiment = "Negative"

                    file.write("=================================\n\n")
                    file.write(f"Citation: {citation}\n\n")
                    file.write(f"Paragraph: {paragraph}\n\n")
                    file.write(f"Negative: {scores['neg']:.3f}\n\n")
                    file.write(f"Neutral: {scores['neu']:.3f}\n\n")
                    file.write(f"Positive: {scores['pos']:.3f}\n\n")
                    file.write(f"Overall sentiment: {sentiment}\n\n")
                    file.write("=================================\n\n")


class CitationUtilsExperimental:
    @classmethod
    def extract_text_from_pdf(cls, pdf_path):
        resource_manager = PDFResourceManager()
        text_stream = StringIO()
        laparams = LAParams()
        device = TextConverter(
            resource_manager, text_stream, laparams=laparams)
        interpreter = PDFPageInterpreter(resource_manager, device)

        with open(pdf_path, 'rb') as file:
            for page in PDFPage.create_pages(file, check_extractable=True):
                interpreter.process_page(page)

        text = text_stream.getvalue()
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        device.close()
        text_stream.close()

        return text

    @classmethod
    def extract_citations(cls, text):
        citations = re.findall(
            r"\b(?!(?:Although|Also)\b)(?:[A-Z][A-Za-z'`-]+)(?:,? (?:(?:and |& )?(?:[A-Z][A-Za-z'`-]+)|(?:et al.?)))*(?:(?:,? |,*)*(?:19|20)[0-9][0-9](?:, p\.? [0-9]+)?| *\((?:19|20)[0-9][0-9](?:, p\.? [0-9]+)?\))", text)
        # Extract the paragraphs containing the citations
        paragraphs = re.split(r'\n\s*\n', text)
        citation_paragraphs = []
        for paragraph in paragraphs:
            if any(citation in paragraph for citation in citations):
                citation_paragraphs.append(paragraph)

        return citations, citation_paragraphs

    @classmethod
    def write_to_file(cls, citations, citation_paragraphs, filename):
        with open(filename, 'w', encoding='utf-8') as file:
            file.write("Citations:\n")
            for citation in citations:
                file.write(citation + '\n')

            file.write("\n\nCitation Paragraphs:\n")
            for paragraph in citation_paragraphs:
                file.write(paragraph + '\n')

    @classmethod
    def remove_non_text_elements(text):
        lines = text.split('\n')
        text_lines = []
        for line in lines:
            if isinstance(line, LTImage):
                continue
            if isinstance(line, (LTTextBox, LTTextLine)):
                text_lines.append(line.get_text())
        return '\n'.join(text_lines)

    @classmethod
    def remove_footer_and_images(text):
        footer_index = text.find('References')
        if footer_index != -1:
            text = text[:footer_index]
        text = re.sub(r'\n\s*\n', '\n', text)
        return text.strip()
