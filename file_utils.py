# -*- coding: utf-8 -*-
"""
Created on Wed May 10 18:39:34 2023

@author: Abhi
"""

import re
from io import StringIO

import nltk

from pdfminer.high_level import extract_text
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTTextBox

from text_utils import TextUtils

from typing import List


class FileUtils:
    '''Contains all the methods related to file operations
    in the context of research papers and citations'''

    # originally ReadFile()
    @classmethod
    def get_words_from_file(cls, file_name: str):
        '''Read the file and return list of words'''
        with open(file_name, 'r') as file:
            return [word.strip() for word in file.readlines()]

    @classmethod
    def read_pdf(cls, file_name: str = None):
        '''Extract text from PDF File'''
        if not isinstance(file_name, str):
            raise TypeError("String must be provided")
        if not file_name:
            raise TypeError("Filename not provided!")
        text = extract_text(file_name)
        text = text.strip().lower()
        text = TextUtils.unicode_to_ascii(text)
        text = TextUtils.remove_unicode(text)
        return text.strip().split('\n\n')

    @classmethod
    def read_pdf_buffered(cls, file_name: str = None):
        '''Extract text from PDF File'''
        if not isinstance(file_name, str):
            raise TypeError("String must be provided")
        if not file_name:
            raise TypeError("Filename not provided!")

        output_string = StringIO()
        with open(file_name, 'rb') as in_file:
            parser = PDFParser(in_file)
            doc = PDFDocument(parser)
            rsrcmgr = PDFResourceManager()
            device = TextConverter(rsrcmgr, output_string, laparams=LAParams())
            interpreter = PDFPageInterpreter(rsrcmgr, device)
            for page in PDFPage.create_pages(doc):
                interpreter.process_page(page)

        return output_string.getvalue()

    @classmethod
    def read_page_text_incremented(cls, file_name: str = None):
        '''Extract text from PDF File page by page'''
        if not isinstance(file_name, str):
            raise TypeError("String must be provided")
        if not file_name:
            raise TypeError("Filename not provided!")

        texts = []
        for page_layout in extract_pages(file_name):
            for element in page_layout:
                if isinstance(element, LTTextContainer):
                    if isinstance(element, LTTextBox):
                        print(f"Textbox index: {element.index}")
                        print("Text content:")
                        print(element.get_text())
                        content = element.get_text().strip()
                        content = TextUtils.unicode_to_ascii(content)
                        content = TextUtils.remove_unicode(content)
                        texts.append(content)

        return texts

    @classmethod
    def save_paragraphs(cls, paragraphs: List[str], output_file: str):
        with open(output_file, 'w', encoding='utf-8') as f:
            for paragraph in paragraphs:
                f.write(paragraph + '\n')

    @classmethod
    def extract_paragraphs(cls, file_name: str) -> List[str]:
        paragraphs = []
        with open(file_name, 'rb') as file:
            resource_manager = PDFResourceManager()
            output_stream = StringIO()
            codec = 'utf-8'
            laparams = LAParams()
            converter = TextConverter(
                resource_manager, output_stream, codec=codec, laparams=laparams)
            interpreter = PDFPageInterpreter(resource_manager, converter)

            for page in PDFPage.get_pages(file, check_extractable=True):
                interpreter.process_page(page)
                extracted_text = output_stream.getvalue()

                # Sanitize the extracted text
                sanitized_text = TextUtils.unicode_to_ascii(extracted_text)
                sanitized_text = re.sub(
                    r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', ' ', extracted_text)
                sanitized_text = re.sub(r'[^\x20-\x7E\n]', ' ', sanitized_text)
                sanitized_text = re.sub(r'\s+', ' ', sanitized_text)

                # paragraphs.extend(sanitized_text.splitlines())
                for line in sanitized_text.splitlines():
                    sent = nltk.sent_tokenize(line)
                    paragraphs.extend(sent)
                output_stream.truncate(0)
                output_stream.seek(0)

            converter.close()
            output_stream.close()

        return paragraphs
