### CiteExtract

A python tool to extract and parse text from PDF files, specifically Research Paper PDFs. The program identifies citations along with the paragraphs they belong to. Additionally, it collects adjectives from the text and categorizes them as positive or negative using a predefined word list.

### Features

- Extract sentences from pdf
- Get citations in text
- Get adjectives present in the text and categorize it as positive or negative based on a pre-built wordlist
- Uses pdfminer.six library
- Removed special characters, formatting tags, unnecessary whitespace, and other artifacts that may have been introduced during PDF parsing.
- Extract data in the form of CSV (See image) and TXT file
- Enhanced the usability and efficiency of the data by making it easier to process and analyze.
 
![image](https://github.com/abhishek7997/cite-extract-python/assets/68701271/db0ea3bd-9bc4-4f1b-bf0c-767d00399c18)

### Usage

- Store pdf files path inside "papers" folder relative to the directory of the program
- Write the key value pair in PAPERS dictionary variable inside CONSTANTS.py file in the following format:
> "textN": "./papers/research-paper-name.pdf"
- Run *citations_text_extractor.py* file

### Libraries

- nltk
- pdfminer.six