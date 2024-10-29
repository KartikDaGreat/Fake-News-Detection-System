import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
import spacy
from datetime import datetime

nlp = spacy.load("en_core_web_md")

global global_list
global_list = [ 
    [
        ['keyword1', 'keyword2', 'keyword3'],
        [
            ['January 1, 2022', 'Text 1'],
            ['January 3, 2022', 'Text 2'],
            ['December 18, 2021', 'Text 3']
        ]
    ],
    [
        ['keyword4', 'keyword5', 'keyword6'],
        [
            ['January 10, 2022', 'Text 4'],
            ['January 30, 2022', 'Text 5'],
            ['April 12, 2021', 'Text 6']
        ]
    ]
]

def extract_keywords(text, num_keywords=5):
    # Tokenize the text
    tokenizer = RegexpTokenizer(r'\b[a-zA-Z]{3,}\b')  # Keep only words with 3 or more characters
    tokens = tokenizer.tokenize(text)
    # Remove stopwords and lemmatize the tokens
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    filtered_tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens if word.lower() not in stop_words]
    # Calculate word frequency
    freq_dist = FreqDist(filtered_tokens)
    # Get the most frequent non-stopword lemmatized words
    top_keywords = freq_dist.most_common(num_keywords)
    return [word for word, _ in top_keywords]

def most_similar_chain(input_keywords):
    max_similarity = 0
    pos = -1
    temp = -1
    for chain in global_list:
        temp += 1
        keywords = chain[0]
        chain_items = chain[1]
        if not chain_items:  # Skip empty chains
            continue
        keyword_similarity = sum(nlp(keyword).similarity(nlp(input_keyword)) for keyword in keywords for input_keyword in input_keywords)
        text_similarity = sum(nlp(text).similarity(nlp(input_keyword)) for date, text in chain_items for input_keyword in input_keywords for date, text in chain_items)
        similarity = (keyword_similarity + text_similarity) / (len(input_keywords) + len(chain_items))
        if similarity > max_similarity:
            max_similarity = similarity
            pos = temp
    if max_similarity < 0.5:
        pos = -1
    return pos


def chain_item_sorting(list_of_lists):
    temp_list = list_of_lists[1]
    sorted_list = sorted(list_of_lists[1:], key=lambda x: datetime.strptime(x[0][0], "%B %d, %Y"))
    list_of_lists[1] = sorted_list

def add_to_global_var(text_input, date_added):
    kw_list = extract_keywords(text_input)
    pos = most_similar_chain(kw_list)
    if pos == -1:
        global_list.append([kw_list, [[date_added, text_input]]])
        chain_item_sorting(global_list[-1])
    else:
        global_list[pos][1].append([date_added, text_input])
        chain_item_sorting(global_list[pos])

def print_global_list():
    for chain in global_list:
        print("Keywords:", chain[0])
        print("Text-Date Pairs:")
        for pair in chain[1]:
            print("  Text:", pair)
        print()

sample_text = "hello world keyword1 keyword2 keyword3"
sample_date = "January 3, 2020"
add_to_global_var(sample_text, sample_date)

sample_text1 = "hello sample world keyword1 keyword2 keyword3"
sample_date1 = "April 10, 2024"
add_to_global_var(sample_text1, sample_date1)

print_global_list()
