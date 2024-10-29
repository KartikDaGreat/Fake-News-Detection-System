import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import PassiveAggressiveClassifier
# from sklearn.metrics import accuracy_score
# from collections import defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
import spacy
from datetime import datetime
nlp = spacy.load("en_core_web_md")
import random
from datetime import timedelta

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
            ['April 12, 20221', 'Text 6']
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
    for chain in global_list:  # Iterate over each chain in global_list
        temp += 1
        keywords = chain[0]  # Keywords are in the first element of the chain
        chain_items = chain[1]  # Text-date pairs are in the second element of the chain
        # Calculate similarity with keywords
        keyword_similarity = sum(nlp(keyword).similarity(nlp(input_keyword)) for keyword in keywords for input_keyword in input_keywords)
        # Calculate similarity with text in each text-date pair
        text_similarity = sum(nlp(text).similarity(nlp(input_keyword)) for date, text in chain_items for input_keyword in input_keywords)
        # Calculate overall similarity as the sum of keyword similarity and text similarity
        similarity = (keyword_similarity + text_similarity) / (len(input_keywords) + len(chain_items))
        if similarity > max_similarity:
            max_similarity = similarity
            pos = temp
    if max_similarity < 0.5:
        pos = -1
    return pos


def chain_item_sorting(list_of_lists):
    # Sort the list of lists based on the date in each inner list
    sorted_list = sorted(list_of_lists[1:], key=lambda x: datetime.strptime(x[0][0], "%B %d, %Y"))
    # Combine the sorted list with the first item (keywords)
    result = [list_of_lists[0]] + sorted_list
    return result

def add_to_global_var(text_input, date_added):
    kw_list = extract_keywords(text_input)
    pos = most_similar_chain(kw_list)
    if pos == -1:
        global_list.append([kw_list, [[date_added, text_input]]])
        chain_item_sorting(global_list[-1])
    else:
        global_list[pos].append([date_added, text_input])
        chain_item_sorting(global_list[pos])


def print_global_list():
    for chain in global_list:
        print("Keywords:", chain[0])
        print("Text-Date Pairs:")
        for pair in chain[1]:
            print("  Text:", pair)
            print()
        print()

# print_global_list()
# # Sample news headlines
# news_headlines = [
#     "Scientists discover new species of marine life in the Pacific Ocean.",
#     "Global leaders convene to discuss climate change initiatives.",
#     "Economic indicators show growth in the manufacturing sector.",
#     "New study suggests potential breakthrough in cancer treatment.",
#     "SpaceX launches satellite into orbit, marking another milestone in space exploration.",
#     "Government announces new policies to promote renewable energy.",
#     "Technology companies unveil latest innovations at annual conference.",
#     "Celebrity couple announces engagement on social media.",
#     "Local community holds fundraiser for charity.",
#     "City council approves construction of new public transportation system.",
#     "Researchers find correlation between diet and mental health.",
#     "Artificial intelligence predicts trends in financial markets.",
#     "Volunteers clean up beach to protect marine wildlife.",
#     "Film industry celebrates record-breaking box office success.",
#     "Health officials warn of flu outbreak in several states.",
#     "Fashion designers showcase new collections at fashion week event.",
#     "Investors show confidence in emerging markets.",
#     "High school students organize protest for climate action.",
#     "New smartphone model released with advanced features.",
#     "Book club hosts author event for latest bestseller.",
#     "Music festival attracts thousands of attendees from around the world.",
#     "Restaurant chain introduces plant-based menu options.",
#     "Medical breakthrough offers hope for rare disease patients.",
#     "Local sports team wins championship title.",
#     "Tech startup secures funding for expansion.",
#     "Study reveals impact of social media on mental well-being.",
#     "Art exhibition opens to critical acclaim.",
#     "Travel industry sees surge in bookings for upcoming holidays.",
#     "Community garden project promotes sustainability.",
#     "Entrepreneur launches app to connect volunteers with local organizations."
# ]

# # Generate 30 random inputs with news and dates
# for i in range(30):
#     # Randomly select a news headline
#     news = (news_headlines[i])
#     # Generate a random date within the past year
#     random_date = datetime.now() - timedelta(days=random.randint(1, 365))
#     # Convert the date to the desired format
#     date_added = random_date.strftime("%B %d, %Y")
#     # Add the news headline to the origin detection system
#     add_to_global_var(news, date_added)

# # Print the updated global list
# print_global_list()
