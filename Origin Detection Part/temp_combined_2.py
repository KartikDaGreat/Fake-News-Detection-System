import random
from datetime import datetime, timedelta
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_md")

# Global list to store text-date pairs
global_list = []

# Function to extract keywords from text
def extract_keywords(text, num_keywords=5):
    tokenizer = RegexpTokenizer(r'\b[a-zA-Z]{3,}\b')  # Keep only words with 3 or more characters
    tokens = tokenizer.tokenize(text)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    filtered_tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens if word.lower() not in stop_words]
    freq_dist = FreqDist(filtered_tokens)
    top_keywords = freq_dist.most_common(num_keywords)
    return [word for word, _ in top_keywords]

# Function to find the most similar chain in the global list
def most_similar_chain(input_keywords):
    max_similarity = 0
    pos = -1
    for idx, chain in enumerate(global_list):
        keywords = chain[0]
        chain_items = chain[1]
        keyword_similarity = 0
        text_similarity = 0
        if keywords:
            # Calculate keyword similarity
            keyword_similarity = sum(nlp(keyword).similarity(nlp(input_keyword)) for keyword in keywords for input_keyword in input_keywords)
            # Calculate average keyword similarity
            avg_keyword_similarity = keyword_similarity / len(keywords)
            # If average keyword similarity is below threshold, continue to next chain
            if avg_keyword_similarity < 0.8:
                continue
        if chain_items:
            # Calculate text similarity
            text_similarity = sum(nlp(text).similarity(nlp(input_keyword)) for date, text in chain_items for input_keyword in input_keywords)
            # Calculate average text similarity
            avg_text_similarity = text_similarity / len(chain_items)
            # If average text similarity is below threshold, continue to next chain
            if avg_text_similarity < 0.6:
                continue
        # Calculate overall similarity
        similarity = (keyword_similarity + text_similarity) / (len(input_keywords) + len(chain_items))
        if similarity > max_similarity:
            max_similarity = similarity
            pos = idx
    if max_similarity < 0.5:
        pos = -1
    return pos


# Function to sort and update chains in the global list
def chain_item_sorting(list_of_lists):
    # Sort the list of lists based on the date in each inner list
    sorted_list = sorted(list_of_lists[1:], key=lambda x: datetime.strptime(x[0][0], "%B %d, %Y"))
    # Combine the sorted list with the first item (keywords)
    result = [list_of_lists[0]] + sorted_list
    return result


# Function to add text to the global list
def add_to_global_list(text_input, date_added):
    kw_list = extract_keywords(text_input)
    pos = most_similar_chain(kw_list)
    if pos == -1:
        global_list.append([kw_list, [[date_added, text_input]]])
        global_list[-1] = chain_item_sorting(global_list[-1])
    else:
        global_list[pos][1].append([date_added, text_input])
        global_list[pos] = chain_item_sorting(global_list[pos])

# Sample news headlines
news_headlines = [
    "Scientists discover new species of marine life in the Pacific Ocean.",
    "Global leaders convene to discuss climate change initiatives.",
    "Economic indicators show growth in the manufacturing sector.",
    "New study suggests potential breakthrough in cancer treatment.",
    "SpaceX launches satellite into orbit, marking another milestone in space exploration.",
    "Government announces new policies to promote renewable energy.",
    "Technology companies unveil latest innovations at annual conference.",
    "Celebrity couple announces engagement on social media.",
    "Local community holds fundraiser for charity.",
    "City council approves construction of new public transportation system.",
    "Researchers find correlation between diet and mental health.",
    "Artificial intelligence predicts trends in financial markets.",
    "Volunteers clean up beach to protect marine wildlife.",
    "Film industry celebrates record-breaking box office success.",
    "Health officials warn of flu outbreak in several states.",
    "Fashion designers showcase new collections at fashion week event.",
    "Investors show confidence in emerging markets.",
    "High school students organize protest for climate action.",
    "New smartphone model released with advanced features.",
    "Book club hosts author event for latest bestseller.",
    "Music festival attracts thousands of attendees from around the world.",
    "Restaurant chain introduces plant-based menu options.",
    "Medical breakthrough offers hope for rare disease patients.",
    "Local sports team wins championship title.",
    "Tech startup secures funding for expansion.",
    "Study reveals impact of social media on mental well-being.",
    "Art exhibition opens to critical acclaim.",
    "Travel industry sees surge in bookings for upcoming holidays.",
    "Community garden project promotes sustainability.",
    "Entrepreneur launches app to connect volunteers with local organizations."
]

# Generate 30 random inputs with news and dates
for i in range(30):
    # Randomly select a news headline
    news = news_headlines[i]
    # Generate a random date within the past year
    random_date = datetime.now() - timedelta(days=random.randint(1, 365))
    # Convert the date to the desired format
    date_added = random_date.strftime("%B %d, %Y")
    # Add the news headline to the origin detection system
    add_to_global_list(news, date_added)
additional_news = [
    ("New research reveals the health benefits of consuming dark chocolate.", "May 25, 2023"),
    ("Artificial intelligence-powered robot assists in delicate surgery for the first time.", "August 7, 2023"),
    ("Major breakthrough in renewable energy technology promises cleaner future.", "June 30, 2023"),
    ("Global summit discusses strategies for achieving sustainable development goals.", "October 14, 2023"),
    ("Local theater group wins prestigious award for groundbreaking production.", "April 3, 2024"),
    ("Social media campaign raises awareness and funds for environmental conservation.", "September 8, 2023"),
    ("Newly discovered archaeological site sheds light on ancient civilization.", "July 12, 2023"),
    ("Advanced drone technology aids in wildlife conservation efforts.", "November 19, 2023"),
    ("International collaboration leads to breakthrough in cancer research.", "February 5, 2024"),
    ("Emerging artist gains recognition for thought-provoking installation art.", "January 20, 2024"),
    ("Innovative startup revolutionizes sustainable packaging industry.", "March 10, 2024"),
    ("Community initiative transforms vacant lot into urban garden oasis.", "August 2, 2023"),
    ("Cutting-edge research paves the way for personalized medicine.", "December 8, 2023"),
    ("Groundbreaking study reveals link between gut health and mental well-being.", "June 15, 2023"),
    ("World-renowned chef launches initiative to tackle food waste crisis.", "October 28, 2023"),
    ("Renowned author releases highly anticipated sequel to bestselling novel.", "April 19, 2024"),
    ("Space agency announces plans for manned mission to Mars.", "November 1, 2023"),
    ("New educational program aims to teach coding to underprivileged youth.", "March 5, 2024"),
    ("Cutting-edge technology enables breakthroughs in clean water access.", "July 30, 2023"),
    ("Local community comes together to rebuild after natural disaster.", "September 29, 2023"),
    ("Renewable energy startup secures investment to scale up operations.", "February 12, 2024"),
    ("Celebrity philanthropist launches initiative to combat homelessness.", "May 10, 2023"),
    ("Advanced AI system predicts and prevents traffic accidents in real-time.", "August 18, 2023"),
    ("Innovative mobile app connects volunteers with opportunities to give back.", "October 6, 2023"),
    ("Breakthrough in quantum computing promises to revolutionize technology.", "December 30, 2023"),
    ("International conference discusses solutions to global food insecurity.", "March 15, 2024"),
    ("Groundbreaking study identifies genetic markers for rare neurological disorder.", "June 8, 2023"),
    ("Renowned architect designs sustainable skyscraper for urban skyline.", "November 22, 2023"),
    ("Cutting-edge nanotechnology leads to breakthroughs in cancer treatment.", "April 28, 2024")
]

# Add the additional news to the origin detection system
for news, date in additional_news:
    add_to_global_list(news, date)


# Print the updated global list
for idx, chain in enumerate(global_list):
    print(f"Chain {idx + 1}:")
    print("Keywords:", chain[0])
    print("Text-Date Pairs:")
    for pair in chain[1]:
        print("  Text:", pair[1])
        print("  Date:", pair[0])
        print()
    print()
