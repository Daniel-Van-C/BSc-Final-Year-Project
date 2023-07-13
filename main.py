# main.py: Main classes used in YouTube Comments Analyser application.
#
# Written by Daniel Van Cuylenburg (k19012373)
#

# Imports.
# GUI Classes.
from gui import *
# Dictionary of (unexpanded contraction) -> (expanded contraction) pairs.
from contractions import contractions

# The Python Standard Library.
from concurrent.futures import process
from sys import argv
from time import time
from datetime import datetime
from os import getcwd, mkdir
from os.path import exists, join
from re import sub
from urllib import request, parse
from json import loads
from pickle import dump, load
from ast import literal_eval
from collections import Counter
# cgitb used for error printing during implementation stage.
import cgitb
# cgitb.enable(format = "text")

from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox

from pandas import DataFrame, read_csv, concat, Series

from numpy import isnan

from googleapiclient.discovery import build

from emoji import UNICODE_EMOJI

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet as wn, sentiwordnet as swn
from nltk.tag import pos_tag

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
# Classifiers used.
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
# Classifiers tested but not used in final version.
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import MultinomialNB

from pylab import figure, close

from wordcloud import WordCloud

from PIL import Image

# Google Cloud Service API keys. If the daily quota is reached,
# use another API key. Here are some alternative API keys.
# API_KEY = "AIzaSyChlJ40_GmFw8n9Sw-2YJdyo4ury3kS2bw"
# API_KEY = "AIzaSyC_jbsqsPiCZZcfdac4DU6AVjcC54FuQ6Y"
# API_KEY = "AIzaSyBDVpl0jBBZj27WmCrGOH7HF-OkCCnwRek"

# Current directory path.
CURRENT_PATH = join(getcwd(), "Python")

# Max features to be used by count vectorizer and TF-IDF vectorizer.
MAX_FEATURES = 2000


class CommentFetcher:
    """Class that downloads comments from a given YouTube URL.

    Attributes:
        url (str): URL entered by user.
        title (str): Title of the video.
        comments (pandas DataFrame): Video's comments.
    """

    def __init__(self, url):
        """Inits CommentFetcher class with url."""
        print("Downloading Video's Comments")

        # Removes any excess whitespace in the URL.
        self.url = url.replace(" ", "")

        # Fetches the Video's title.
        query = parse.urlencode({"format": "json", "url": self.url})
        response = request.urlopen("https://www.youtube.com/oembed?" +
                                   query).read()
        data = loads(response.decode())
        self.title = data["title"]
        self.title = CommentCleaner().remove_punctuation_and_symbols(
            self.title).replace("  ", " ")

        # Removes any excess whitespace at the end of a video's title.
        while self.title[-1] == " ":
            self.title = self.title[:-1]

        self.retrieve_comments()

    def retrieve_comments(self):
        """Downloads a YouTube video's comments. Saves them in a DataFrame."""
        # Stores the URL's video ID.
        url_parsed = parse.urlparse(self.url)
        qsl = parse.parse_qs(url_parsed.query)
        video_id = qsl["v"][0]

        # Fetches the current API key from a text file.
        with open(join(CURRENT_PATH, "API Key.txt"), "r") as file:
            api_key = file.read().splitlines()

        # Creates a youtube resource object.
        youtube = build("youtube", "v3", developerKey=api_key)
        video_response = youtube.commentThreads().list(
            part="snippet", videoId=video_id).execute()

        fetched_comments = [[], []]
        # Iterates through the video's response.
        while video_response:
            # Extracts the comment's text and timestamp from each result object.
            for item in video_response["items"]:
                # Comment's text.
                fetched_comments[0].append(item["snippet"]["topLevelComment"]
                                           ["snippet"]["textDisplay"])
                # Comments date-time stamp.
                fetched_comments[1].append(item["snippet"]["topLevelComment"]
                                           ["snippet"]["publishedAt"])

            # If there are more comments that have not been downloaded,
            # go to the next results page.
            if "nextPageToken" in video_response:
                p_token = video_response["nextPageToken"]
                video_response = youtube.commentThreads().list(
                    part="snippet", videoId=video_id,
                    pageToken=p_token).execute()
            else:  # If all of the comments have been downloaded, stop.
                break

        # Removes any HTML tags from the dataset.
        for comment in range(len(fetched_comments[0])):
            new_comment = sub(
                '''<a href="|</a>|<br>|<b>|</b>|&quot;|&amp;|&#39;''', """'""",
                fetched_comments[0][comment])
            fetched_comments[0][comment] = new_comment

        # Saves the comments as a pandas DataFrame.
        self.comments = DataFrame({
            "Comment": fetched_comments[0],
            "Time": fetched_comments[1]
        })

        self.format_comments()

    def format_comments(self):
        # Replaces any new lines in the dataset with whitespaces.
        self.comments = self.comments.replace("\n", " ", regex=True)

        # Option of removing duplicate comments. Not used.
        # self.comments = self.comments.drop_duplicates(subset = "Comment")

        # Creates a folder for the current video.
        # Folder's name is the title of the video.
        path = join(CURRENT_PATH, "Videos", str(self.title))
        if not exists(join(CURRENT_PATH, "Videos", self.title)):
            mkdir(path)
            mkdir(join(path, "Charts"))

        # Saves the title of the video.
        with open(join(path, "title.pkl"), "wb") as file:
            dump((self.title), file)

        # Saves the raw comments of the video.
        self.comments.to_csv(join(path, "data.csv"), index=False)


class CommentCleaner:
    """Class that cleans the comments dataset passed into it.
    
    Attributes:
        comments (pandas DataFrame, optional): Video's comments.
            Defaults to None.
        spell_check (bool): If spelling correction should be
            performed on misspelled words. Defaults to None.
        corpus (bool): If the comments are a corpuses comments.
    """

    def __init__(self, comments=None, spell_check=False, corpus=False):
        """Inits CommentCleaner with comments."""
        self.comments = comments
        self.spell_check = spell_check
        self.corpus = corpus

    def clean(self):
        """Makes calls to each function in the class to clean the dataset."""
        # If the dataset is a corpus, removes additional Twitter tags.
        if self.corpus:  
            print("Cleaning Corpus")
            self.remove_twitter_tags_comments()
        else:
            print("Cleaning Data")

        self.remove_excess_whitespace()
        self.remove_symbols_comments()
        self.remove_punctuation_and_symbols_comments()
        self.remove_urls_comments()
        self.lower_case_comments()
        self.remove_excess_whitespace()
        self.expand_contractions_comments()
        self.correct_spelling_comments()
        self.tokenise_comments()
        self.lemmatize_comments()
        self.remove_stopwords_comments()
        self.remove_excess_whitespace()

        # If the dataset is not a corpus, reformats the date-time stamps.
        if not self.corpus:  
            self.format_date_time()

        return self.comments

    def remove_twitter_tags_comments(self):
        """Calls remove_twitter_tags() on each comment.
        
        Creates a new column with the returned data.
        """
        self.comments["Comment"] = self.comments["Comment"].apply(
            self.remove_twitter_tags)

    def remove_twitter_tags(self, comment):
        """Removes any Twitter tags from a string."""
        return sub(r"\@user|\@USER", "", comment)

    def remove_excess_whitespace(self):
        """Removes any rows from the data that contain whitespace only."""
        self.comments = self.comments.dropna()

    def remove_symbols(self, comment):
        """Removes symbols from a string.

        Args:
            comment (str): A single comment from the dataset.

        Returns:
            str: Comment with no symbols.
        """
        return "".join([
            x for x in comment
            if (x.isalpha() or x in [" ", ".", ",", "!", "?"] or
                x not in UNICODE_EMOJI)
        ])  # or x not in emoji.UNICODE_EMOJI

    def remove_symbols_comments(self):
        """Calls remove_symbols() on each comment
        
        Creates a new column with the returned data.
        """
        self.comments["no_symbols"] = self.comments["Comment"].apply(
            self.remove_symbols)

    def remove_punctuation_and_symbols(self, comment):
        """Removes punctuation and symbols from a string.

        Args:
            comment (str): A single comment from the dataset.

        Returns:
            str: Comment with no punctuation and symbols.
        """
        return "".join([x for x in comment if (x.isalpha() or x == " ")])

    def remove_punctuation_and_symbols_comments(self):
        """Calls remove_punctuation_and_symbols() on each comment
        
        Creates a new column with the returned data.
        """
        self.comments["no_punctuation_and_symbols"] = self.comments[
            "Comment"].apply(self.remove_punctuation_and_symbols)

    def remove_urls(self, comment):
        """Removes URLs from a string.

        Args:
            comment (str): A single comment from the dataset.

        Returns:
            str: Comment with no URLs.
        """
        return sub(r"\S*https?\S*", "", comment)

    def remove_urls_comments(self):
        """Calls remove_urls() on each comment
        
        Creates a new column with the returned data.
        """
        self.comments["no_urls"] = self.comments["no_symbols"].apply(
            self.remove_urls)
        self.comments["no_urls2"] = self.comments[
            "no_punctuation_and_symbols"].apply(self.remove_urls)

    def lower_case(self, comment):
        """Makes all characters in a string lower case.

        Args:
            comment (str): A single comment from the dataset.

        Returns:
            str: Comment with all characters in lower case.
        """
        lower = ""
        for character in comment:
            if character.isalpha():
                lower += character.lower()
            else:
                lower += character
        return lower

    def lower_case_comments(self):
        """Calls lower_case() on each comment
        
        Creates a new column with the returned data.
        """
        self.comments["lower_case"] = self.comments["no_urls2"].apply(
            self.lower_case)

    def expand_contractions(self, comment):
        """Expands all contractions in a string.

        Args:
            comment (str): A single comment from the dataset.

        Returns:
            str: Comment with expanded contractions.
        """
        expanded = ""
        for word in comment.split():
            if word in contractions:
                expanded += contractions[word] + " "
            else:
                expanded += word + " "
        return expanded

    def expand_contractions_comments(self):
        """Calls expand_contractions() on each comment
        
        Creates a new column with the returned data.
        """
        self.comments["expanded_contractions"] = self.comments[
            "lower_case"].apply(self.expand_contractions)
        self.comments["expanded_contractions2"] = self.comments[
            "no_urls"].apply(self.expand_contractions)

    def correct_spelling(self, comment):
        """Corrects the spelling of words in a string.

        Args:
            comment (str): A single comment from the dataset.

        Returns:
            str: Comment with corrected spelling.
        """
        new_comment = []
        for word in comment.split():
            try:
                new_comment.append(str(TextBlob(word).correct()))
            except:
                new_comment.append(word)
        return " ".join(new_comment)

    def correct_spelling_comments(self):
        """Calls correct_spelling() on each comment if the user has selected the
        spell correction option; creates a new column with the returned data.
        """
        if self.spell_check:
            print("Starting spelling correction. May take a while if the dataset is large. A statement will print when spelling correction finishes.")
            self.comments["spelling"] = self.comments[
                "expanded_contractions"].apply(self.correct_spelling)
            self.comments["spelling2"] = self.comments[
                "expanded_contractions2"].apply(self.correct_spelling)
            print("Spelling correction finished.")
        else:
            self.comments["spelling"] = self.comments["expanded_contractions"]
            self.comments["spelling2"] = self.comments["expanded_contractions2"]

    def tokenise(self, comment):
        """Turns a string into a list, splitting at whitespaces.

        Args:
            comment (list): A single comment from the dataset.

        Returns:
            list: Comment as a list.
        """
        return word_tokenize(comment)

    def tokenise_comments(self):
        """Calls tokenise() on each comment
        
        Creates a new column with the returned data.
        """
        self.comments["tokens"] = self.comments["spelling"].apply(self.tokenise)

    def lemmatize(self, comment):
        """Turns a string into its dictionary form, if it has one.

        Args:
            comment (list): A single comment from the dataset.

        Returns:
            list: Lemmatized comment.
        """
        lemmas = []
        for word in comment:
            lemmas.append(WordNetLemmatizer().lemmatize(word))
        return lemmas

    def lemmatize_comments(self):
        """Calls lemmatize() on each comment
        
        Creates a new column with the returned data.
        """
        self.comments["lemmas"] = self.comments["tokens"].apply(self.lemmatize)

    def remove_stopwords(self, comment):
        """Removes stop words from a string.

        Args:
            comment (list): A single comment from the dataset.

        Returns:
            list: Comment with no stop words.
        """
        new_comment = []
        for word in comment:
            if not word in stopwords.words("english"):
                new_comment.append(word)
        if new_comment == []:
            return None
        else:
            return new_comment

    def remove_stopwords_comments(self):
        """Calls remove_stopwords() on each comment
        
        Creates a new column with the returned data.
        """
        self.comments["lemmas_no_stopwords"] = self.comments["lemmas"].apply(
            self.remove_stopwords)

    def format_date_time(self):
        """Changes each comments timestamp into a more readable format."""
        self.comments["Time"] = self.comments["Time"].apply(
            lambda x: (x[0:10] + "\n" + x[11:19]))


class CommentSentiment:
    """Class that assigns sentiment and subjectivity scores to each comment.
    
    Attributes:
        comments (pandas DataFrame): Video's comments.
        vader_analyzer (SentimentIntensityAnalyzer): VADER lexicon analyser.
    """

    def __init__(self, comments):
        """Inits CommentSentiment class with comments."""
        self.comments = comments
        self.vader_analyzer = SentimentIntensityAnalyzer()

    def analyse(self):
        """Calls assign_sentiment_comments() andassign_subjectivity_comments()."""
        print("Assigning Sentiment to Comments")
        self.assign_sentiment_comments()
        self.assign_subjectivity_comments()
        return self.comments

    def assign_sentiment_comments(self):
        """Assigns sentiment scores and tags to each comment."""
        self.comments["Polarity"] = self.comments["spelling2"].apply(
            self.assign_valence)
        for _, row in self.comments.iterrows():  # For each comment
            # If the VADER and Textblob methods was not able to assign a
            # polarity to the current comment, tries the SentiWordNet method.
            if row["Polarity"] == 0.0:
                self.apply_swn(row)

        # Tags each comment with a sentiment label based on its polarity score.
        self.comments["Sentiment"] = self.comments["Polarity"].apply(
            self.assign_sentiment)

    def assign_valence(self, comment):
        """Assign a valence score to the comment."""
        vader_sentiment_dict = self.vader_analyzer.polarity_scores(comment)
        # If VADER was able to assign a polarity, returns this score.
        if vader_sentiment_dict["compound"] != 0:
            return vader_sentiment_dict["compound"]
        else:  # If VADER could not assign a polarity score.
            # Assigns a polarity score with TextBlob.
            return self.assign_textblob_valence(comment)

    def assign_textblob_valence(self, comment):
        """Assigns and returns a polarity score using the TextBlob lexicon."""
        return TextBlob(comment).sentiment.polarity

    def apply_swn(self, row):
        """Assigns a polarity score using the SentiWordNet method to a comment."""
        sentiment = 0.0
        # For each word and its POS tag in the comment.
        for word, tag in pos_tag(row["tokens"]):
            # Converts the POS tag into a WordNet POS tag.
            wn_tag = self.penn_to_wn(tag)
            # Lemmatizes the word.
            lemma = WordNetLemmatizer().lemmatize(word, pos=wn_tag)
            # Generates a subset of synonyms for the given word and its POS tag.
            synsets = wn.synsets(lemma, pos=wn_tag)
            if synsets:  # If a synset was found for the word.
                # Takes the first, most common sense (synset) of the word.
                synset = synsets[0]
                # Retrieves the sentiment of the word,
                # compound this to the total sentiment of the comment.
                swn_synset = swn.senti_synset(synset.name())
                sentiment += swn_synset.pos_score() - swn_synset.neg_score()
        row["Polarity"] == sentiment

    def penn_to_wn(self, tag):
        """Converts a POS tag to a WordNet POS tag."""
        if tag.startswith("J"):
            return wn.ADJ
        elif tag.startswith("N"):
            return wn.NOUN
        elif tag.startswith("R"):
            return wn.ADV
        elif tag.startswith("V"):
            return wn.VERB
        return wn.NOUN

    def assign_subjectivity_comments(self):
        """Calls assign_subjectivity() on each comment."""
        self.comments["Subjectivity"] = self.comments["spelling2"].apply(
            self.assign_subjectivity)

    def assign_subjectivity(self, comment):
        """Assigns a subjectivity score using the TextBlob lexicon.
        
        Args:
            comment (str): A single comment from the dataset.

        Returns:
            float: Subjectivity score of the comment.
        """
        return TextBlob(comment).sentiment.subjectivity

    def assign_sentiment(self, comment):
        """Assigns a sentiment label based on the given comments polarity score.
        
        Args:
            comment (list): A single comment from the dataset.

        Returns:
            str: Comment's sentiment label.
        """
        if comment >= 0.05:
            return "positive"
        elif comment <= -0.05:
            return "negative"
        else:
            return "neutral"


class CommentVectorizer:
    """Class that vectorizes the dataset passed into it.
    
    Attributes:
        comments (pandas DataFrame): Video's comments.
        frequencies (list): Count vectorized data for each n-gram.
        importance (list): TF-IDF vectorized data for each n-gram.
    """

    def __init__(self, comments):
        """Inits CommentVectorizer with comments."""
        self.comments = comments

    def process(self):
        """Calls vectorize() and tfidf()."""
        self.vectorize()
        self.tfidf()

    def vectorize(self):
        """Performs count vectorization on the dataset for unigrams, bigrams, trigrams."""
        self.frequencies = []
        ngrams = [(1, 1), (2, 2), (3, 3)]
        for ngram in ngrams:  # For each n-gram.
            count_vectorizer = CountVectorizer(ngram_range=ngram,
                                               max_features=MAX_FEATURES)
            count = count_vectorizer.fit_transform(
                self.comments["lemmas_no_stopwords"].apply(
                    lambda x: " ".join(x)))
            df = DataFrame(count.toarray(),
                           columns=count_vectorizer.get_feature_names_out())
            # Generates a list of n-gram-frequency pairs, sorted by most frequent.
            df = df.T.sum(axis=1).sort_values(ascending=False)
            df.columns = ["Word", "Frequency"]
            self.frequencies.append(df)

    def tfidf(self):
        """Performs TF-IDF vectorization on the dataset for unigrams, bigrams, trigrams."""
        self.importance = []
        ngrams = [(1, 1), (2, 2), (3, 3)]
        for ngram in ngrams:  # For each n-gram.
            tfidf_vectorizer = TfidfVectorizer(use_idf=True,
                                               smooth_idf=False,
                                               ngram_range=ngram,
                                               max_features=MAX_FEATURES)
            score = tfidf_vectorizer.fit_transform(
                self.comments["lemmas_no_stopwords"].apply(lambda x: " ".join(x)))
            df = DataFrame(score.toarray(),
                           columns=tfidf_vectorizer.get_feature_names_out())
            # Generates a list of n-gram-importance pairs, sorted by most important.
            self.importance.append(
                df.T.sum(axis=1).sort_values(ascending=False))


class CommentClassifier:
    """Class that classifies the dataset into categories.
    
    Attributes:
        comments (pandas DataFrame): Video's comments.
        
        offensive_path (str): File path of OffensEval corpus.
        emotion_path (str): File path of GoEmotions corpus.
        
        offensive_corpus (pandas DataFrame): OffensEval corpus.
        emotions_corpus (pandas DataFrame): GoEmotions corpus.
    """

    def __init__(self, comments):
        """Inits CommentClassifier with comments; saved cleaned corpus as CSV"""
        self.comments = comments

        self.offensive_path = join(CURRENT_PATH, "Corpora", "OffensEval")
        self.emotion_path = join(CURRENT_PATH, "Corpora", "Emotions")

        self.clean_corpora()

    def clean_corpora(self):
        """If the corpus has not been cleaned, clean it and save this as a CSV file."""
        # If the OffensEval corpus has not yet been cleaned, cleans it.
        if not exists(join(self.offensive_path, "cleaned_offense_corpus.csv")):
            corpus = read_csv(join(self.offensive_path, "offence_corpus.csv"))
            corpus["Comment"] = corpus["Comment"].apply(str)
            corpus = CommentCleaner(corpus, False, True).clean()
            corpus.to_csv(join(self.offensive_path,
                               "cleaned_offense_corpus.csv"),
                               index=False)
        # Reads the cleaned OffensEval corpus into a DataFrame.
        self.offensive_corpus = read_csv(
            join(self.offensive_path, "cleaned_offense_corpus.csv"))

        # If the GoEmotions corpus has not yet been cleaned, cleans it.
        if not exists(join(self.emotion_path, "cleaned_emotions_corpus.csv")):
            corpus = read_csv(join(self.emotion_path, "goemotions_1.csv"))
            corpus = concat([corpus,
                             read_csv(join(self.emotion_path, "goemotions_2.csv"))],
                             ignore_index=True)
            corpus = concat(
                [corpus,
                 read_csv(join(self.emotion_path, "goemotions_3.csv"))],
                 ignore_index=True)
            corpus = corpus.rename(columns={'text': 'Comment'})
            corpus["Category"] = corpus.apply(self.assign_emotion, axis=1)
            corpus = CommentCleaner(corpus, False, True).clean()
            corpus[["Comment", "Category", "lemmas_no_stopwords"
                   ]].to_csv(join(self.emotion_path,
                                  "cleaned_emotions_corpus.csv"),
                                  index=False)
        # Reads the cleaned GoEmotions corpus into a DataFrame.
        self.emotions_corpus = read_csv(
            join(self.emotion_path, "cleaned_emotions_corpus.csv"))

    def assign_emotion(self, row):
        """When cleaning the GoEmotions corpus, assign an emotion to each comment.

        Args:
            row (pandas DataFrame single row): Row of a comment from the corpus.

        Returns:
            str: Assigned emotion category.
        """
        emotions = [
            "admiration", "amusement", "anger", "annoyance", "approval",
            "caring", "confusion", "curiosity", "desire", "disappointment",
            "disapproval", "disgust", "embarrassment", "excitement", "fear",
            "gratitude", "grief", "joy", "love", "nervousness", "optimism",
            "pride", "realization", "relief", "remorse", "sadness", "surprise",
            "neutral"
        ]
        for emotion in emotions:
            if row[emotion] == 1:
                return emotion
        return "neutral"

    def classify(self):
        """Classifies the comments.

        Returns:
            pandas DataFrame: Classified comments dataset.
        """
        print("Classifying Data")
        # Elements needed for each corpus.
        elements = [[self.offensive_corpus, self.emotions_corpus],
                    ["offence_svm", "emotions_decision_tree"],
                    [DecisionTreeClassifier(), SVC(kernel="linear", gamma="auto")],
                    [self.offensive_path, self.emotion_path],
                    ["Offensive?", "Emotion"]]
        
        for n in range(len(elements[0])):  # For each of the corpora.
            # Transforms the labels into numbers.
            encoder = LabelEncoder()
            labels = encoder.fit_transform(elements[0][n]["Category"])

            # Performs count vectorization on the cleaned corpus.
            count_vectorizer = CountVectorizer(max_features=5000)
            count_vectorizer.fit(
                self.emotions_corpus["lemmas_no_stopwords"].apply(
                    lambda x: " ".join(literal_eval(x))))
            classifier_input = count_vectorizer.transform(
                self.comments["lemmas_no_stopwords"].apply(
                    lambda x: " ".join(x)))

            # If the classifier has not been trained on the corpus previously, do so.
            if not exists(join(elements[3][n], elements[1][n] + ".pkl")):
                training_data = count_vectorizer.transform(
                    self.emotions_corpus["lemmas_no_stopwords"].apply(
                        lambda x: " ".join(literal_eval(x))))
                # Fits the training data into the classifier.
                classifier = elements[2][n]
                classifier.fit(training_data, labels)
                # Saves the trained classifier as a pickle file.
                with open(join(elements[3][n], elements[1][n] + ".pkl"), "wb") as file:
                    dump((classifier), file)
            else:  # If the classifier has already been trained, loads this data.
                with open(join(elements[3][n], elements[1][n] + ".pkl"), "rb") as file:
                    classifier = load(file)

            # Assigns a category to each comment in the user input dataset.
            predictions = classifier.predict(classifier_input)
            # Converts the numerical assigned categories to their labels.
            labels = encoder.inverse_transform(predictions)
            self.comments[elements[4][n]] = labels

        return self.comments


class Main:
    """Main class. Makes calls to all other classes.
    
    Time ranges refer to the subset of comments posted in the last
    30 days, 7 days, 48 hours, and all time.
    
    Attributes:
        app (QApplication): PyQt5 application used to display windows.
        menu (Menu): Menu window.
        start_time (time): Start of processing time, used when calculating
            the total processing time.
        file_name (str): Name of video's file.
        file_path (str): File path to the current video.
        
        comments (pandas DataFrame): Video's comments.
        comments_30_days (pandas DataFrame): Video's comments from the last 30 days.
        comments_7_days (pandas DataFrame): Video's comments from the last 7 days.
        comments_48_hours (pandas DataFrame): Video's comments from the last 48 hours.
        comments_time_ranges (list): List of all of the above comments DataFrames.
        
        vectorizer_data (list): Vectorizer data for unigrams, bigrams, trigrams.
        report (pandas DataFrame): Report about the comments.
    """

    def __init__(self):
        """Inits Main class."""
        print("Loading Application")
        self.generate_directories()
        self.app = QApplication(argv)

        # Applies a stylesheet to the whole application.
        qss = open(join(CURRENT_PATH, "StyleSheets", "Combinear.qss"), "r").read()
        self.app.setStyleSheet(qss)

        # Initially, the user has not picked any menu option.
        option_selected = False
        # While the user has not picked an option or has entered an invalid URL.
        while not option_selected:
            self.display_window("Menu")
            if self.menu.option == "YouTube":  # If the user wants to use a YouTube URL.
                try:  # Attempts to download comments from the specified URL.
                    self.start_time = time()  # Start of processing time tracking.
                    fetcher = CommentFetcher(self.menu.url)
                    self.file_name = fetcher.title
                    self.file_path = join(CURRENT_PATH, "Videos",
                                          self.file_name)
                    self.comments = fetcher.comments
                    if self.comments.empty:  # If the video does not have any comments.
                        self.display_error("This video does not have any comments. Please enter a valid YouTube URL.")
                    else:
                        self.process_comments()
                        option_selected = True
                except:  # If there was a problem when downloading or processing the comments.
                    self.display_error("Please enter a valid YouTube URL.")
            elif self.menu.option == "CSV":  # If the user wants to view a previously processed dataset.
                self.file_name = self.menu.file_selection
                self.file_path = join(CURRENT_PATH, "Videos", self.file_name)
                self.read_from_files()
                option_selected = True

            # If the user has selected an option and there were no problems downloading and processing the comments.
            if self.menu.option != None and option_selected:
                # Displays the main window.
                self.display_window("Main")
                option_selected = False
            elif self.menu.option == None:  # Else if the user has pressed the close button on the menu.
                break

    def generate_directories(self):
        """Generate any directories used by the system if they do not already exist."""
        # If the "Videos" folder does not exist, creates it.
        if not exists(join(CURRENT_PATH, "Videos")):
            mkdir(join(CURRENT_PATH, "Videos"))

    def display_window(self, selection):
        """Displays specified window.

        Args:
            selection (str): Window to be displayed.
        """
        window = QMainWindow()
        if selection == "Menu":
            self.menu = Menu(window)
            window.show()
        else:
            # Saves the main window object as a variable so that it does
            # not go out of scope, closing the window.
            main_window = MainWindow(window, self.file_name, self.report,
                                     self.comments_time_ranges,
                                     self.emotional_analysis,
                                     self.vectorizer_data)

            # If the screen size of the current machine is small, maximizes the window.
            if self.app.primaryScreen().size().height() < 960:
                window.showMaximized()
            else:
                window.show()
        self.app.exec_()

    def display_error(self, message):
        """Displays error message window.

        Args:
            message (str): Error message to be displayed.
        """
        error_dialog = QMessageBox()
        error_dialog.setText(message)
        error_dialog.exec()

    def generate_empty_dataframes(self):
        """Generates empty DataFrames for the different time ranges."""
        self.comments_30_days = DataFrame(columns=self.comments.columns)
        self.comments_7_days = DataFrame(columns=self.comments.columns)
        self.comments_48_hours = DataFrame(columns=self.comments.columns)

    def process_comments(self):
        """Makes calls to other functions and classes to clean and process the comments."""
        self.comments = CommentCleaner(self.comments,
                                       self.menu.spell_check).clean()
        self.comments = CommentSentiment(self.comments).analyse()
        classifier = CommentClassifier(self.comments)
        self.comments = classifier.classify()

        self.generate_empty_dataframes()
        self.split_data_by_time_ranges()
        self.comments_time_ranges = [
            self.comments, self.comments_30_days, self.comments_7_days,
            self.comments_48_hours
        ]

        self.vectorize_data()

        self.generate_report()
        self.generate_charts()
        self.generate_files()

    def split_data_by_time_ranges(self):
        """Fills time range DataFrames with any comments that fit the specified time range."""
        for index, row in self.comments.iterrows():  # For each comment.
            current_datatime = datetime(year=int(row["Time"][0:4]),
                                        month=int(row["Time"][5:7]),
                                        day=int(row["Time"][8:10]))
            # Length of time since the current comment was posted.
            datatime_difference = (datetime.today() - current_datatime).days
            # If the comment was posted in the last 30 days,
            # appends it to the relevant DataFrame.
            if datatime_difference < 30:
                self.comments_30_days = concat([
                    self.comments_30_days,
                    self.comments.loc[index].to_frame().transpose()
                ], ignore_index=True)
                # If the comment was posted in the last 7 days,
                # appends it to the relevant DataFrame.
                if datatime_difference < 7:
                    self.comments_7_days = concat([
                        self.comments_7_days,
                        self.comments.loc[index].to_frame().transpose()
                    ], ignore_index=True)
                    # If the comment was posted in the last 48 hours,
                    # appends it to the relevant DataFrame.
                    if datatime_difference < 2:
                        self.comments_48_hours = concat([
                            self.comments_48_hours,
                            self.comments.loc[index].to_frame().transpose()
                        ], ignore_index=True)

        # Formats the data.
        for dataframe in [
                self.comments_30_days, self.comments_7_days,
                self.comments_48_hours
        ]:
            dataframe["Polarity"] = dataframe["Polarity"].apply(
                lambda x: x.item())
            dataframe["Subjectivity"] = dataframe["Subjectivity"].apply(
                lambda x: x.item())

    def vectorize_data(self):
        """Makes calls to the vectorizer for all of the data by time range."""
        print("Vectorizing Data")
        self.vectorizer_data = []
        self.emotional_data = []
        # For each time range of comments.
        for dataframe in self.comments_time_ranges: 
            vectorizer = CommentVectorizer(dataframe)
            # If data exists for this time range, appends the relevant vectorizer data.
            if not dataframe.empty:
                vectorizer.process()
                self.vectorizer_data.append([
                    vectorizer.frequencies[0], vectorizer.importance[0],
                    vectorizer.frequencies[1], vectorizer.importance[1],
                    vectorizer.frequencies[2], vectorizer.importance[2]
                ])
            # If no data exists for the time range, appends empty DataFrames.
            else:
                self.vectorizer_data.append([
                    DataFrame(),
                    DataFrame(),
                    DataFrame(),
                    DataFrame(),
                    DataFrame(),
                    DataFrame()
                ])

    def generate_report(self):
        """Generates a report about the comments."""
        print("Generating Report")
        self.report = DataFrame({
            " ": [
                "Positive", "Neutral", "Negative", "Total", "Average Valence",
                "Average Subjectivity", "Offensive", "Inoffensive",
                "Processing Time"
            ]
        })
        self.emotional_analysis = [[], [], [], []]
        time_frame_labels = [
            "All Time", "Last Month", "Last Week", "Last 48 Hours"
        ]
        # For each time range subset.
        for dataframe in range(len(self.comments_time_ranges)):  
            sentiment_dict = {"positive": 0, "negative": 0, "neutral": 0}
            current_subset = self.comments_time_ranges[dataframe]
            if not current_subset.empty:  # If data exists for this time range.
                # Calculates the total number of positive, negative, and neutral comments.
                for index, value in current_subset["Sentiment"].value_counts(
                ).items():
                    sentiment_dict[index] = int(value)

                # Calculates the total number of comments.
                total = 0
                for value in sentiment_dict.values():
                    total += value

                # Calculates the average polarity and subjectivity from the scores.
                comments_valence_no_0 = current_subset["Polarity"].drop(
                    current_subset["Polarity"][current_subset["Polarity"] == 0.0].index)
                averages = [
                    comments_valence_no_0.sum().item() /
                    comments_valence_no_0.count(),
                    current_subset["Subjectivity"].sum() /
                    current_subset["Subjectivity"].count()
                ]
                for n in range(len(averages)):
                    # If the value has more than there decimal places, shortens it.
                    if str(averages[n].item())[::-1].find('.') > 3:
                        averages[n] = round(averages[n], 3)
                    if isnan(averages[n]):
                        averages[n] = 0
                    else:
                        averages[n] = averages[n].item()

                # If the current subset is of time range "all time",
                # calculates the processing time.
                processing_time = " "
                if dataframe == 0:
                    processing_time = str(round(time() - self.start_time,
                                                1)) + " Secs"

                # Counts the number of offensive and inoffensive comments.
                offensive_dict = {"Yes": 0, "No": 0}
                offensive_count = current_subset["Offensive?"].value_counts()
                for label in ["Yes", "No"]:
                    try:
                        offensive_dict[label] = offensive_count[label].item()
                    except:
                        pass

                # Appends all of the statistics to the report.
                self.report[time_frame_labels[dataframe]] = [
                    sentiment_dict.get("positive"),
                    sentiment_dict.get("neutral"),
                    sentiment_dict.get("negative"), total, averages[0],
                    averages[1], offensive_dict["Yes"], offensive_dict["No"],
                    processing_time
                ]

                # Counts the frequency of each emotion in the dataset.
                emotion_dict = {"admiration": 0, "amusement": 0, "anger": 0,
                                "annoyance": 0, "approval": 0, "caring": 0,
                                "confusion": 0, "curiosity": 0, "desire": 0,
                                "disappointment": 0, "disapproval": 0,
                                "disgust": 0, "embarrassment": 0,
                                "excitement": 0, "fear": 0, "gratitude": 0,
                                "grief": 0, "joy": 0, "love": 0,
                                "nervousness": 0, "optimism": 0, "pride": 0,
                                "realization": 0, "relief": 0, "remorse": 0,
                                "sadness": 0, "surprise": 0, "neutral": 0
                }
                
                for index, value in current_subset["Emotion"].value_counts().items():
                    emotion_dict[index] = int(value)
                # Sorts the dictionary by the highest frequency value.
                emotion_dict = dict(
                    sorted(emotion_dict.items(),
                           key=lambda item: item[1],
                           reverse=True))
                # Converts the dictionary into a list of key, value pairs.
                for key, value in emotion_dict.items():
                    self.emotional_analysis[dataframe].append([key, value])

                # File names for each of the emotional analysis charts.
                chart_names = [
                    "emotions bar chart all", "emotions bar chart 30",
                    "emotions bar chart 7", "emotions bar chart 48"
                ]
                cloud_names = [
                    "emotions word cloud all", "emotions word cloud 30",
                    "emotions word cloud 7", "emotions word cloud 48"
                ]
                # Charts do not include the neutral emotion category.
                emotion_dict.pop("neutral")
                emotional_subset = Series(emotion_dict)
                # If data exists in this subset, generate charts for this data.
                if not emotional_subset.empty and emotional_subset.iloc[0] != 0:
                    self.generate_chart(emotional_subset,
                                        chart_names[dataframe],
                                        cloud_names[dataframe])
            else:  # If no data exists in this time range.
                self.report[time_frame_labels[dataframe]] = [
                    0, 0, 0, 0, 0, 0, 0, 0, " "
                ]

    def generate_chart(self, subset, bar_chart_name, word_cloud_name):
        """_summary_

        Args:
            subset (pandas DataFrames): Current subset to generate charts for.
            bar_chart_name (str): File name to save the bar chart as.
            word_cloud_name (str): File name to save the word cloud as.
        """
        # Generates a bar chart for the current DataFrame.
        # Sets chart's background colour as grey.
        fig = figure(facecolor="#3a3a3a")
        ax = fig.gca()
        ax.set_autoscale_on(False)
        # Takes the first 20 values.
        ax.bar(subset.iloc[:20].index.values, subset.iloc[:20])
        ax.axis([0, 20, 0, subset.max()])
        # Rotates the axis labels 90 degrees clockwise.
        ax.tick_params(axis="x", rotation=90, labelcolor="#ffffff")
        ax.tick_params(axis="y", rotation=90, labelcolor="#ffffff")

        # Styles the charts axes.
        ax.grid(axis="y", linewidth=0.5, alpha=0.65)
        ax.set_facecolor("#3a3a3a")

        # Save the bar chart as a PNG file.
        current_image_path = join(self.file_path, "Charts",
                                  bar_chart_name + ".png")
        fig.savefig(current_image_path, bbox_inches="tight")
        close(fig)

        # Rotates the bar chart 90 degrees clockwise.
        if exists(current_image_path):
            image = Image.open(current_image_path)
            rotated = image.rotate(-90, expand=True)
            rotated.save(current_image_path, "PNG")

        # Generates a word cloud for the current DataFrame.
        # Saves the word cloud as a PNG file.
        WordCloud(background_color="#3a3a3a",
                  max_words=50).generate_from_frequencies(subset).to_file(
                      join(self.file_path, "Charts", word_cloud_name + ".png"))

    def generate_files(self):
        """Outputs any processed data to CSV files."""
        # Store cleaned comments as CSV file.
        self.comments[[
            "Comment", "no_symbols", "no_punctuation_and_symbols", "no_urls",
            "no_urls2", "lower_case", "expanded_contractions",
            "expanded_contractions2", "spelling", "spelling2", "tokens",
            "lemmas", "lemmas_no_stopwords"
        ]].to_csv(join(self.file_path, "cleaned_data.csv"), index=False)

        # Store report as CSV file.
        self.report.to_csv(join(self.file_path, "report.csv"), index=False)

        # Store processed comments as CSV files.
        file_names = [
            "processed_all", "processed_30_days", "processed_7_days",
            "processed_48_hours"
        ]
        for comments_set in range(len(self.comments_time_ranges)):
            self.comments_time_ranges[comments_set][[
                "Comment", "Time", "Polarity", "Sentiment", "Subjectivity",
                "Offensive?", "Emotion"
            ]].to_csv(join(self.file_path, file_names[comments_set] + ".csv"), index=False)

        # Store emotional analysis data as pickle file.
        with open(join(self.file_path, "emotional_analysis.pkl"), "wb") as file:
            dump((self.emotional_analysis), file)

        # Store vectorizer data as pickle file.
        with open(join(self.file_path, "vectorizer_data.pkl"), "wb") as file:
            dump((self.vectorizer_data), file)

    def generate_charts(self):
        """Generates bar charts and word clouds based on vectorizer data."""
        print("Generating Charts")
        # Bar chart file names for each vectorizer and time range subset.
        chart_names = [
            ["frequency unigram bar chart all", "frequency unigram bar chart 30",
            "frequency unigram bar chart 7", "frequency unigram bar chart 48"],
                       
            ["importance unigram bar chart all", "importance unigram bar chart 30",
             "importance unigram bar chart 7", "importance unigram bar chart 48"],
            
            ["frequency bigram bar chart all", "frequency bigram bar chart 30",
             "frequency bigram bar chart 7", "frequency bigram bar chart 48"],
            
            ["importance bigram bar chart all", "importance bigram bar chart 30",
             "importance bigram bar chart 7", "importance bigram bar chart 48"],
            
            ["frequency trigram bar chart all", "frequency trigram bar chart 30",
             "frequency trigram bar chart 7", "frequency trigram bar chart 48"],
            
            ["importance trigram bar chart all", "importance trigram bar chart 30",
             "importance trigram bar chart 7", "importance trigram bar chart 48"]
        ]
        # Word cloud file names for each vectorizer and time range subset.
        cloud_names = [
            ["frequency unigram word cloud all", "frequency unigram word cloud 30",
             "frequency unigram word cloud 7", "frequency unigram word cloud 48"],
            
            ["importance unigram word cloud all", "importance unigram word cloud 30",
             "importance unigram word cloud 7", "importance unigram word cloud 48"],
            
            ["frequency bigram word cloud all", "frequency bigram word cloud 30",
             "frequency bigram word cloud 7", "frequency bigram word cloud 48"],
            
            ["importance bigram word cloud all", "importance bigram word cloud 30",
             "importance bigram word cloud 7", "importance bigram word cloud 48"],
            
            ["frequency trigram word cloud all", "frequency trigram word cloud 30",
             "frequency trigram word cloud 7", "frequency trigram word cloud 48"],
            
            ["importance trigram word cloud all", "importance trigram word cloud 30",
             "importance trigram word cloud 7", "importance trigram word cloud 48"]
        ]
        # For each vectorizer n-gram.
        for data_type in range(len(chart_names) - 1, -1, -1):
            for time_range in range(0, 4):  # For each time range.
                # The current vectorizer n-gram and time range data subset.
                vectorizer_subset = self.vectorizer_data[time_range][data_type]
                if not vectorizer_subset.empty:  # If data exists in this subset.
                    self.generate_chart(vectorizer_subset,
                                        chart_names[data_type][time_range],
                                        cloud_names[data_type][time_range])

    def read_from_files(self):
        """Reads in a previously processed dataset."""
        with open(join(self.file_path, "title.pkl"), "rb") as file:
            self.title = load(file)  # Title of the video.
        self.comments = read_csv(join(self.file_path, "data.csv"))  # Raw comments.
        self.report = read_csv(join(self.file_path, "report.csv"))  # Report.

        self.generate_empty_dataframes()
        self.comments_time_ranges = [
            self.comments, self.comments_30_days,
            self.comments_7_days, self.comments_48_hours
        ]

        # Reads in processed comments for each time range.
        file_names = [
            "processed_all", "processed_30_days",
            "processed_7_days", "processed_48_hours"
        ]
        for comments_set in range(len(self.comments_time_ranges)):
            self.comments_time_ranges[comments_set] = read_csv(
                join(self.file_path, file_names[comments_set] + ".csv"))

        # Reads in emotional analysis data.
        with open(join(self.file_path, "emotional_analysis.pkl"), "rb") as file:
            self.emotional_analysis = load(file)

        # Reads in vectorizer data.
        with open(join(self.file_path, "vectorizer_data.pkl"), "rb") as file:
            self.vectorizer_data = load(file)


if __name__ == "__main__":
    Main()
