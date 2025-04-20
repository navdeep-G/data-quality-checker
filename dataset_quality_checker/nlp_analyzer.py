from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from collections import Counter
from langdetect import detect
from sklearn.metrics.pairwise import cosine_similarity
import gensim.downloader as api
import pandas as pd
from textblob import TextBlob
from rake_nltk import Rake
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
import spacy
from nltk.tokenize import sent_tokenize
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.tree import Tree
from difflib import SequenceMatcher
import textstat
import matplotlib.pyplot as plt
import seaborn as sns


class NLPAnalyzer:
    def __init__(self, data):
        self.data = data
        self.model = self._load_word2vec_model()

    @staticmethod
    def _load_word2vec_model():
        """
        Load Word2Vec model (cached for reuse).

        Returns:
            model: Pre-trained Word2Vec model.
        """
        print("🔄 Loading Word2Vec model (Google News 300)...")
        return api.load('word2vec-google-news-300')

    def correct_spelling(self, column):
        """
        Correct spelling errors in a text column.

        Args:
            column (str): The text column to correct.

        Returns:
            pd.Series: Text column with corrected spelling.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist in the dataset.")
        if not pd.api.types.is_string_dtype(self.data[column]):
            raise ValueError(f"Column '{column}' must be of string type.")

        return self.data[column].apply(lambda x: str(TextBlob(x).correct()) if pd.notnull(x) else x)

    def topic_modeling(self, column, n_topics=5, n_top_words=5):
        """
        Perform topic modeling on a text column using Latent Dirichlet Allocation (LDA).

        Args:
            column (str): The text column to analyze.
            n_topics (int): Number of topics to identify.
            n_top_words (int): Number of top words per topic.

        Returns:
            list: A list of topics with top words.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist in the dataset.")
        if not pd.api.types.is_string_dtype(self.data[column]):
            raise ValueError(f"Column '{column}' must be of string type.")

        text_data = self.data[column].dropna().astype(str).tolist()
        vectorizer = CountVectorizer(stop_words='english')
        text_matrix = vectorizer.fit_transform(text_data)

        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda.fit(text_matrix)

        feature_names = vectorizer.get_feature_names_out()
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
            topics.append(f"Topic {topic_idx + 1}: {' '.join(top_words)}")
        return topics

    def check_text_length(self, column, max_length=255):
        return self.data[self.data[column].str.len() > max_length]

    def category_feature_interaction(self, categorical_column, numeric_column):
        """
        Analyze interaction between categorical and numeric columns.
        """
        if categorical_column not in self.data.columns or numeric_column not in self.data.columns:
            raise ValueError("One or both specified columns do not exist.")
        interaction_stats = self.data.groupby(categorical_column)[numeric_column].describe()
        return interaction_stats

    def word_length_distribution(self, column):
        """
        Compute and visualize the distribution of word lengths.

        Args:
            column (str): The text column to analyze.

        Returns:
            pd.Series: Word length frequency distribution.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist.")

        word_lengths = self.data[column].dropna().apply(lambda x: [len(word) for word in x.split()])
        all_lengths = [length for sublist in word_lengths for length in sublist]

        plt.figure(figsize=(10, 5))
        sns.histplot(all_lengths, bins=20, kde=True)
        plt.title("Word Length Distribution")
        plt.xlabel("Word Length")
        plt.ylabel("Frequency")
        plt.show()

        return pd.Series(all_lengths).value_counts().sort_index()

    def sentence_length_distribution(self, column):
        """
        Compute and visualize sentence length distribution.

        Args:
            column (str): The text column to analyze.

        Returns:
            pd.Series: Sentence length frequency distribution.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist.")

        nltk.download("punkt", quiet=True)

        sentence_lengths = self.data[column].dropna().apply(lambda x: [len(sent.split()) for sent in sent_tokenize(x)])
        all_lengths = [length for sublist in sentence_lengths for length in sublist]

        plt.figure(figsize=(10, 5))
        sns.histplot(all_lengths, bins=20, kde=True)
        plt.title("Sentence Length Distribution")
        plt.xlabel("Number of Words in Sentence")
        plt.ylabel("Frequency")
        plt.show()

        return pd.Series(all_lengths).value_counts().sort_index()

    def character_count_distribution(self, column):
        """
        Compute and visualize the distribution of character counts.

        Args:
            column (str): The text column to analyze.

        Returns:
            pd.Series: Character count distribution.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist.")

        char_counts = self.data[column].dropna().apply(len)

        plt.figure(figsize=(10, 5))
        sns.histplot(char_counts, bins=20, kde=True)
        plt.title("Character Count Distribution")
        plt.xlabel("Character Count")
        plt.ylabel("Frequency")
        plt.show()

        return char_counts.value_counts().sort_index()

    def n_gram_distribution(self, column, n=2, top_n=20):
        """
        Identify most common n-grams in text data.

        Args:
            column (str): The text column to analyze.
            n (int): Size of the n-gram (2 for bigrams, 3 for trigrams).
            top_n (int): Number of top n-grams to display.

        Returns:
            dict: Most common n-grams with counts.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist.")

        vectorizer = CountVectorizer(ngram_range=(n, n), stop_words="english")
        n_grams = vectorizer.fit_transform(self.data[column].dropna())

        n_gram_counts = dict(zip(vectorizer.get_feature_names_out(), n_grams.toarray().sum(axis=0)))
        sorted_n_grams = dict(sorted(n_gram_counts.items(), key=lambda item: item[1], reverse=True)[:top_n])

        return sorted_n_grams

    def check_text_redundancy(self, column, n=3):
        """
        Identify commonly repeated phrases in text data.

        Args:
            column (str): The text column to analyze.
            n (int): Minimum number of occurrences to consider redundancy.

        Returns:
            dict: Repeated phrases and their counts.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist.")

        phrases = self.data[column].dropna().str.split().explode()
        phrase_counts = Counter(phrases)

        return {phrase: count for phrase, count in phrase_counts.items() if count >= n}

    def named_entity_analysis(self, column, model='spacy', entity_types=None, return_frequency=False):
        """
        Perform Named Entity Recognition (NER) and optionally compute entity frequency.

        Args:
            column (str): The text column to analyze.
            model (str): NLP model for NER ('spacy' or 'nltk').
            entity_types (list, optional): List of entity types to filter (e.g., ['PERSON', 'ORG', 'DATE']).
            return_frequency (bool): If True, return entity frequency instead of per-row entity extraction.

        Returns:
            - If return_frequency=False: A pandas Series with extracted named entities for each row.
            - If return_frequency=True: A dictionary with entity frequency counts.

        Raises:
            ValueError: If the column does not exist, is not string type, or invalid model is specified.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist in the dataset.")
        if not pd.api.types.is_string_dtype(self.data[column]):
            raise ValueError(f"Column '{column}' must be of string type.")
        if model not in ['spacy', 'nltk']:
            raise ValueError("Invalid model. Choose from 'spacy' or 'nltk'.")

        entity_counts = Counter()
        results = []

        if model == 'spacy':
            nlp = spacy.load('en_core_web_sm')

            def extract_entities_spacy(text):
                if pd.isnull(text) or not text.strip():
                    return {}
                doc = nlp(text)
                entities = {ent.label_: [] for ent in doc.ents}
                for ent in doc.ents:
                    if not entity_types or ent.label_ in entity_types:
                        entities.setdefault(ent.label_, []).append(ent.text)
                        entity_counts[ent.text] += 1  # Count occurrences
                return entities

            results = self.data[column].apply(extract_entities_spacy)

        elif model == 'nltk':
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('maxent_ne_chunker', quiet=True)
            nltk.download('words', quiet=True)

            def extract_entities_nltk(text):
                if pd.isnull(text) or not text.strip():
                    return {}
                tokens = word_tokenize(text)
                tagged = pos_tag(tokens)
                chunked = ne_chunk(tagged)
                entities = {}
                for subtree in chunked:
                    if isinstance(subtree, Tree):
                        entity_label = subtree.label()
                        entity_text = " ".join([token for token, pos in subtree.leaves()])
                        if not entity_types or entity_label in entity_types:
                            entities.setdefault(entity_label, []).append(entity_text)
                            entity_counts[entity_text] += 1  # Count occurrences
                return entities

            results = self.data[column].apply(extract_entities_nltk)

        return dict(sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)) if return_frequency else results

    def text_tokenization_analysis(self, column, level="word", n_gram=None, language="english"):
        """
        Perform text tokenization and n-gram analysis.

        Args:
            column (str): The text column to tokenize.
            level (str): "word" for word tokenization, "sentence" for sentence tokenization.
            n_gram (int): Set to an integer (e.g., 2 for bigrams) to compute n-grams.
            language (str): Language for tokenization.

        Returns:
            pd.Series or dict: Tokenized text or n-gram counts.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist.")

        nltk.download("punkt", quiet=True)

        text_data = self.data[column].dropna().astype(str)

        if n_gram:
            vectorizer = CountVectorizer(ngram_range=(n_gram, n_gram), stop_words="english")
            n_grams = vectorizer.fit_transform(text_data)
            n_gram_counts = dict(zip(vectorizer.get_feature_names_out(), n_grams.toarray().sum(axis=0)))
            return dict(sorted(n_gram_counts.items(), key=lambda x: x[1], reverse=True))

        if level == "word":
            return text_data.apply(lambda x: word_tokenize(x, language=language))

        elif level == "sentence":
            return text_data.apply(lambda x: sent_tokenize(x, language=language))

        else:
            raise ValueError("Invalid level. Choose 'word' or 'sentence'.")

    def analyze_text_complexity(self, column):
        """
        Analyze text complexity using readability scores, text length, and compression ratio.

        Args:
            column (str): The text column to analyze.

        Returns:
            pd.DataFrame: Readability scores, text length statistics, and compression ratios.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist.")

        def compute_metrics(text):
            if pd.isnull(text):
                return None
            words = text.split()
            unique_words = set(words)
            return {
                "Text_Length": len(text),
                "Word_Count": len(words),
                "Unique_Word_Ratio": len(unique_words) / len(words) if words else 0,
                "Flesch_Reading_Ease": textstat.flesch_reading_ease(text),
                "SMOG_Index": textstat.smog_index(text),
                "Dale_Chall_Score": textstat.dale_chall_readability_score(text),
            }

        scores = self.data[column].apply(compute_metrics).dropna()
        return pd.DataFrame(scores.tolist(), index=self.data.index)

    def analyze_text_keywords(self, column, method="rake", top_n=10, exclude_stopwords=True):
        """
        Extract keywords or analyze word frequency.

        Args:
            column (str): The text column to analyze.
            method (str): Keyword extraction method ("rake" or "word_freq").
            top_n (int): Number of top results to return.
            exclude_stopwords (bool): Whether to exclude stopwords in word frequency analysis.

        Returns:
            dict: Extracted keywords or word frequency distribution.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist.")

        text_data = self.data[column].dropna().astype(str)

        if method == "rake":
            rake = Rake()
            rake.extract_keywords_from_text(" ".join(text_data))
            return {phrase: score for phrase, score in rake.get_word_degrees().items()[:top_n]}

        if method == "word_freq":
            words = text_data.str.split().explode()
            if exclude_stopwords:
                stop_words = set(stopwords.words("english"))
                words = words[~words.isin(stop_words)]
            return dict(Counter(words).most_common(top_n))

        raise ValueError("Invalid method. Choose 'rake' or 'word_freq'.")

    def analyze_text_similarity(self, column, similarity_method="word2vec", similarity_threshold=0.8, max_features=100):
        """
        Perform comprehensive text similarity analysis by combining:
        - Word2Vec similarity (pairwise comparisons)
        - High-similarity text pairs detection
        - TF-IDF vectorization for structured text comparison

        Args:
            column (str): The text column to analyze.
            similarity_method (str): Similarity method ("word2vec", "tfidf", or "cosine").
            similarity_threshold (float): Threshold for similarity detection (only for text pairs).
            max_features (int): Maximum features for TF-IDF vectorization.

        Returns:
            dict: Contains:
                - 'similar_text_pairs': List of high-similarity text pairs.
                - 'word2vec_similarity': Pairwise similarity scores (if applicable).
                - 'tfidf_matrix': DataFrame of TF-IDF vector representations.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist.")
        if not pd.api.types.is_string_dtype(self.data[column]):
            raise ValueError(f"Column '{column}' must be of string type.")

        results = {}

        # Step 1: Find highly similar text pairs
        results["similar_text_pairs"] = self._find_text_pairs(column, similarity_threshold)

        # Step 2: Compute word2vec similarity (if selected)
        if similarity_method == "word2vec":
            results["word2vec_similarity"] = self._compute_text_similarity(column, method="word2vec")

        # Step 3: Compute TF-IDF vectorization
        results["tfidf_matrix"] = self._text_vectorization_analysis(column, method="tfidf", max_features=max_features)

        return results

    def analyze_topics_and_keywords(self, column, n_topics=5, n_top_words=5, keyword_method="rake", top_keywords=10,
                                    ngram_n=2, top_ngrams=20):
        """
        Perform a comprehensive topic and keyword extraction analysis, combining:
        - Topic modeling (LDA) to identify dominant themes.
        - Keyword extraction using RAKE.
        - N-gram analysis to find common multi-word expressions.

        Args:
            column (str): The text column to analyze.
            n_topics (int): Number of topics to extract via LDA.
            n_top_words (int): Number of top words per topic.
            keyword_method (str): Keyword extraction method ('rake' or 'word_freq').
            top_keywords (int): Number of keywords to extract.
            ngram_n (int): Size of the n-gram (e.g., 2 for bigrams).
            top_ngrams (int): Number of top n-grams to return.

        Returns:
            dict: Contains:
                - 'topics': Extracted topic descriptions.
                - 'keywords': Extracted keywords with scores.
                - 'ngrams': Most common n-grams with their counts.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist.")
        if not pd.api.types.is_string_dtype(self.data[column]):
            raise ValueError(f"Column '{column}' must be of string type.")

        # Step 1: Extract topics using LDA

        # Step 2: Extract keywords using RAKE or word frequency

        # Step 3: Identify common n-grams (bigrams, trigrams, etc.)

        results = {"topics": self.topic_modeling(column, n_topics=n_topics, n_top_words=n_top_words),
                   "keywords": self.analyze_text_keywords(column, method=keyword_method, top_n=top_keywords),
                   "ngrams": self.n_gram_distribution(column, n=ngram_n, top_n=top_ngrams)}

        return results

    def analyze_deep_linguistics(self, column, ner_model="spacy"):
        """
        Perform deep linguistic analysis on a text column, including:
        - Named entity recognition (NER) with frequency counts.
        - Part-of-speech (POS) distribution.
        - Lexical diversity measurement (overall dataset-level).
        - Language detection for multilingual data handling.

        Args:
            column (str): The text column to analyze.
            ner_model (str): NLP model for NER ('spacy' or 'nltk').

        Returns:
            dict: Contains:
                - 'named_entities': Named entity frequency counts.
                - 'pos_distribution': Count of parts of speech.
                - 'lexical_diversity': Overall lexical diversity score.
                - 'detected_languages': Language distribution across text.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist.")
        if not pd.api.types.is_string_dtype(self.data[column]):
            raise ValueError(f"Column '{column}' must be of string type.")

        results = {}

        # Step 1: Named Entity Recognition (NER) - Counts entities
        results["named_entities"] = self.named_entity_analysis(column, model=ner_model, return_frequency=True)

        # Step 2: Part-of-Speech (POS) distribution
        results["pos_distribution"] = self._pos_distribution(column)

        # Step 3: Lexical diversity (ratio of unique words to total words in dataset)
        results["lexical_diversity"] = self._lexical_diversity(column, mode="overall")

        # Step 4: Language detection
        results["detected_languages"] = self._detect_language(column).value_counts().to_dict()

        return results

    def detect_text_variability(self, column, entity_type="ORG", language="english"):
        """
        Evaluate variability of text data by combining:
        - Named entity consistency (checks variations in entity usage).
        - Stopword density measurement.
        - Sentiment distribution analysis.
        - Subjectivity scoring.

        Args:
            column (str): The text column to analyze.
            entity_type (str): Named entity type to check consistency (e.g., 'ORG', 'PERSON', 'GPE').
            language (str): Language for stopword detection.

        Returns:
            dict: Contains:
                - 'entity_consistency': Entities with inconsistent spellings or casing.
                - 'stopword_counts': Stopword frequency per row.
                - 'sentiment_distribution': Distribution of sentiment scores.
                - 'subjectivity_scores': Subjectivity levels across the dataset.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist.")
        if not pd.api.types.is_string_dtype(self.data[column]):
            raise ValueError(f"Column '{column}' must be of string type.")

        results = {}

        # Step 1: Named Entity Consistency - Detect inconsistencies in entity usage
        results["entity_consistency"] = self._named_entity_consistency(column, entity_type=entity_type)

        # Step 2: Stopword Count - Measures how much of the text is made up of stopwords
        results["stopword_counts"] = self._count_stopwords(column, language=language).to_list()

        # Step 3: Sentiment Analysis - Get sentiment distribution (negative, neutral, positive)
        results["sentiment_distribution"] = self._sentiment_analysis(column, return_distribution=True)

        # Step 4: Subjectivity Analysis - Determines how much of the text is opinion-based
        results["subjectivity_scores"] = self._subjectivity_analysis(column).to_list()

        return results

    def analyze_text_complexity_overview(self, column):
        """
        Perform a comprehensive readability and text complexity analysis, including:
        - Readability scores and word-level statistics
        - Word length distribution
        - Sentence length distribution

        Args:
            column (str): The text column to analyze.

        Returns:
            dict: Contains:
                - 'complexity_scores': DataFrame with readability metrics and text-level stats.
                - 'word_length_distribution': Series with word length frequencies.
                - 'sentence_length_distribution': Series with sentence length frequencies.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist.")
        if not pd.api.types.is_string_dtype(self.data[column]):
            raise ValueError(f"Column '{column}' must be of string type.")

        results = {}

        # Step 1: Readability scores, unique word ratios, and text length
        results["complexity_scores"] = self.analyze_text_complexity(column)

        # Step 2: Word length histogram and frequency
        results["word_length_distribution"] = self.word_length_distribution(column)

        # Step 3: Sentence length histogram and frequency
        results["sentence_length_distribution"] = self.sentence_length_distribution(column)

        return results

    def clean_and_standardize_text(self, column, max_length=255):
        """
        Perform automated text cleaning and standardization:
        - Detect language of each entry.
        - Identify rows exceeding a max character length.
        - Apply spelling correction to text.

        Args:
            column (str): The text column to process.
            max_length (int): Maximum allowed character length for text.

        Returns:
            dict: Contains:
                - 'language_detection': Series of detected language codes.
                - 'long_texts': DataFrame of rows with excessively long text.
                - 'corrected_text': Series with spelling-corrected text.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist.")
        if not pd.api.types.is_string_dtype(self.data[column]):
            raise ValueError(f"Column '{column}' must be of string type.")

        results = {}

        # Step 1: Detect language
        results["language_detection"] = self._detect_language(column)

        # Step 2: Find long text entries
        results["long_texts"] = self.check_text_length(column, max_length=max_length)

        # Step 3: Correct spelling
        results["corrected_text"] = self.correct_spelling(column)

        return results

    def analyze_text_redundancy_structure(self, column, redundancy_threshold=3, ngram_n=3, top_ngrams=20):
        """
        Analyze text for structural patterns and redundancies:
        - Identify overused phrases or words.
        - Tokenize text into sentences to observe structure and flow.
        - Extract most common n-grams (e.g., trigrams) for repetitive pattern detection.

        Args:
            column (str): The text column to analyze.
            redundancy_threshold (int): Minimum number of repetitions for a word/phrase to be flagged.
            ngram_n (int): Size of the n-gram (e.g., 3 for trigrams).
            top_ngrams (int): Number of top n-grams to return.

        Returns:
            dict: Contains:
                - 'redundant_phrases': Repeated phrases with their counts.
                - 'sentence_tokens': Tokenized text as lists of sentences.
                - 'common_ngrams': Most frequent n-grams in the dataset.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist.")
        if not pd.api.types.is_string_dtype(self.data[column]):
            raise ValueError(f"Column '{column}' must be of string type.")

        results = {}

        # Step 1: Detect repeated words/phrases
        results["redundant_phrases"] = self.check_text_redundancy(column, n=redundancy_threshold)

        # Step 2: Sentence tokenization
        results["sentence_tokens"] = self.text_tokenization_analysis(column, level="sentence")

        # Step 3: Top n-grams (e.g., trigrams)
        results["common_ngrams"] = self.n_gram_distribution(column, n=ngram_n, top_n=top_ngrams)

        return results

    def semantic_search_analysis(self, column, similarity_threshold=0.8, max_features=100):
        """
        Perform advanced semantic search analysis:
        - Vectorize text using TF-IDF.
        - Compute cosine similarity matrix.
        - Identify similar text pairs exceeding a similarity threshold.

        Args:
            column (str): The text column to analyze.
            similarity_threshold (float): Minimum cosine similarity to consider two entries similar.
            max_features (int): Maximum features for TF-IDF vectorization.

        Returns:
            dict: Contains:
                - 'tfidf_vectors': TF-IDF feature matrix (DataFrame).
                - 'similarity_matrix': Cosine similarity matrix (DataFrame).
                - 'similar_text_pairs': List of (text1, text2, similarity_score).
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist.")
        if not pd.api.types.is_string_dtype(self.data[column]):
            raise ValueError(f"Column '{column}' must be of string type.")

        results = {}

        # Step 1: TF-IDF Vectorization
        results["tfidf_vectors"] = self._text_vectorization_analysis(column, method="tfidf", max_features=max_features)

        # Step 2: Cosine Similarity Matrix
        similarity_matrix = self._text_vectorization_analysis(column, method="similarity_matrix",
                                                             max_features=max_features)
        results["similarity_matrix"] = similarity_matrix

        # Step 3: Similar Text Pairs using thresholded cosine similarity
        results["similar_text_pairs"] = self._find_text_pairs(column, similarity_threshold)

        return results

    def analyze_emotional_tone(self, column):
        """
        Analyze sentiment and emotional tone of text by combining:
        - Sentiment polarity distribution.
        - Subjectivity scoring (opinion vs. fact).
        - Lexical diversity per row.

        Args:
            column (str): The text column to analyze.

        Returns:
            dict: Contains:
                - 'sentiment_distribution': Histogram of sentiment polarity values.
                - 'subjectivity_scores': List of subjectivity scores per row.
                - 'lexical_diversity_scores': List of lexical diversity ratios per row.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist.")
        if not pd.api.types.is_string_dtype(self.data[column]):
            raise ValueError(f"Column '{column}' must be of string type.")

        results = {}

        # Step 1: Sentiment polarity histogram
        results["sentiment_distribution"] = self._sentiment_analysis(column, return_distribution=True)

        # Step 2: Subjectivity scores (0 = objective, 1 = subjective)
        results["subjectivity_scores"] = self._subjectivity_analysis(column).tolist()

        # Step 3: Lexical diversity (unique words / total words per row)
        results["lexical_diversity_scores"] = self._lexical_diversity(column, mode="row").tolist()

        return results

    def _find_text_pairs(self, column, similarity_threshold=0.8):
        """
        Identify pairs of text entries in a column with high similarity.

        Args:
            column (str): Name of the column to check.
            similarity_threshold (float): Threshold for similarity (0 to 1).

        Returns:
            list of tuples: Pairs of similar text entries.
        """

        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist in the dataset.")
        text_data = self.data[column].dropna().astype(str).tolist()
        similar_pairs = []

        for i, text1 in enumerate(text_data):
            for j, text2 in enumerate(text_data):
                if i < j:  # Avoid duplicate comparisons
                    similarity = SequenceMatcher(None, text1, text2).ratio()
                    if similarity >= similarity_threshold:
                        similar_pairs.append((text1, text2, similarity))
        return similar_pairs

    def _text_vectorization_analysis(self, column, method="tfidf", max_features=100):
        """
        Perform text vectorization and similarity analysis.

        Args:
            column (str): The text column to analyze.
            method (str): "tfidf" for TF-IDF vectors, "cosine" for cosine similarity, "similarity_matrix" for full similarity matrix.
            max_features (int): Max number of features for vectorization.

        Returns:
            - If method="tfidf": Returns a TF-IDF DataFrame.
            - If method="cosine": Returns cosine similarity scores.
            - If method="similarity_matrix": Returns a similarity matrix DataFrame.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist.")

        text_data = self.data[column].dropna().astype(str)
        vectorizer = TfidfVectorizer(max_features=max_features)
        tfidf_matrix = vectorizer.fit_transform(text_data)

        if method == "tfidf":
            return pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

        elif method == "cosine":
            return cosine_similarity(tfidf_matrix)

        elif method == "similarity_matrix":
            similarity_matrix = cosine_similarity(tfidf_matrix)
            return pd.DataFrame(similarity_matrix, index=text_data.index, columns=text_data.index)

        else:
            raise ValueError("Invalid method. Choose 'tfidf', 'cosine', or 'similarity_matrix'.")

    def _compute_text_similarity(self, column, method="word2vec", word1=None, word2=None, threshold=0.2):
        """
        Compute text similarity using different methods.

        Args:
            column (str): The text column to analyze.
            method (str): Similarity method ("word2vec", "tfidf", or "cosine").
            word1 (str, optional): First word for Word2Vec similarity (required for "word2vec").
            word2 (str, optional): Second word for Word2Vec similarity (required for "word2vec").
            threshold (float, optional): Cosine similarity threshold for anomaly detection.

        Returns:
            Depending on method:
            - "word2vec": Cosine similarity scores.
            - "tfidf": Cosine similarity matrix.
            - "cosine": Outliers detected based on similarity.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist.")
        text_data = self.data[column].dropna().astype(str)

        if method == "word2vec":
            if not word1 or not word2:
                raise ValueError("Both word1 and word2 must be specified for Word2Vec similarity.")
            return self.data[column].apply(lambda x: self.model.similarity(word1, word2) if pd.notnull(x) else None)

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(text_data)

        if method == "tfidf":
            return pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

        if method == "cosine":
            similarity_matrix = cosine_similarity(tfidf_matrix)
            avg_similarity = similarity_matrix.mean(axis=1)
            anomalies = text_data[avg_similarity < threshold]
            return anomalies.tolist()

        raise ValueError("Invalid method. Choose 'word2vec', 'tfidf', or 'cosine'.")

    def _pos_distribution(self, column):
        """
        Compute the distribution of parts of speech (POS) in text data.

        Args:
            column (str): The text column to analyze.

        Returns:
            dict: POS tag counts.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist.")

        # Load pre-trained spaCy NLP model
        nlp = spacy.load("en_core_web_sm")
        pos_counts = Counter()

        # Process each text entry in the column
        for text in self.data[column].dropna():
            doc = nlp(text)
            pos_counts.update([token.pos_ for token in doc])  # Count POS tags

        # Plot POS distribution
        plt.figure(figsize=(10, 5))
        sns.barplot(x=list(pos_counts.keys()), y=list(pos_counts.values()))
        plt.title("Part-of-Speech (POS) Distribution")
        plt.xlabel("POS Tag")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.show()

        return dict(pos_counts)

    def _lexical_diversity(self, column, mode="row"):
        """
        Compute lexical diversity (ratio of unique words to total words).

        Args:
            column (str): The text column to analyze.
            mode (str): "row" for row-wise diversity (default), "overall" for dataset-wide diversity.

        Returns:
            float or pd.Series:
                - If mode="row": Returns a Pandas Series with lexical diversity scores for each row.
                - If mode="overall": Returns a single float representing lexical diversity for the entire dataset.

        Raises:
            ValueError: If the column does not exist or mode is invalid.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist.")

        # Tokenize and flatten words for row-wise computation
        if mode == "row":
            return self.data[column].dropna().apply(
                lambda text: len(set(text.split())) / len(text.split()) if text.strip() else 0
            )

        # Compute overall lexical diversity for the entire dataset
        elif mode == "overall":
            all_words = self.data[column].dropna().str.split().explode()
            unique_words = set(all_words)
            return len(unique_words) / len(all_words) if len(all_words) > 0 else 0

        else:
            raise ValueError(
                "Invalid mode. Choose 'row' for per-row diversity or 'overall' for dataset-wide diversity.")

    def _detect_language(self, column):
        """
        Detect the language of text data.

        Args:
            column (str): The text column to analyze.

        Returns:
            pd.Series: Detected language codes.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist.")

        return self.data[column].dropna().apply(lambda x: detect(x))

    def _named_entity_consistency(self, column, entity_type="ORG"):
        """
        Detect inconsistent usage of named entities.

        Args:
            column (str): The text column to analyze.
            entity_type (str): Type of named entity (e.g., 'ORG', 'PERSON', 'GPE').

        Returns:
            dict: Entities with inconsistent casing or spelling variations.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist.")

        nlp = spacy.load("en_core_web_sm")
        entity_dict = {}

        for text in self.data[column].dropna():
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ == entity_type:
                    entity_dict.setdefault(ent.text.lower(), set()).add(ent.text)

        return {k: list(v) for k, v in entity_dict.items() if len(v) > 1}

    def _count_stopwords(self, column, language="english"):
        stop_words = set(stopwords.words(language))
        return self.data[column].apply(lambda x: sum(1 for w in str(x).split() if w.lower() in stop_words))

    def _sentiment_analysis(self, column, return_distribution=False):
        """
        Perform sentiment analysis on a text column.

        Args:
            column (str): The text column to analyze.
            return_distribution (bool): If True, return a distribution of sentiment scores.

        Returns:
            pd.Series or dict:
                - If return_distribution=False: Returns a Series with sentiment polarity scores.
                - If return_distribution=True: Returns a histogram of sentiment scores.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist.")

        sentiments = self.data[column].dropna().apply(lambda x: TextBlob(x).sentiment.polarity)

        if return_distribution:
            return dict(sentiments.value_counts().sort_index())

        return sentiments

    def _subjectivity_analysis(self, column):
        """
        Compute subjectivity scores (0 = objective, 1 = highly subjective).

        Args:
            column (str): The text column to analyze.

        Returns:
            pd.Series: Subjectivity scores.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist.")

        return self.data[column].dropna().apply(lambda x: TextBlob(x).sentiment.subjectivity)