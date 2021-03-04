####################################
### Neccessary Import Statements ###
####################################
import pandas as pd
import numpy as np
import re
from connect_bucket import Connect_Bucket
import data_cleaning
from spacy.matcher import Matcher

##############################################
### Write the Class That Identifies Errors ###
##############################################
class Error_Identifier:
    """
    Class Purpose
    -------------
    To identify and categorize errors in recognized entity from SpaCy model and output a
    Pandas DF of categorical data associated with error types.

    Attributes
    ----------
    identified_errors_table - (DataFrame)
    error_example - (Dict)

    Features:
    url - (str)

    frag - (int) count of fragmentatation errors found for a given article
    sos_frag - (int) count of sos frag errors for a given article
    num_frag
    title_colon_frag
    title_prefix_frag

    concat
    sos_concat
    noun_ent_concat
    company_product_concat
    conj_adv_concat
    interior_ent_concat
    contractional_concat
    comma_list_concat
    comma_ent_concat
    sports_concat
    athlete_pos_concat
    team_score_rank_concat
    hyphen_concat
    colon_concat
    diseases_frag()

    disambig - (int) sum of all disambig errors
    noun_ent_ambig
    punct_ambig
    sports_nickname_ambig
    mixed_casing_ambig

    strict - Both entity text and label correct
    exact - Entity text correct but not necessarily label
    partial - Entity text partially correct
    spurious -


    TP -
    FP -
    FN -
    TN -

    Methods
    -------

    is_Frag(ent Entity, List golden_annotation):
    returns Boolean

    is_Concat(ent Entity, List golden_annotation):
    returns Boolean

    is_Disamb(ent Entity, List golden_annotation):
    returns Boolean

    Init Parameters
    ---------------

    References
    ----------
    1.
    """

    table_schema = [
        'art_title',
        'ner_process',
        'tp',
        'fp',
        'fn',
        'frag',
        'sos_frag',
        'num_frag',
        'title_colon_frag',
        'title_prefix_frag',
        'concat',
        'sos_concat',
        'noun_ent_concat',
        'conj_adv_concat',
        'interior_ent_concat',
        'contractional_concat',
        'comma_list_concat',
        'comma_ent_frag',
        'sports_concat',
        'athlete_pos_concat',
        'team_score_rank_concat',
        'hyphen_concat',
        'one_hyphen_concat',
        'colon_concat',
        'diseases_frag',
        'spurious',
        'missing'
    ]
    identified_errors_table = pd.DataFrame(columns = table_schema)
    # error_examples = {
    # 	'sos_frag': [],
    # 	'num_frag': [],
    # 	'title_colon_frag': [],
    # 	'title_prefix_frag': [],
    # 	'sos_concat': [],
    # 	'noun_ent_concat': [],
    # 	'conj_adv_concat': [],
    # 	'interior_ent_concat': [],
    # 	'contractional_concat': [],
    # 	'comma_list_concat': [],
    # 	'comma_ent_concat': [],
    # 	'sports_concat': [],
    # 	'athlete_pos_concat': [],
    # 	'team_score_rank_concat': [],
    # 	'hyphen_concat': [],
    # 	'one_hyphen_concat': [],
    # 	'colon_concat': [],
    # 	'diseases_frag': []
    # }
    # creates an instance of Connect_Bucket that can be used to access the s3 bucket
    connect_bucket = Connect_Bucket()
    # directly obtaining the saved error examples
    error_examples = connect_bucket.get_err_examples()
    honorifics = [
        'Mr',
        'Mrs',
        'Miss',
        'Ms',
        'Dr',
        'Professor',
        'Rabbi',
        'Canon',
        'Dame',
        'Chief',
        'Sister',
        'Brother',
        'Reverend',
        'Major',
        'Sir',
        'Lord',
        'Lady',
        'Mx',
        'St',
        'Saint',
        'Cantor',
        'Chancellor',
        'President'
    ]
    # Creates a cursor object that allows us to create tables and query into existing tables using SQL language
    conn, cursor = data_cleaning.connect_to_db(
                                            database="postgres",
                                            hostname="xxxx.rds.amazonaws.com",
                                            port="5432",
                                            userid="xxx",
                                            passwrd="xxx")

    # ----------------------------------------------------------------------------------------------------------------------

    def __init__(self, nlp, article_text, article_url, article_title, annotated_entities, language, ner_process, labels = None, found_entities=None):
        """
        Parameters:
        * NLP SpaCy model
        * Dictionary of annotated entities (keys) and labels (values)
        * Article Text
        * found_entities (list of found entity strings)

        Attributes:
        * self.article_doc: Contains all article tokens (SpaCy Doc Object)
        * self.found_entity_list: List of all entities found by the model (entities are SpaCy Span objects)
        * self.ground_truth_dict: Ground truth dict of entities for article (keys) and labels (values)
        """
        # The title and url for the article, can be used for indexing into dataframe
        self.title = article_title
        self.url = article_url
        self.language = language
        if len(ner_process) == 0 or not found_entities:
            ner_process = 'vanilla spacy'
        row_schema = {
            'art_title' : article_title,
            'ner_process' : ner_process,
            'tp': 0,
            'fp': 0,
            'fn': 0,
            'frag': 0,
            'sos_frag': 0,
            'num_frag': 0,
            'title_colon_frag': 0,
            'title_prefix_frag': 0,
            'concat': 0,
            'sos_concat': 0,
            'noun_ent_concat': 0,
            'conj_adv_concat': 0,
            'interior_ent_concat': 0,
            'contractional_concat': 0,
            'comma_list_concat': 0,
            'comma_ent_frag': 0,
            'sports_concat': 0,
            'athlete_pos_concat': 0,
            'team_score_rank_concat': 0,
            'hyphen_concat': 0,
            'one_hyphen_concat': 0,
            'colon_concat': 0,
            'diseases_frag': 0,
            'spurious': 0,
            'missing': 0
        }
        # Adds an additional row of data for new article instance
        article_row = pd.DataFrame(row_schema, columns = row_schema.keys(), index = [article_url])
        if article_url in Error_Identifier.identified_errors_table.index:
            Error_Identifier.identified_errors_table.loc[article_url].values[2:] = 0
        else:
            Error_Identifier.identified_errors_table = Error_Identifier.identified_errors_table.append(article_row)
        # SpaCy Doc object that contains all tokens for article text
        self.article_doc = nlp(article_text)
        # if found_entities weren't provided then default to finding entities using Vanilla SpaCy trained on
        # en_core_web_md
        if not found_entities:
            # A list of found entities
            if "All labels" in labels:
                self.found_entity_list_minus_mapped = [ent for ent in self.article_doc.ents]
            else:
                self.found_entity_list_minus_mapped = [ent for ent in self.article_doc.ents if ent.label_ in labels]
        # if found entities were provided then find corresponding span objects using found_entity_creator
        else:
            # A list of found entities
            self.found_entity_list_minus_mapped = self.entity_creator([ent.strip() for ent in found_entities], nlp, self.article_doc)
        # A sorted list of entities that were found by the model for the given article (List of SpaCy Span Objects)
        self.found_entity_list_sorted = sorted(self.found_entity_list_minus_mapped, key= lambda ent: ent.text)
        # A list of annotated entities (Span Objects)
        self.ground_truth_list = self.entity_creator([ent.strip() for ent in annotated_entities], nlp, self.article_doc)
        # A sorted list of annotated entities
        self.ground_truth_sorted = sorted(self.ground_truth_list, key= lambda ann_ent: ann_ent.text)
        # A dictionary of all entities and their corresponding error-type
        self.error_types_dictionary = {}
        for ent in self.found_entity_list_minus_mapped:
            start_index = len(self.article_doc[:ent.start].text)
            end_index = len(self.article_doc[:ent.end].text)
            self.error_types_dictionary[ent] = [start_index, end_index, "SPURIOUS"]
# ----------------------------------------------------------------------------------------------------------------------
    # Kaelan

    def entity_creator(self, entities_list, nlp, doc):
        """
        Purpose
        -------
        This method is used to find the corresponding Span objects within the article's spacy.Doc container object for
        each found or annotated entity string. This allows us to easily access the context of an entity within an
        article by utilizing SpaCy's tokenizer.

        *Note: that if an entities text (python str) appears multiple times within the article we have no way
        of determining which instance within the article corresponds to the entity, so we will always select the
        FIRST occurrence of the entities text within the article.

        Method Parameters
        -----------------
        entities_list (List of str) - All the entities that were found by the users NER model or annotated for a given
                                        article
        nlp (spacy.nlp object) - the spacy nlp object that was used to tokenize the article text and contains the
                                        article's vocab
        doc (spacy.Doc object) - the spacy doc object that contains all of the article's tokens


        Returns
        -------
        A list of the entities' (found or annotated) corresponding spacy.Span objects that is sorted in ascending order
        by entity start position.

        References
        ----------
        1. https://course.spacy.io/en/chapter1
        """
        # There must be at least one found entity otherwise this method is unnecessary
        assert (len(entities_list) > 0)
        sorted_entities_list = sorted(entities_list)
        i = 1
        entity_counts = {}
        entity_counts[sorted_entities_list[0]] = 1
        while i < len(sorted_entities_list):
            ent = sorted_entities_list[i]
            if ent == sorted_entities_list[i - 1]:
                entity_counts[ent] += 1
            else:
                entity_counts[ent] = 1
            i += 1
        matcher = Matcher(nlp.vocab)
        ent_index = 0
        for ent in set(entities_list):
            # list that will contain all separate tokens for the purpose of matching
            pattern = []
            # break the entities up into individual tokens
            split_commas = ent.replace(",", " , ")
            split_colon = split_commas.replace(":", " : ")
            split_dash = split_colon.replace("-", " - ")
            split_apostrophe = split_dash.replace("'", " '")
            tokens = split_apostrophe.split()
            for token in tokens:
                word_pattern = {"TEXT": token}
                pattern.append(word_pattern)
            matcher.add("entity_{}".format(ent_index), None, pattern)
            ent_index += 1
        matches = matcher(doc)
        spans = []
        index = -1
        for match_id, start, end in matches:
            index += 1
            matched_span = doc[start:end]
            if entity_counts[matched_span.text] > 0:
                if index + 1 < len(matches):
                    next_match = matches[index + 1]
                    if start == next_match[1]:
                        continue
                if index - 1 > 0:
                    last_match = matches[index - 1]
                    if start < last_match[2] and end - start < last_match[2] - last_match[1]:
                        continue
                entity_counts[matched_span.text] -= 1
                spans.append(matched_span)
        return spans

# ----------------------------------------------------------------------------------------------------------------------
    # Kaelan

    def is_frag(self, ent, possible_ground_truth):
        """
        Purpose
        -------
        This method is used to determine whether or not an identified entity (found by a specified NER model or by some
        sort of pre-extraction scheme) is in fact a fragmented portion of its corresponding ground truth entity AND
        establish the one-to-one mapping between the identified entity and its corresponding ground truth IF it is
        indeed a fragmentation.
             1. { "Proposed" : "iPhone",
                  "Actual"   : "iPhone X" }
             2. { "Proposed" : "Bean",
                  "Actual" : "Mr. Bean" }
        *Note that these examples show that a fragmentation can occur in several different ways. Hence the granularity

        Method Parameters
        -----------------
        ent - (spacy.Span object) This is the Span object that was created to represent the collection
                                  of tokens that make up the identified entity.
        possible_ground_truth (list of spacy.Span objects) - This is a list comprised of ground truth entities that are
        similar (based on Spacy's similarity function) but not equal (in terms of text), each of which is potentially
        the identified entity's corresponding ground truth entity.


        Returns
        -------
        Tuple of 2 elements that respectively indicate 1. Whether or not the proposed entity is the result of a
        fragmentation error through a Boolean, and 2. The fragmented entity's corresponding ground_truth entity as a
        spacy.Span object OR None if the identified entity is not a fragmentation

        References
        ----------
        1.
        """
        ground_truth_unique = set([ann_ent.text for ann_ent in possible_ground_truth])
        for ann_ent in possible_ground_truth:
            # identified entity must be shorter in text length than the ground truth to be a fragment
            if ent.text in ann_ent.text and len(ent.text) < len(ann_ent.text):
                # check both the left and right neigboring tokens of the entity to check whether or not they are
                # contained within the ground truth
                if self.article_doc[ent.start - 1].text in ann_ent.text or self.article_doc[ent.end].text in ann_ent.text:
                    # Update frag column of DataFrame
                    Error_Identifier.identified_errors_table.at[self.url, 'frag'] += 1
                    # Update fp column of DataFrame
                    Error_Identifier.identified_errors_table.at[self.url, 'fp'] += 1
                    # Update the error types dictionary
                    self.error_types_dictionary[ent][2] = 'FRAGMENT'
                    # No longer need to compare ann_ent to any other found entity now that we have found its match
                    self.ground_truth_list.remove(ann_ent)
                    # returns a tuple containing: Is it fragmented, what portion is missing, corresponding ground_truth
                    return (True, ann_ent)
        return (False, None)

# ----------------------------------------------------------------------------------------------------------------------
    # Kaelan

    def is_sos_frag(self, ent, ground_truth_ent):
        """
        Purpose
        -------
        This method is used to determine whether or not a fragmented entity (as determined by is_frag) is a
        Start-of-Sentence fragment meaning that the missing (fragmented) text consists of tokens at the very beginning of a
        sentence.

        Method Parameters
        -----------------
        ent - (spacy.Span object) This is the Span object that was created to represent the collection
                                  of tokens that make up the identified entity.
        ground_truth_ent (spacy.Span object) - The identified entities corresponding ground truth entity


        Returns
        -------
        Whether or not the fragmented entity is a start-of-sentence fragment through a Boolean

        References
        ----------
        1.
        """
        #determine the difference in number of tokens between identified entity and ground truth entity
        length_of_ann = ground_truth_ent.end - ground_truth_ent.start
        length_of_ent = ent.end - ent.start
        diff_of_length = length_of_ann - length_of_ent
        #should always be true if entity is fragmented
        assert(diff_of_length > 0)
        #check whether or not the token before the first fragmented token is an end-of-sentence character
        # if so then the fragment is an sos fragment
        sos = False
        end_of_sentence_token = self.article_doc[ent.start - diff_of_length - 1]
        if end_of_sentence_token.text == '"':
            end_of_sentence_token = self.article_doc[ent.start - diff_of_length - 2]
        if end_of_sentence_token.text in [
            "!",
            "?",
        ]:
            sos = True
        elif end_of_sentence_token.text == ".":
            start_of_sentence_token = self.article_doc[ent.start - diff_of_length]
            token_before_period = self.article_doc[end_of_sentence_token.i - 1]
            # check whether or not the period is part of an abbreviation to avoid falsely classifying the period as
            # an end of sentence character
            # Note that spacy will not classify the period as a stand-alone token if the model recognizes the
            # abbreviation
            is_abbreviation = (len(token_before_period.text) == 1 and token_before_period.is_upper)
            if is_abbreviation:
                return False
            # period only comes after punctuation if the punctuation is the final token in a sentence
            elif token_before_period.is_punct:
                sos = True
            # if the period is not part of an abbreviation, and the next word is capitalized then we can assume the
            # period is an end of sentence character
            elif start_of_sentence_token.is_upper:
                sos = True
        if sos:
            Error_Identifier.identified_errors_table.at[self.url, "sos_frag"] += 1
            if self.error_types_dictionary[ent][2] == 'FRAGMENT':
                self.error_types_dictionary[ent][2] = 'SOSFRAG'
            else:
                self.error_types_dictionary[ent][2] = 'MULTIFRAG'
            if len(Error_Identifier.error_examples["sos_frag"]) < 3:
                Error_Identifier.error_examples["sos_frag"].append(
                    'Got "{}" but should be "{}"'.format(ent.text, ground_truth_ent.text)
                )
            return True
        return False

# ----------------------------------------------------------------------------------------------------------------------
    # Kaelan

    def is_num_frag(self, ent, ground_truth_ent):
        """
        Purpose
        -------
        This method is used to determine whether or not a fragmented entity (as determined by is_frag) is a
        Numeral Fragment meaning that the missing (fragmented) text begins with a numeral.

        Method Parameters
        -----------------
        ent - (spacy.Span object) This is the Span object that was created to represent the collection
                                  of tokens that make up the identified entity.
        ground_truth_ent (spacy.Span object) - The identified entities corresponding ground truth entity


        Returns
        -------
        Whether or not the fragmented entity is a numeral fragment through a Boolean

        References
        ----------
        1.
        """
        missing_text = ground_truth_ent.text.replace(ent.text, '', 1)
        # checks whether or not the missing text begins with a numeral
        if missing_text[0].isdigit():
            Error_Identifier.identified_errors_table.at[self.url, "num_frag"] += 1
            if self.error_types_dictionary[ent][2] == 'FRAGMENT':
                self.error_types_dictionary[ent][2] = 'NUMFRAG'
            else:
                self.error_types_dictionary[ent][2] = 'MULTIFRAG'
            if len(Error_Identifier.error_examples["num_frag"]) < 3:
                Error_Identifier.error_examples["num_frag"].append(
                    'Got "{}" but should be "{}"'.format(
                        ent.text, ground_truth_ent.text
                    )
                )
            return True
        return False

# ----------------------------------------------------------------------------------------------------------------------
    # Kaelan

    def is_title(self, text):
        """
        Purpose
        -------
        This method determines whether or not a string is a title and acts as a utility function for title_prefix_frag and
        title_colon_frag

        Method Parameters
        -----------------
        text (spacy.Span object) - consists of an entities text

        Returns
        -------
        Whether or not the entity text is a title or not via a Boolean

        References
        ----------
        1.
        """
        for token in text:
            # If the character is uppercase we continue to check the next word in the text
            if token.is_title or token.is_digit or token.is_punct:
                continue
            word = token.text
            # If the character is lowercase then check to see whether or not it is a stop word
            apostrophe_split = word.split("'")
            # this step prevents the SQL statement below from breaking if the word contains an apostrophe
            if len(apostrophe_split) > 1:
                word = "''".join(apostrophe_split)
            stop_words_select_execute = Error_Identifier.cursor.execute(
                "SELECT word FROM stop_words WHERE LOWER(word) = '{}'".format(word.lower()))
            is_stop_word = len(Error_Identifier.cursor.fetchall()) > 0
            if is_stop_word:
                continue
            else:
                return False
        return True

# ----------------------------------------------------------------------------------------------------------------------
    # Kaelan

    def is_title_colon_frag(self, ent, ground_truth_ent):
        """
        Purpose
        -------
        This method is used to determine whether or not a fragmented entity (as determined by is_frag) is a
        Title Colon Fragment meaning that the missing (fragmented) text is the portion of a title that follows a colon
        (i.e.
        Actual- "Avatar: The Last Airbender"
        Proposed - "Avatar"
        Missing- ": The Last Airbender"
        )

        Method Parameters
        -----------------
        ent - (spacy.Span object) This is the Span object that was created to represent the collection
                                  of tokens that make up the identified entity.
        ground_truth_ent (spacy.Span object) - The identified entities corresponding ground truth entity


        Returns
        -------
        Whether or not the fragmented entity is a title colon fragment through a Boolean

        References
        ----------
        1.
        """
        missing_text = ground_truth_ent.text.replace(ent.text, '', 1)
        # checks whether or not a colon is even contained within the missing text
        if ":" in missing_text:
            # checks whether or not the colon (and the rest of the fragmented text) is the part of a title
            if self.is_title(ground_truth_ent):
                Error_Identifier.identified_errors_table.at[
                    self.url, "title_colon_frag"
                ] += 1
                if self.error_types_dictionary[ent][2] == 'FRAGMENT':
                        self.error_types_dictionary[ent][2] = 'TITLECOLONFRAG'
                else:
                    self.error_types_dictionary[ent][2] = 'MULTIFRAG'
                if len(Error_Identifier.error_examples["title_colon_frag"]) < 3:
                    Error_Identifier.error_examples["title_colon_frag"].append(
                        'Got "{}" but should be "{}"'.format(
                            ent.text, ground_truth_ent.text
                        )
                    )
                return True
        return False

# ----------------------------------------------------------------------------------------------------------------------
    # Kaelan

    def is_title_prefix_frag(self, ent, ground_truth_ent):
        """
        Purpose
        -------
        This method is used to determine whether or not a fragmented entity (as determined by is_frag) is a
        Title Prefix Fragment meaning that the missing (fragmented) text begins with an honorific that is part of a title
        (i.e.
        Actual- "Mr. Bean"
        Proposed - "Bean"
        Missing- "Mr. "
        )

        Method Parameters
        -----------------
        ent - (spacy.Span object) This is the Span object that was created to represent the collection
                                  of tokens that make up the identified entity.
        ground_truth_ent (spacy.Span object) - The identified entities corresponding ground truth entity


        Returns
        -------
        Whether or not the fragmented entity is a title prefix fragment through a Boolean

        References
        ----------
        1.
        """
        missing_text = ground_truth_ent.text.replace(ent.text, '', 1).strip()
        first_word = missing_text.split()[0]
        honorific = False
        for honor in Error_Identifier.honorifics:
            if honor in first_word:
                honorific = True
        if honorific:
            if self.is_title(ground_truth_ent):
                Error_Identifier.identified_errors_table.at[
                    self.url, "title_prefix_frag"
                ] += 1
                if self.error_types_dictionary[ent][2] == 'FRAGMENT':
                        self.error_types_dictionary[ent][2] = 'TITLEPREFIXFRAG'
                else:
                    self.error_types_dictionary[ent][2] = 'MULTIFRAG'
                if len(Error_Identifier.error_examples["title_prefix_frag"]) < 3:
                    Error_Identifier.error_examples["title_prefix_frag"].append(
                        'Got "{}" but should be "{}"'.format(
                            ent.text, ground_truth_ent.text
                        )
                    )
                return True
        return False


# ----------------------------------------------------------------------------------------------------------------------
    # Sebastian

    def is_concat(self, ent, possible_ground_truth):
        """
        Purpose
        -------
        This method is meant to identify whether or not an entity that has been identified (either by the specified
        Spacy model or by some sort of pre-extraction scheme) is actually NOT an entity due to the fact that it is
        the result of a concatenation error. General example of such an error include:
             1. { "Proposed" : "and George Washington",
                  "Actual"   : "George Washington" }
             2. { "Proposed" : "Samsung Note", :
                  "Actual" : "Samsung" }
        Note that these examples show that a concatenation can occur in several different ways. Hence the granularity

        Method Parameters
        -----------------
        ent - (Spacy Span object) This is the Span object that was created to represent the collection
                                  of tokens that make up this proposed entites.
        possible_ground_truth (list of Spacy Span object) - This is a list comprised of Span objects that were all
                                                            created during the class initialization as the specified
                                                            golden annotated entities were compiled.

        Note that these examples show that a concatenation can occur in several different ways. Hence the granularity

        Returns
        -------
        tuple of 3 elements that respectively indicate 1. Whether or not the proposed entity is the result of a
        concatentation error through a Boolean, 2. The string that was incorrectly added on, and 3. What the correct
        entity actually is. All that being said, if the first element is False, then the other two will simply be a
        `None` value.

        References
        ----------
        1.
        """
        to_return = (False, None, None)
        ### Perform the neccessary comparisons to see if this entity (ent) is in fact the result of a concat error.
        for ann_ent in possible_ground_truth:
            # Have to match up the passed in proposed entity with all of the possibilites for what actual entity
            # it corresponds to.
            if ann_ent.text in ent.text and len(ent) > len(ann_ent):
                if self.article_doc[ann_ent.start - 1].text in ent.text or self.article_doc[ann_ent.end].text in ent.text:
                    # note that if there is indeed a concatentation, the proposed entity (ent) will be longer than
                    # the ground truth entity (ann_ent). Thus the latter would have to be in the former.
                    Error_Identifier.identified_errors_table.at[self.url, "concat"] += 1
                    # goes without saying, we have identified a concatenation error so we should indicate
                    # that on the master dataframe.
                    self.error_types_dictionary[ent][2] = 'CONCAT'
                    Error_Identifier.identified_errors_table.at[self.url, "fp"] += 1
                    # Since we have the proposal of this entity as incorrect due to the fact that it was the
                    self.ground_truth_list.remove(ann_ent)
                    # We have also been able to pair a proposed entity with a golden entity, so we should indicate
                    # this in our lists of mapped entites.
                    to_return = (True, ann_ent)
                    return to_return
        return to_return

# ----------------------------------------------------------------------------------------------------------------------
    # Sebastian

    def conj_adv_concat(self, ent, ground_truth_ent):
        """
        Purpose
        -------
        This method determines whether or not an identified concatenation is an example of

        Method Parameters
        -----------------
        ent - (Spacy Span object) This is the Span object that was created to represent the collection
                                  of tokens that make up this proposed entites.
        ground_truth_ent - (Spacy Span object) This is the Span object that was created during the class
                                               initialization as the specified golden annotated entities
                                               were compiled.

        Returns
        -------
        to_return - (Boolean) This indicates whether or not the concatenation error that occured
                              with this entity is due to

        References
        ----------
        1. https://en.wikipedia.org/wiki/Conjunctive_adverb
        2. https://www.chompchomp.com/terms/conjunctiveadverb.htm
        3. https://stackoverflow.com/questions/1912095/how-to-insert-a-value-that-contains-an-apostrophe-single-quote
        """
        to_return = False
        added_text = ent.text.replace(ground_truth_ent.text, '', 1).strip()
        # Makes the assumption that if a conjunctive adverb does exist it is at the start of the concatenated text
        first_concatenated_word = added_text.split()[0]
        # if an apostrophe exists within the first concatenated word then the word is not a conjunctive adverb
        # Note: the presence of an apostrophe would break the SQL statement below, but is handled by this condition
        if "'" in first_concatenated_word:
            return to_return
        conj_adv_select_execute = Error_Identifier.cursor.execute(
            "SELECT conjunctive_adverb FROM conjunctive_adverbs_table WHERE LOWER(conjunctive_adverb) = '{}'".format(
                first_concatenated_word.lower()))
        # checks whether or not the first concatenated word is indeed a conjunctive adverb
        if len(Error_Identifier.cursor.fetchall()) > 1:
            to_return = True
        if to_return:
            Error_Identifier.identified_errors_table.at[self.url, 'conj_adv_concat'] += 1
            if self.error_types_dictionary[ent][2] == 'CONCAT':
                self.error_types_dictionary[ent][2] = 'CONJADVCONCAT'
            else:
                self.error_types_dictionary[ent][2] = 'MULTICONCAT'
            if len(Error_Identifier.error_examples['conj_adv_concat']) < 3:
                Error_Identifier.error_examples['conj_adv_concat'].append(
                    'Got "{}" but should be "{}"'.format(ent.text, ground_truth_ent.text))
        return to_return

# ----------------------------------------------------------------------------------------------------------------------
    # Sebastian

    def sos_concat(self, ent, ground_truth_ent):
        """
        Purpose
        -------
        This method determines whether or not an identified concatenation is an example of

        Method Parameters
        -----------------
        ent - (Spacy Span object) This is the Span object that was created to represent the collection
                                  of tokens that make up this proposed entites.
        ground_truth_ent - (Spacy Span object) This is the Span object that was created during the class
                                               initialization as the specified golden annotated entities
                                               were compiled.

        Returns
        -------
        to_return - (Boolean) This indicates whether or not the concatenation error that occured
                              with this entity is due to

        References
        ----------
        1.
        """
        ### Check to see if the first letter of this entity is capitalized and if there is a period
        ### two character indices before it (which is equivalent to saying at the end of the preceeding
        ### token). NOTE that we can also check for any other form of end of sentence punctuation, namely,
        ### "?" and "!". This would indicate that the proposed entity lives at the beginning of a sentence
        ### which is the only way for a start of sentence concatenation to occur.
        # capitalization
        capitalization = ent.text[0:1:].isupper()
        if not capitalization:
            return False
        # punctuation
        puncuations_list = [".", "!", "?"]
        preceding_token = self.article_doc[ent.start - 1]
        is_punct_right_before = preceding_token.text in puncuations_list if len(preceding_token) == 1 else \
        preceding_token.text[-1] in puncuations_list

        ### One can argue that the classical example of a sos concatenation is that the word "The" (which
        ### starts the sentence) gets attached to the rest of the entity. For that reason, we can check if
        ### "The" is present in the entity text (through a case-SENSITIVE search). What cememnts our ability
        ### to do this is that since the entity had to go through the `is_concat()` method, we know that this
        ### "The" should NOT be included in the entity text (since there are cases such as movie titles and
        ### songs where "The" must be included). IN FACT, we can generalize this to any stop word such as "A",
        ### "And", "But", etc.. which is why we loaded in a full list of them above. We will be checking for
        ### all of them. NOTE that we are working off of the assumption that a start of sentence concatenation
        ### can only occur if the first word is in fact a stop word.
        # NOTE again that this search is case-SENSITIVE which is important since we are using the "title"
        # (first letter capitalized and rest lower-case) version of the stop word! ALSO NOTE that for each
        # stop word, we are searching for it in a LAZY way; that is, once we find the first instance of it,
        # we stop the search. We do this because we are taking advantage of the fact that we only care about
        # whether or not the stop word is at the beginning of the sentence.
        if not is_punct_right_before:
            return False
        stop_words_select_execute = Error_Identifier.cursor.execute(
            "SELECT word FROM stop_words WHERE LOWER(word)  = '{}'".format(ent[0].text.lower()))
        beginning_is_stop_word = len(Error_Identifier.cursor.fetchall()) > 0
        if beginning_is_stop_word:
            Error_Identifier.identified_errors_table.at[self.url, 'sos_concat'] += 1
            if self.error_types_dictionary[ent][2] == 'CONCAT':
                self.error_types_dictionary[ent][2] = 'SOSCONCAT'
            else:
                self.error_types_dictionary[ent][2] = 'MULTICONCAT'
            if len(Error_Identifier.error_examples['sos_concat']) < 3:
                Error_Identifier.error_examples['sos_concat'].append(
                    'Got "{}" but should be "{}"'.format(ent.text, ground_truth_ent.text))
            return True
        return False
# ----------------------------------------------------------------------------------------------------------------------
    # Sebastian

    def athlete_pos_concat(self, ent, ground_truth_ent):
        """
        Purpose
        -------
        This method determines whether or not an identified concatenation is an example of

        Method Parameters
        -----------------
        ent - (Spacy Span object) This is the Span object that was created to represent the collection
                                  of tokens that make up this proposed entites.
        ground_truth_ent - (Spacy Span object) This is the Span object that was created during the class
                                               initialization as the specified golden annotated entities
                                               were compiled.

        Returns
        -------
        to_return - (Boolean) This indicates whether or not the concatenation error that occured
                              with this entity is due to

        References
        ----------
        1.
        """
        to_return = False
        ### First, load in the list of athlete positions that can be found in tables in the AWS
        ### PostgreSQL database that will be connected to when this script is ran in a different one.
        athlete_position_select_execute = Error_Identifier.cursor.execute(
            "SELECT position FROM athlete_positions_table"
        )
        uncleaned_athlete_positions_list = Error_Identifier.cursor.fetchall()
        athlete_positions_list = [
            position_tuple[0] for position_tuple in uncleaned_athlete_positions_list
        ]

        ### Now see if the entity string contains any of these positions
        position_find_results = [
            re.match(position, ent.text, re.IGNORECASE)
            for position in athlete_positions_list
        ]
        if np.any(position_find_results):
            # if we did NOT find any athlete positions in the entity, then it it not possible for
            # there to be an athlete position concatenation error.

            # extract part of string that does NOT have the position
            matched_obj = np.any(position_find_results)
            start_position_index, end_position_index = matched_obj.span()
            entity_length = len(ent.text)

            if start_position_index == 0:
                # if the position string is the start of the entity
                uncleaned_non_position_str = ent.text[end_position_index + 1 : :]
                non_position_str = (
                    uncleaned_non_position_str[1::]
                    if uncleaned_non_position_str[0].isspace()
                    else uncleaned_non_position_str
                )
                # if there is a space between the conjunctive adverb and the rest of the entity, then this will ensure
                # that the span object that represents the part of the entity that does NOT include the conjunctive
                # adverb will NOT BEGIN with a space.
            elif (
                end_position_index == entity_length
                or end_position_index == entity_length - 1
            ):
                # if the conjunctive adverb string makes up the ending of the entity. The minus 1 is to protect against
                # cases where some sort of puncuation might be present.
                uncleaned_non_position_str = ent.text[0:end_position_index:]
                uncleaned_length = len(uncleaned_non_position_str)
                non_position_str = (
                    uncleaned_non_position_str[0 : uncleaned_length - 1 :]
                    if uncleaned_non_position_str[-1].isspace()
                    else uncleaned_non_position_str
                )
                # if there is a space between the conjunctive adverb and the rest of the entity, then this will ensure
                # that the span object that represents the part of the entity that does NOT include the conjunctive
                # adverb will NOT END with a space.
            else:
                # if, for some reason, the conjunctive adverb string is in the middle of the entity.
                uncleaned_left_non_position_str = ent.text[0:start_position_index:]
                left_uncleaned_length = len(uncleaned_left_non_position_str)
                left_non_position_str = (
                    uncleaned_left_non_position_str[0 : left_uncleaned_length - 1 :]
                    if uncleaned_left_non_position_str[-1].isspace()
                    else uncleaned_left_non_position_str
                )

                uncleaned_right_non_position_str = ent.text[
                    end_position_index + 1 : entity_length :
                ]
                right_uncleaned_length = len(uncleaned_right_non_position_str)
                right_non_position_str = (
                    uncleaned_right_non_position_str[1:right_uncleaned_length:]
                    if uncleaned_right_non_position_str[0].isspace()
                    else uncleaned_right_non_position_str
                )
                non_position_str = "{} {}".format(
                    left_non_position_str, right_non_position_str
                )

            # see how this comparies to the golden label entity.
            if (
                non_position_str == ground_truth_ent.text
                or ground_truth_ent.text in non_position_str
            ):
                to_return = True

        if to_return:
            Error_Identifier.identified_errors_table.at[
                self.url, "athlete_pos_concat"
            ] += 1
            if self.error_types_dictionary[ent][2] == 'CONCAT':
                self.error_types_dictionary[ent][2] = 'ATHLETEPOSCONCAT'
            else:
                self.error_types_dictionary[ent][2] = 'MULTICONCAT'
            if len(Error_Identifier.error_examples["athlete_pos_concat"]) < 3:
                Error_Identifier.error_examples["athlete_pos_concat"].append(
                    'Got "{}" but should be "{}"'.format(
                        ent.text, ground_truth_ent.text
                    )
                )
        return to_return

# ----------------------------------------------------------------------------------------------------------------------
    # Sebastian

    def colon_concat(self, ent, ground_truth_ent):
        """
        Purpose
        -------
        This method determines whether or not an identified concatenation is an example of a colon
        concatenation where two tokens (possibly two seperate entities) seperated by a colon were put
        together due to the (statistical/procedural) model thinking that the colon meant it was all
        one entity.

        Method Parameters
        -----------------
        ent - (Spacy Span object) This is the Span object that was created to represent the collection
                                  of tokens that make up this proposed entites.
        ground_truth_ent - (Spacy Span object) This is the Span object that was created during the class
                                               initialization as the specified golden annotated entities
                                               were compiled.

        Returns
        -------
        to_return - (Boolean) This indicates whether or not the concatenation error that occured
                              with this entity is due to the midhandling of a colon that was
                              present in the context around the entity.

        References
        ----------
        1. https://spacy.io/api/span
        """
        ### First check if there's a colon in the entity
        if ":" not in ent.text:
            to_return = False
        else:
            ### split up on colon and see if either side matches the golden annotation
            colon_index = ent.text.find(":")
            # left side of colon
            left_side_str = ent.text[0:colon_index:]
            # right side of colon
            uncleaned_right_side_str = ent.text[colon_index + 1 : len(ent.text) :]
            # Recall that entities are Spacy Span objects.
            # Note that the specified indices are CHARACTER indicies and NOT token indicies.
            right_side_str = (
                uncleaned_right_side_str[1::]
                if uncleaned_right_side_str[0].isspace()
                else uncleaned_right_side_str
            )
            # The reason for why we are keeping everything as a Span object is so that we can
            # use the similarity method in the elif statement below.
            if (
                left_side_str == ground_truth_ent.text
                or right_side_str == ground_truth_ent.text
            ):
                # this is where we are trying to get an exact match
                to_return = True
            elif np.any(
                [
                    left_side_str in ground_truth_ent.text,
                    right_side_str in ground_truth_ent.text,
                ]
            ):
                # in the case that we do not get an exact match, test for similarity and return truth
                # if either is above the pre-determined threshold. This is to protect against cases where
                # the reason why there were subtle differences that were not accounted for and thus led to
                # there not being an exact match.
                to_return = True
            else:
                # cases where the incorporation of the colon was correct.
                to_return = False

        if to_return:
            Error_Identifier.identified_errors_table.at[self.url, "colon_concat"] += 1
            if self.error_types_dictionary[ent][2] == 'CONCAT':
                self.error_types_dictionary[ent][2] = 'COLONCONCAT'
            else:
                self.error_types_dictionary[ent][2] = 'MULTICONCAT'
            if len(Error_Identifier.error_examples["colon_concat"]) < 3:
                Error_Identifier.error_examples["colon_concat"].append(
                    'Got "{}" but should be "{}"'.format(
                        ent.text, ground_truth_ent.text
                    )
                )
        return to_return

# ----------------------------------------------------------------------------------------------------------------------
    # Sebastian

    def comma_list_concat(self, ent, ground_truth_ent):
        """
        Purpose
        -------
        This method

        Method Parameters
        -----------------
        ent - (Spacy Span object) This is the Span object that was created to represent the collection
                                  of tokens that make up this proposed entites.
        ground_truth_ent - (Spacy Span object) This is the Span object that was created during the class
                                               initialization as the specified golden annotated entities
                                               were compiled.

        Returns
        -------
        to_return - (Boolean) This indicates whether or not the concatenation error that occured
                              with this entity is due to

        References
        ----------
        1. https://www.ocpsoft.org/tutorials/regular-expressions/or-in-regex/
        2. https://en.wikipedia.org/wiki/Serial_comma
        """
        to_return = False
        ### First, check for the presence of both commas and the use of the words "and" or "or" since
        ### essentially all lists include use of these.
        comma_matched_objs_iter = re.finditer(
            pattern=r"(?<=[a-zA-Z]),(?= )", string=ent.text
        )
        # NOTE that this will also match the comma that comes right before that special instance
        # of either "and" or "or" (a.k.a the serial comma).
        comma_matched_objs_list = [match for match in comma_matched_objs_iter]
        and_or_matched_objs_iter = re.finditer(
            pattern=r", and|, or|, and/or", string=ent.text
        )
        and_or_matched_objs_list = [match for match in and_or_matched_objs_iter]
        # Recall that "|" is the OR operator in regular expressions.

        # NOTE that we are only searching for instances of "and" and "or" that come right after a
        # comma and a space. This is to protect against returning matches of these words that occur
        # in the entities themselves; we only want to be looking at the last instance. FOR this reason,
        # if whoever wrote the text that corresponds to that entity decided to not use a Serial Comma
        # (which, in my humble opinion is a sin in the English Language), then that will not get picked
        # up by this search (see reference no. 1 in the docstring; need to figure out a good way to be
        # able to do this). Hard to incorporate this Serial Comma flexibility while still guarding
        # against NOT picking up instance of "and" and "or" that occur within the list elemnts. Currently
        # brainstorming ways to do that.
        if len(comma_matched_objs_list) >= 2 and len(and_or_matched_objs_list) >= 1:
            # if this entity is made up of a list.

            # Split up the elements that make up the word list. As always we still want to keep the span objects around for use below.
            split_list_spans = []
            for index, match in enumerate(comma_matched_objs_iter):
                if index == 0:
                    # if we are working with the first comma

                    # we have to append both the first and second elements
                    current_start_comma_index, current_end_comma_index = match.span()
                    split_list_spans.append(ent.char_span(0, current_start_comma_index))
                    # This will append the first element.
                    next_start_comma_index, _ = comma_matched_objs_list[
                        index + 1
                    ].span()
                    split_list_spans.append(
                        ent.char_span(
                            current_end_comma_index + 1, next_start_comma_index
                        )
                    )
                elif 0 < index < len(comma_matched_objs_list) - 1:
                    # if we are working with the commas that show up in the middle.
                    _, current_end_comma_index = match.span()
                    next_start_comma_index, _ = comma_matched_objs_list[
                        index + 1
                    ].span()
                    split_list_spans.append(
                        ent.char_span(current_end_comma_index, next_start_comma_index)
                    )

                else:
                    # if we are working with the final comma
                    _, current_end_comma_index = match.span()
                    text = ent.text[current_end_comma_index + 1 : :]
                    if text[0:3:] == "or ":
                        #
                        to_append = ent.char_span(
                            current_end_comma_index + 4, len(ent.text)
                        )
                    elif text[0:4:] == "and ":
                        #
                        to_append = ent.char_span(
                            current_end_comma_index + 5, len(ent.text)
                        )

                    split_list_spans.append(to_append)

                assert len(split_list_spans) == len(comma_matched_objs_list) + 1

            # check to see if any other those elements matches up with the golden annotation.
            conditions_list = [
                span.text == ground_truth_ent.text for span in split_list_spans
            ] + [span.similarity(ground_truth_ent) > 0.9 for span in split_list_spans]
            if np.any(conditions_list):
                to_return = True

        if to_return:
            Error_Identifier.identified_errors_table.at[
                self.url, "comma_list_concat"
            ] += 1
            if self.error_types_dictionary[ent][2] == 'CONCAT':
                self.error_types_dictionary[ent][2] = 'COMMALISTCONCAT'
            else:
                self.error_types_dictionary[ent][2] = 'MULTICONCAT'
            if len(Error_Identifier.error_examples["comma_list_concat"]) < 3:
                Error_Identifier.error_examples["comma_list_concat"].append(
                    'Got "{}" but should be "{}"'.format(
                        ent.text, ground_truth_ent.text
                    )
                )
        return to_return

# ----------------------------------------------------------------------------------------------------------------------
    # Sophia

    def hyphen_concat(self, ent, ground_truth_ent):
        # First check if hyphen is in entity
        if "-" in ent.text:
            # checks if the golden entities are in the entity being examined
            if ent.text[: ent.text.index("-")] in ground_truth_ent.text:
                Error_Identifier.identified_errors_table.at[
                    self.url, "hyphen_concat"
                ] += 1
                if self.error_types_dictionary[ent][2] == 'CONCAT':
                    self.error_types_dictionary[ent][2] = 'HYPHENCONCAT'
                else:
                    self.error_types_dictionary[ent][2] = 'MULTICONCAT'
                if len(Error_Identifier.error_examples["hyphen_concat"]) < 3:
                    Error_Identifier.error_examples["hyphen_concat"].append(
                        'Got "{}" but should be "{}"'.format(
                            ent.text, ground_truth_ent.text
                        )
                    )
                return True
        return False

# ----------------------------------------------------------------------------------------------------------------------
    # Sophia

    def one_hyphen_concat(self, ent, ground_truth_ent):
        # First check if hyphen is in entity
        if (
            ent.text.startswith("-") == False
            and ent.text.endswith("-") == False
        ):
            return False
        # checks if the golden entity is in the entity being examined
        if ent.text.strip("-") == ground_truth_ent.text:
            Error_Identifier.identified_errors_table.at[
                self.url, "one_hyphen_concat"
            ] += 1
            if self.error_types_dictionary[ent][2] == 'CONCAT':
                self.error_types_dictionary[ent][2] = 'ONEHYPHENCONCAT'
            else:
                self.error_types_dictionary[ent][2] = 'MULTICONCAT'
            if len(Error_Identifier.error_examples["one_hyphen_concat"]) < 3:
                Error_Identifier.error_examples["one_hyphen_concat"].append(
                    'Got "{}" but should be "{}"'.format(
                        ent.text, ground_truth_ent.text
                    )
                )
            return True
        return False

# ----------------------------------------------------------------------------------------------------------------------
    # Sophia

    def contractional_concat(self, ent, ground_truth_ent):
        # First check if an apostrophe is in the entity
        if "'s" not in ent.text:
            return False
        # Check if possessive noun is part of ground truth entity
        if (
            ent.text[: ent.text.index("'")] in ground_truth_ent.text
        ):  # checks possessive noun
            Error_Identifier.identified_errors_table.at[
                self.url, "contractional_concat"
            ] += 1
            if self.error_types_dictionary[ent][2] == 'CONCAT':
                self.error_types_dictionary[ent][2] = 'CONTRACTIONALCONCAT'
            else:
                self.error_types_dictionary[ent][2] = 'MULTICONCAT'
            if len(Error_Identifier.error_examples["contractional_concat"]) < 3:
                Error_Identifier.error_examples["contractional_concat"].append(
                    'Got "{}" but should be "{}"'.format(
                        ent.text, ground_truth_ent.text
                    )
                )
            return True
        return False

    # ----------------------------------------------------------------------------------------------------------------------
    # Sophia

    def sports_concat(self, ent, ground_truth_ent):
        # error type 1: 10 Lionel Messi (number & name)
        words = ent.text.split(" ")
        if len(words) < 2:
            return False
        # checks for player number
        if words[0].isdigit():
            name = " ".join(words[1:])
            # checks if name is in golden entity
            if name in ground_truth_ent.text:
                if "'" in name:
                    name = name[: name.index("'")] + "'" + name[name.index("'") :]
                # checks if player name is in database
                sqlStatement = "select * from player_table where name = '" + name + "';"
                Error_Identifier.cursor.execute(sqlStatement)
                results = Error_Identifier.cursor.fetchall()
                if len(results) > 0:  # player name was found
                    Error_Identifier.identified_errors_table.at[
                        self.url, "sports_concat"
                    ] += 1
                    if self.error_types_dictionary[ent][2] == 'CONCAT':
                        self.error_types_dictionary[ent][2] = 'SPORTSCONCAT'
                    else:
                        self.error_types_dictionary[ent][2] = 'MULTICONCAT'
                    if len(Error_Identifier.error_examples["sports_concat"]) < 3:
                        Error_Identifier.error_examples["sports_concat"].append(
                            'Got "{}" but should be "{}"'.format(
                                ent.text, ground_truth_ent.text
                            )
                        )
                    return True
        # error type 2: Barcelona's Lionel Messi (team & name)
        # remove contraction to find actual team name
        j = -1
        for i in range(len(words)):
            if words[i].endswith("s'"):
                words[i] = words[i].replace("s'", "s")
                j = i
            elif words[i].endswith("'s"):
                words[i] = words[i].strip("'s")
                j = i
        if j != -1:
            team_name = words[: j + 1]
            team_name = " ".join(team_name)
        else:
            return False
        # checks if team name is in golden entity
        if team_name in ground_truth_ent.text:
            if "'" in team_name:
                team_name = (
                    team_name[: team_name.index("'")]
                    + "'"
                    + team_name[team_name.index("'") :]
                )
            sqlStatement = "select * from team_table where name = '" + team_name + "';"
            Error_Identifier.cursor.execute(sqlStatement)
            results = Error_Identifier.cursor.fetchall()
            if len(results) > 0:  # found the team name
                player_name = " ".join(words[1:])
                if "'" in player_name:
                    player_name = (
                        player_name[: player_name.index("'")]
                        + "'"
                        + player_name[player_name.index("'") :]
                    )
                sqlStatement = (
                    "select * from team_table where name = '" + team_name + "';"
                )
                Error_Identifier.cursor.execute(sqlStatement)
                results = Error_Identifier.cursor.fetchall()
                if len(results) > 0:  # found the player name
                    Error_Identifier.identified_errors_table.at[
                        self.url, "sports_concat"
                    ] += 1
                    if self.error_types_dictionary[ent][2] == 'CONCAT':
                        self.error_types_dictionary[ent][2] = 'SPORTSCONCAT'
                    else:
                        self.error_types_dictionary[ent][2] = 'MULTICONCAT'
                    if len(Error_Identifier.error_examples["sports_concat"]) < 3:
                        Error_Identifier.error_examples["sports_concat"].append(
                            'Got "{}" but should be "{}"'.format(
                                ent.text, ground_truth_ent.text
                            )
                        )
                    return True
        return False

    # ----------------------------------------------------------------------------------------------------------------------
    # Ivy
    ## The function is called when the error is neither interior_ent_concat err or comma_ent_concat err
    def noun_ent_concat(self, ent, ground_truth_ent):
        """
        This method determines whether the entity incorrectly concatenates some token with
        a ground truth entity that is NOUN. For example, the ground truth entity is
        "Western Baptist Church" but the ent is "First Western Baptist Church".

        :param ent: This is the Span object that was created to reprersent the collection of
                    tokens that make up the proposed entity.
        :param ground_truth_ent: This is a list of Span objects that were all created during the
                                 class initialization as the specified golden annotated entities
                                 were compared.
        :type ent: Spacy Span object
        :type ground_truth_ent: List of Spacy Span objects
        :return: This indicates whether or not the concatenation error that occured with this
                 entity is due to the addition of a noun token.
        :rtype: Boolean
        """
        ## Check the pos_ of the token
        if (
            self.article_doc[ent.start].pos_ != "NOUN"
            and self.article_doc[ent.end].pos_ != "NOUN"

        ):
            return False
        if ground_truth_ent.text in ent.text:
            Error_Identifier.identified_errors_table.at[
                self.url, "noun_ent_concat"
            ] += 1
            if self.error_types_dictionary[ent][2] == 'CONCAT':
                self.error_types_dictionary[ent][2] = 'NOUNENTCONCAT'
            else:
                self.error_types_dictionary[ent][2] = 'MULTICONCAT'
            if len(Error_Identifier.error_examples["noun_ent_concat"]) < 3:
                Error_Identifier.error_examples["noun_ent_concat"].append(
                    'Got "{}" but should be "{}"'.format(
                        ent.text, ground_truth_ent.text
                    )
                )
            return True
        return False

    # ----------------------------------------------------------------------------------------------------------------------
    # Ivy
    ## two truth entities are incorrected concatenated
    def interior_ent_concat(self, ent, similar_truth_ents):
        """
        This method determines whether the identified entity incorrectly concatenates two ground
        truth entities together. For example, the ground truth entities are "Brazil" and "China",
        but the ent is "Brazil and China".

        :param ent: This is the Span object that was created to reprersent the collection of
                    tokens that make up the proposed entity.
        :param ground_truth_ent: This is a list of Span objects that were all created during the
                                 class initialization as the specified golden annotated entities
                                 were compared.
        :type ent: Spacy Span object
        :type ground_truth_ent: List of Spacy Span objects
        :return: This indicates whether or not the concatenation error that occured with this
                 entity is due to the concatenation of two entities.
        :rtype: Boolean
        """
        count = 0  # count the number of entities concatenated
        ent_text = ent.text.strip()
        interior_ents = []
        example = 'Got "{}" but should be '.format(ent.text)
        ent_range = range(ent.start, ent.end)
        for similar_ent in similar_truth_ents:
            if similar_ent.start in ent_range:
                count += 1
                interior_ents.append(similar_ent)
                if self.ground_truth_list.count(similar_ent) > 0:
                    self.ground_truth_list.remove(similar_ent)
                ent_text = ent_text.replace(similar_ent.text, "", 1)
                example += '"{}", '.format(similar_ent.text)
        if count > 1:
            Error_Identifier.identified_errors_table.at[
                self.url, "interior_ent_concat"
            ] += 1
            if self.error_types_dictionary[ent][2] == 'CONCAT':
                self.error_types_dictionary[ent][2] = 'INTERIORENTCONCAT'
            else:
                self.error_types_dictionary[ent][2] = 'MULTICONCAT'
            if len(Error_Identifier.error_examples["interior_ent_concat"]) < 3:
                Error_Identifier.error_examples["interior_ent_concat"].append(
                   example.strip(', ')
                )
            return True
        return False

    # ----------------------------------------------------------------------------------------------------------------------
    # Ivy
    def comma_ent_frag(self, ent, ground_truth_ent):
        """
        This method determines whether the identified entity incorrectly fragments the ground
        truth entity into several different entities.

        :param ent: This is the Span object that was created to reprersent the collection of
                    tokens that make up the proposed entity.
        :param ground_truth_ent: This is a list of Span objects that were all created during the
                                 class initialization as the specified golden annotated entities
                                 were compared.
        :type ent: Spacy Span object
        :type ground_truth_ent: List of Spacy Span objects
        :return: This indicates whether or not the concatenation error that occured with this
                 entity is due to the framentation of two entities.
        :rtype: Boolean
        """
        # determine whether this is a multi comma entity
        ent_text = ent.text
        splitted = ground_truth_ent.text.split(",")
        splitted = [word.strip() for word in splitted]
        splitted = [word.strip("and") for word in splitted]
        # if all the splitted substrings are in ent, then it is comma_ent_concat err
        count = len(splitted)
        for each in splitted:
            if each.strip() in ent_text:
                ent_text = ent_text.replace(each.strip(), "", 1)
                count -= 1
        if count == 0:
            Error_Identifier.identified_errors_table.at[self.url, "comma_ent_frag"] += 1
            if self.error_types_dictionary[ent][2] == 'FRAGMENT':
                self.error_types_dictionary[ent][2] = 'COMMAENTFRAG'
            else:
                self.error_types_dictionary[ent][2] = 'MULTIFRAG'
            if len(Error_Identifier.error_examples["comma_ent_frag"]) < 3:
                Error_Identifier.error_examples["comma_ent_frag"].append(
                    'Got "{}" but should be "{}"'.format(
                        ent.text, ground_truth_ent.text
                    )
                )
            return True
        return False

    # ----------------------------------------------------------------------------------------------------------------------
    # Ivy
    def team_score_rank_concat(self, ent, ground_truth_ent):
        """
        This method determines whether the ground truth entity is a sport team and the identified
        entity incorrectly concatenates the ground truth entity with the score or rank of the team.

        :param ent: This is the Span object that was created to reprersent the collection of
                    tokens that make up the proposed entity.
        :param ground_truth_ent: This is a list of Span objects that were all created during the
                                 class initialization as the specified golden annotated entities
                                 were compared.
        :type ent: Spacy Span object
        :type ground_truth_ent: List of Spacy Span objects
        :return: This indicates whether or not the concatenation error that occured with this
                 entity is due to the concatenation of the score or rank of the team.
        :rtype: Boolean
        """
        # Check whether the ent name is correct or not
        ent_text = ent.text
        if "'" in ent_text:
            ent_text = (
                ent_text[: ent_text.index("'")] + "'" + ent_text[ent_text.index("'") :]
            )
        sqlStatement = "select * from team_table where name = '" + ent_text + "';"
        Error_Identifier.cursor.execute(sqlStatement)
        results = Error_Identifier.cursor.fetchall()
        if results:
            return False
        if ground_truth_ent.text in ent.text:
            team_name = ground_truth_ent.text
            if "'" in team_name:
                team_name = (
                    team_name[: team_name.index("'")]
                    + "'"
                    + team_name[team_name.index("'") :]
                )
            sqlStatement = "select * from team_table where name = '" + team_name + "';"
            self.cursor.execute(sqlStatement)
            results = self.cursor.fetchall()
            concatenated = ent.text.replace(ground_truth_ent.text, "").strip()
            # check whether the ent has any ints included
            score_rank = any(c.isdigit() for c in concatenated)
            for i in range(ent.start, ent.end+1):
                if score_rank:
                    break
                score_rank = score_rank or (self.article_doc[i].pos_ == "NUM")
            if results and score_rank:
                Error_Identifier.identified_errors_table.at[
                    self.url, "team_score_rank_concat"
                ] += 1
                if self.error_types_dictionary[ent][2] == 'CONCAT':
                    self.error_types_dictionary[ent][2] = 'TEAMSCORECONCAT'
                else:
                    self.error_types_dictionary[ent][2] = 'MULTICONCAT'
                if len(Error_Identifier.error_examples["team_score_rank_concat"]) < 3:
                    Error_Identifier.error_examples["team_score_rank_concat"].append(
                        'Got "{}" but should be "{}"'.format(
                            ent.text, ground_truth_ent.text
                        )
                    )
                return True
        return False

    # ----------------------------------------------------------------------------------------------------------------------
    ## Ivy

    def diseases_frag(self, ent, ground_truth_ent):
        """
        This method determines whether the ground truth entity is a disease and the identified
        entity incorrectly fragments the ground truth entity.

        :param ent: This is the Span object that was created to reprersent the collection of
                    tokens that make up the proposed entity.
        :param ground_truth_ent: This is a list of Span objects that were all created during the
                                 class initialization as the specified golden annotated entities
                                 were compared.
        :type ent: Spacy Span object
        :type ground_truth_ent: List of Spacy Span objects
        :return: This indicates whether or not the concatenation error that occured with this
                 entity is due to the fragmentation of a disease name.
        :rtype: Boolean
        """
        # Check whether the ent name is correct or not
        ent_text = ent.text
        if "'" in ent_text:
            ent_text = ent_text.split("'")[0]
        sqlStatement = "select * from diseases where LOWER(name) = '{}'".format(ent_text.lower())
        Error_Identifier.cursor.execute(sqlStatement)
        results = Error_Identifier.cursor.fetchall()
        if len(results) > 0:
            Error_Identifier.identified_errors_table.at[self.url, 'diseases_frag'] += 1
            if self.error_types_dictionary[ent][2] == 'FRAGMENT':
                self.error_types_dictionary[ent][2] = 'DISEASEFRAG'
            else:
                self.error_types_dictionary[ent][2] = 'MULTIFRAG'
            if len(Error_Identifier.error_examples['diseases_frag']) < 3:
                Error_Identifier.error_examples['diseases_frag'].append(
                    'Got "{}" but should be "{}"'.format(ent.text, ground_truth_ent.text))
            return True
        return False

    # ----------------------------------------------------------------------------------------------------------------------
    # The container function that calls all subsequent identifier functions for all identified entities
    def main(self):
        """
        Purpose
        -------
        This method acts as the engine for the identifier class with several functions:
        1. First it begins by finding all TP pairs of found entities and corresponding annotated entities
        2. For each remaining unmapped (erroneous) found entity, the main function finds all similar annotated entities and passes
        the entity and these similar annotated entities to the top-most layer of our error identification hierarchy (is_frag
        and is_concat)
        3. If a fragment or concatenation is found, the main function passes the now mapped (and fragmented or concatenated)
        entity and it's corresponding annotated entity to the more granular fragmentation or concatenation functions
        (depending on if it is a fragment or concatenation) to more specifically label the error
        4. If the error type was not a fragmentation or a concatenation the main function will label the remaining entities
        and annotated entities as spurious or missing


        Method Parameters
        -----------------

        Returns
        -------

        References
        ----------

        """
        similar_ent_ann = {}
        index = 0
        for ent in self.found_entity_list_sorted:
            while index < len(self.ground_truth_sorted):
                ann_ent = self.ground_truth_sorted[index]
                if ent.text > ann_ent.text:
                    index += 1
                    continue
                elif ent.text == ann_ent.text:
                    self.found_entity_list_minus_mapped.remove(ent)
                    index += 1
                    self.ground_truth_list.remove(ann_ent)
                    Error_Identifier.identified_errors_table.at[self.url, 'tp'] += 1
                    self.error_types_dictionary[ent][2] = 'CORRECT'
                    break
                else:
                    break
        ################################################
        # TODO: Add Japanese and Spanish module
        ################################################

        # now look through proposed entities that have NOT been mapped yet.
        for ent in self.found_entity_list_minus_mapped:
            similar_ent_ann[ent] = []
            for ann_ent in self.ground_truth_list:
                # still going through all possible pairs (just in case).
                similarity_threshold = 0.65
                if ent.similarity(
                        ann_ent) > similarity_threshold or ent.text in ann_ent.text or ann_ent.text in ent.text:
                    similar_ent_ann[ent].append(ann_ent)
            frag = self.is_frag(ent, similar_ent_ann[ent])
            if frag[0]:
                # Call all more granular frag functions
                # If the fragment is a sos_frag then we know the fragment is at the beginning of a sentence
                if self.language == 'English':
                    sos_frag = self.is_sos_frag(ent, frag[1])
                    self.is_title_prefix_frag(ent, frag[1])
                    self.diseases_frag(ent, frag[1])
                    self.is_num_frag(ent, frag[1])
                    self.is_title_colon_frag(ent, frag[1])
                    self.comma_ent_frag(ent, frag[1])

                if self.language == 'Japanese':
                    self.is_num_frag(ent, frag[1])
                    self.is_title_colon_frag(ent, frag[1])

                continue
            concat = self.is_concat(ent, similar_ent_ann[ent])
            if concat[0]:
                left = False
                index_of_ent = ent.text.find(concat[1].text)
                if index_of_ent > 0:
                    left = True
                if self.language == 'English':
                    # All more specific concat functions
                    is_interior_ent_concat = self.interior_ent_concat(ent, similar_ent_ann[ent])

                    self.hyphen_concat(ent, concat[1])
                    self.one_hyphen_concat(ent, concat[1])
                    self.contractional_concat(ent, concat[1])
                    self.sports_concat(ent, concat[1])
                    self.conj_adv_concat(ent, concat[1])
                    self.sos_concat(ent, concat[1])
                    self.athlete_pos_concat(ent, concat[1])
                    self.colon_concat(ent, concat[1])
                    self.comma_list_concat(ent, concat[1])
                    # noun_ent_concat called when the error is not interior_ent_concat err or comma_ent_concat err
                    if not is_interior_ent_concat:
                        is_noun_ent_concat = self.noun_ent_concat(ent, concat[1])
                    # team_score_rank_concat is called when the ent has any ints included
                    if any([char.isdigit() for char in ent.text]):
                        self.team_score_rank_concat(ent, concat[1])
                if self.language == 'Japanese':
                    self.interior_ent_concat(ent, similar_ent_ann[ent])
                    self.colon_concat(ent, concat[1])
                    self.hyphen_concat(ent, concat[1])
                    self.one_hyphen_concat(ent, concat[1])
                    self.noun_ent_concat(ent, concat[1])
                continue
            Error_Identifier.identified_errors_table.at[self.url, 'spurious'] += 1
            Error_Identifier.identified_errors_table.at[self.url, 'fp'] += 1
        print("article not processed before, processing it now")
        Error_Identifier.identified_errors_table.at[self.url, 'missing'] = len(self.ground_truth_list)
        for ann_ent in self.ground_truth_list:
            start_index = len(self.article_doc[:ann_ent.start].text)
            end_index = len(self.article_doc[:ann_ent.end].text)
            self.error_types_dictionary[ann_ent.text] = [start_index, end_index, "MISSING"]
        Error_Identifier.identified_errors_table.at[self.url, 'fn'] = len(self.ground_truth_list)
        # save the data to pickle file in our bucket
        Error_Identifier.connect_bucket.save_identified_errors(self.url,
                                                               Error_Identifier.identified_errors_table.loc[
                                                                   self.url])
        print(Error_Identifier.identified_errors_table.loc[self.url])
        # save the error examples in our bucket
        Error_Identifier.connect_bucket.save_err_examples(Error_Identifier.error_examples)
        return sorted([[start, end, type] for start, end, type in self.error_types_dictionary.values()], key=lambda ent: ent[0])
# ----------------------------------------------------------------------------------------------------------------------
