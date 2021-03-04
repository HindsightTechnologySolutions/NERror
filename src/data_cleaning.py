####################################
### Neccessary Import Statements ###
####################################
import pandas as pd
from spacy.tokens import Doc
import psycopg2
import re
import json
import os

########################
### Load in the Data ###
########################
def custom_json_to_dataframe(
        file_name: str,
        rel_path_to_file="../data/",
        print_warnings=False):
    """
    The purpose of this function is to load a specified JSON file that
    contains all of the articles (specifically their URLs, titles,
    contents, correct entities, and perhaps even entities identified by
    a custom model) that are in the dataset the user wishes to use into
    a Pandas DataFrame for future use in the frontend portion of this
    project.
    :param file_name: This string specifies the file name of the JSON
                      file that contains the user's custom data. Note
                      that including the ".json" extension in this
                      string is optional.
    :type file_name: str
    :param rel_path_to_file: This string specifies where that JSON file
                             lives relative to this script. Its default
                             value is in the data directory that can be
                             reached from the root directory of this
                             project.
    :type rel_path_to_file: str
    :returns: This function returns a Pandas DataFrame that contains all
              of the custom data that lived in the specified JSON.
    :rtype: Pandas DataFrame
    **Notes**
    1. https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_json.html
    2. https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.reset_index.html
    3. https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rename.html
    """
    to_return = None
    final_file_name = "{}.json".format(file_name) if ".json" not in file_name else file_name
    # Get the absolute path to the file that we wish to load in
    file_path = os.path.dirname(__file__)
    full_path_to_file = os.path.join(file_path, rel_path_to_file)
    ### Now, load it in and double-check to see that it is suitable to be
    ### loaded in as a Pandas DataFrame.
    # Load as a dictionary.
    file_obj = open("{}/{}".format(full_path_to_file, final_file_name))
    raw_json_dict = json.load(file_obj)
    # Check that each principal key is in fact a URL.
    principal_key_checker = [
            key for key in raw_json_dict.keys() if all(["http" in key.lower(),
                                                        ".com" in key.lower()])
            ]
    try:
        # Use a try-except block for this check to handle the
        # possibility that the keys are not article URLs.
        assert len(principal_key_checker) == len(raw_json_dict)
    except AssertionError:
        if print_warnings:
            print(
                "**Warning**: The principal keys for this JSON file were found to NOT all be URLs. \
                \nIf this was not intended check the outputed DataFrame and specified JSON file to see if it satisfies the required schema.."
                )
    # Check to see that each instance has the same required column keys.
    # No need for a try-except block with this check.
    columns_checker = [
            value.keys() for value in raw_json_dict.values() if all(["title" in value.keys(),
                                                                     "content" in value.keys(),
                                                                     "entities" in value.keys()])
            ]
    assert len(columns_checker) == len(raw_json_dict)
    ### After pasing these checks, we are now ready to load the JSON
    ### file as a Pandas DataFrame
    loaded_df = pd.read_json(
                    open("{}/{}".format(full_path_to_file, final_file_name)),
                    orient="index"
                ).reset_index().rename(columns = {"index" : "url"})
    ### Clean up the text column
    final_df = loaded_df.copy()
    final_df["content"] = final_df.apply(remove_html_tags, axis = 1)
    to_return = final_df
    file_obj.close()
    return to_return

def remove_html_tags(text_obj, pandas_apply_mode=True):
    """
    The purpose of this function is to clean up the strings of article
    contents by removing any HTML tags present.
    :param text_obj: This object specifies what will be cleaned. It can
                     either be a single string that represents the
                     content for a single article or the row of a Pandas
                     DataFrame that represent the content of a single
                     article that belongs to a collection of articles.
                     If it is a DataFrame row, then that means that this
                     funciton is being used in the `.apply()` method of
                     a Pandas DataFrame.
    :type text_obj: str or row of a Pandas DataFrame.
    :param pandas_apply_mode: This Boolean controls whether or not this
                              function is being used to clean a single
                              string or an entire column of a DataFrame
                              (which would be the case if this parameter
                              is set to True which is its default
                              value).
    :type pandas_apply_mode: Bool
    :returns: The function itself returns a string the represents the
    cleaned text. Of course, if this function is used with the
                 `.apply()` DataFrame method, then you will get a Pandas
                 Series that contains of all the cleaned content
                 strings.
    :rtype: str
    **Notes**
    1. https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.apply.html
    """
    # Instantiate object that will look for text within html tags.
    cleanr = re.compile("<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")
    # Determine how we want to go about the cleaning.
    if pandas_apply_mode:
        # If the user is using this function to clean multiple strings
        # that live in a column of a Pandas DataFrame.
        content_str = text_obj.content
        cleantext = re.sub(cleanr, "", content_str)
    else:
        # If the user is simply trying to use this function to clean out
        # a single string.
        # removes anything between <> and any other unneeded html tags
        cleantext = re.sub(cleanr, "", text_obj)
    return cleantext

def large_data_extractor():
    data = pd.read_csv("../data/GMB_dataset.txt", sep="\t", encoding="latin1").drop(
        columns="Unnamed: 0"
    )
    # Make sure you run this in your "src" folder.

    # Note that this data was downloaded from https://www.kaggle.com/shoumikgoswami/annotated-gmb-corpus/home.
    # Super big shout out to them.

    ###############################
    ### Extract all of entities ###
    ###############################
    # Get a list of all the tags for every single row.
    tags_list = data.Tag.tolist()

    # Now determine where the entities occur.
    list_of_entity_dfs = []
    # this is a list that stores all of the sub-DataFrames of the original DataFrame
    # that make up the entities. This is mainly done just in case you want to keep
    # track of information like the sentence ID of the entity, the POS tags for each
    # token in the entity, etc...
    found_a_B_tag = False
    # We're just starting the search so it makes sense that would be initialized to
    # False.
    starting_index, ending_index = 0, 0
    for index, tag in enumerate(tags_list):
        if not found_a_B_tag and tag == "O":
            # If you're at a point where you've run into a series of "O" tags since you
            # are in the meat of the text.
            pass
        elif "B-" in tag:
            # If you have finally found the beginning of an entity.
            found_a_B_tag = True
            starting_index = index
            # We need to keep track of where to begin the entity.
        if found_a_B_tag and tag == "O":
            # If you have reached the end of the entity. Notice how we are NOT checking
            # for "I"-tags. We don't have to and we choose not to since it allows for
            # flexibility to handle entities that are either comprised of one token or
            # multiple.
            found_a_B_tag = False
            # We have to revert this back to False since we are done with the entity
            # that we found.
            ending_index = index
            # We will be going UP to this index when saving this entity in our list.

            df_to_append = data[starting_index:ending_index:]
            assert df_to_append.iloc[0].Tag[0:2:] == "B-"
            # if the first token of this entity isn't marked at the beginning, then
            # we know that we did something wrong.
            if len(df_to_append) > 1:
                # if we have an entity that is made up of more than token.
                assert df_to_append["Sentence #"].std() == 0
                # this is checking whether or not all of the tokens that comprise the
                # entity are in the same sentence. They have to be since there is no
                # way that an entity can span more than one sentence.
            list_of_entity_dfs.append(df_to_append)

    list_of_entity_strs = [
        entity_compiler(entity_df) for entity_df in list_of_entity_dfs
    ]
    # also compile the tags for each of these entities
    return (data, list_of_entity_strs)

def entity_compiler(entity_df, *args, **kwargs):
    """
    """
    to_return = (None, None, 0)
    ###
    entity_str_to_format = (len(entity_df) - 1) * "{} " + "{}"
    to_return = (
        entity_str_to_format.format(*entity_df.Word.tolist()),
        entity_df.Tag.iloc[0][2::],
        entity_df["Sentence #"].iloc[0],
    )
    return to_return
# ----------------------------------------------------------------------------------------------------------------------
"""
geo = Geographical Entity
org = Organization
per = Person
gpe = Geopolitical Entity
tim = Time indicator
art = Artifact
eve = Event
nat = Natural Phenomenon
"""

def entity_label_mapper(label):
    """
    Purpose
    -------

    Parameters
    ----------

    Returns
    -------

    References
    ----------
    1. https://spacy.io/api/annotation
    """
    if label == "per":
        mapped_label = "PERSON"
    elif label == "org":
        mapped_label = "ORG"
    elif label == "eve" or label == "nat":
        mapped_label = "EVENT"
    elif label == "geo":
        mapped_label = "LOC"
    elif label == "art":
        mapped_label = "WORK_OF_ART"
    elif label == "tim":
        mapped_label = "TIME"
    elif label == "gpe":
        mapped_label = "GPE"
    return mapped_label


# ---------------------------------------------------------------------------------------------------------------------
def article_doc_creator(words_iter):
    call_next = True
    spaces = []
    token = next(words_iter)[1]
    article_text = ""
    inside_quote = False
    # Concats article text together and determines whether or not their should be spaces in between tokens
    while True:
        article_text += token
        try:
            if call_next:
                next_token = next(words_iter)[1]  # the next row in the series
            call_next = True
            if next_token == '"' and not inside_quote:
                inside_quote = True
                article_text += " " + next_token
                token = next(words_iter)[1]
                token = next(words_iter)[1]
            elif next_token == '"' and inside_quote:
                inside_quote = False
                article_text += next_token + " "
                token = next(words_iter)[1]
            elif next_token not in [",", "'", ":", ".", "-", "'s"] and token not in [
                "'",
                "-",
                '"',
            ]:
                try:
                    next_next_token = next(words_iter)[1]
                    # # the next row in the series
                    if next_next_token == "." and token == ".":
                        spaces.append(False)
                    else:
                        article_text += " "
                    token = next_token
                    next_token = next_next_token
                    call_next = False
                except StopIteration:
                    article_text += " "
                    token = next_token
            else:
                if next_token == "'" and token.endswith("s"):
                    try:
                        article_text += next_token + " "
                        token = next(words_iter)[1]
                        continue
                    except StopIteration:
                        article_text += next_token
                        break
                spaces.append(False)
                token = next_token
        except StopIteration:
            spaces.append(False)
            break
    # returns a doc objects containing all article tokens and entities
    return article_text


# ----------------------------------------------------------------------------------------------------------------------
"""
Inputs:
golden_annotation: A list of tuples or lists that contain the entity text and the label
ex. [(China, GPE), (Tom Hanks, PERSON)]
nlp: SpaCy model

Outputs:
gold_truth_list: A list of all gold truth entities as Span Objects
gold_truth_dict: A dict of all gold truth entities as Span Objects (keys) and corresponding label (value)
"""


def gold_truth_creator(golden_annotation, nlp):
    """
    Purpose
    -------

    Parameters
    ----------

    Returns
    -------

    References
    ----------
    1.
    """
    ent_position = []
    i, j = 0, 0
    spaces = []
    words = []
    # splits annotated entities into individual tokens
    for ann_entity in golden_annotation:
        # strip the annotation
        split_commas = ann_entity[0].replace(",", " , ")
        split_colon = split_commas.replace(":", " : ")
        split_apostrophe = split_colon.replace("'", " ' ")
        split_dash = split_apostrophe.replace("-", " - ")
        split_period = split_dash.replace(".", " . ")
        tokens = split_period.split()
        j += len(tokens)
        token_iter = iter(tokens)
        ent_position.append([i, j])
        i = j
        # Creates the words and spaces lists to be used in creating the annotated data doc
        call_next = True
        token = next(token_iter)
        while True:
            words.append(token)
            try:
                if call_next:
                    next_token = next(token_iter)  # the next row in the series
                call_next = True
                if next_token not in [",", "'", ":", ".", "-", "'s"] and token not in [
                    "'",
                    "-",
                    '"',
                ]:
                    try:
                        next_next_token = next(token_iter)
                        # # the next row in the series
                        if next_next_token == "." and token == ".":
                            spaces.append(False)
                        else:
                            spaces.append(True)
                        token = next_token
                        next_token = next_next_token
                        call_next = False
                    except StopIteration:
                        spaces.append(True)
                        token = next_token
                else:
                    if next_token == "'" and token.endswith("s"):
                        try:
                            spaces.append(False)
                            words.append(next_token)
                            spaces.append(True)
                            token = next(token_iter)
                            continue
                        except StopIteration:
                            spaces.append(False)
                            words.append(next_token)
                            spaces.append(False)
                            break
                    spaces.append(False)
                    token = next_token
            except StopIteration:
                spaces.append(False)
                break
    annotated_data_doc = Doc(nlp.vocab, words=words, spaces=spaces)
    ground_truth_dict = {}
    ground_truth_list = []
    i = 0
    for start, end in ent_position:
        ann_ent = annotated_data_doc[start:end]
        ground_truth_list.append(ann_ent)
        ground_truth_dict[ann_ent] = golden_annotation[i][1]
        i += 1
    return (ground_truth_list, ground_truth_dict)

# ----------------------------------------------------------------------------------------------------------------------
def connect_to_db(database, hostname, port, userid, passwrd):
    # create string
    conn_string = "host={} port={} dbname={} user={} password={}".format(
        hostname, port, database, userid, passwrd
    )
    # connect to the database with the connection string
    conn = psycopg2.connect(conn_string)
    # commits all queries you execute
    conn.autocommit = True
    cursor = conn.cursor()
    return conn, cursor

#-----------------------------------------------------------------------------------------------------------------------
