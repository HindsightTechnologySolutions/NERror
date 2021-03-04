import boto3
import pickle
import json
from s3_credentials import ACCESS_KEY, SECRET_KEY

class Connect_Bucket:

    def __init__(self):
        self.s3client = boto3.client('s3', aws_access_key_id = ACCESS_KEY, \
                                    aws_secret_access_key = SECRET_KEY)
        read_response = self.s3client.get_object(Bucket = '', \
                                        Key = 'identified_errors_table.pkl')
        body = read_response['Body'].read()
        self.dataframe = pickle.loads(body)

        read_response = self.s3client.get_object(Bucket = '', \
                                        Key = 'error_examples.json')
        body = read_response['Body'].read()
        self.error_examples = json.loads(body)

    # check if the article has already been processed, return the article's info
    # if not, return None
    def if_existed(self, article_url):
        # obtain data from bucket
        try:
            result = self.dataframe.loc[[article_url]]
            return result
        except KeyError:
            return None

    # def enough_err_examples():
    #     # check if the number of error examples are enough for each type of errors
    #     return all(len(i) >= 3 for i in error_examples.values())

    # call this function only when the data already exists in the dataframe
    # def get_error_type_list(self, url):
    #     self.dataframe.loc[url]

    def get_err_examples(self):
        return self.error_examples

    def save_err_examples(self, err_examples):
        self.s3client.put_object(
            Bucket = '',\
            Body = str(json.dumps(err_examples)), \
            Key = 'error_examples.json')

    def save_identified_errors(self, url, identified_errors):
        # if the database does not have the url saved
        self.dataframe.at[url] = identified_errors
        to_write = pickle.dumps(self.dataframe)
        self.s3client.put_object(
                Bucket = '', \
                Body = to_write, \
                Key = 'identified_errors_table.pkl')
