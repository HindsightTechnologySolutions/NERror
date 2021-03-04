import csv
import psycopg2

# Connects to AWS Database containing necessary tables
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


# Creates a cursor object that allows us to create tables and query into existing tables using SQL language
conn, cursor = connect_to_db(
    database="postgres",
    hostname="xxxx.us-east-1.rds.amazonaws.com",
    port="5432",
    userid="x",
    passwrd="x",
)


def create_test_text():
    # transform the golden annotation into a dictionary
    annotated_entities = []
    with open("../data/annotated_entities.csv", "r") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            annotated_entities.append(row[0].strip())
    found_entities = []
    with open('../data/article_example_found_entities.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            found_entities.append(row[0].strip())
    # read article text
    file = open("../data/example_article.txt")
    article_text = file.read()
    file.close()
    return (annotated_entities, article_text, found_entities)
