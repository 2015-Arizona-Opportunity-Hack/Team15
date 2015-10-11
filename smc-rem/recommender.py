import re
import string

import gensim
import html2text
import nltk
import numpy as np
import pandas
from sklearn.feature_extraction import DictVectorizer


def nullToEmptyString(x):
    return '' if pandas.isnull(x) else x


def body2text(x):
    return html2text.html2text(nullToEmptyString(x))


def why_select(bodytxt):
    res = re.search('why:(.*)', bodytxt, re.IGNORECASE | re.DOTALL)
    return '' if res is None else res.group(1).strip(' \t\n*')


def club_select(tags):
    clubs = {'art club',
             'basketball',
             'dance',
             'fashion club'
             'golf',
             'robotics',
             'soccer',
             'softball',
             'student council',
             'track and field'
             }
    tags = set([tag.strip() for tag in tags.lower().split(',')])
    club = list(clubs & tags)
    return club[0] if len(club) > 0 else ''


orders = pandas.read_csv('orders_export.csv')
customers = pandas.read_csv('customers_export.csv')
products = pandas.read_csv('products_export.csv')

# Must use the same column name to join
orders['SKU'] = orders['Lineitem sku']
products['SKU'] = products['Variant SKU']
products['IntIndex'] = range(len(products.index))
products['BodyText'] = products['Body (HTML)'].apply(body2text)
products['Reason'] = products['BodyText'].apply(why_select)
products['Club'] = products['Tags'].apply(lambda x: club_select(str(x)))
# Drop orders that do not have an SKU, like recurring donations, gift cards
item_orders = orders[~pandas.isnull(orders['SKU'])]
# Inner join orders with products, so we do not keep orders without an SKU
# such as recurring donations
orders_plus = pandas.merge(item_orders, products, on='SKU', how='inner',
                           suffixes=('.o', '.p'))
lower_strip = lambda x: x.lower().strip()
orders_plus['CanonEmail'] = orders_plus['Email'].apply(lower_strip)
# Inner join orders_plus with customers, so we do not keep orders without a
# matching customer
orders_plus = pandas.merge(orders_plus, customers, on='Email', how='inner',
                           suffixes=('.o', '.c'))


school_rewrite = {'Arcadia': 'Arcadia High',
                  'Bioscience': 'Bioscience High',
                  'Thunderbird': 'Thunderbird High'}
products['School'] = [school_rewrite[vendor] if vendor in school_rewrite
                      else vendor for vendor in products['Vendor']]

d_school = [({} if pandas.isnull(school) else {'School': school}) for school
            in products['School']]
v_school = DictVectorizer(sparse=False)
x_school = v_school.fit_transform(d_school)
d_type = [({} if pandas.isnull(ptype) else {'Type': ptype}) for ptype in 
          products['Type']]
v_type = DictVectorizer(sparse=False)
x_type = v_school.fit_transform(d_type)


documents = products['BodyText']
# remove common words and tokenize
stoplist = nltk.corpus.stopwords.words('english')
stripPunc = lambda s: s.strip(string.punctuation)
texts = [[word for word in map(stripPunc, document.lower().split())
          if word not in stoplist] for document in documents]
# add tags
tag_lists = products['Tags'].apply(nullToEmptyString)
tags = [[word.lower().strip() for word in tag_list.split(',')] for tag_list
        in tag_lists]
texts = [tags[idx] + texts[idx] for idx in range(len(texts))]
# remove words that appear less than threshold
threshold_count = 1
from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

# construct the dictionary and corpus
dictionary = gensim.corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
tfidf = gensim.models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]
num_topics = 15
lsi = gensim.models.LsiModel(corpus_tfidf, id2word=dictionary,
                             num_topics=num_topics)
corpus_lsi = lsi[corpus_tfidf]
x_text = gensim.matutils.corpus2dense(corpus_lsi, num_topics).T

school_offset = 0
school_n = x_school.shape[1]
school_cols = range(school_offset, school_offset + school_n)
type_offset = school_n
type_n = x_type.shape[1]
type_cols = range(type_offset, type_offset + type_n)
text_offset = type_offset + type_n
text_n = x_text.shape[1]
text_cols = range(text_offset, text_offset + text_n)

x = np.hstack((x_school, x_type, x_text))


def get_customers():
    orders_plus['total_price'] = orders_plus['Lineitem price'] * \
        orders_plus['Lineitem quantity']
    cgroups = orders_plus.groupby('Email')
    df = pandas.DataFrame(data={'total_orders': cgroups.size()})
    df['total_spent'] = cgroups['total_price'].agg({'total_spent': np.sum})
    df['email'] = cgroups['Email'].agg({'email': lambda g: g.iloc[0]})
    df['city'] = cgroups['City'].agg({'city': lambda g: g.iloc[0]})
    return df


def get_orders(email):
    email = email.lower().strip()
    prev_orders = orders_plus[orders_plus['CanonEmail'] == email]
    return prev_orders


def get_recommendation(email, topN=10, school_weight=1, type_weight=1,
                       text_weight=1):
    prev_orders = get_orders(email)
    n_prev_orders = len(prev_orders.index)
    if n_prev_orders < 1:
        return None
    prev_indices = prev_orders['IntIndex'].values
    if n_prev_orders == 1:
        avg_order = x[prev_orders['IntIndex']]
    else:
        avg_order = np.sum(x[prev_orders['IntIndex']], axis=0) / \
            (1.0 * n_prev_orders)
    query = np.concatenate((avg_order[school_cols] * school_weight,
                            avg_order[type_cols] * type_weight,
                            avg_order[text_cols] * text_weight))
    recs = products.iloc[[idx for idx in np.argsort(np.dot(x, query))[::-1]
                          if idx not in prev_indices]].iloc[:topN]
    # recs = recs.loc[:, ('Handle', 'Vendor', 'Reason', 'Club',
    #                     'Variant Inventory Qty', 'Variant Price')]
    return recs



