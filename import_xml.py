import xml.etree.ElementTree as ET
import pandas as pd
import langid
from langdetect import detect
import textblob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.data import load
from nltk.stem import SnowballStemmer
from string import punctuation
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import io



#Funciones que me detectan el idioma de un
def landig_safe(tweets):
    try:
        return langid.classify(tweets)[0]
    except Exception as e:
        pass

def langdetect(tweets):
    try:
        return detect(tweets)
    except Exception as e:
        pass

def textblob_safe(tweets):
    try:
        return textblob.TextBlob(tweets).detect_language()
    except Exception as e:
        pass




def process():
    tree = ET.parse('general-train-tagged-3l_large.xml')
    root = tree.getroot()

    # for i in range(len(root)):
    #     print(root[i][2].text)
        

    tweets = []
    sentiment = []
    
    find_sentiments = root.findall("./tweet/sentiments")
    find_content = root.findall("./tweet/content")


    for i in range(len(find_content)):
        if find_sentiments[i][0][0].text != 'NONE':
            tweets.extend([find_content[i].text])
            if find_sentiments[i][0][0].text.lower()=='p' or find_sentiments[i][0][0].text.lower() == 'NEU':
                sentiment.extend([1])
            if find_sentiments[i][0][0].text.lower()=='n':
                sentiment.extend([0])
            print(find_sentiments[i][0][0].text.lower(), i)



    tweets_pd = pd.DataFrame(zip(tweets, sentiment), columns=['tweets', 'sentiment'])
    tweets_pd.to_csv('tweets_extraidos_large1.csv', encoding='utf-8')


    return tweets_pd

non_words = list(punctuation)
non_words.extend(['¿', '¡'])
non_words.extend(map(str, range(10)))


def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    text = ''.join([c for c in text if c not in non_words])
    tokens = word_tokenize(text)

    #steaming
    try:
        stem = stem_tokens(tokens, stemmer)
    except Exception as e:
        print(e)
        print(text)
        stem = ['']
    return stem


print('Que operación desea?\n1----->Extraer tweets\n2----->En que idioma están los tweets\n3------>SVM grid serach and fit')
opt = input()
#extrayendo los tweets del archivo xml
if opt=='1':
    print('--------------------procesando---------------------------------------')
    tweets_id = process()

#funciones que detectan el idioma de los tweets
if opt =='2':
    print('--------------------detector idiomas---------------------------------------')
    tweets_id = pd.read_csv('tweets_extraidos_large.csv', encoding='utf-8')
    tweets_id['first_len_fun'] = tweets_id.tweets.apply(landig_safe)
    tweets_id['second_len_fun'] = tweets_id.tweets.apply(langdetect)
    tweets_id['third_len_fun'] = tweets_id.tweets.apply(textblob_safe)
    #guardando lo anterior
    tweets_id.to_csv('tweets_extraidos_id_large.csv', encoding='utf-8')



if opt=='3':
    print('-------------------------------------------al vectorizdor-----------------------------------------------------')

    tweets_id = pd.read_csv('tweets_extraidos_large.csv', encoding='utf-8')
    #preguntandole al dataframe
    #tweets_id = tweets_id.query('first_len_fun == "es" or second_len_fun == "es" or third_len_fun == "es"')
    spanish_stopwords = stopwords.words('spanish')
    stemmer = SnowballStemmer('spanish')

    #iniciaizando 
    english_stopwords = [line.rstrip('\n') for line in open('english_stopwords.txt')]
    english_stopwords.extend(['algun', 'com','contr', 'cuand', 'desd', 'dond', 'durant', 'eram', 'estab', 'estais', 'estam', 'estan', 'estand', 'estaran', 'estaras', 'esteis', 'estem', 'esten', 'estes', 'estuv', 'fuer', 'fues', 'fuim', 'fuist', 'hab', 'habr', 'habran', 'habras', 'hast', 'hem', 'hub', 'mas', 'mia', 'mias', 'mio', 'mios', 'much', 'nad', 'nosotr', 'nuestr', 'par', 'per', 'poc', 'porqu', 'qui', 'seais', 'seam', 'sent', 'ser', 'seran', 'seras', 'si', 'sient', 'sint', 'sobr', 'som', 'suy', 'tambien', 'tant', 'ten', 'tendr', 'tendran', 'tendras', 'teng', 'tien', 'tod', 'tuv', 'tuy', 'vosotr', 'vuestr', 'abov', 'becaus', 'befor', 'betw', 'furth', 'hav', 'mor', 'ourselv', 'sam', 'tambi', 'themselv', 'ther', 'thes', 'thos', 'wer', 'wher', 'whil', 'yourselv'])
    spanish_stopwords.extend(english_stopwords)
    print(spanish_stopwords)

    vectorizer = CountVectorizer(analyzer = 'word', tokenizer = tokenize, lowercase=True, stop_words=spanish_stopwords)

    vector = vectorizer.fit_transform(tweets_id.tweets)

    #ahora se está haciendo el entrenamiento
    pipeline = Pipeline([('vect', vectorizer), ('cls', LinearSVC())])

    #se va a hacer un barrido de parámetros con la función GridSearchCV
    parameters = {
        'vect__max_df':(0.5, 1.9),
        'vect__min_df':(10, 20, 50),
        'vect__max_features':(2000),
        'vect__ngram_range':((1, 1), (1, 2)),
        'cls__C':(0.2, 0.5, 0.7),
        'cls__loss':('hinge', 'squared_hinge'),
        'cls__max_iter':(500, 1000)
    }

    print('----------------------------y el entrenamiento---------------------')
    grind_search = GridSearchCV(pipeline, parameters, n_jobs=-1, scoring='roc_auc')
    grind_search.fit(tweets_id.tweets, tweets_id.sentiment)

    print(grind_search.best_params_)
    best_param = pd.DataFrame(grind_search.best_params_)
    best_prams.to_csv('best_params.csv', encoding='utf-8')