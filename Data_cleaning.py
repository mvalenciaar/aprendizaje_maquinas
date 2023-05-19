"""Importar librerías de trabajo"""
import numpy as np
import pandas as pd
import re
from bs4 import BeautifulSoup
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import nltk
import ssl
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.tokenize.toktok import ToktokTokenizer

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


def process_data():
    """
    Descarga de corpus necesarios para hacer el procesamiento de lenguaje natural
    """
    nltk.download('wordnet')  
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')

    """Lectura del dataset"""
    df_raw = pd.read_csv('./IMDB Dataset.csv')
    df_raw.head()

    """Marcación de 1 "uno" para reseñas positivas y 0 "cero" para negativas"""
    df_raw['sentiment'].replace(['positive', 'negative'], [1, 0], inplace = True)
    df_raw.head()

    """Validación de valores duplicados o nulos"""
    duplicates = df_raw[df_raw.duplicated()]
    df_raw.isnull().sum()
    df_raw.drop_duplicates(inplace=True)

    """Gráfica de distribución para evaluar si existe desbalanceo de datos"""
    print(sns.countplot(x=df_raw['sentiment']))

    """Limpieza de reseñas"""
    stopword_list=nltk.corpus.stopwords.words('english')
    stop=set(stopwords.words('english'))

    def strip_html(text):
        '''
        Eliminación de etiquetas HTML
        '''
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text()

    def remove_between_square_brackets(text):
        '''
        Eliminación de corchetes
        '''
        return re.sub('\[[^]]*\]', '', text)

    def denoise_text(text):
        """
        Eliminación tags de e-mail y corchetes
        """
        text = strip_html(text)
        text = remove_between_square_brackets(text)
        return text

    def remove_Emails(text):
        """
        Eliminación de e-mai
        l"""
        pattern=r'\S+@\S+'
        text=re.sub(pattern,'',text)
        return text

    def remove_URLS(text):
        '''
        Eliminación de  etiquetas de URLs
        '''
        pattern=r'http\S+'
        text=re.sub(pattern,'',text)
        return text

    def remove_special_characters(text, remove_digits=True):
        """
        Eliminación de caracteres especiales
        """
        pattern=r'[^a-zA-z0-9\s]'
        text=re.sub(pattern,'',text)
        return text

    def remove_numbers(text):
        """
        Eliminación de números
        """
        pattern = r'\d+'
        text = re.sub(pattern, '', text)
        return text

    def lowercase_text(text):
        """
        Convertir texto a minúscula
        """
        return text.lower()

    def reviewsCleaning(df):
        """
        Limpieza de revisiones
        """
        df['review']=df['review'].apply(denoise_text)
        df['review']=df['review'].apply(remove_URLS)
        df['review']=df['review'].apply(remove_Emails)
        df['review']=df['review'].apply(remove_special_characters)
        df['review']=df['review'].apply(remove_numbers)
        df['review']=df['review'].apply(lowercase_text)
        return df
    
    df_raw = reviewsCleaning(df_raw)

    positivedata = df_raw[df_raw['sentiment'] == 1]
    positivedata = positivedata['review']
    negdata = df_raw[df_raw['sentiment']== 0]
    negdata = negdata['review']
    nltk.download('stopwords')

    def wordcloud_draw(data, color, s):
        '''
        Creación de nube de palabras, donde se verifican palabras comunes en los dos tipos de reseñas
        '''
        words = ' '.join(data)
        cleaned_word = " ".join([word for word in words.split() if(word!='movie' and word!='film')])
        wordcloud = WordCloud(stopwords=STOPWORDS,background_color=color,width=2500,height=2000).generate(cleaned_word)
        plt.imshow(wordcloud)
        plt.title(s)
        plt.axis('off')

    plt.figure(figsize=[20,10])
    plt.subplot(1,2,1)
    wordcloud_draw(positivedata,'white','Palabras positivas más comunes')

    plt.subplot(1,2,2)
    wordcloud_draw(negdata, 'white','Palabras negativas más comunes')
    plt.show()

    def get_wordnet_pos(tag):
        """
        Mapeo del etiquetado gramatical para el proceso de lematización
        """
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def preprocess_review(review):
        """
        Preprocesado de reseñas usando la técnica de lematización
        """
        lemmatizer = WordNetLemmatizer()
        # Tokeniza las reseñas en palabras
        words = nltk.word_tokenize(review)
        
        # Etiqueta las palabras en su categoría gramatical
        tagged_words = nltk.pos_tag(words)
        
        # Lematiza las palabras usando las etiquetas gramaticales
        lemmatized_words = []
        for word, tag in tagged_words:
            pos = get_wordnet_pos(tag)
            lemmatized_word = lemmatizer.lemmatize(word, pos=pos)
            lemmatized_words.append(lemmatized_word)
        
        # Devuelve las palabras en un solo texto
        preprocessed_review = ' '.join(lemmatized_words)
        
        return preprocessed_review

    print ('Antes de Lematización.. \n',df_raw['review'][2])
    df_raw['review']=df_raw['review'].apply(preprocess_review)
    print ('Después de Lematización .. \n',df_raw['review'][2])

    tokenizer=ToktokTokenizer()

    def remove_stopwords(text, is_lower_case=True):
        '''
        Se remueven palabras comunes innecesarias (articulos, preposiciones, pronombres)
        '''
        tokens = tokenizer.tokenize(text)
        tokens = [token.strip() for token in tokens]
        if is_lower_case:
            filtered_tokens = [token for token in tokens if token not in stopword_list]
        else:
            filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
        filtered_text = ' '.join(filtered_tokens)    
        return filtered_text
    
    print ('Antes de remover stopwords.. \n',df_raw['review'][2])
    df_raw['review']=df_raw['review'].apply(remove_stopwords)
    print ('Después de remover stopwords .. \n',df_raw['review'][2])

    return df_raw #.to_csv('IMDB_reviews_cleaned.csv', index = False)