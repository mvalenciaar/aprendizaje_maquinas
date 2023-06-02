import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import nltk
from nltk.corpus import stopwords, wordnet
import matplotlib.pyplot as plt
from data_cleaning import clean_data

def plot_data_analysis(df_raw):
    '''Gráficas de desbalanceo y nube de palabras'''
    print('======================== NUBE DE PALABRAS =================================')

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