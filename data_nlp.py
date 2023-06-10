from Data_cleaning import preprocess_review, remove_stopwords, generate_stopword_tokenizer
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import seaborn as sns

def draw_n_gram(string,i = 1, title = "Basic unigram"):
    n_gram = (pd.Series(nltk.ngrams(string, i)).value_counts())[:10]
    n_gram_df=pd.DataFrame(n_gram)
    n_gram_df = n_gram_df.reset_index()
    n_gram_df = n_gram_df.rename(columns={"index": "word", 0: "count"})
    print(n_gram_df.head(n = 10))
    plt.figure(figsize = (10,5))
    plt.title(title)

    return sns.barplot(x='count',y='word', data=n_gram_df)

def show_nlp_results(imdb_df):
    '''Función para mostrar el proceso hecho en el data_cleaning'''
   
    (tokenizer, stopword_list) = generate_stopword_tokenizer()

    print ('Antes de preprocesar reviews... \n',imdb_df['review'][2])
    imdb_df['review'] = imdb_df['review'].apply(preprocess_review)
    print ('Después de preprocesar reviews...\n',imdb_df['review'][2]  + '\n')

    print ('================================================Proceso para remover stopwords================================================')
    imdb_df['review'] = imdb_df['review'].apply(lambda x: remove_stopwords(tokenizer, stopword_list, x))
    print ('Después de remover stoprwords... \n',imdb_df['review'][2] + '\n')

    return imdb_df