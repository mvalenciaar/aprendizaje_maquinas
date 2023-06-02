from data_cleaning import clean_data, preprocess_review, remove_stopwords, generate_stopword_tokenizer
def show_nlp_results(imdb_df):
    '''Función para mostrar el proceso hecho en el data_cleaning'''
   
    (tokenizer, stopword_list) = generate_stopword_tokenizer()

    print ('Antes de preprocesar reviews... \n',imdb_df['review'][2])
    imdb_df['review'] = imdb_df['review'].apply(preprocess_review)
    print ('Después de preprocesar reviews...\n',imdb_df['review'][2])

    print ('Antes de remover stopwords... \n',imdb_df['review'][2])
    imdb_df['review'] = imdb_df['review'].apply(lambda x: remove_stopwords(tokenizer, stopword_list, x))
    print ('Después de remover stoprwords... \n',imdb_df['review'][2])

    return imdb_df