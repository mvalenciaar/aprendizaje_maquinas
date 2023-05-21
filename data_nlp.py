from data_cleaning import clean_data, preprocess_review, remove_stopwords, generate_stopword_tokenizer
def show_nlp_results():
    imdb_df = clean_data()
    (tokenizer, stopword_list) = generate_stopword_tokenizer()

    print ('BEFORE (remove_stopwords).. \n',imdb_df['review'][2])
    imdb_df['review'] = imdb_df['review'].apply(preprocess_review)
    print ('AFTER (remove_stopwords) .. \n',imdb_df['review'][2])

    print ('BEFORE (remove_stopwords).. \n',imdb_df['review'][2])
    imdb_df['review'] = imdb_df['review'].apply(lambda x: remove_stopwords(tokenizer, stopword_list, x))
    print ('AFTER (remove_stopwords) .. \n',imdb_df['review'][2])