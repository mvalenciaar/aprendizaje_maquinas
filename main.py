import sys
import Data_cleaning
import data_plot
import data_nlp
import data_tokenizing
import models_CNN
import models_RL
import models_LSTM


if __name__ == "__main__":
    Data_cleaning.clean_data()
    data_plot.plot_data_analysis()
    data_nlp.show_nlp_results()
    data_tokenizing.tokenize_data()
    models_CNN.run_cnn_model()
    models_LSTM.run_lstm_model()
    models_RL.run_lr_model()


