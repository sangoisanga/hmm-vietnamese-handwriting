from main.feature.specialized_hmm import SpecializedHMM
from main.word.word_classifier import WordClassifier
from main.word.word_examples_generator import generate_examples_for_words


def create_word_classifier(word_list, save_to_file_path):
    training_examples = generate_examples_for_words(words=word_list,
                                                    number_of_examples=800,
                                                    poelap=0.03,
                                                    poelenl=0.7,
                                                    powlap=0.1,
                                                    polmap=0.03)
    classifier = WordClassifier(training_examples,
                                nr_of_hmms_to_try=1,
                                fraction_of_examples_for_test=0,
                                train_with_examples=True,
                                initialisation_method=SpecializedHMM.InitMethod.count_based)
    classifier_string = classifier.to_string()
    file = open(save_to_file_path, 'w')
    file.write(classifier_string)
    file.close()


if __name__ == '__main__':
    word_list = ["dog", "cat", "pig", "love", "hate",
                 "scala", "python", "summer", "winter", "night",
                 "daydream", "nightmare", "animal", "happiness", "sadness",
                 "tennis", "feminism", "fascism", "socialism", "capitalism"]
    create_word_classifier(word_list, "word_classifier.dat")
