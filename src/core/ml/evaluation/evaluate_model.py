import numpy as np

from core.ml.pipeline.pipeline import Pipeline
from core.ml.prepare_dataset import get_test_data

test_dataset = get_test_data()
pipeline = Pipeline()
pipeline.cls.load_weights(pipeline.checkpoint_path)
# pipeline.cls.evaluate([test_dataset['input_sentence'], test_dataset['output_sentence']])

output_vocab = pipeline.output_vectorizer.vectorizer.get_vocabulary()
output_index_lookup = dict(zip(range(len(output_vocab)), output_vocab))
max_decoded_sentence_length = pipeline.params['decoded_sequence_length']

pipeline.input_vectorizer.vectorizer.adapt(test_dataset['input_sentence'])
pipeline.output_vectorizer.vectorizer.adapt(test_dataset['output_sentence'])


def decode_sequence(input_sentence):
    tokenized_input_sentence = pipeline.input_vectorizer.vectorizer([input_sentence])
    decoded_sentence = "[start]"
    for i in range(max_decoded_sentence_length):
        tokenized_target_sentence = pipeline.output_vectorizer.vectorizer([decoded_sentence])[:, :-1]
        predictions = pipeline.cls([tokenized_input_sentence, tokenized_target_sentence])

        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = output_index_lookup[sampled_token_index]
        decoded_sentence += " " + sampled_token

        if sampled_token == "[end]":
            break
    return decoded_sentence


for _ in range(10):
    sentence = test_dataset.sample(n=1)
    translated = decode_sequence(sentence['input_sentence'])
    print(sentence['input_sentence'].values, '===>', translated,
          f"Ground truth: {sentence['output_sentence'].values}")
