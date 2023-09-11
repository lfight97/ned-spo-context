import os
import json
import numpy as np
import sys
import math
from ipywidgets import IntProgress

from utils import bin_data_into_buckets, get_words, infer_vector_from_word
import pandas as pd
import torch
import logging
from roberta_classification import load_and_cache_examples,train
from transformers import (WEIGHTS_NAME, RobertaForSequenceClassification, RobertaTokenizer)
from arguments import args


_path = os.path.dirname(__file__)
_saving_dir = os.path.join(_path, '../data/')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
_bucket_size = 10
_minimum_trace = 10
# Create a TransformerModel
#model = TransformerModel('roberta', 'roberta-base')


MODEL_CLASSES = {
    #'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    #'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    #'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
}

config_class, model_class, tokenizer_class = MODEL_CLASSES[args['model_type']]
task = args['task_name']
tokenizer = tokenizer_class.from_pretrained(args['model_name'])
model = model_class.from_pretrained(args['model_name'])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def get_answers_and_questions_from_json(filename):
    questions_and_answers = []
    dataset_dicts = json.loads(open(filename).read())
    for item in dataset_dicts:
        questions_and_answers.append({'question': item['qText'], 'answers': item['answers']})
    return questions_and_answers


def find_position_of_best_match(candidate_vectors, answer_vector):
    old_distance = 10000
    position = -1
    for index, candidate in enumerate(candidate_vectors):
        distance = np.linalg.norm(candidate - answer_vector)
        if distance < old_distance:
            position = index
            old_distance = distance
    return position


def get_vector_list_from_sentence(model, sentence):
    words = get_words(sentence)
    vectors = []
    for word in words:
        vectors.append(infer_vector_from_word(model, word))
    return vectors

if __name__ == '__main__':
    train_dataset = load_and_cache_examples(task, tokenizer)
    global_step, tr_loss = train(train_dataset, model, tokenizer)
    logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
    
    if not os.path.exists(args['output_dir']):
            os.makedirs(args['output_dir'])
    logger.info("Saving model checkpoint to %s", args['output_dir'])
    
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(args['output_dir'])
    tokenizer.save_pretrained(args['output_dir'])
    torch.save(args, os.path.join(args['output_dir'], 'training_args.bin'))