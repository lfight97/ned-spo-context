import os
import json
import numpy as np
import glob

from utils import get_words, infer_vector_from_word
import torch
from roberta_evaluate import evaluate
import logging
from roberta_classification import load_and_cache_examples
from pytorch_transformers import (WEIGHTS_NAME, 
                                  #BertConfig, BertForSequenceClassification, BertTokenizer,
                                  #XLMConfig, XLMForSequenceClassification, XLMTokenizer, 
                                  #XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer,
                                  RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                                TensorDataset)


from arguments import args

_path = os.path.dirname(__file__)
_saving_dir = os.path.join(_path, '../data/')
_bucket_size = 10

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
MODEL_CLASSES = {
    #'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    #'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    #'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
}

config_class, model_class, tokenizer_class = MODEL_CLASSES[args['model_type']]
task = args['task_name']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = tokenizer_class.from_pretrained(args['model_name'])



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


_is_relevant = 1
_is_not_relevant = 0

def erase_edges_with_mask(A, mask):
    for i, erase_row in enumerate(mask):
        if erase_row:
            A[i] = 0
    return A


def get_prediction_from_models(A_fw, A_bw, vectors, types, question_vectors, models):
    predictions = {}
    for i, model in enumerate(models):
        prediction = model.predict(A_fw, A_bw, vectors, types, question_vectors)
        for j, item in enumerate(prediction):
            predictions[(j, i)] = item
    prediction_list = []
    for j in range(len(vectors)):
        all_predictions = [predictions[(j, i)] for i in range(len(models))]
        item_dict = {}
        for item in all_predictions:
            try:
                item_dict[tuple(item)] += 1
            except:
                item_dict[tuple(item)] = 1
        final_prediction = list(sorted(item_dict.items(), key=lambda x: -x[1])[0][0])
        prediction_list.append(final_prediction)
    return prediction_list

if __name__ == '__main__':
    #with open(os.path.join('/home/IAIS/cprabhu/Thesis/ned-graphs/', 'dataset/wikidata-disambig-test.json')) as f:
    #    json_data = json.load(f)
    #test_df = get_json_data(json_data)
    #test_df.to_csv(os.path.join(args['data_dir'], 'dev.tsv'), sep='\t', index=False, header=False, columns=test_df.columns)
    test_dataset = load_and_cache_examples(task, tokenizer, True)
    results = {}
    checkpoints = [args['output_dir']]

    if args['eval_all_checkpoints']:
        checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args['output_dir'] + '/**/' + WEIGHTS_NAME, recursive=True)))
        logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
    logger.info("Evaluate the following checkpoints: %s", checkpoints)

    #for checkpoint in checkpoints:
    global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
    # checkpoint = './data/outputs_context_merged_top3'
    # checkpoint = './data/outputs_context_merged_top3_with_desc'
    checkpoint = './data/outputs_context_merged_window3_with_desc'
    print('loading from %s'%checkpoint)
    model = model_class.from_pretrained(checkpoint)
    model.to(device)
    torch.cuda.empty_cache()
    result, wrong_preds = evaluate(test_dataset, model, tokenizer, prefix=global_step)
    result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
    results.update(result)