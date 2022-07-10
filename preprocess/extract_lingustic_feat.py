import nltk
import pickle
import numpy as np
import pandas as pd
import json
import argparse
import os
import spacy # 2.1.0
import neuralcoref

nlp = spacy.load('en') #python -m spacy download en
neuralcoref.add_to_pipe(nlp)

def encode(seq_tokens, token_to_idx, allow_unk=False):
    seq_idx = [token_to_idx['<SOS>']]
    for token in seq_tokens:
        if token not in token_to_idx:
            if allow_unk:
                token = '<UNK>'
            else:
                raise KeyError('Token "%s" not in vocab' % token)
        seq_idx.append(token_to_idx[token])
    seq_idx.append(token_to_idx['<EOS>'])
    return seq_idx

def openeded_encoding_data(args, vocab, questions, video_names, answers, dialogs, mode='train'):
    ''' Encode question tokens'''
    print('Encoding data')
    dialogs_encoded = [] 
    dialogs_len = []
    questions_encoded = []
    questions_len = []
    answers_encoded = []
    answers_len = []
    video_ids = []
    for idx, question in enumerate(questions):
        # question = question.lower()
        # question_tokens = nltk.word_tokenize(question)
        question_tokens = question.split()
        question_encoded = encode(question_tokens, vocab['question_token_to_idx'], allow_unk=True)

        cap_sum_dialog = dialogs[idx]
        dialog_encoded_accumulate = []
        dialog_encoded = [[vocab['question_token_to_idx']['<NULL>']]] # first answer is <NULL>
        dialog_len = [1]
        
        for sentence in cap_sum_dialog:
            # sentence = sentence.lower()
            # sentence_tokens = nltk.word_tokenize(sentence)
            sentence_tokens = sentence.split()
            sentence_encoded = encode(sentence_tokens, vocab['question_token_to_idx'], allow_unk=True)
            dialog_encoded_accumulate.extend(sentence_encoded)
            if args.no_accumulate:
                dialog_encoded.append(sentence_encoded)
                dialog_len.append(len(sentence_encoded))
            else:
                dialog_encoded.append(dialog_encoded_accumulate.copy())
                dialog_len.append(len(dialog_encoded_accumulate.copy()))
        
        if args.question_accumulate:
            questions_encoded.append(dialog_encoded_accumulate + question_encoded)
            questions_len.append(min(len(dialog_encoded_accumulate + question_encoded), 80))
        else:
            questions_encoded.append(question_encoded)
            questions_len.append(len(question_encoded))

        while len(dialog_encoded) < 19: #fixed number of cap_sum_dialogs per video
            if args.no_accumulate:
                # dialog_encoded.append([vocab['question_token_to_idx']['<NULL>']])
                # dialog_len.append(1)

                # Pre-padding
                dialog_encoded = [[vocab['question_token_to_idx']['<NULL>']]] + dialog_encoded
                dialog_len = [1] + dialog_len
            else:
                dialog_encoded.append(dialog_encoded_accumulate.copy())
                dialog_len.append(len(dialog_encoded_accumulate.copy()))

        # if len(dialog_encoded) == 0: # pad zero dialog
        #     if args.no_accumulate:
        #         dialog_encoded.append([vocab['question_token_to_idx']['<NULL>']])
        #         dialog_len.append(1)
        #         dialog_encoded.append([vocab['question_token_to_idx']['<NULL>']])
        #         dialog_len.append(1)
        #     else:
        #         dialog_encoded.append(dialog_encoded_accumulate.copy())
        #         dialog_len.append(len(dialog_encoded_accumulate.copy()))
        #         dialog_encoded.append(dialog_encoded_accumulate.copy())
        #         dialog_len.append(len(dialog_encoded_accumulate.copy()))
        
        dialogs_encoded.append(dialog_encoded)
        dialogs_len.append(dialog_len)
        
        answer = answers[idx]
        # answer = answer.lower()
        # answer_tokens = nltk.word_tokenize(answer)
        answer_tokens = answer.split()
        answer_encoded = encode(answer_tokens, vocab['question_token_to_idx'], allow_unk=True)
        answers_encoded.append(answer_encoded)
        answers_len.append(len(answer_encoded))

        video_ids.append(video_names[idx])

    # Concat all dialogs
    # dialogs_encoded = [np.concatenate(dialog_encoded) for dialog_encoded in dialogs_encoded]
    # dialogs_len = [sum(dialog_len) for dialog_len in dialogs_len]

    # Pad encoded questions
    if args.question_accumulate:
        max_question_length = 80
    else:
        max_question_length = max(len(x) for x in questions_encoded)
    print(f"Max question length: {max_question_length}")
    new_questions_encoded = np.ones((len(questions_encoded), max_question_length), dtype=np.int32) * vocab['question_token_to_idx']['<NULL>']
    for i, question_encoded in enumerate(questions_encoded):
        new_questions_encoded[i, :len(question_encoded)] = question_encoded
    print("Question ENCODED")
    print(new_questions_encoded)
    questions_encoded = np.asarray(new_questions_encoded, dtype=np.int32)
    questions_len = np.asarray(questions_len, dtype=np.int32)
    print(questions_encoded.shape)

    # Pad encoded dialogs
    new_dialogs_encoded = []
    max_dialog_length = max(len(x) for dialog_encoded in dialogs_encoded for x in dialog_encoded)
    print(f"Max dialog length: {max_dialog_length}")
    for dialog_encoded in dialogs_encoded:
        new_dialog_encoded = np.ones((len(dialog_encoded), max_dialog_length), dtype=np.int32) * vocab['question_token_to_idx']['<NULL>']
        for i, sentence in enumerate(dialog_encoded):
            new_dialog_encoded[i, :len(sentence)] = sentence
        new_dialogs_encoded.append(new_dialog_encoded)

    # max_dialog_length = max(len(x) for x in dialogs_encoded)
    # print(f"Max dialog length: {max_dialog_length}")

    # new_dialogs_encoded = np.ones((len(dialogs_encoded), max_dialog_length), dtype=np.int32) * vocab['question_token_to_idx']['<NULL>']
    # for i, dialog_encoded in enumerate(dialogs_encoded):
    #     new_dialogs_encoded[i, -len(dialog_encoded):] = dialog_encoded
    print("Dialog ENCODED")
    # print(new_dialogs_encoded)
    dialogs_encoded = np.asarray(new_dialogs_encoded, dtype=np.int32)
    dialogs_len = np.asarray(dialogs_len, dtype=np.int32)
    print(dialogs_encoded.shape)

    # Pad encoded answers
    max_answer_length = max(len(x) for x in answers_encoded)
    print(f"Max answer length: {max_answer_length}")
    new_answers_encoded = np.ones((len(answers_encoded), max_answer_length), dtype=np.int32) * vocab['question_token_to_idx']['<NULL>']
    for i, answer_encoded in enumerate(answers_encoded):
        new_answers_encoded[i, :len(answer_encoded)] = answer_encoded

    print("answers ENCODED")
    print(new_answers_encoded)
    answers_encoded = np.asarray(new_answers_encoded, dtype=np.int32)
    answers_len = np.asarray(answers_len, dtype=np.int32)
    print(answers_encoded.shape)

    glove_matrix = None
    # if mode == 'train':
    #     token_itow = {i: w for w, i in vocab['question_token_to_idx'].items()}
    #     print("Load glove from %s" % args.glove_pt)
    #     glove = pickle.load(open(args.glove_pt, 'rb'))
    #     dim_word = glove['the'].shape[0]
    #     glove_matrix = []
    #     for i in range(len(token_itow)):
    #         vector = glove.get(token_itow[i], np.zeros((dim_word,)))
    #         glove_matrix.append(vector)
    #     glove_matrix = np.asarray(glove_matrix, dtype=np.float32)
    #     print(glove_matrix.shape)

    print('Writing ', args.output_pt.format(mode))
    obj = {
        'questions': questions_encoded,
        'questions_len': questions_len,
        'dialogs': dialogs_encoded,
        'dialogs_len': dialogs_len,
        'answers': answers_encoded,
        'answers_len': answers_len,
        'video_ids': np.asarray(video_ids),
        'glove': glove_matrix,
    }
    with open(args.output_pt.format(mode), 'wb') as f:
        pickle.dump(obj, f)

def process_questions_openended(args):
    map_id_name = pickle.load(open("input/map_id_name_8c22f_full.pkl", "rb"))
    list_name_available = [v for k,v in map_id_name.items()]
    if args.question_coref:
        fcoref = open('linguistic/coref.txt', 'w')
    print(f'Loading data {args.mode}')
    csv_data = pd.read_json(args.annotation_file.format(args.mode))
    dialogs = []
    questions = []
    answers = []
    video_names = []

    if args.mode == 'test':
        for idx in range(len(csv_data)):
            cap_sum_qa = []
            # cap_sum_qa.append(csv_data['dialogs'][idx]["caption"])
            # cap_sum_qa.append(csv_data['dialogs'][idx]["summary"])
            # if True:
            if csv_data['dialogs'][idx]['image_id'] in list_name_available:
                for i in range(len(csv_data['dialogs'][idx]['dialog'])):
                    if csv_data['dialogs'][idx]['dialog'][i]['answer'] == "__UNDISCLOSED__":
                        dialogs.append(cap_sum_qa.copy())
                        if args.question_coref:
                            question_coref = dialogs[-1].copy()
                            question_coref.append(csv_data['dialogs'][idx]['dialog'][i]['question'])
                            question_coref = ' | '.join(question_coref)
                            fcoref.write(f"ORG: {question_coref}\n")
                            question_coref = nlp(question_coref)._.coref_resolved
                            fcoref.write(f"CRF: {question_coref}\n")
                            questions.append(question_coref.split(' | ')[-1])
                            fcoref.write(f"QUES: {question_coref.split(' | ')[-1]}\n\n")
                        else:
                            questions.append(csv_data['dialogs'][idx]['dialog'][i]['question'])
                        answers.append(csv_data['dialogs'][idx]['dialog'][i]['answer'])
                        video_names.append(csv_data['dialogs'][idx]['image_id'])
                        
                    cap_sum_qa.append(csv_data['dialogs'][idx]['dialog'][i]['question'])
                    cap_sum_qa.append(csv_data['dialogs'][idx]['dialog'][i]['answer'])
    else:
        for idx in range(len(csv_data)):
            cap_sum_qa = []
            # cap_sum_qa.append(csv_data['dialogs'][idx]["caption"])
            # cap_sum_qa.append(csv_data['dialogs'][idx]["summary"])
            # if True:
            if csv_data['dialogs'][idx]['image_id'] in list_name_available:
                for i in range(len(csv_data['dialogs'][idx]['dialog'])):
                    dialogs.append(cap_sum_qa.copy())
                    if args.question_coref:
                        question_coref = dialogs[-1].copy()
                        question_coref.append(csv_data['dialogs'][idx]['dialog'][i]['question'])
                        question_coref = ' | '.join(question_coref)
                        fcoref.write(f"ORG: {question_coref}\n")
                        question_coref = nlp(question_coref)._.coref_resolved
                        fcoref.write(f"CRF: {question_coref}\n")
                        questions.append(question_coref.split(' | ')[-1])
                        fcoref.write(f"QUES: {question_coref.split(' | ')[-1]}\n\n")
                    else:
                        questions.append(csv_data['dialogs'][idx]['dialog'][i]['question'])
                    answers.append(csv_data['dialogs'][idx]['dialog'][i]['answer'])
                    video_names.append(csv_data['dialogs'][idx]['image_id'])

                    cap_sum_qa.append(csv_data['dialogs'][idx]['dialog'][i]['question'])
                    cap_sum_qa.append(csv_data['dialogs'][idx]['dialog'][i]['answer'])
    if args.question_coref:
        fcoref.close()
    print('Number of questions: %s' % len(questions))
    # Either create the vocab or load it from disk
    if args.mode == 'train':
        word_freq = {}
        print('Building vocab')
        for i, q in enumerate(questions):
            # question = q.lower()
            # for token in nltk.word_tokenize(question):
            question = q.split()
            for token in question:
                if token not in word_freq:
                    word_freq[token] = 1
                else:
                    word_freq[token] += 1

        for i, a in enumerate(answers):
            # answer = a.lower()
            # for token in nltk.word_tokenize(answer):
            answer = a.split()
            for token in answer:
                if token not in word_freq:
                    word_freq[token] = 1
                else:
                    word_freq[token] += 1

        print('Len word_freq new')
        print(len(word_freq))

        cutoffs = [1,2,3,4,5]
        for cutoff in cutoffs:
            token_to_idx = {'<NULL>': 1, '<UNK>': 0, '<SOS>': 2, '<EOS>': 3}
            for word, freq in word_freq.items():
                if freq > cutoff:
                    token_to_idx[word] = len(token_to_idx) 
            print("{} words for cutoff {}".format(len(token_to_idx), cutoff))
        vocab = {
            'question_token_to_idx': token_to_idx,
        }

        print('Write into %s' % args.vocab_json)
        with open(args.vocab_json, 'w') as f:
            json.dump(vocab, f, indent=4)

        openeded_encoding_data(args, vocab, questions, video_names, answers, dialogs, args.mode)
    else:
        print('Loading vocab')
        with open(args.vocab_json, 'r') as f:
            vocab = json.load(f)
        openeded_encoding_data(args, vocab, questions, video_names, answers, dialogs, mode=args.mode)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--glove_pt',
    #                     help='glove pickle file, should be a map whose key are words and value are word vectors represented by numpy arrays. Only needed in train mode')
    parser.add_argument('--output_pt', type=str, default='data/{}_questions.pt')
    parser.add_argument('--vocab_json', type=str, default='data/vocab.json')
    parser.add_argument('--mode', choices=['train', 'valid', 'test'])
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--no_accumulate', action='store_true') #dialog turn accumulation
    parser.add_argument('--question_accumulate', action='store_true') #question accumulate with prev dialog
    parser.add_argument('--question_coref', action='store_true') #question replace coref

    args = parser.parse_args()
    np.random.seed(args.seed)

    args.annotation_file = 'data/{}_set4DSTC7-AVSD.json'
    args.output_pt = 'linguistic/{}_full_questions_answers.pt'
    args.vocab_json = 'linguistic/vocab.json'
    # check if data folder exists
    if not os.path.exists('linguistic'):
        os.makedirs('linguistic')

    process_questions_openended(args)