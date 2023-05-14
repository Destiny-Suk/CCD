import argparse
import copy
import json
import random
import re
import copy
import tqdm
import numpy as np
import os
def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='./data/train.json')
    parser.add_argument('--dev_path', type=str, default='./data/dev.json')
    parser.add_argument('--test_path', type=str, default='./data/test.json')
    return parser.parse_args()

def read_json(file_path):
    results = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            results.append(json.loads(line))
    f.close()
    return results

def main(args):
    train_dataset = read_json(args.train_path)
    dev_dataset = read_json(args.dev_path)
    test_dataset = read_json(args.test_path)

    train_dataset_error = [result for result in train_dataset if result['正错判断'] == '错误' and len(result['具体错误']) > 0]
    dev_dataset_error = [result for result in dev_dataset if result['正错判断'] == '错误' and len(result['具体错误']) > 0]
    test_dataset_error = [result for result in test_dataset if result['正错判断'] == '错误' and len(result['具体错误']) > 0]

    for file_name in ['./benchmark{}'.format(i+1) for i in range(6)]:
        if not os.path.exists(file_name):
            os.mkdir(file_name)

    createbenchmark1(train_dataset, './benchmark1/train.json')
    createbenchmark1(dev_dataset, './benchmark1/dev.json')
    createbenchmark1(test_dataset,'./benchmark1/test.json')

    # Benchmark 2

    createbenchmark2(train_dataset_error, './benchmark2/train.json')
    createbenchmark2(dev_dataset_error, './benchmark2/dev.json')
    createbenchmark2(test_dataset_error,'./benchmark2/test.json')


    # Benchmark 3
    print("Train Example have:{}".format(len(train_dataset_error)))
    print("Dev Example have:{}".format(len(dev_dataset_error)))
    print("Test Example have:{}".format(len(test_dataset_error)))


    createbenchmark3_misewbased(train_dataset_error, './benchmark3/train.json')
    createbenchmark3_misewbased(dev_dataset_error, './benchmark3/dev.json')
    createbenchmark3_misewbased(test_dataset_error,'./benchmark3/test.json')

    # Benchmark 4
    print("Train Example have:{}".format(len(train_dataset_error)))
    print("Dev Example have:{}".format(len(dev_dataset_error)))
    print("Test Example have:{}".format(len(test_dataset_error)))
    createbenchmark4(train_dataset_error, './benchmark4/train.json')
    createbenchmark4(dev_dataset_error, './benchmark4/dev.json')
    createbenchmark4(test_dataset_error,'./benchmark4/test.json')



    print("Train Example have:{}".format(len(train_dataset_error)))
    print("Dev Example have:{}".format(len(dev_dataset_error)))
    print("Test Example have:{}".format(len(test_dataset_error)))
    createbenchmark5(train_dataset_error, './benchmark5/train.json')
    createbenchmark5(dev_dataset_error, './benchmark5/dev.json')
    createbenchmark5(test_dataset_error,'./benchmark5/test.json')

    


    #train_ben6, dev_ben6, test_ben6 = split_data(error_text, train_rate, dev_rate)
    createbenchmark6(train_dataset_error, './benchmark6/train.json')
    createbenchmark6(dev_dataset_error, './benchmark6/dev.json')
    createbenchmark6(test_dataset_error,'./benchmark6/test.json')

def createbenchmark1(dataset, save_path):
    nums = 0
    for data in dataset:
        if data['正错判断'] == '正确':
            nums += 1
    print(nums, nums / len(dataset) * 100)
    with open(save_path, 'w') as f:
        for d in dataset:
            example = {'text':d['原文'], 'label':d['正错判断']}
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    f.close()
def createbenchmark2(dataset, save_path):
    results = []
    rates = []
    for d in tqdm.tqdm(dataset):
        example = {}
        txt = d['原文']
        #example['label'] = [0] * len(example['text'])
        span_ids = []
        for de in d['具体错误']:
            mini_word_idx = [int(r_s) for r_s in re.findall(r'\d+', de['错误词集位置'])]
            for idx in range(len(mini_word_idx) // 2):
                span_0 = mini_word_idx[2 * idx]
                span_1 = mini_word_idx[2 * idx + 1]
                span_ids.extend(list(range(span_0, span_1+1)))
                #span_ids.append([span_0, span_1])
            #span_ids = sorted(span_ids, key=lambda x:x[0])
        span_ids = list(set(span_ids))
        rates.append(len(span_ids) / len(txt))
        span_ids.sort()
        merged_span = []
        if len(span_ids) == 1:
            merged_span = [[span_ids[0], span_ids[0] + 1]]
        for i in range(len(span_ids)):
            if i == 0:
                s = span_ids[0]
                tmp_span = []
                tmp_span.append(s)
                continue
            if i == len(span_ids)-1:
                tmp_span.append(span_ids[i])
                merged_span.append(tmp_span[:])
                continue
            if span_ids[i] == s +1:
                s += 1
            else:
                tmp_span.append(s)
                merged_span.append(tmp_span[:])
                s = span_ids[i]
                tmp_span = [s]
        start = 0
        split_text = []
        labels = []
        for span in merged_span:
            end = span[0]
            if txt[start:end] != '':
                split_text.append(txt[start:end])
                labels.append(0)
            start = span[0]
            end = span[1]+1
            if txt[start:end] != '':
                split_text.append(txt[start:end])
                labels.append(1)
            start = end
        if txt[start:] != '':
            split_text.append(txt[start:])
            labels.append(0)

        if len(split_text) != 0:
            example['text'] = ' '.join(split_text)
            example['label'] = labels
            results.append(copy.deepcopy(example))
        else:
            print(txt)
            print(span_ids)
            print(merged_span)
    print("{} have total {}, {}".format(save_path, len(results), np.mean(rates)))
    with open(save_path, 'w') as f:
        for example in results:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    f.close()

def createbenchmark3_misewbased(dataset, save_path):
    results = []
    example_id = 0
    nums = 0

    for d in tqdm.tqdm(dataset):
        example = {}
        txt = d['原文']
        #example['misew'] = [0] * len(example['text'])
        misew_spans = []

        for de in d['具体错误']:
            example['id'] = example_id
            span = [int(x) for x in de['错误位置'][1:-1].split(',')]
            if span[0] > span[1]:
                span[0], span[1] = span[1], span[0]
            error_span = list(range(span[0], span[1] + 1))

            mini_word_idx = [int(r_s) for r_s in re.findall(r'\d+', de['错误词集位置'])]
            for idx in range(len(mini_word_idx) // 2):
                span_0 = mini_word_idx[2 * idx]
                span_1 = mini_word_idx[2 * idx + 1]
                misew_spans.extend(list(range(span_0, span_1+1)))

            misew_spans = list(set(misew_spans))

            misew_spans.sort()
            merged_span = []
            if len(misew_spans) == 1:
                merged_span = [[misew_spans[0], misew_spans[0] + 1]]
            for i in range(len(misew_spans)):
                if i == 0:
                    s = misew_spans[0]
                    tmp_span = []
                    tmp_span.append(s)
                    continue
                if i == len(misew_spans) - 1:
                    tmp_span.append(misew_spans[i])
                    merged_span.append(tmp_span[:])
                    continue
                if misew_spans[i] == s + 1:
                    s += 1
                else:
                    tmp_span.append(s)
                    merged_span.append(tmp_span[:])
                    s = misew_spans[i]
                    tmp_span = [s]
            start = 0
            split_text = []
            misew_labels = []
            for span in merged_span:
                end = span[0]
                if txt[start:end] != '':
                    split_text.append(txt[start:end])
                    misew_labels.append(0)
                start = span[0]
                end = span[1] + 1
                if txt[start:end] != '':
                    split_text.append(txt[start:end])
                    misew_labels.append(1)
                start = end
            if txt[start:] != '':
                split_text.append(txt[start:])
                misew_labels.append(0)
            '''
            print(txt)
            print(misew_labels)
            print(split_text)
            print(error_span)
            '''
            label = []
            split_text_with_error_span = []
            split_text_with_error_label = []
            start = 0
            flag = False
            i = 0
            error_start, error_end = error_span[0], error_span[-1]
            while( i < (len(split_text))):
                span_text = split_text[i]
                span_len = len(span_text)
                label_id = misew_labels[i]
                if error_start >= start and error_end < start + span_len:
                    span_0 = span_text[:error_start-start]
                    span_1 = span_text[error_start-start:error_end-start+1]
                    span_3 = span_text[error_end-start+1:]

                    if span_0 != '':
                        split_text_with_error_span.append(span_0)
                        split_text_with_error_label.append(label_id)

                    split_text_with_error_span.append(span_1)
                    split_text_with_error_label.append(2)
                    if span_3 != '':
                        split_text_with_error_span.append(span_3)
                        split_text_with_error_label.append(label_id)
                elif error_end < start or error_start >= start + span_len:
                    split_text_with_error_span.append(split_text[i])
                    split_text_with_error_label.append(label_id)
                else:
                    if error_start < start + span_len:
                        span0 = span_text[:error_start-start]
                        span1 = span_text[error_start-start:]
                        if span0 != '':
                            split_text_with_error_span.append(span0)
                            split_text_with_error_label.append(label_id)
                        if span1 != '':
                            split_text_with_error_span.append(span1)
                            split_text_with_error_label.append(2)
                        error_start = start + span_len
                    elif error_end >= start:
                        span0 = span_text[:error_end-start]
                        span1 = span_text[error_end-start:]
                        if span0 != '':
                            split_text_with_error_span.append(span0)
                            split_text_with_error_label.append(2)
                        if span1 != '':
                            split_text_with_error_span.append(span1)
                            split_text_with_error_label.append(label_id)
                        else:
                            nums += 1
                            flag = True
                start += span_len
                i += 1
            if flag:
                print('=========================')
                print(error_span)
                print(split_text)
                print(split_text_with_error_label)
                print(split_text_with_error_span)

            example['text'] = ' '.join(split_text_with_error_span)
            example['label'] = split_text_with_error_label

            results.append(copy.deepcopy(example))
            example_id += 1

    print("{} have total {}".format(save_path, len(results)))
    with open(save_path, 'w') as f:
        for example in results:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    f.close()

def createbenchmark4(dataset, save_path):
    results = []
    for d in dataset:
        example = {}
        raw_text = d['原文']
        for de in d['具体错误']:
            span = [int(x) for x in de['错误位置'][1:-1].split(',')]
            label = 0
            txt = []
            span = [span[0], span[1]] if span[0] <= span[1] else [span[1], span[0]]
            if raw_text[0:span[0]] != '':
                txt.append(raw_text[0:span[0]])
                label += 1
            txt.append(raw_text[span[0]:span[1]+1])
            txt.append(raw_text[span[1]+1:])
            example['text'] = ' '.join(txt)
            example['span'] = label
            example['class1'] = de['错误大类']
            example['class2'] = de['错误小类']
            #example['detail'] = d
            results.append(copy.deepcopy(example))
    print("{} have total {}".format(save_path, len(results)))
    with open(save_path, 'w') as f:
        for example in results:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    f.close()


def createbenchmark5(dataset, save_path, with_correct=False):
    results = []
    cnt, total = 0, 0
    nums = {}
    for d in tqdm.tqdm(dataset):
        example = {}
        dtmp = [x for x in d['原文']]
        corrected =  [x for x in d['改正']]
        masked_text= copy.deepcopy(corrected)
        example['label'] = copy.deepcopy(dtmp)
        masked_text_show = copy.deepcopy(dtmp)
        if d['正错判断'] == '错误':
            total += 1
            for de in d['具体错误']:
                span = [int(x) for x in de['错误位置'][1:-1].split(',')]
                if span[0] > span[1]:
                    span[0], span[1] = span[1], span[0]


                if de['修正词'] == 'error':
                    cnt += 1
                    error = True
                    continue

                masked_text[span[0]:span[1]+1] = ['[MASK]'] * (span[1] - span[0] + 1)

            res_tmp = []
            for idx, mt in enumerate(masked_text):
                if mt != '[MASK]':
                    res_tmp.append(mt)
                else:
                    if (res_tmp and  res_tmp[-1] != '[MASK]') or idx == 0:
                        res_tmp.append(mt)

            example['mask_text'] = ''.join(res_tmp)
            example['label'] = ''.join(example['label'])
            example['detail'] = d
            if '[MASK]' in example['mask_text']:
                results.append(copy.deepcopy(example))
            else:
                print(d)

        else:
            if with_correct:
                start_idx = np.random.randint(0, len(masked_text)-4)
                end_idx = start_idx + np.random.randint(1, 5)
                masked_text[start_idx:end_idx] = ['[MASK]'] * (end_idx - start_idx)
                res_tmp = []
                for mt in masked_text:
                    if mt != '[MASK]':
                        res_tmp.append(mt)
                    else:
                        if res_tmp and res_tmp[-1] != '[MASK]':
                            res_tmp.append(mt)
                example['mask_text'] = ''.join(res_tmp)
                example['label'] = ''.join(example['label'])
                example['detail'] = d
                if '[MASK]' in example['mask_text']:
                    results.append(copy.deepcopy(example))



            #nums[nu] = nums.get(nu, 0) + 1
            #print(example['masked_text'])
    print(len(results))
    with open(save_path, 'w') as f:
        for example in results:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    f.close()

def createbenchmark6(dataset, save_path):
    results = []
    for d in dataset:
        example = {}
        if random.random() > 0.5:
            sent0 = d['原文']
            sent1 = d['改正']
            label = 1
        else:
            sent0 = d['改正']
            sent1 = d['原文']
            label = 0
        example={'sent0':sent0,
                'sent1':sent1,
                'label':label}

        results.append(copy.deepcopy(example))
    print("{} have total {}".format(save_path, len(results)))
    with open(save_path, 'w') as f:
        for example in results:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    f.close()

if __name__ == '__main__':
    parse = parse_config()
    main(parse)