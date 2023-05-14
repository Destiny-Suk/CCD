import json
def read_json(file_path, output_file):
    res_tmp = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            txt = json.loads(line)
            d = txt['detail']['具体错误']
            text = 'S ' + ' '.join([x for x in txt['detail']['原文']])
            res_tmp.append(text)

            for de in d:
                span = [int(x) for x in de['错误位置'][1:-1].split(',')]
                de_info = 'A {} {}|||M|||{}|||REQUIRED|||-NONE-|||0'.format(span[0], span[1] + 1, de['修正词'])
                res_tmp.append(de_info)
            res_tmp.append('')
    f.close()
    with open(output_file, 'w') as f:
        for line in res_tmp:
            f.write(line + '\n')
    f.close()



def main():
    org_dev_path = './data/benchmark5/dev.json'
    dev_path = './data/benchmark5/gec_dev_golden.txt'
    read_json(org_dev_path, dev_path)

    org_test_path = './data/benchmark5/test.json'
    test_path = './data/benchmark5/gec_test_golden.txt'
    read_json(org_test_path, test_path)

if __name__ == '__main__':
    main()