# -*- coding: utf-8 -*-
import json

if __name__ == '__main__':
    with open('README.t', 'r', encoding="utf-8") as file:
        text = file.readlines()
    text = ''.join(text)

    with open('results.json', 'r', encoding="utf-8") as file:
        results = json.load(file)

    print(results[0])

    for i, result in enumerate(results):
        out_key = 'out_list_{}'.format(i)
        text = text.replace('$({})'.format(out_key), '<br>'.join(result[out_key]))
        gt_key = 'gt_{}'.format(i)
        text = text.replace('$({})'.format(gt_key), result[gt_key])

    text = text.replace('<sos>', '')
    text = text.replace('<eos>', '')

    with open('README.md', 'w', encoding="utf-8") as file:
        file.write(text)
