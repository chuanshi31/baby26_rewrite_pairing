#!/usr/bin/env python3
"""
Split a text file (one article per line) into sentences.
Output: one sentence per line, min_words <= len <= max_words.
"""
import argparse
import nltk
nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import sent_tokenize

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--min_words', type=int, default=10)
    parser.add_argument('--max_words', type=int, default=200)
    args = parser.parse_args()

    total_in = 0
    total_out = 0
    with open(args.input, encoding='utf-8') as fin, \
         open(args.output, 'w', encoding='utf-8') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total_in += 1
            sents = sent_tokenize(line)
            for s in sents:
                s = s.strip()
                words = s.split()
                if args.min_words <= len(words) <= args.max_words:
                    fout.write(s + '\n')
                    total_out += 1

    print(f'Articles in: {total_in:,}')
    print(f'Sentences out: {total_out:,}')

if __name__ == '__main__':
    main()
