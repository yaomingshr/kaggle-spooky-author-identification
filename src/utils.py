# encoding: utf-8
# Author: patrickyao

from absl import app
from absl import flags
from absl import logging

import csv
import random

FLAGS = flags.FLAGS

flags.DEFINE_string('ori_train_file', '../data/train.csv', 'original train data file')
flags.DEFINE_string('ft_train_file', '../data/ft_train.txt', 'fasttext train data file')
flags.DEFINE_string('ft_cv_file', '../data/ft_cv.txt', 'fasttext cv data file')
flags.DEFINE_float('cv_percentage', 0.3, 'cross validation percentage')

# global
author_dict = {"EAP":1, "HPL":2, "MWS":3}
fasttext_format = "__label__%d %s\n"

def shuffle_sample(total_list, percentage):
    random.seed(7)
    random.shuffle(total_list)
    split_index = int(round(percentage * len(total_list)))
    return total_list[:split_index], total_list[split_index:]

def ori_input_to_fasttext():
    with open(FLAGS.ori_train_file, 'r') as fin:
        f_csv = csv.reader(fin)
        _ = next(f_csv)
        lines = []
        class_count = {}
        for row in f_csv:
            line = fasttext_format % (author_dict[row[2]], row[1])
            lines.append(line)
            if (not class_count.has_key(author_dict[row[2]])):
                class_count[author_dict[row[2]]] = 0
            else:
                class_count[author_dict[row[2]]] += 1

    print "class_count: %d_%d, %d_%d, %d_%d" % (1, class_count[1], 2, class_count[2], 3, class_count[3])
    train_lines, cv_lines = shuffle_sample(lines, 1 - FLAGS.cv_percentage)
    with open(FLAGS.ft_train_file, 'w') as train_fout, open(FLAGS.ft_cv_file, 'w') as cv_fout:
        train_fout.writelines(train_lines)
        cv_fout.writelines(cv_lines)

def main(argv):
    del argv
    ori_input_to_fasttext()

if __name__ == '__main__':
    app.run(main)