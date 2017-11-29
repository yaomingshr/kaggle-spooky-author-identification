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
flags.DEFINE_string('fasttext_out_file', '../../../public/fastText/result/naive_ft.txt', 'fasttext output file')
flags.DEFINE_string('test_file', '../data/test.csv', 'kaggle test file')
flags.DEFINE_string('sub_file', '../data/sub.csv', 'kaggle submission file')

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

def fasttext_ouput_to_sub():
    with open(FLAGS.fasttext_out_file, 'r') as f_fast, open(FLAGS.test_file, 'r') as f_tag, \
        open(FLAGS.sub_file, 'w') as f_sub:
        tag_lines = f_tag.readlines()
        lines = f_fast.readlines()
        assert len(tag_lines) == len(lines)
        f_sub.write("id,EAP,HPL,MWS\n")
        for i in range(len(lines)):
            # get in sort
            res_map = {}
            fields = lines[i].split(' ')
            for j in range(0, len(fields), 2):
                res_map[fields[j]] = 1.0 if (float(fields[j + 1]) > 1.0) else float(fields[j + 1])
            new_line = tag_lines[i].split(',')[0] + ',' + str(res_map["__label__1"]) + ',' + \
                       str(res_map["__label__2"]) + ',' + str(res_map["__label__3"]) + '\n'
            f_sub.write(new_line)

def main(argv):
    del argv
    # ori_input_to_fasttext()
    fasttext_ouput_to_sub()

if __name__ == '__main__':
    app.run(main)