import tensorflow as tf
from scipy import misc
import tensorflow.contrib.slim as slim

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def test_read(txt_file):
    file = open(txt_file).readlines()
    results = []
    for line in file:
        line = line[:-1].split()
        results.append(float(line[0]))
    return results