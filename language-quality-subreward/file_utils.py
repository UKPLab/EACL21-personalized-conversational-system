# encoding:utf-8
import os
import sys
import pickle as pk
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import ticker as ticker
from collections import Counter
import numpy as np


def save_model(folder, model, type, name):
    base_dir = os.path.dirname(__file__)
    if os.path.isdir(os.path.join(base_dir, folder)):
        pass
    else:
        os.makedirs(os.path.join(base_dir, folder))

    if os.path.isfile(os.path.join(base_dir, folder, name + type)):
        os.remove(os.path.join(base_dir, folder, name + type))

    f = open(os.path.join(base_dir, folder, name + type), 'wb')
    try:
        pk.dump(model, f)
    except:
        print("Unexpected SAVING error:", sys.exc_info())
    finally:
        f.close()


def read_model(folder, name, type):
    base_dir = os.path.dirname(__file__)
    f = open(os.path.join(base_dir, folder, name + type), 'rb')
    try:
        model = pk.load(f)
    except:
        print("Unexpected READING error:", sys.exc_info())
        model = None
    finally:
        f.close()
        return model


def save_pic(plot, folder, name):
    base_dir = os.path.dirname(__file__)
    if os.path.isdir(os.path.join(base_dir, folder)):
        pass
    else:
        os.makedirs(os.path.join(base_dir, folder))
    plot.savefig(os.path.join(base_dir, folder, name))


def counter_plot(counter, folder, title, num):
    c = list(counter.items())
    c.sort(key=lambda x: -x[1])
    keys, values = zip(*c)
    indexes = sorted(np.arange(num), reverse=True)

    plt.barh(indexes, values[0:num], align='center', alpha=0.4)
    plt.yticks(indexes, keys[0:num])
    plt.xlabel('Count')
    # plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    # plt.title(title.replace('original_', '').replace('revised_', ''))
    plt.title(title)
    plt.tight_layout()
    save_pic(plt, folder, title)
    plt.show()
    # plt.savefig("barh.eps", format="eps")


def ranking_output(dictionary, folder, file_name, file_type):
    res_arr = np.array(sorted(dictionary.items(), key=lambda d: d[1], reverse=True))
    res_arr = np.column_stack((np.array([i + 1 for i in range(0, len(res_arr))]), res_arr))

    base_dir = os.path.dirname(__file__)
    if os.path.isdir(os.path.join(base_dir, folder)):
        pass
    else:
        os.makedirs(os.path.join(base_dir, folder))
    np.savetxt(os.path.join(base_dir, folder, file_name + file_type), res_arr, fmt='%s', encoding='UTF-8')


def save_txt(folder, arr, type, name):
    base_dir = os.path.dirname(__file__)
    if os.path.isdir(os.path.join(base_dir, folder)):
        pass
    else:
        os.makedirs(os.path.join(base_dir, folder))

    if os.path.isfile(os.path.join(base_dir, folder, name + type)):
        os.remove(os.path.join(base_dir, folder, name + type))

    np.savetxt(os.path.join(base_dir, folder, name + type), arr, fmt='%s', encoding='UTF-8')
