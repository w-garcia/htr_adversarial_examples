import numpy as np
import pickle
import os
import argparse
import matplotlib.pyplot as plt

from collections import defaultdict


def similarity(ref, hyp):
    toks_ref = set(ref.split())
    toks_hyp = set(hyp.split())

    inte = toks_ref.intersection(toks_hyp)

    return float(len(inte)) / float(len(toks_ref))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('from_path', help="Path to experiment.pkl file")
    parser.add_argument('gt_path', help="Path where to find ground truth file.")

    args = parser.parse_args()

    targets = ['hello',
               'hello there',
               'hello there how',
               'hello there how are',
               'hello there how are you',
               'hello there how are you today']

    file_to_label = {}

    with open(args.gt_path, 'r') as f:
        lines = f.readlines()
        for file_space_label in lines:
            toks = file_space_label.split(' ')
            file = toks[0]
            gt = u' '.join([t.decode('utf-8', 'strict') for t in toks[1:]])
            file_to_label[file + '.jpg'] = gt

    with open(args.from_path, 'rb') as f:
        o = pickle.load(f)
        # {'target': target, 'image': image_basename, 'from': orig, 'to': attak, 'seed': seed}

        total_similarity = defaultdict(list)
        tos = set()
        froms = set()
        vocab = set()
        matches = defaultdict(list)

        for t in targets:
            for word in t.split(' '):
                vocab.add(word)

        for exp in o:
            target = exp['target']
            image = exp['image']
            orig = exp['from']
            to = exp['to']
            seed = exp['seed']
            label = file_to_label[image].strip('\n')

            this_key = (label, target)
            froms.add(label)
            tos.add(target)

            sim = similarity(target, to)

            matches[label].append(to)
            total_similarity[this_key].append(sim)

        # print total_similarity
        froms.remove('sort of rocket ? \"')
        y_std = []
        x = []
        y = []
        linestyles = ['solid', 'dashed', 'dashdot', 'dotted', '-', '--', '-.', ':']
        markers = np.random.choice(['.', '^'], len(froms))
        using = np.random.choice(linestyles, len(froms))
        for i, fr in enumerate(sorted(froms)):
            for to in sorted(tos):
                x.append(len(to.split(' ')))
                avg = np.average(total_similarity[(fr, to)])
                std = np.std(total_similarity[(fr, to)])
                y.append(avg)
                y_std.append(std)

            fr_toks = fr.split(' ')
            for c in ('.', '?', '\"'):
                if c in fr_toks:
                    fr_toks.remove(c)

            plt.errorbar(x, y, yerr=y_std, linestyle=linestyles[i], marker=markers[i], capsize=3, label="Source Len. {}".format(len(fr_toks)))
            print(fr, y)
            x = []
            y = []
            y_std = []

        plt.legend(loc='lower left')
        plt.ylabel('Character-level Similarity')
        plt.xlabel('Desired Target Length (# words)')
        plt.show()
        # plt.savefig('my_attacks/figure1.pdf')
