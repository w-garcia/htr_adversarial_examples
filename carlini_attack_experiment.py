import shutil
import numpy as np
import torch
import argparse
import logging
import pickle
import os
import sys
import operator

from multiprocessing import Pool as ThreadPool, cpu_count


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

targets = ['hello',
           'hello there',
           'hello there how',
           'hello there how are',
           'hello there how are you',
           'hello there how are you today']
mem_per_gpu = 12000
mem_per_task = 1000


def go(args):
    """
    Experiment harness for carlini HTR attack. Assume batch size=1.
    :param args:
    :return: nil
    """

    devices = len(args.cuda_devices)
    tasks_per_device = int(np.floor(float(mem_per_gpu) / float(mem_per_task)))

    task_ledger = {idx: [] for idx in range(devices)}
    device_broker = [ThreadPool(tasks_per_device) for _ in range(devices)]
    device_counter = 0

    candidate_images = os.listdir(args.indir)

    for seed in args.seeds:
        root_out_dir = os.path.join(args.outdir, str(seed))

        for image_basename in candidate_images:
            for target in targets:
                out_dir = os.path.join(root_out_dir, image_basename, target)
                if not os.path.isdir(out_dir):
                    os.makedirs(out_dir)

                device = device_counter % devices
                task_ledger[device].append((args, image_basename, out_dir, target, device, seed))

                device_counter += 1

    device2res = [device_broker[d].apply_async(test_combination, task)
                  for d in range(devices) for task in task_ledger[d]]
    [device_broker[d].close() for d in range(devices)]
    [device_broker[d].join() for d in range(devices)]

    device2res = [i.get() for i in device2res]

    print(device2res)

    pickle.dump(device2res, open(os.path.join(args.outdir, 'experiment.pkl'), 'wb'))


def test_combination(args, image_basename, out_dir, target, device, seed):
    torch.manual_seed(seed)
    logger.info("START: {} -> {} | seed {}".format(image_basename, target, seed))

    input = os.path.join(args.indir, image_basename)
    ckpt = args.ckpt
    alphabet = args.alphabet
    CUDA_flag = "CUDA_VISIBLE_DEVICES={}".format(device)
    prologue = "conda activate nephi"
    cmd = "{} python carlini_attack.py --input \"{}\" --target \"{}\" --out \"{}\" --ckpt {} --alphabet {} --cuda --seed {} > /dev/null".format(
        CUDA_flag, input, target, out_dir, ckpt, alphabet, seed
    )
    os.system(cmd)

    logger.info("FINISH: {} -> {} | seed {}".format(image_basename, target, seed))

    pkl_path = os.path.join(out_dir, 'result.pkl')
    orig, attak = pickle.load(open(pkl_path, 'rb'))

    return {'target': target, 'image': image_basename, 'from': orig, 'to': attak, 'seed': seed}


def source_images(args):
    """
    We need to find a balanced set of candidate images to run the experiments on.
    This fn takes care of that by just copying the good set to input directory.
    We need a ground truth text file performed by doing the nephi pre-processing
    :param args
    :return: nil
    """
    if len(os.listdir(args.indir)) != 0:
        logger.warning("Given input directory has images! Might be overwritten.")
        iijj = raw_input("Continue? [y/n] ").lower()
        if iijj == 'n':
            sys.exit(0)

    print(
        "As a reminder, you should have performed the pre-conditions outlined under nephi/dataset.py for reading Laia data.\n"
        "This is what it looks like for IAM:\n"
        "\tcp /path/to/Laia/egs/iam/data/imgs/lines_h128/ /path/to/nephi/datasets/iam/ -r\n"
        "\tcp /path/to/Laia/egs/iam/data/part/lines/aachen/ /path/to/nephi/datasets/iam/aachen_splits -r\n"
        "\tcp /path/to/Laia/egs/iam/data/lang/lines/word/aachen/ /path/to/nephi/datasets/iam/aachen_gt -r\n"
        "\n"
        "This script can read some .txt ground truth file from aachet_gt/ to find candidate images.\n")

    # iill = raw_input("If you have done this already, hit any key, or hit q to exit.").lower()
    # if iill == 'q':
    #     sys.exit(0)

    in_dir = args.indir
    # args.lmdb_path, args.img_db_base_path
    sourcing_seed = 9
    np.random.seed(sourcing_seed)

    file_to_label = {}
    file_to_lengths = {}

    with open(args.gt_path, 'r') as f:
        lines = f.readlines()
        for file_space_label in lines:
            toks = file_space_label.split(' ')
            file = toks[0]
            img_file = os.path.join(args.img_db_base_path, file + '.jpg')
            if not os.path.isfile(img_file):
                logger.warning("Image was not found at {}. Not fatal, skipping...".format(img_file))
                continue

            gt = u' '.join([t.decode('utf-8', 'strict') for t in toks[1:]])
            file_to_label[file + '.jpg'] = gt
            file_to_lengths[file + '.jpg'] = len(gt)

    sorted_items = np.array(sorted(file_to_lengths.items(), key=operator.itemgetter(1)))
    # The average length is pretty high for any IAM split (for our use anyway)s.
    # Instead sample from a subset on lower end
    sorted_items = sorted_items[:95]
    idx = np.random.choice(range(len(sorted_items)), args.num_images, replace=False)
    our_picks = sorted_items[idx]

    logger.info("Chose these labels:")
    for key, _ in our_picks:
        logger.info(file_to_label[key].strip('\n'))

    for key, _ in our_picks:
        img_file = os.path.join(args.img_db_base_path, key)
        dst_file = os.path.join(args.indir, key)
        logger.debug("cp {} -> {}".format(img_file, dst_file))
        shutil.copy2(img_file, dst_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', help="Random seeds to try.", default=[9, 555, 12, 24, 3])
    parser.add_argument('--indir', help="Directory where to grab candidate images.", default='my_images/')
    parser.add_argument('--outdir', help="Directory to save adversarial examples.", default='out_images/')
    parser.add_argument('--ckpt', required=True, help="Path to the best trained netCRNN .pth file.")
    parser.add_argument('--alphabet', required=True, help="Path to alphabet of the trained checkpoint.")
    parser.add_argument('--cuda_devices', nargs='+', default=[0], help="Use given CUDA devices.")

    ss = parser.add_argument_group('SourcingArgs')
    ss.add_argument('--source', action='store_true', help="Should we source images and populate indir?")
    ss.add_argument('--num_images', default=6, help="How many candidate images in experiment.")
    ss.add_argument('--gt_path', help="Path where to find ground truth files.", required=False)
    ss.add_argument('--img_db_base_path', help="Directory where to grab candidate images.", required=False)

    args = parser.parse_args()

    for dir in (args.indir, args.outdir):
        if not os.path.isdir(dir):
            os.makedirs(dir)

    try:
        # if args.source:
        #     source_images(args)
        go(args)
    except KeyboardInterrupt:
        pass
