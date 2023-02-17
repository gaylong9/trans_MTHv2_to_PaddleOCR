import codecs
import datetime
import json
import logging
import logging.handlers as handlers
import os
import sys
from random import shuffle

import cv2
import numpy as np
from PIL import Image

modes = {'TWO_MODE': ['train', 'eval'], 'THREE_MODE': ['train', 'eval', 'test']}
divide_ratios = {'TWO_MODE': [0.8, 0.2], 'THREE_MODE': [0.8, 0.1, 0.1]}

# 仅划分为train/eval，还是train/eval/test
MODE_TYPE = modes['TWO_MODE']
DIVIDE_RATIO = divide_ratios['TWO_MODE']


def set_logger() -> logging.Logger:
    """
    设置日志：文件日志在当前目录/log/<date>、终端打印
    """
    global logger

    # 创建日志目录
    if not os.path.exists("./log"):
        os.mkdir('./log')

    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)

    # 文件日志输出DEBUG级别
    file_handler = handlers.TimedRotatingFileHandler(filename='./log/' + str(datetime.date.today()),
                                                     when='midnight',
                                                     interval=1,
                                                     backupCount=5,
                                                     encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(funcName)s - %(message)s'))

    # 终端打印输出INFO级别
    terminal_handler = logging.StreamHandler(sys.stdout)
    terminal_handler.setLevel(logging.INFO)
    terminal_handler.setFormatter(logging.Formatter('%(asctime)s - %(funcName)s - %(message)s'))

    logger.addHandler(file_handler)
    logger.addHandler(terminal_handler)

    return logger


logger = set_logger()


def build_det_mthv2(dataset_root_dir: str) -> None:
    """
    构建paddleocr检测所需的数据集

    :param dataset_root_dir: MTHv2的绝对路径，其下应有MTH1200、MTH1000、TKH三个目录
    """
    dataset_names = os.listdir(dataset_root_dir)
    assert len(dataset_names) == 3
    for dataset_name in dataset_names:
        logger.info('dataset name:%s' % dataset_name)
        dataset_dir = os.path.join(dataset_root_dir, dataset_name)
        img_dir = os.path.join(dataset_dir, 'img')
        text_dir = os.path.join(dataset_dir, 'label_textline')

        # 获得子集的图像名称列表
        img_names = os.listdir(img_dir)
        shuffle(img_names)
        dataset_size = len(img_names)
        if dataset_name == 'MTH1200':
            assert dataset_size == 1200
        elif dataset_name == 'MTH1000':
            assert dataset_size == 1000
        else:
            assert dataset_size == 999

        assert sum(DIVIDE_RATIO) == 1
        train_size = int(dataset_size * DIVIDE_RATIO[0])
        if len(MODE_TYPE) == 2:
            # 仅划分train/eval
            eval_size = int(dataset_size - train_size)
            subset_size = [train_size, eval_size]
            subset_idx = [0, subset_size[0]]
            train_img_names = img_names[:subset_idx[1]]
            eval_img_names = img_names[subset_idx[1]:]
            subset_img_names = [train_img_names, eval_img_names]
            logger.info('划分完成 %d:%d（共%d）' % (len(train_img_names), len(eval_img_names), dataset_size))
        else:
            eval_size = int(dataset_size * DIVIDE_RATIO[1])
            test_size = int(dataset_size - train_size - eval_size)
            subset_size = [train_size, eval_size, test_size]
            subset_idx = [0, subset_size[0], subset_size[0] + subset_size[1]]
            train_img_names = img_names[:subset_idx[1]]
            eval_img_names = img_names[subset_idx[1]:subset_idx[2]]
            test_img_names = img_names[subset_idx[2]:]
            subset_img_names = [train_img_names, eval_img_names, test_img_names]
            logger.info('划分完成 %d:%d:%d（共%d）' % (
                len(train_img_names), len(eval_img_names), len(test_img_names), dataset_size))

        # 对每个子集分别操作，每个子集生成一个文件
        for i in range(len(MODE_TYPE)):
            logger.info('mode: %s' % MODE_TYPE[i])
            img_names = subset_img_names[i]
            mode = MODE_TYPE[i]
            label_file = codecs.open(os.path.join(dataset_dir, 'det_' + mode + '_' + dataset_name + '_label.txt'), 'w',
                                     encoding='utf-8')
            # 每个图片分别操作，每个图片生成一行信息label，写入label_file
            for j in range(len(img_names)):
                if j % 100 == 0:
                    logger.info('%d/%d' % (j, len(img_names)))
                img_name = img_names[j]
                logger.debug(img_name)
                # label先加入图像路径与图像名
                label = 'img/' + img_name + '\t'
                # label再加入图像中所有的文本信息与边框信息
                text_file = codecs.open(os.path.join(text_dir, img_name[:-4] + '.txt'), 'r', encoding='utf-8')
                text_info = []
                # 每个图像对应的文本信息有一个文件，要从中读出所有文本，存入text_info，最终加入label，写入label_file
                for line in text_file.readlines():
                    text_content = {}
                    contents = line.strip('\n').split(',')
                    assert len(contents) == 9
                    text_content['transcription'] = contents[0]
                    text_content['points'] = [[contents[2 * k + 1], contents[2 * k + 2]] for k in range(4)]
                    text_info.append(text_content)
                text_file.close()
                label = label + json.dumps(text_info, ensure_ascii=False) + '\n'
                label_file.write(label)
            label_file.close()


def build_rec_mthv2(dataset_root_dir: str) -> None:
    """
    构建paddleocr识别所需的数据集

    :param dataset_root_dir: MTHv2的绝对路径，其下应有MTH1200、MTH1000、TKH三个目录
    """
    dataset_names = os.listdir(dataset_root_dir)
    assert len(dataset_names) == 3
    for dataset_name in dataset_names:
        logger.info('dataset name:%s' % dataset_name)
        dataset_dir = os.path.join(dataset_root_dir, dataset_name)
        full_img_dir = os.path.join(dataset_dir, 'img')
        text_dir = os.path.join(dataset_dir, 'label_textline')

        # 获得子集的图像名称列表
        img_names = os.listdir(full_img_dir)
        # 图像乱序后划分
        shuffle(img_names)
        dataset_size = len(img_names)
        if dataset_name == 'MTH1200':
            assert dataset_size == 1200
        elif dataset_name == 'MTH1000':
            assert dataset_size == 1000
        else:
            assert dataset_size == 999

        assert sum(DIVIDE_RATIO) == 1
        train_size = int(dataset_size * DIVIDE_RATIO[0])
        if len(MODE_TYPE) == 2:
            # 仅划分train/eval
            eval_size = int(dataset_size - train_size)
            subset_size = [train_size, eval_size]
            subset_idx = [0, subset_size[0]]
            train_img_names = img_names[:subset_idx[1]]
            eval_img_names = img_names[subset_idx[1]:]
            subset_img_names = [train_img_names, eval_img_names]
            logger.info('划分完成 %d:%d（共%d）' % (len(train_img_names), len(eval_img_names), dataset_size))
        else:
            # train/eval/test
            eval_size = int(dataset_size * DIVIDE_RATIO[1])
            test_size = int(dataset_size - train_size - eval_size)
            subset_size = [train_size, eval_size, test_size]
            subset_idx = [0, subset_size[0], subset_size[0] + subset_size[1]]
            train_img_names = img_names[:subset_idx[1]]
            eval_img_names = img_names[subset_idx[1]:subset_idx[2]]
            test_img_names = img_names[subset_idx[2]:]
            subset_img_names = [train_img_names, eval_img_names, test_img_names]
            logger.info('划分完成 %d:%d:%d（共%d）' % (
                len(train_img_names), len(eval_img_names), len(test_img_names), dataset_size))

        # 对每个子集分别操作，每个子集生成一个label文件
        for i in range(len(MODE_TYPE)):
            logger.info('mode: %s' % MODE_TYPE[i])
            img_names = subset_img_names[i]
            mode = MODE_TYPE[i]
            label_file = codecs.open(os.path.join(dataset_dir, 'rec_' + mode + '_' + dataset_name + '_label.txt'),
                                     'w', encoding='utf-8')
            # 创建图片目录
            img_des_dir = os.path.join(dataset_dir, mode)
            if not os.path.exists(img_des_dir):
                os.mkdir(img_des_dir)

            # 每个图片分别操作，分割成许多小图片，并写入label_file
            for j in range(len(img_names)):
                if j % 100 == 0:
                    logger.info('%d/%d' % (j, len(img_names)))
                img_name = img_names[j]
                logger.debug(img_name)
                full_img_name = os.path.join(full_img_dir, img_name)
                # 读入图片
                full_img = cv2.imread(full_img_name)
                # 读入text_file
                text_file = codecs.open(os.path.join(text_dir, img_name[:-4] + '.txt'), 'r', encoding='utf-8')
                # 每个图像，对应一个text_file，其中是文本行与坐标，
                # 根据坐标，切割出文本行局部图片存入img_des_dir
                # 子图名 与 文本内容 存入label_file
                cnt = 0
                for line in text_file.readlines():
                    cnt += 1
                    contents = line.strip('\n').split(',')
                    assert len(contents) == 9
                    # 数据集的坐标，以图片左上为原点，横轴x，纵轴y
                    # 数据集原始label中坐标顺序不定，先排序为[[左上横坐标, 左上纵坐标], [右上], [右下], [左下]]
                    raw_locations = contents[1:]
                    position = [[int(raw_locations[2 * k]), int(raw_locations[2 * k + 1])] for k in range(4)]
                    # 根据纵坐标排序，数字较小的两个点为上方两点，较大的两点为下方两点
                    sorted_list = sorted(position, key=lambda x: x[1])
                    upper_positions = sorted_list[0:2]
                    lower_positions = sorted_list[2:4]
                    # 分别按照横坐标排序
                    upper_positions = sorted(upper_positions, key=lambda x: x[0])
                    lower_positions = sorted(lower_positions, key=lambda x: x[0], reverse=True)
                    text_position = upper_positions + lower_positions
                    # crop时，原形状可能不规则，crop出最小包围矩形
                    # 区域距离 上边 最近的距离，即左上右上两点纵坐标中的较小值
                    top = int(min(text_position[0][1], text_position[1][1]))
                    # 区域距离 上边 最远的距离，即左下右下两点纵坐标中的较大值
                    bottom = int(max(text_position[2][1], text_position[3][1]))
                    # 区域距离 左边 最近的距离，即左下左上两点横坐标中的较小值
                    left = int(min(text_position[0][0], text_position[3][0]))
                    # 区域距离 左边 最远的距离，即右下右上两点横坐标中的较大值
                    right = int(max(text_position[1][0], text_position[2][0]))
                    # crop
                    sub_img = full_img[top:bottom, left:right]
                    # save
                    sub_img_name = img_name[:-4] + '_' + str(cnt) + full_img_name[-4:]
                    sub_img_path = os.path.join(img_des_dir, sub_img_name)
                    # logger.debug('%s %s %s' % (str(contents[1:]), str(text_position), str(sub_img.shape)))
                    cv2.imwrite(sub_img_path, sub_img)

                    label = mode + '/' + sub_img_name + '\t' + contents[0] + '\n'
                    label_file.write(label)
                text_file.close()

            label_file.close()


def generate_rec_dict(dataset_root_dir: str) -> None:
    """
    生成数据集的字典文件
    生成位置如：dataset_root_dir/MTH1200/dict.txt

    :param dataset_root_dir: MTHv2的绝对路径，其下应有MTH1200、MTH1000、TKH三个目录
    """
    dataset_names = os.listdir(dataset_root_dir)
    assert len(dataset_names) == 3
    for dataset_name in dataset_names:
        logger.info('dataset name:%s' % dataset_name)
        dataset_dir = os.path.join(dataset_root_dir, dataset_name)
        # 集合，数据集原始label中，不清楚的文字用'#'代表
        dict_set = {'#', }
        # 逐个读入text_line中的文件
        label_textlines_dir = os.path.join(dataset_dir, 'label_textline')
        text_files = os.listdir(label_textlines_dir)
        cnt = 0
        for text_file_name in text_files:
            logger.debug(text_file_name)
            cnt += 1
            if cnt % 100 == 0:
                logger.info('%d/%d' % (cnt, len(text_files)))
            if cnt == len(text_files):
                logger.info('%d/%d - done' % (cnt, cnt))

            # 逐行读入一个label文件
            text_file = codecs.open(os.path.join(label_textlines_dir, text_file_name), 'r', encoding='utf-8')
            for line in text_file.readlines():
                contents = line.strip('\n').split(',')
                assert len(contents) == 9
                content = contents[0]
                # 逐字遍历文本
                for i in range(len(content)):
                    character = content[i]
                    dict_set.add(character)

            text_file.close()

        # 生成dict.txt
        dict_file = codecs.open(os.path.join(dataset_dir, 'dict.txt'), 'w', encoding='utf-8')
        for character in dict_set:
            dict_file.write(character + '\n')
        dict_file.close()


def test_rec_img(img_path: str, raw_locations: list) -> None:
    """
    测试分割识别数据集时遇到问题的数据

    :param img_path: 图片的完整路径，包括图片名
    :param raw_locations: 8位list，原始label中的边框坐标
    """
    img = cv2.imread(img_path)
    # cv2.imshow("", img)
    # print(img.shape)    # (1560, 2400, 3) HWC
    position = [[int(raw_locations[2 * k]), int(raw_locations[2 * k + 1])] for k in range(4)]
    # 根据纵坐标排序，数字较小的两个点为上方两点，较大的两点为下方两点
    sorted_list = sorted(position, key=lambda x: x[1])
    upper_positions = sorted_list[0:2]
    lower_positions = sorted_list[2:4]
    # 分别按照横坐标排序
    upper_positions = sorted(upper_positions, key=lambda x: x[0])
    lower_positions = sorted(lower_positions, key=lambda x: x[0], reverse=True)
    text_position = upper_positions + lower_positions
    print(text_position)
    # 区域距离 上边 最近的距离，即左上右上两点纵坐标中的较小值
    top = int(min(text_position[0][1], text_position[1][1]))
    # 区域距离 上边 最远的距离，即左下右下两点纵坐标中的较大值
    bottom = int(max(text_position[2][1], text_position[3][1]))
    # 区域距离 左边 最近的距离，即左下左上两点横坐标中的较小值
    left = int(min(text_position[0][0], text_position[3][0]))
    # 区域距离 左边 最远的距离，即右下右上两点横坐标中的较大值
    right = int(max(text_position[1][0], text_position[2][0]))

    sub_img = img[top:bottom, left:right]
    print(sub_img.shape)

    for i in range(len(text_position)):
        cv2.circle(img, text_position[i], 20, (0, 255, 0), -1)
        cv2.namedWindow("img", 0)
        cv2.imshow("img", img)
        cv2.waitKey(0)

    cv2.namedWindow("img", 0)
    cv2.imshow("img", sub_img)
    cv2.waitKey(0)


if __name__ == '__main__':
    dataset_root_dir = 'D:/File/Postgraduate/graduation project/OCR database/rec/TKHMTH2200'
    # build_det_mthv2(dataset_root_dir)
    # build_rec_mthv2(dataset_root_dir)
    # test_rec_img('D:/File/Postgraduate/graduation project/OCR database/rec/TKHMTH2200/MTH1000/img/01-V100P0656.png',
    #             [1172,1441,1095,1440,1104,145,1181,146])
    generate_rec_dict(dataset_root_dir)
    pass
