#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : file_utils.py
# @Author: sl
# @Date  : 2021/4/10 -  上午11:24
import csv
import glob
import json
import math
import os
import pickle
from functools import partial

import shutil
from typing import List

import collections

import jsonlines as jsonlines
import pandas as pd
from tqdm import tqdm

from nlptravel.process.entity.ner_entity import InputExample
from nlptravel.utils.base_utils import BaseUtil, DataFileType, NlpPretrain
from nlptravel.utils.constant import Constants
from nlptravel.utils.logger_utils import logger
from nlptravel.utils.time_utils import TimeUtils

'''
文件处理的工具类

'''


class FileUtils(BaseUtil):
    """
    文件工具类
    """

    def init(self):
        pass

    @staticmethod
    def get_content(path, encoding='gbk'):
        """
        读取文本内容
        :param path:
        :param encoding:
        :return:
        """
        with open(path, 'r', encoding=encoding, errors='ignore') as f:
            content = ''
            for l in f:
                l = l.strip()
                content += l
            return content

    @staticmethod
    def save_to_text(filename, content, mode='w'):
        """
        保存为文本
        :param filename:
        :param content:
        :return:
        """
        FileUtils.check_file_exists(filename)
        with open(filename, mode, encoding='utf-8') as f:
            f.writelines(content)

    @staticmethod
    def save_to_json(filename, content):
        """
        保存map 数据
        :param filename:
        :param maps:
        :return:
        """
        FileUtils.check_file_exists(filename)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(content, f, ensure_ascii=False)

    @staticmethod
    def load_json(filename) -> json:
        if not os.path.exists(filename):
            return dict()

        with open(filename, 'r', encoding='utf8') as f:
            return json.load(f)

    @staticmethod
    def load_json_line(filename, encoding='utf8') -> List:
        """
        读取json line 格式的数据
        :param filename:
        :param encoding:
        :return:
        """
        result = []
        if not os.path.exists(filename):
            return result

        with open(filename, 'r', encoding=encoding) as f:
            for line in f:
                result.append(json.loads(line))
        return result

    @staticmethod
    def dump_json(fp, obj, sort_keys=False, indent=4, show_info=True):
        try:
            fp = os.path.abspath(fp)
            if not os.path.exists(os.path.dirname(fp)):
                os.makedirs(os.path.dirname(fp))
            with open(fp, 'w', encoding='utf8') as f:
                json.dump(obj, f, ensure_ascii=False, sort_keys=sort_keys, indent=indent, separators=(',', ':'))
            if show_info:
                logger.info(f'json line 文件保存成功，{fp}')
            return True
        except Exception as e:
            logger.info(f'json line 文件 {obj} 保存失败, {e}')
            return False

    @staticmethod
    def dump_json_line(file_name, data, encoding='utf8', show_info=True):
        """
        保存json line 格式
        :param file_name: 
        :param data: 
        :param encoding:
        :param show_info:
        :return:
        """
        try:
            file_name = os.path.abspath(file_name)
            if not os.path.exists(os.path.dirname(file_name)):
                os.makedirs(os.path.dirname(file_name))
            with open(file_name, 'w', encoding=encoding) as f:
                for record in data:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
                # json.dump(data, f, ensure_ascii=False, indent=4, separators=(',', ':'))
            if show_info:
                logger.info(f'json line 文件保存成功，{file_name}')
            return True
        except Exception as e:
            logger.info(f'json line 文件 {data} 保存失败, {e}')
            return False

    @staticmethod
    def get_file_name_list(path, type="*.txt"):
        """获取指定路径下的指定类型的所有文件"""
        files = glob.glob(os.path.join(path, type))
        return files

    @staticmethod
    def check_file_exists(filename, delete=False):
        """检查文件是否存在"""
        dir_name = os.path.dirname(filename)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print("文件夹不存在,创建目录:{}".format(dir_name))
        return os.path.exists(filename)

    @staticmethod
    def read_to_text(path, encoding='utf-8'):
        """读取txt 文件"""
        with open(path, 'r', encoding=encoding) as f:
            content = f.read()
            return content

    @staticmethod
    def read_to_text_list(path, encoding='utf-8'):
        """
        读取txt文件,默认utf8格式,
        :param path:
        :param encoding:
        :return:
        """
        list_line = []
        if not os.path.exists(path):
            return list_line
        with open(path, 'r', encoding=encoding) as f:
            list_line = f.readlines()
            list_line = [row.rstrip("\n") for row in list_line]
            return list_line

    @staticmethod
    def list_file(file_dir, endswith="", add_dir=False):
        """
        获取目录下的 指定后缀的文件列表
        :param file_dir:
        :param endswith:
        :param add_dir:
        :return:
        """
        file_list = []
        if not os.path.exists(file_dir):
            return file_list

        for file_name in os.listdir(file_dir):
            if file_name.endswith(endswith):
                if add_dir:
                    file_list.append(os.path.join(file_dir, file_name))
                else:
                    file_list.append(file_name)

        return file_list

    @staticmethod
    def get_dir_sub_dir(file_dir, add_dir=False):
        """
        获取当前目录下的 所有子目录
        :param file_dir:
        :param add_dir:
        :return:
        """
        file_list = []
        if not os.path.exists(file_dir):
            return file_list

        for file_name in os.listdir(file_dir):
            file_path = os.path.join(file_dir, file_name)
            file_path = os.path.abspath(file_path)
            if os.path.isdir(file_path):
                if add_dir:
                    file_list.append(file_path)
                else:
                    file_list.append(file_name)

        return file_list

    @staticmethod
    def delete_file_all(path):
        """
        删除一个目录下的所有文件
        :param path:
        :return:
        """

        for i in os.listdir(path):
            path_children = os.path.join(path, i)
            if os.path.isfile(path_children):
                os.remove(path_children)
            else:  # 递归, 删除目录下的所有文件
                FileUtils.delete_file(path_children)

    @staticmethod
    def list_dir_or_file(file_dir, add_parent=False, sort=False, start_with=None, is_dir=True):
        """
        读取文件夹下的所有子文件夹
        :param file_dir:
        :param add_parent:
        :param sort:
        :param start_with:
        :param is_dir:
        :return:
        """
        dir_list = []
        if not os.path.exists(file_dir):
            return dir_list
        for name in os.listdir(file_dir):
            run_dir = os.path.join(file_dir, name)
            flag = os.path.isdir(run_dir) if is_dir else os.path.isfile(run_dir)
            if flag:
                if start_with is not None and not str(name).startswith(start_with):
                    continue
                if add_parent:
                    run_dir = os.path.join(file_dir, name)
                else:
                    run_dir = name
                dir_list.append(run_dir)

        if sort:
            dir_list.sort(key=lambda k: str(k), reverse=False)

        return dir_list

    @staticmethod
    def list_dir(file_dir, add_parent=False, sort=False, start_with=None):
        """
        读取文件夹下的所有子文件夹
        :param file_dir:
        :param add_parent:
        :param sort:
        :param start_with:
        :param is_dir:
        :return:
        """
        return FileUtils.list_dir_or_file(file_dir=file_dir, add_parent=add_parent, sort=sort, start_with=start_with,
                                          is_dir=True)

    @staticmethod
    def list_file_prefix(file_dir, add_parent=False, sort=False, start_with=None):
        """
        读取文件夹下的所有文件
        :param file_dir:
        :param add_parent:
        :param sort:
        :param start_with: 文件前缀
        :param is_dir:
        :return:
        """
        return FileUtils.list_dir_or_file(file_dir=file_dir, add_parent=add_parent, sort=sort, start_with=start_with,
                                          is_dir=False)

    @staticmethod
    def delete_file(path):
        """
        删除一个文件
        :param path:
        :return:
        """
        if os.path.exists(path):
            if os.path.isfile(path):
                os.remove(path)
                logger.info(f"删除文件：{path}")

    @staticmethod
    def get_path_dir(file_dir, add_parent=False, sort=False, start_with=None):
        """
        读取文件夹下的所有子文件夹
        :param file_dir:
        :param add_parent:
        :param sort:
        :param start_with:
        :return:
        """
        dir_list = []
        # files_list = []
        for root, dirs, files in os.walk(file_dir):
            for run_dir in dirs:
                if start_with is not None and not str(run_dir).startswith(start_with):
                    continue
                if add_parent:
                    run_dir = os.path.join(file_dir, run_dir)
                dir_list.append(run_dir)
            # dir_list.extend(dirs)
            # files_list.extend(files)
        if sort:
            dir_list.sort(key=lambda k: str(k), reverse=False)

        return dir_list

    @staticmethod
    def save_to_pickle(model, file_name):
        """
        保存模型
        :param model:
        :param file_name:
        :return:
        """
        FileUtils.check_file_exists(file_name)
        pickle.dump(model, open(file_name, "wb"))

    @staticmethod
    def load_to_model(file_name):
        """
         使用pickle加载模型文件
        :param file_name:
        :return:
        """
        loaded_model = pickle.load(open(file_name, "rb"))
        return loaded_model

    @staticmethod
    def get_file_name(file_name):
        """
        获取文件名称
        :param file_name:
        :return:
        """
        begin_index = str(file_name).rfind("/")
        end_index = str(file_name).rfind(".")
        return file_name[(begin_index + 1):end_index]

    @staticmethod
    def remove_file_name_end(file_name):
        """
        移除文件后缀
        :param file_name:
        :return:
        """
        end_index = str(file_name).rfind(".")
        return file_name[0:end_index]

    @staticmethod
    def get_file_size(file_name):
        """
        获取文件大小
        :param file_name:
        :return:
        """
        file_size = os.path.getsize(file_name)
        return file_size

    @staticmethod
    def get_file_line(file_name):
        """
        获取文件 行数
        :param file_name:
        :return:
        """
        count = 0
        with open(file_name, encoding="utf-8") as f:
            for line in f:
                count += 1
        return count

    @staticmethod
    def get_dir_file_name(file_name):
        """
        获取文件目录名称
        :param file_name:
        :return:
        """
        return os.path.dirname(file_name)

    @staticmethod
    def read_file(file_name, block_size=1024 * 8):
        """
        读取文件
        :param block_size:
        :param file_name:
        :return:
        """

        count = 0
        with open(file_name) as fp:
            for chunk in FileUtils.chunked_file_reader(fp, block_size):
                count += 1
        return count

    @staticmethod
    def chunked_file_reader(fp, block_size=1024 * 8):
        """生成器函数：分块读取文件内容，使用 iter 函数
        """
        # 首先使用 partial(fp.read, block_size) 构造一个新的无需参数的函数
        # 循环将不断返回 fp.read(block_size) 调用结果，直到其为 '' 时终止
        for chunk in iter(partial(fp.read, block_size), ''):
            yield chunk

    @staticmethod
    def read_lines(file_name, encoding='utf-8'):
        """
        给定一个文件路径，读取文件并返回一个迭代器，这个迭代器将顺序返回文件的每一行
        """
        with open(file_name, 'r', encoding=encoding) as f:
            for line in f:
                yield line

    @staticmethod
    def get_parent_dir_name(file_name):
        """
        获取文件父目录名称
        :param file_name:
        :return:
        """
        dir_name = FileUtils.get_dir_file_name(file_name)
        begin_index = str(dir_name).rfind("/")
        return dir_name[(begin_index + 1):]

    @staticmethod
    def copy_file(src, dst, cp_metadata=False):
        """
        拷贝文件
        :param src:
        :return:
        """
        file_name = os.path.basename(src)
        dst_file_name = os.path.join(dst, file_name)

        if src is None or len(src) == 0 or not os.path.exists(src):
            return dst_file_name
        FileUtils.check_file_exists(os.path.join(dst, "tmp.txt"))

        if cp_metadata:
            # Copy files, but preserve metadata (cp -p src dst)
            shutil.copy2(src, dst)
        else:
            #  Copy src to dst. (cp src dst)
            shutil.copy(src, dst)

        file_name = os.path.basename(src)
        dst_file_name = os.path.join(dst, file_name)
        return dst_file_name

    @staticmethod
    def copy_file_rename(src, dst):
        """
        拷贝文件
        :param src:
        :return:
        """
        if src is None or len(src) == 0 or not os.path.exists(src) or os.path.exists(dst):
            return dst
        FileUtils.check_file_exists(dst)

        shutil.copyfile(src, dst)

        file_name = os.path.basename(src)
        dst_file_name = os.path.join(dst, file_name)
        return dst_file_name

    @staticmethod
    def copy_dir(src, dst, symlinks=True):
        """
        拷贝目录
        :param src:
        :return:
        """
        FileUtils.check_file_exists(os.path.join(dst, "tmp.txt"))
        # Copy directory tree (cp -R src dst)
        shutil.copytree(src, dst, symlinks)

    @staticmethod
    def move_file(src, dst):
        """
        移动文件
        :param src:
        :return:
        """
        FileUtils.check_file_exists(os.path.join(dst, "tmp.txt"))
        shutil.move(src, dst)

    @staticmethod
    def add_time_sub_dir(save_dir):
        """
        添加时间子目录： %Y-%m-%d
        :param save_dir:
        :return:
        """
        file_path = os.path.join(save_dir, TimeUtils.get_time())
        return file_path

    @staticmethod
    def load_set_file(path):
        words = set()
        with open(path, 'r', encoding='utf-8') as f:
            for w in f:
                w = w.strip()
                if w.startswith('#'):
                    continue
                if w:
                    words.add(w)
        return words

    @staticmethod
    def load_same_pinyin(file_name, sep='\t'):
        """
        加载同音字
        :param file_name:
        :param sep:
        :return:
        """
        result = dict()
        if not os.path.exists(file_name):
            logger.warn("file not exists:" + file_name)
            return result
        with open(file_name, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#'):
                    continue
                parts = line.split(sep)
                if parts and len(parts) > 2:
                    key_char = parts[0]
                    same_pron_same_tone = set(list(parts[1]))
                    same_pron_diff_tone = set(list(parts[2]))
                    value = same_pron_same_tone.union(same_pron_diff_tone)
                    if key_char and value:
                        result[key_char] = value
        return result

    @staticmethod
    def load_same_stroke(file_name, sep='\t'):
        """
        加载形似字
        :param file_name:
        :param sep:
        :return:
        """
        result = dict()
        if not os.path.exists(file_name):
            logger.warn("file not exists:" + file_name)
            return result
        with open(file_name, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#'):
                    continue
                parts = line.split(sep)
                if parts and len(parts) > 1:
                    for i, c in enumerate(parts):
                        exist = result.get(c, set())
                        current = set(list(parts[:i] + parts[i + 1:]))
                        result[c] = exist.union(current)
        return result

    @staticmethod
    def load_word_freq_dict(file_name):
        """
        加载切词词典
        :param file_name:
        :return:
        """
        word_freq = {}
        if file_name:
            if not os.path.exists(file_name):
                logger.warning('file not found.%s' % file_name)
                return word_freq
            else:
                with open(file_name, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith('#'):
                            continue
                        info = line.split()
                        if len(info) < 1:
                            continue
                        word = info[0]
                        # 取词频，默认1
                        freq = int(info[1]) if len(info) > 1 else 1
                        word_freq[word] = freq
        return word_freq

    @staticmethod
    def read_data_from_file(file_name, sep=Constants.DELIMITER_TAB, column_name=Constants.COLUMN_NAME_4,
                            data_type=DataFileType.CSV, show_info=False) -> List[InputExample]:
        """
        读取数据

        :param file_name:
        :param sep:
        :param column_name:
        :param data_type: 数据格式类型
        :param show_info: 数据格式类型
        :return:
        """
        if data_type == DataFileType.CSV:
            # return FileUtils.read_data_from_pandas(file_name=file_name, sep=sep, column_name=column_name)
            return FileUtils.read_data_from_csv(file_name=file_name, sep=sep, column_name=column_name,
                                                show_info=show_info)
        elif data_type == DataFileType.JSON_LINE:
            return FileUtils.read_data_from_json_line(file_name=file_name)
        else:
            return FileUtils.read_ner_data_from_file(file_name=file_name, data_type=data_type,
                                                     input_type="train", markup='bios')

    @staticmethod
    def read_data_with_file_end(file_name, sep=Constants.DELIMITER_TAB, column_name=Constants.COLUMN_NAME_4,
                                show_info=False, eval_ture_input=False, return_str=True, ) -> List[InputExample]:
        """
        通过文件后缀结尾判断文件类型进行文件读取

        :param file_name:
        :param sep:
        :param column_name:
        :param show_info:
        :param text_index:
        :param label_index:
        :return:
        """

        if str(file_name).endswith("pkl"):
            input_examples = FileUtils.read_csc_data_from_pickle(file_name=file_name, show_info=show_info,
                                                                 eval_ture_input=eval_ture_input, return_str=return_str)
        else:
            input_examples = FileUtils.read_data_from_csv(file_name=file_name, show_info=show_info, sep=sep,
                                                          column_name=column_name, return_str=return_str)

        return input_examples

    @staticmethod
    def read_data_from_csv(file_name, sep=Constants.DELIMITER_TAB, column_name=Constants.COLUMN_NAME_4,
                           show_info=False, text_index=0, label_index=1, return_str=True, ) -> List[
        InputExample]:
        """
        通过pandas 读取数据
        :return:
        """
        examples = []

        total = 0
        if show_info:
            total = FileUtils.get_file_line(file_name)
            logger.info(f"total line : {total}")
        with open(file_name, 'r') as reader:
            run_num = 0
            for index, line in enumerate(tqdm(reader)):
                line = line.strip()
                split_line = str(line).split(sep)
                text_a = split_line[text_index]
                if not return_str:
                    text_a = str(text_a).split()

                if len(split_line) > 1:
                    label = split_line[label_index]
                    if not return_str:
                        label = str(label).split()

                else:
                    label = None
                run_num += 1

                if show_info and total > 100000 and run_num % 100000 == 0:
                    logger.info(f"read data: {run_num} / {total} - {run_num / total:.3f}")
                examples.append(InputExample(guid=index, text_a=text_a, label=label))

        return examples

    @staticmethod
    def read_data_from_pandas(file_name, sep=Constants.DELIMITER_TAB, column_name=Constants.COLUMN_NAME_4,
                              show_info=False) -> List[
        InputExample]:
        """
        通过pandas 读取数据
        :return:
        """
        df = pd.read_csv(file_name, sep=sep, names=column_name, quoting=csv.QUOTE_NONE)

        examples = []

        total = len(df)
        if total > 100000:
            for index in tqdm(range(total)):
                text = df.iloc[index, 0]
                label = df.iloc[index, 1]
                examples.append(InputExample(guid=index, text_a=text, label=label))
        else:
            for index in range(total):
                text = df.iloc[index, 0]
                label = df.iloc[index, 1]
                examples.append(InputExample(guid=index, text_a=text, label=label))

        return examples

    @staticmethod
    def read_data_cls_label_from_csv(file_name, sep=Constants.DELIMITER_TAB, column_name=Constants.COLUMN_NAME_4,
                                     show_info=False, return_str=True, ) -> List[InputExample]:
        """
        读取 cls slot label 数据
        :param file_name:
        :param sep:
        :param column_name:
        :param show_info:
        :param return_str:
        :return:
        """
        examples = []
        total = 0

        raw_list = FileUtils.read_to_text_list(file_name)

        if show_info:
            total = FileUtils.get_file_line(file_name)
            logger.info(f"total line : {total}")

        for index, line in enumerate(raw_list):
            splits = str(line).strip().split(sep)
            text_a = splits[0]
            text_b = splits[1]

            if not return_str:
                text_a = text_a.split()
                text_b = text_b.split()

            label = splits[2]
            seq_len = splits[3]

            examples.append(InputExample(guid=index, text_a=text_a, label=label, text_b=text_b, length=seq_len))

        return examples

    @staticmethod
    def read_data_from_json_line(file_name, input_type="train") -> List[
        InputExample]:
        """
        读取 json line 格式的文件

            {"label": "108", "label_desc": "news_edu", "sentence": "上课时学生手机响个不停，老师一怒之下把手机摔了，家长拿发票让老师赔，大家怎么看待这种事？", "keywords": ""}
        :param file_name:
        :param input_type:
        :param column_name:
        :return:
        """
        examples = []

        with jsonlines.open(file_name, 'r') as reader:
            for index, line in enumerate(reader):
                guid = str(index)
                if "sentence" in line:
                    text_a = line.get('sentence')
                else:
                    text_a = line.get('text')
                if "sentence" in line:
                    label = line.get('label')
                else:
                    label = ""

                text_a = " ".join(list(text_a))
                examples.append(InputExample(guid=guid, text_a=text_a, label=label))
        return examples

    @staticmethod
    def read_ner_data_from_json_line_without_label(input_file, input_type="train", markup='bios') -> List[InputExample]:
        """
            读取 json line 格式的文件 没有标签

            - 示例
            {"id": 0, "text": "\"日产的社长兼首席执行官（ceo）西川广人在会议伊始向股东\"\"代表公司深表歉意\"\"。\""}

        :param input_file:
        :param input_type:
        :param markup:
        :return:
        """

        # 读取文件
        raw_lines = FileUtils.read_to_text_list(input_file)

        examples = []
        for index, line in enumerate(raw_lines):
            line = json.loads(line.strip())
            text = line['text']
            guid = str(line.get('id', -1))
            examples.append(InputExample(guid=guid, text_a=text))
        return examples

    @staticmethod
    def read_ner_data_from_json(input_file, input_type="train", markup='bios') -> List[InputExample]:
        """
            读取 json 格式的文件

            - 示例
             {
                "context": "当 希 望 工 程 救 助 的 百 万 儿 童 成 长 起 来 ， 科 教 兴 国 蔚 然 成 风 时 ， 今 天 有 收 藏 价 值 的 书 你 没 买 ， 明 日 就 叫 你 悔 不 当 初 ！",
                "end_position": [],
                "entity_label": "NS",
                "impossible": true,
                "qas_id": "0.1",
                "query": "按照地理位置划分的国家,城市,乡镇,大洲",
                "span_position": [],
                "start_position": []
              },

        :param input_file:
        :param input_type:
        :param markup:
        :return:
        """

        # 读取文件
        raw_lines = FileUtils.load_json(input_file)

        examples = []
        for index, line in enumerate(raw_lines):
            # guid = "%s-%s" % (input_type, index)
            guid = str(index)
            text_a = line['context'].split()
            text_b = line['query'].split()
            labels = line['entity_label']
            start_ids = line['start_position']
            end_ids = line['end_position']
            qas_id = line['qas_id']

            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=labels, start_ids=start_ids,
                                         end_ids=end_ids, qas_id=qas_id))

        return examples

    @staticmethod
    def read_ner_data_from_text_line(input_file, sep="\t", input_type=None) -> List[InputExample]:
        """
            读取 text line 格式的文件

            - 示例
                常 B-NAME
                建 M-NAME
                良 E-NAME
                ， O
                男 O
                ， O


        :param input_file:  一行  一个字和一个标注，以空行表示不同的句子
        :param sep:
        :param input_type:
        :return:
        """

        # 读取文件
        raw_lines = FileUtils.read_to_text_list(input_file)

        examples = []

        text_a = []
        labels = []

        # 解析一行 中的 text 和label
        for line in raw_lines:
            line = line.strip()
            # 一条文本结束
            if len(line) == 0:
                # guid = "%s-%s" % (input_type, len(examples))
                guid = str(len(examples))
                examples.append(InputExample(guid=guid, text_a=text_a, label=labels))
                text_a = []
                labels = []
                continue
            splits = line.split(sep=sep)
            if len(splits) == 2:
                text_a.append(splits[0])
                labels.append(splits[1])

        return examples

    @staticmethod
    def read_ner_data_from_file(file_name, data_type=DataFileType.NER_JSON_LINE, input_type="train", markup='bios') -> \
            List[InputExample]:
        """
        读取 NER 数据集
        :param file_name:
        :param data_type:
        :param input_type:
        :param markup:
        :return:
        """

        if data_type == DataFileType.NER_JSON_LINE:
            return FileUtils.read_ner_data_from_json_line(input_file=file_name, input_type=input_type, markup=markup)
        elif data_type == DataFileType.NER_VERTICAL_BIO:
            return FileUtils.read_ner_data_from_text_line(input_file=file_name, input_type=input_type)
        elif data_type == DataFileType.NER_JSON:
            return FileUtils.read_ner_data_from_json(input_file=file_name, input_type=input_type, markup=markup)
        elif data_type == DataFileType.NER_JSON_LINE_WITHOUT_LABEL:
            return FileUtils.read_ner_data_from_json_line_without_label(input_file=file_name, input_type=input_type,
                                                                        markup=markup)
        else:
            # elif data_type == DataFileType.NER_TEXT:
            return FileUtils.read_ner_data_from_text_line(input_file=file_name, input_type=input_type)

    @staticmethod
    def read_csc_data_from_pickle(file_name, data_type=None, return_str=False, show_info=False,
                                  eval_ture_input=False) -> List[InputExample]:
        """
        读取 csc 训练数据
        :param file_name:
        :param data_type:
        :param return_str:
        :param show_info:
        :param eval_ture_input:
        :return:
        """
        dataset = FileUtils.load_to_model(file_name)

        examples = []
        run_dataset = tqdm(tqdm(enumerate(dataset))) if show_info else enumerate(dataset)
        for index, data in run_dataset:
            # guid = str(index)
            guid = data['id']
            text_a = list(data['src'])
            labels = list(data['tgt'])
            if eval_ture_input:
                text_a = labels

            if return_str:
                text_a = " ".join(text_a)
                labels = " ".join(labels)

            start_ids = data['src_idx']
            end_ids = data['tgt_idx']
            other = data['tokens_size']
            length = data['lengths']
            examples.append(InputExample(guid=guid, text_a=text_a, label=labels, start_ids=start_ids,
                                         end_ids=end_ids, length=length, other=other))
        return examples

    @staticmethod
    def get_checkpoint_num(file_path):
        """
        获取 checkpoint_num
        :param file_path:
        :return:
        """
        dir_name = FileUtils.get_parent_dir_name(f"{file_path}/test.txt")
        check_num = str(dir_name).split("-")[1]
        return int(check_num)

    @staticmethod
    def get_step_num(file_name, index=2):
        """
        获取 epoch_step
        :param file_name: eval_0_7000_metric.json
        :param index:
        :return:
        """
        raw_file_name = FileUtils.get_file_name(file_name)
        check_num = str(raw_file_name).split("_")[index]
        res = math.floor(float(check_num))
        return res

    @staticmethod
    def get_lm_model_bin(nlp_pretrain: NlpPretrain):
        """
        获取 pretrain 的 bin path
        :param nlp_pretrain:
        :return:
        """
        model_path = os.path.join(nlp_pretrain.path, "pytorch_model.bin")
        return model_path

    @staticmethod
    def load_vocab(vocab_file):
        """Loads a vocabulary file into a dictionary."""
        vocab = collections.OrderedDict()
        with open(vocab_file, "r", encoding="utf-8") as reader:
            tokens = reader.readlines()
        for index, token in enumerate(tokens):
            token = token.rstrip("\n")
            vocab[token] = index
        return vocab

    @staticmethod
    def get_output_predict_path(output_dir, metric_key_prefix="eval"):
        """
        获取RLS预测的文件路径
        :param output_dir:
        :param metric_key_prefix:
        :return:
        """
        pred_txt_path = os.path.join(output_dir, metric_key_prefix, "preds.txt")
        pred_lbl_path = os.path.join(output_dir, metric_key_prefix, "labels.txt")
        return pred_lbl_path, pred_txt_path

    @staticmethod
    def get_file_path_csc_rls_original_truth(eval_file_path):
        """
        获取 原始的标签的路径
        :param eval_file_path:
        :return:
        """
        file_dir = Constants.CSC_RLS_DATA_OUTPUT_DIR
        file_name = FileUtils.get_file_name(eval_file_path)
        out_file_name = f"{file_dir}/{file_name}_original_truth_{TimeUtils.get_time()}_rls.txt"
        return out_file_name

    @staticmethod
    def get_file_path_csc_rls_output(file_path, sep_name="predict_result", end_with="rls.txt", add_time=True):
        """
        获取 原始的标签的路径
        :param file_path:
        :param sep_name:
        :param end_with:
        :param add_time:
        :return:
        """
        file_dir = Constants.CSC_RLS_DATA_OUTPUT_DIR
        file_name = FileUtils.get_file_name(file_path)
        out_file_name = f"{file_dir}/{file_name}_{sep_name}_{TimeUtils.get_time()}_{end_with}"
        if not add_time:
            out_file_name = f"{file_dir}/{file_name}_{sep_name}_{end_with}"
        return out_file_name

    @staticmethod
    def stopwords(filename=Constants.STOPWORDS_PATH):
        """
        中文停用词
        :param filename:
        :return:
        """
        return FileUtils.read_to_text_list(filename)

    @staticmethod
    def read_confusion_set(filename=Constants.CONFUSION_SET_PATH):
        """
        读取混淆集
        施:司事十试弛尸师失诗时食史狮实饰旋使是拖腿石释
        :param filename:
        :return:
        """
        confusion_set = {}
        all_lines = FileUtils.read_to_text_list(filename)
        for line in all_lines:
            splits = str(line).split(":")
            if len(splits) != 2:
                continue
            words = list(splits[1])
            confusion_set[splits[0]] = words

        return confusion_set

    @staticmethod
    def read_same_mean_word(filename=Constants.SAME_MEAN_WORD_PATH, same_mean=True):
        """
        读取近义词
        Hj32C01= 休息 休憩 歇息 喘息 停歇 喘气 休 歇 息 喘喘气 作息 歇歇
        :param filename:
        :param same_mean:
        :return:
        """
        combine_dict = {}
        endwith = "=" if same_mean else "#"
        all_lines = FileUtils.read_to_text_list(filename)
        for line in all_lines:
            seperate_word = line.strip().split(" ")
            # 仅保留真正近义词，过滤相关词和独立词
            if not seperate_word[0].endswith(endwith):
                continue
            num = len(seperate_word)
            for i in range(1, num):
                wi = seperate_word[i]
                combine_dict[wi] = seperate_word[1:]
        return combine_dict

    @staticmethod
    def get_dataset_path_csc(dataset_name="csc"):
        """
        根据任务名称获取 训练数据
        :param dataset_name:
        :return:
        """

        data_file_list = {
            "train": Constants.CSC_RLS_DATA_TRAIN_DIR,
            "eval": Constants.CSC_RLS_DATA_EVAL_15_DIR,
            "eval_label": Constants.CSC_RLS_DATA_EVAL_LABEL_15_DIR,
            "test": Constants.CSC_RLS_DATA_EVAL_15_DIR,
        }

        if dataset_name == Constants.DATASET_NAME_CSC_DCN_BUSINESS:
            data_file_list = {
                "train": Constants.CSC_DCN_BUSINESS_TRAIN_BIG,
                # "train": Constants.CSC_DCN_BUSINESS_DEV_ZG,
                "eval": Constants.CSC_DCN_BUSINESS_DEV_ZG,
                "test": Constants.CSC_DCN_BUSINESS_DEV_ZG,
            }
        elif dataset_name == Constants.DATASET_NAME_CSC_DCN_WIKI_28:
            data_file_list = {
                "train": Constants.CSC_DCN_WIKI_28_TRAIN_DIR,
                "eval": Constants.CSC_DCN_DATA_EVAL_15_DIR_CSV,
                "test": Constants.CSC_DCN_DATA_EVAL_15_DIR_CSV,
            }
        elif dataset_name == Constants.DATASET_NAME_CSC_PLOME:
            data_file_list = {
                "train": Constants.CSC_PLOME_DATA_TRAIN_DIR,
                "eval": Constants.CSC_PLOME_DATA_TEST_SIGHAN_DIR,
                "test": Constants.CSC_PLOME_DATA_TEST_SIGHAN_DIR,
            }

        return data_file_list

    @staticmethod
    def get_dataset_path_cls(dataset_name="cls"):
        """
        根据任务名称获取 训练数据
        :param dataset_name:
        :return:
        """

        data_file_list = {
            # "train": Constants.CLS_DATA_TRAIN_ALL,
            "train": Constants.CLS_RLS_DATA_TRAIN_DIR_CSV,
            "eval": Constants.CLS_RLS_DATA_EVAL_15_CSV,
            "test": Constants.CLS_RLS_DATA_EVAL_15_CSV,
        }

        if dataset_name == Constants.DATASET_NAME_CLS_TNEWS:
            data_file_list = {
                "train": Constants.CLS_DATA_TNEWS_TRAIN,
                "eval": Constants.CLS_DATA_TNEWS_DEV,
                "test": Constants.CLS_DATA_TNEWS_TEST,
            }
        elif dataset_name == Constants.DATASET_NAME_CLS_RLS:
            data_file_list = {
                "train": Constants.CSC_RLS_DATA_TRAIN_DIR_CSV,
                "eval": Constants.CSC_RLS_DATA_EVAL_15_DIR_CSV,
                "test": Constants.CSC_RLS_DATA_EVAL_15_DIR_CSV,
            }
        elif dataset_name == Constants.DATASET_NAME_CLS_SLOT:
            data_file_list = {
                "train": Constants.CSC_DATA_TRAIN_28_WIKI_DIR_CSV,
                "eval": Constants.CSC_DATA_EVAL_15_DIR_CSV,
                "test": Constants.CSC_DATA_EVAL_15_DIR_CSV,
            }

        return data_file_list


if __name__ == '__main__':
    pass
    FileUtils.get_checkpoint_num(Constants.CSC_RLS_CHECKPOINT_83000_DIR)
    # FileUtils.read_csc_data_from_pickle()
    print(FileUtils.get_file_name("./test.txt"))
