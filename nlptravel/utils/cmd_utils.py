#!/user/bin/env python
# -*- coding: utf-8 -*-
# @Project nlptravel
# @File  : cmd_utils.py
# @Author: sl
# @Date  : 2022/4/16 - 下午3:15
import os
import subprocess
from typing import List

from nlptravel.utils.base_utils import BaseUtil
from nlptravel.utils.constant import Constants
from nlptravel.utils.file_utils import FileUtils

"""
命令行处理的工具类

"""


class CmdUtils(BaseUtil):
    """
    命令行工具类
    """

    def init(self):
        pass

    @staticmethod
    def run_cmd1(cmds):
        """
        run cmd
        :param cmds:
        :return:
        """
        res = subprocess.run(cmds, stdout=subprocess.PIPE)
        result = str(res.stdout).strip().replace("\n", "")
        return result

    @staticmethod
    def run_cmd(cmds):
        """
        run cmd
        :param cmds:
        :return:
        """
        if isinstance(cmds, List):
            cmds = "".join(cmds)
        res = os.system(cmds)
        result = str(res).strip().replace("\n", "")
        return result

    @staticmethod
    def modify_constants_py_file(run_env=None):
        current_src_dir = f"{Constants.SRC_DIR}"
        if run_env is None:
            current_dir = os.path.abspath(CmdUtils.run_cmd(["pwd"]))
            print(f"current_dir: {current_dir}")

            run_env = "win"
            if current_dir.startswith("/kaggle"):
                run_env = "kaggle"
                current_src_dir = f"/kaggle/working/nlptravel/nlptravel"
            elif current_dir.startswith("/content"):
                run_env = "colab"
                current_src_dir = f"/content/drive/MyDrive/NLP/nlptravel/nlptravel"
            print(f"current run env = {run_env}")
            if run_env == "win":
                return

        constants_file_name = f"{current_src_dir}/utils/constant.py"
        src_list = FileUtils.read_to_text_list(constants_file_name)

        result_file = f"{current_src_dir}/utils/constant.py"
        results = []
        # run_env = "kaggle"
        # run_env = "colab"

        for line in src_list:
            new_line = line
            if str(line).replace(" ", "").startswith("WORK_DIR="):
                print(line)
                if run_env == "kaggle":
                    new_line = '    WORK_DIR = "/kaggle/working/nlptravel"'
                elif run_env == "colab":
                    new_line = '    WORK_DIR = "/content/drive/MyDrive/NLP/nlptravel"'
            if str(line).replace(" ", "").startswith("NLP_DATA_DIR="):
                print(line)
                if run_env == "kaggle":
                    new_line = '    NLP_DATA_DIR = f"/kaggle/input"'
                elif run_env == "colab":
                    new_line = '    NLP_DATA_DIR = f"/content/drive/MyDrive/NLP"'
            if str(line).replace(" ", "").startswith("NLP_CSC_DATA_DIR="):
                print(line)
                if run_env == "kaggle":
                    new_line = '    NLP_CSC_DATA_DIR = f"{NLP_DATA_DIR}/nlp-csc"'
                elif run_env == "colab":
                    new_line = '    NLP_CSC_DATA_DIR = f"{NLP_DATA_DIR}/csc"'

            results.append(new_line)

        FileUtils.save_to_text(result_file, "\n".join(results))
        print(f"rewrite the modify constants file")

if __name__ == '__main__':
    CmdUtils.modify_constants_py_file()