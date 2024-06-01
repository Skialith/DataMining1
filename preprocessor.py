import os
import pickle
from tqdm import tqdm
from collections import Counter, OrderedDict
from jieba import posseg as pseg


class CutResult(object):
    """
    分词结果
    char_counter：字频统计
    author_counter：作者计数
    word_set：词汇表
    word_counter：词汇计数
    word_property_counter_dict：词汇词性
    author_poetry_dict：解析后的结果，作者与他对应的诗
    """

    def __init__(self):
        self.char_counter = Counter()
        self.author_counter = Counter()
        self.word_set = set()
        self.word_counter = Counter()
        self.word_property_counter_dict = {}
        self.author_poetry_dict = OrderedDict()

    def add_cut_poetry(self, author, divided_lines):
        """为author_poetry_dict添加对象"""
        ctp = self.author_poetry_dict.get(author)
        if ctp is None:
            self.author_poetry_dict[author] = ""
        else:
            self.author_poetry_dict[author] += ' '
        self.author_poetry_dict[author] += ' '.join(divided_lines)


def _is_chinese(c):
    return '\u4e00' <= c <= '\u9fff'


from tqdm import tqdm

def cut_poetry(filename, saved_dir):
    """
    对全唐诗分词
    :param filename: 全唐诗输入文件位置
    :param saved_dir: 结果存储位置
    :return: 分词结果
    """
    target_file_path = os.path.join(saved_dir, 'cut_result.pkl')
    if not os.path.exists(saved_dir):
        os.mkdir(saved_dir)
    if os.path.exists(target_file_path):
        print('load existed cut result.')
        with open(target_file_path, 'rb') as f:
            result = pickle.load(f)
    else:
        print('begin cutting poetry...')
        result = CutResult()
        current_author = None
        divided_lines = []
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()  # 读取所有行，便于用tqdm显示进度条

        # 使用tqdm包装循环，显示进度条
        for line in tqdm(lines, desc="Processing", unit=" lines"):
            if line.strip() == "":
                continue
            # 解析作者
            if "【" in line:
                header = line.split()[1]
                author = header[header.find("】") + 1:].strip()
                result.author_counter[author] += 1
                divided_lines.append("\n")
                # 将当前分词后的结果加入结果表中
                if current_author is not None:
                    result.add_cut_poetry(current_author, divided_lines)
                    divided_lines = []
                current_author = author
                continue
            # 解析诗句
            chars = [c for c in line if _is_chinese(c)]
            for char in chars:
                result.char_counter[char] += 1
            cut_line = pseg.cut(line)
            for word, property in cut_line:
                if not _is_chinese(word):
                    continue
                if result.word_property_counter_dict.get(property) is None:
                    result.word_property_counter_dict[property] = Counter()
                result.word_property_counter_dict[property][word] += 1
                result.word_set.add(word)
                result.word_counter[word] += 1
                divided_lines.append(word)
        # 加入最后一次解析的结果
        if current_author:
            result.add_cut_poetry(current_author, divided_lines)

        with open(target_file_path, 'wb') as f:
            pickle.dump(result, f)

    return result

