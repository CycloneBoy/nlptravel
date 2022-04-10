#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：nlptravel
# @File    ：ner_entity.py
# @Author  ：sl
# @Date    ：2022/1/17 19:29
import copy
import json


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, subject=None, start_ids=None, end_ids=None,
                 qas_id=None, other=None, length=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.subject = subject
        self.start_ids = start_ids
        self.end_ids = end_ids
        self.qas_id = qas_id
        self.other = other
        self.length = length

    def __str__(self):
        show_list = [str(self.guid), str(self.text_a), str(self.text_b), str(self.label), str(self.subject)]
        return " ,".join(show_list)

    def to_str(self, sep="|"):
        show_list = [str(self.guid), str(self.length), " ".join(self.text_a), " ".join(self.label)]
        return sep.join(show_list)


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, input_len, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.input_len = input_len

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class Tag(object):
    """
    NER MRC tag
    """

    def __init__(self, term, tag, begin, end):
        self.term = term
        self.tag = tag
        self.begin = begin
        self.end = end

    def to_tuple(self):
        return tuple([self.term, self.begin, self.end])

    def __str__(self):
        return str({key: value for key, value in self.__dict__.items()})

    def __repr__(self):
        return str({key: value for key, value in self.__dict__.items()})
