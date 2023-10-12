import numpy as np
from .corelib import (
    find_top_k_sents,
    get_alignment_types,
    find_first_search_path,
    first_pass_align,
    first_back_track,
    find_second_search_path,
    second_pass_align,
    second_back_track,
)
from .utils import clean_text, detect_lang, split_sents, LANG
from .encoder import Encoder


class Bertalign:
    def __init__(
        self,
        src,
        tgt,
        max_align=5,
        top_k=3,
        win=5,
        skip=-0.1,
        margin=True,
        len_penalty=True,
        is_split=False,
        src_lang="",
        tgt_lang="",
    ):
        self.max_align = max_align
        self.top_k = top_k
        self.win = win
        self.skip = skip
        self.margin = margin
        self.len_penalty = len_penalty

        src = clean_text(src)
        tgt = clean_text(tgt)
        if src_lang == "":
            src_lang = detect_lang(src)
        if tgt_lang == "":
            tgt_lang = detect_lang(tgt)

        if is_split:
            src_sents = src.splitlines()
            tgt_sents = tgt.splitlines()
        else:
            src_sents = split_sents(src, src_lang)
            tgt_sents = split_sents(tgt, tgt_lang)

        src_num = len(src_sents)
        tgt_num = len(tgt_sents)

        src_lang = LANG.ISO[src_lang]
        tgt_lang = LANG.ISO[tgt_lang]

        print(f"Source language: {src_lang}, Number of sentences: {src_num}")
        print(f"Target language: {tgt_lang}, Number of sentences: {tgt_num}")

        model_name = "LaBSE"
        model = Encoder(model_name)
        print(f"Embedding source and target text using {model.model_name} ...")
        src_vecs, src_lens = model.transform(src_sents, max_align - 1)
        tgt_vecs, tgt_lens = model.transform(tgt_sents, max_align - 1)

        char_ratio = np.sum(src_lens[0,]) / np.sum(tgt_lens[0,])

        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.src_sents = src_sents
        self.tgt_sents = tgt_sents
        self.src_num = src_num
        self.tgt_num = tgt_num
        self.src_lens = src_lens
        self.tgt_lens = tgt_lens
        self.char_ratio = char_ratio
        self.src_vecs = src_vecs
        self.tgt_vecs = tgt_vecs

    def align_sents(self):
        print("Performing first-step alignment ...")
        D, I = find_top_k_sents(self.src_vecs[0, :], self.tgt_vecs[0, :], k=self.top_k)
        first_alignment_types = get_alignment_types(2)  # 0-1, 1-0, 1-1
        first_w, first_path = find_first_search_path(self.src_num, self.tgt_num)
        first_pointers = first_pass_align(
            self.src_num, self.tgt_num, first_w, first_path, first_alignment_types, D, I
        )
        first_alignment = first_back_track(
            self.src_num,
            self.tgt_num,
            first_pointers,
            first_path,
            first_alignment_types,
        )

        print("Performing second-step alignment ...")
        second_alignment_types = get_alignment_types(self.max_align)
        second_w, second_path = find_second_search_path(
            first_alignment, self.win, self.src_num, self.tgt_num
        )
        second_pointers = second_pass_align(
            self.src_vecs,
            self.tgt_vecs,
            self.src_lens,
            self.tgt_lens,
            second_w,
            second_path,
            second_alignment_types,
            self.char_ratio,
            self.skip,
            margin=self.margin,
            len_penalty=self.len_penalty,
        )
        second_alignment = second_back_track(
            self.src_num,
            self.tgt_num,
            second_pointers,
            second_path,
            second_alignment_types,
        )

        print(
            f"Finished! Successfully aligning {self.src_num} {self.src_lang} sentences to {self.tgt_num} {self.tgt_lang} sentences\n"
        )
        self.result = second_alignment
