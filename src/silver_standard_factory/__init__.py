from os.path import join, splitext, split
from glob import iglob
from tqdm import tqdm
import string
import math
import pandas as pd
from utils_typings.annotations import (
    AnnotationCollection,
    EntityAnnotation,
    NoteAnnotation,
    Fragment,
    DocumentCollection,
)
from utils_io.file_read import read_and_load_ann_file
from utils_io.ann_write import write_ann_file
from labse_sentence_alignment.aligner import Bertalign
from bert_word_alignment.WordAlignment import WordAlignment


def contains_letter(mystring):
    return any(c.isalpha() for c in mystring)


def get_sentences_indices(full_txt, sentences) -> list[tuple[int, int]]:
    """Given the full txt and the list of sentences of size K, give a list of size K giving the start and end indices for each sentence"""
    sents_inds = []
    idx_start_count = 0
    for s in sentences:
        start = full_txt.find(s, idx_start_count)
        end = start + len(s)
        sents_inds.append((start, end))
        start = end
    return sents_inds


def find_word_indices(sentence, sentence_start, word_list) -> list[tuple[int, int]]:
    """Same as get_sentences_indices, but for words in sentence instead of sentences in full text"""
    indices = []
    buffer_start = 0
    for word in word_list:
        local_start = sentence.find(word, buffer_start)
        if local_start == -1:
            # should never happen
            if word == "[PAD]" or word == "[SEP]" or word == "[CLS]":
                print("It should not happen though")
                indices.append((0, 0))
            else:
                start = sentence.find(word) + sentence_start
                end = start + len(word)
                indices.append((start, end))
        start = local_start + sentence_start
        end = start + len(word)
        buffer_start = local_start + len(word)
        indices.append((start, end))
    return indices


def count_trailing_punctuations(txt) -> int:
    count = 0
    for char in reversed(txt):
        if char in string.punctuation:
            count += 1
        else:
            break
    return count


def count_heading_punctuations(txt) -> int:
    count = 0
    for char in txt:
        if char in string.punctuation:
            count += 1
        else:
            break
    return count


def remove_trailing_punctations(txt):
    count = 0
    for char in reversed(txt):
        if char in string.punctuation:
            count += 1
        else:
            break
    txt = txt.rstrip(string.punctuation)
    return txt, count


def remove_heading_punctations(txt):
    count = 0
    for char in txt:
        if char in string.punctuation:
            count += 1
        else:
            break
    txt = txt.lstrip(string.punctuation)
    return txt, count


def remove_trailing_spaces(txt):
    count = 0
    for char in reversed(txt):
        if char == " ":
            count += 1
        else:
            break
    txt = txt.rstrip()
    return txt, count


def remove_heading_spaces(txt):
    count = 0
    for char in txt:
        if char == " ":
            count += 1
        else:
            break
    txt = txt.lstrip()
    return txt, count


def clean_puncts_annotation(es_content, dstlang_content, dstlang_start, dstlang_end):
    new_content, new_dstlang_start, new_dstlang_end = (
        dstlang_content,
        dstlang_start,
        dstlang_end,
    )
    # check if es_content does NOT have trailing puncts, then remove trailing puncts in dst translation, and update end
    if count_trailing_punctuations(es_content) == 0:
        new_content, removed_count = remove_trailing_punctations(new_content)
        new_dstlang_end -= removed_count
    # do same with heading puncts and start index
    if count_heading_punctuations(es_content) == 0:
        new_content, removed_count = remove_heading_punctations(new_content)
        new_dstlang_start += removed_count
    # if any remaining spaces...
    new_content, removed_count = remove_trailing_spaces(new_content)
    new_dstlang_end -= removed_count
    new_content, removed_count = remove_heading_spaces(new_content)
    new_dstlang_start += removed_count
    # if it deleted everything from the string found, cancel the clean
    if len(new_content) == 0:
        new_content, new_dstlang_start, new_dstlang_end = (
            dstlang_content,
            dstlang_start,
            dstlang_end,
        )
    return new_content, new_dstlang_start, new_dstlang_end


def find_max_overlap(intervals, my_interval):
    """Given a list of intervals, and interval my_interval, outputs the bigger overlapping interval"""
    max_overlap = 0
    # best_interval = None
    perfect = False
    winning_indice = 0

    for i, interval in enumerate(intervals):
        start, end = interval
        overlap = min(end, my_interval[1]) - max(start, my_interval[0])

        if overlap > max_overlap:
            winning_indice = i
            max_overlap = overlap
            # best_interval = interval
            if start == my_interval[0] and end == my_interval[1]:
                perfect = True

    return winning_indice, perfect


def find_overlapping_intervals(intervals, my_interval):
    """Given a list of intervals, and interval my_interval, outputs all overlapping intervals"""
    # overlapping_intervals = []
    winning_indices = []

    for i, interval in enumerate(intervals):
        start, end = interval
        overlap = min(end, my_interval[1]) - max(start, my_interval[0])

        if overlap > 0:
            winning_indices.append(i)
            # overlapping_intervals.append(interval)

    return winning_indices


def translate_ann_file(
    src_txt, src_ann, dst_txt, df, word_aligner, filename, srclang, tgtlang
) -> AnnotationCollection:
    print("translation for file:", filename)
    # sentence aligner
    sent_aligner = Bertalign(
        src=src_txt, tgt=dst_txt, src_lang=srclang, tgt_lang=tgtlang
    )
    # Align sentences
    sent_aligner.align_sents()

    # List of sentences in both sides
    src_sents, tgt_sents = sent_aligner.src_sents, sent_aligner.tgt_sents

    # Results as dict of entities
    sent_aligner_res: dict[int, list[int]] = dict()
    for id_srclang, id_dstlang in sent_aligner.result:
        for id_srclang_el in id_srclang:
            if id_srclang_el in list(sent_aligner_res.keys()):
                sent_aligner_res[id_srclang_el].extend(id_dstlang)
            else:
                sent_aligner_res[id_srclang_el] = id_dstlang
    # remove dups and sort lists
    for k in sent_aligner_res.keys():
        undup = list(set(sent_aligner_res[k]))
        undup.sort()
        sent_aligner_res[k] = undup

    # get ind_start/ind_end of each sentence
    src_sents_inds = get_sentences_indices(src_txt, src_sents)
    tgt_sents_inds = get_sentences_indices(dst_txt, tgt_sents)

    res = AnnotationCollection(annotations=[])
    for ents in src_ann.get_annotations():
        if isinstance(ents, EntityAnnotation):
            # find the corresponding srclang sentence
            for index_sentence_srclang, sentids in enumerate(src_sents_inds):
                if ents.fragments[0].start > sentids[1]:
                    # not reached yet
                    continue
                # sentence found
                elif (
                    ents.fragments[0].start >= sentids[0]
                    and ents.fragments[0].end <= sentids[1]
                ):
                    try:  # in case translation of one sentence missed by MT
                        # get dst lang sentence
                        indices_sentence_dstlang = sent_aligner_res[
                            index_sentence_srclang
                        ]
                        sentence_start_dstlang, _ = tgt_sents_inds[
                            indices_sentence_dstlang[0]
                        ]
                    except Exception:
                        breakpoint()
                        print(
                            "Careful: a issue has been found in the sentence alignment... Mistranslated sentence?",
                            "\nFilename =",
                            filename,
                        )
                        exit()
                    # run word aligner and decode
                    tgt_sentence = tgt_sents[indices_sentence_dstlang[0]]
                    if len(indices_sentence_dstlang) > 1:
                        for k in range(1, len(indices_sentence_dstlang)):
                            tgt_sentence += " "
                            tgt_sentence += tgt_sents[indices_sentence_dstlang[k]]

                    indice_align, decoded_srclang_dstlang = word_aligner.get_alignment(
                        src_sents[index_sentence_srclang].split(),
                        tgt_sentence.split(),
                        calculate_decode=True,
                    )
                    srclangwords, dstlangwords = [], []
                    for srclang_w, dstlang_w in decoded_srclang_dstlang:
                        srclangwords.append(srclang_w)
                        dstlangwords.append(dstlang_w)
                    indices_spain_words = find_word_indices(
                        src_sents[index_sentence_srclang], sentids[0], srclangwords
                    )
                    if " " in ents.content:
                        isperfect = None
                        # annotation is multiword (M-to-1 OR M-to-M)
                        mode = "mtm"
                        # print("Multiword", ents.content)
                        winning_indices = find_overlapping_intervals(
                            indices_spain_words,
                            (ents.fragments[0].start, ents.fragments[0].end),
                        )
                        # chopper les indices des mots dans le texte
                        indices_words_dstlang = find_word_indices(
                            tgt_sentence,
                            sentence_start_dstlang,
                            tgt_sentence.split(),
                        )
                        # remove from winning indices those that are out of range (PAD, CLS tokens...)
                        winning_indices = [
                            wi
                            for wi in winning_indices
                            if indice_align[wi] < len(tgt_sentence.split())
                        ]
                        dstlang_start, dstlang_end = (
                            indices_words_dstlang[indice_align[winning_indices[0]]][0],
                            indices_words_dstlang[indice_align[winning_indices[-1]]][1],
                        )
                        dstlang_content = dst_txt[dstlang_start:dstlang_end]
                        if dstlang_end <= dstlang_start:
                            # should be manually fixed
                            tmp = dstlang_end
                            dstlang_end = dstlang_start
                            dstlang_start = tmp
                            isperfect = False
                        dstlang_content = dst_txt[dstlang_start:dstlang_end]
                        if dstlang_content in dstlangwords:
                            mode = "mt1"
                        (
                            dstlang_content,
                            dstlang_start,
                            dstlang_end,
                        ) = clean_puncts_annotation(
                            ents.content, dstlang_content, dstlang_start, dstlang_end
                        )
                        assert dstlang_content == dst_txt[dstlang_start:dstlang_end]
                        res.add_annotation(
                            EntityAnnotation(
                                ents.id,
                                "expression_complete",
                                [Fragment(dstlang_start, dstlang_end)],
                                dstlang_content,
                            )
                        )
                        df = pd.concat(
                            [
                                df,
                                pd.DataFrame(
                                    [
                                        {
                                            "filename": filename,
                                            "matchingstyle": mode,
                                            "srclang_content": ents.content,
                                            "srclang_start": ents.fragments[0].start,
                                            "srclang_end": ents.fragments[0].end,
                                            "dstlang_content": dstlang_content,
                                            "dstlang_start": dstlang_start,
                                            "dstlang_end": dstlang_end,
                                            "isperfect": isperfect,
                                        }
                                    ]
                                ),
                            ],
                            ignore_index=True,
                        )
                    else:
                        # annotation is monoword (1-to-1 OR 1-to-M)
                        # check if it could be 1-to-many
                        src_sent_splitted = src_sents[index_sentence_srclang].split()
                        (
                            ind_align_dst_src,
                            _decoded_dst_src,
                        ) = word_aligner.get_alignment(
                            tgt_sentence.split(),
                            src_sent_splitted,
                            calculate_decode=True,
                        )
                        found_dstlang_words = []
                        magical_indice = 105248
                        for ind_dstlang, (_dstlang_word, ind_srclang) in enumerate(
                            zip(tgt_sentence.split(), ind_align_dst_src)
                        ):
                            if ind_srclang >= len(src_sent_splitted):
                                continue  # the element is a special token
                            if src_sent_splitted[ind_srclang] == ents.content:
                                if magical_indice == 105248:
                                    magical_indice = ind_srclang
                                if (
                                    magical_indice == ind_srclang
                                ):  # actually one to many
                                    found_dstlang_words.append(ind_dstlang)
                        if len(found_dstlang_words) > 1:
                            # it is 1-to-many
                            dstlangwords = tgt_sentence.split()
                            winning_indices_words = found_dstlang_words
                            indices_words_dstlang = find_word_indices(
                                tgt_sentence,
                                sentence_start_dstlang,
                                dstlangwords,
                            )
                            # récupérer l'indice start du premier mot et l'indice end du dernier mot
                            dstlang_start, dstlang_end = (
                                indices_words_dstlang[winning_indices_words[0]][0],
                                indices_words_dstlang[winning_indices_words[-1]][1],
                            )
                            dstlang_content = dst_txt[dstlang_start:dstlang_end]
                            # rajouter dans CSV le one to many: filename, 1tm, srclang_content, idx,es, dstlang_content, idxfr, isperfect (#NA)
                            # print("onetomany added", ents.content, dstlang_content)
                            (
                                dstlang_content,
                                dstlang_start,
                                dstlang_end,
                            ) = clean_puncts_annotation(
                                ents.content,
                                dstlang_content,
                                dstlang_start,
                                dstlang_end,
                            )
                            assert dstlang_content == dst_txt[dstlang_start:dstlang_end]
                            res.add_annotation(
                                EntityAnnotation(
                                    ents.id,
                                    "expression_complete",
                                    [Fragment(dstlang_start, dstlang_end)],
                                    dstlang_content,
                                )
                            )
                            if dstlang_end <= dstlang_start:
                                # should NOT happen
                                print(
                                    "dstlang_end is bigger or equal than dstlang_start... please check"
                                )
                                # breakpoint()
                                exit()
                            df = pd.concat(
                                [
                                    df,
                                    pd.DataFrame(
                                        [
                                            {
                                                "filename": filename,
                                                "matchingstyle": "1tm",
                                                "srclang_content": ents.content,
                                                "srclang_start": ents.fragments[
                                                    0
                                                ].start,
                                                "srclang_end": ents.fragments[0].end,
                                                "dstlang_content": dstlang_content,
                                                "dstlang_start": dstlang_start,
                                                "dstlang_end": dstlang_end,
                                                "isperfect": None,
                                            }
                                        ]
                                    ),
                                ],
                                ignore_index=True,
                            )
                        else:
                            # 1to1
                            winning_indice, isperfect = find_max_overlap(
                                indices_spain_words,
                                (ents.fragments[0].start, ents.fragments[0].end),
                            )
                            dstlang_content = dstlangwords[winning_indice]
                            dstlang_start = dst_txt.find(
                                dstlang_content, sentence_start_dstlang
                            )
                            dstlang_end = dstlang_start + len(dstlang_content)
                            (
                                dstlang_content,
                                dstlang_start,
                                dstlang_end,
                            ) = clean_puncts_annotation(
                                ents.content,
                                dstlang_content,
                                dstlang_start,
                                dstlang_end,
                            )
                            try:
                                mstyle = "1t1"
                                assert (
                                    dstlang_content
                                    == dst_txt[dstlang_start:dstlang_end]
                                )
                            except:
                                # might have found a pad or something like that
                                mstyle = "1t1tofix"
                                dstlang_content = dst_txt[dstlang_start:dstlang_end]
                            # rajouter dans CSV le one to one: filename, 1t1, srclang_content, idxes, dstlang_content, idxfr, isperfect
                            # print("onetoone added", ents.content)
                            res.add_annotation(
                                EntityAnnotation(
                                    ents.id,
                                    "expression_complete",
                                    [Fragment(dstlang_start, dstlang_end)],
                                    dstlang_content,
                                )
                            )
                            if dstlang_end <= dstlang_start:
                                # should NOT happen
                                print(
                                    "dstlang_end is bigger or equal than dstlang_start... please check"
                                )
                                # breakpoint()
                                exit()
                            df = pd.concat(
                                [
                                    df,
                                    pd.DataFrame(
                                        [
                                            {
                                                "filename": filename,
                                                "matchingstyle": mstyle,
                                                "srclang_content": ents.content,
                                                "srclang_start": ents.fragments[
                                                    0
                                                ].start,
                                                "srclang_end": ents.fragments[0].end,
                                                "dstlang_content": dstlang_content,
                                                "dstlang_start": dstlang_start,
                                                "dstlang_end": dstlang_end,
                                                "isperfect": isperfect,
                                            }
                                        ]
                                    ),
                                ],
                                ignore_index=True,
                            )
                    break
                else:
                    # to be manually fixed
                    dstlang_start, dstlang_end = (
                        ents.fragments[0].start,
                        ents.fragments[0].end,
                    )
                    if dstlang_start >= len(dst_txt):
                        dstlang_start = len(dst_txt) - 2
                        dstlang_end = len(dst_txt) - 1
                    elif dstlang_end >= len(dst_txt):
                        dstlang_end = len(dst_txt) - 1
                    dstlang_content = dst_txt[dstlang_start:dstlang_end]
                    print(ents.content, "ouch!")
                    res.add_annotation(
                        EntityAnnotation(
                            ents.id,
                            "expression_a_corriger",
                            [Fragment(dstlang_start, dstlang_end)],
                            dstlang_content,
                        )
                    )
                    df = pd.concat(
                        [
                            df,
                            pd.DataFrame(
                                [
                                    {
                                        "filename": filename,
                                        "matchingstyle": "NOMATCH",
                                        "srclang_content": ents.content,
                                        "srclang_start": ents.fragments[0].start,
                                        "srclang_end": ents.fragments[0].end,
                                        "dstlang_content": dstlang_content,
                                        "dstlang_start": dstlang_start,
                                        "dstlang_end": dstlang_end,
                                        "isperfect": False,
                                    }
                                ]
                            ),
                        ],
                        ignore_index=True,
                    )
                    break
        elif isinstance(ents, NoteAnnotation):
            res.add_annotation(
                NoteAnnotation(ents.id, "AnnotatorNotes", ents.value, ents.component)
            )
    return res, df


def translate_all_files(
    src_lang_txtann_folder,
    dst_lang_txt_folder,
    dst_lang_ann_folder_output,
    csv_output,
    srclang,
    tgtlang,
) -> None:
    # iterate each file txt srclang + ann, each file txt dst and go to translate_ann_file function
    srclang_txt_contents: dict[str, str] = dict()
    dstlang_txt_contents: dict[str, str] = dict()
    srclang_ann_contents: dict[str, AnnotationCollection] = dict()

    print("Read txt src_language")
    # Loop through each file in the folder SRC LANGUAGE TXT
    for filename in tqdm(iglob(src_lang_txtann_folder + "/**/*.txt", recursive=True)):
        if filename.endswith(".txt"):
            # file_path = join(src_lang_txtann_folder, filename)
            # Read the contents of the file
            with open(filename, "r", encoding="utf-8") as file:
                content = file.read()
                _, filebase = split(filename)
                srclang_txt_contents[splitext(filebase)[0]] = content

    print("Read translated txt dst_language")
    # Loop through each file in the folder DST LANGUAGE TXT
    for filename in tqdm(iglob(dst_lang_txt_folder + "/**/*.txt", recursive=True)):
        if filename.endswith(".txt"):
            # file_path = join(dst_lang_txt_folder, filename)
            # Read the contents of the file
            with open(filename, "r", encoding="utf-8") as file:
                content = file.read()
                _, filebase = split(filename)
                dstlang_txt_contents[splitext(filebase)[0]] = content

    print("Read ann srclang")
    # Loop through each file in the folder srclang ANN
    for filename in tqdm(iglob(src_lang_txtann_folder + "/**/*.ann", recursive=True)):
        if filename.endswith(".ann"):
            # file_path = join(src_lang_txtann_folder, filename)
            ann_content: AnnotationCollection = read_and_load_ann_file(
                filename,
                no_duplicates=False,
                sorting=False,
                renumerotize=False,
                grammar_check=False,
            )
            _, filebase = split(filename)
            srclang_ann_contents[splitext(filebase)[0]] = ann_content

    wa = WordAlignment(
        model_name="bert-base-multilingual-cased",
        tokenizer_name="bert-base-multilingual-cased",
        device="cpu",
        fp16=False,
    )
    df = pd.DataFrame(
        columns=[
            "filename",
            "matchingstyle",
            "srclang_content",
            "srclang_start",
            "srclang_end",
            "dstlang_content",
            "dstlang_start",
            "dstlang_end",
            "isperfect",
        ]
    )
    # Loop each dstlang key: get corresponding srclang txt, srclang ann and run translate_ann_file method
    for dstlang_filename, dstlang_txt in sorted(dstlang_txt_contents.items()):
        srclang_txt = srclang_txt_contents[dstlang_filename]
        srclang_ann = srclang_ann_contents[dstlang_filename]
        # translate and get ann_collection
        ann_collection, df = translate_ann_file(
            src_txt=srclang_txt,
            src_ann=srclang_ann,
            dst_txt=dstlang_txt,
            df=df,
            word_aligner=wa,
            filename=dstlang_filename,
            srclang=srclang,
            tgtlang=tgtlang,
        )
        # write the resulting ann file
        write_ann_file(
            ann_collection, join(dst_lang_ann_folder_output, dstlang_filename + ".ann")
        )

    df.to_csv(csv_output, encoding="UTF-8")


def count_parentheses(text):
    open_count = text.count("(")
    close_count = text.count(")")
    return open_count, close_count


def has_punctuation(text) -> bool:
    return any(char in string.punctuation for char in text)


def post_process_tsv(input_tsv_path, output_tsv_path) -> None:
    """This is the CSV postprocessing to help for manual correction
    This code is resource-specific (CANTEMIST is commented).
    This code must be changed depending on:
    - your corpus' annotations (through a first csv analysis)
    - the language of your corpus (e.g. bothstartend_word_redflag contains french coordinating conjunctions and determiners)
    """
    # exact_redflag = [
    #     "pulmonaire",
    #     "pulmonaires",
    #     "cérébrale",
    #     "chirurgical",
    #     "de type",
    #     "droit",
    #     "endodermique",
    #     "excisée",
    #     "gauche",
    #     "hépatiques",
    #     "hépatique",
    #     "histologique",
    #     "holocrânienne",
    #     "lymphadénectomie",
    #     "occupant l'espace",
    #     "palpation",
    #     "pleural",
    #     "probable",
    #     "vascularisation",
    #     "vasculaire",
    #     "tomodensitométrie",
    #     "syndrome",
    #     "Tomodensitométrie",
    # ]
    exact_redflag = [
        "sur",
        "un",
        "une",
        "Une",
        "Un",
        " ",
        "réalisée",
        "réalisées",
        "ni",
        "week-end",
    ]
    contain_orangeflag = ["a été"]
    # startswith_redflag = [
    #     "lymphatique",
    #     "saignement",
    # ]
    startswith_redflag = ["avec", "patient", "patiente", ":", ";", "?"]
    # endswith_redflag = ["à grande", "à grandes", "à petite", "à petites", ".", ","]
    endswith_redflag = [".", ","]

    bothstartend_word_redflag = [
        "mais",
        "ou",
        "et",
        "donc",
        "or",
        "ni",
        "car",
        "de",
        "du",
        "un",
        "une",
        "ainsi",
        "que",
        "étant",
        "des",
        "qu'un",
        "à",
        "le",
        "la",
        "les",
        "dans",
    ]

    # exact_orangeflag = [
    #     "gastro-intestinale",
    #     "lésion",
    #     "lésions",
    #     "maladie",
    #     "nodules",
    #     "rétropéritonéale",
    # ]
    exact_orangeflag = ["observée"]
    startswith_orangeflag = [
        "l'",
        "d'",
    ]

    # Read the TSV file into a pandas DataFrame
    df = pd.read_csv(input_tsv_path, encoding="UTF-8", sep="\t")

    # Add an empty column with header "Flag"
    df["Flag"] = "Empty"

    # We tag R if...
    # ...UNMATCHED annotations
    df.loc[df["matchingstyle"] == "NOMATCH", "Flag"] = "R_ToFix"

    # ...duplicated translated annotations
    duplicates = df.duplicated(
        subset=["filename", "dstlang_start", "dstlang_end"], keep=False
    )
    df.loc[duplicates, "Flag"] = "R_Dup"

    for i, row in df.iterrows():
        if row["Flag"] == "Empty":
            dst_open, dst_close = count_parentheses(str(row["dstlang_content"]))
            src_open, src_close = count_parentheses(str(row["srclang_content"]))
            if len(str(row["dstlang_content"])) < 2:
                df.at[i, "Flag"] = "R_TooShort"
                continue
            elif not contains_letter(str(row["dstlang_content"])):
                df.at[i, "Flag"] = "R_NoLetter"
                continue
            # ...if not the same amount of parentheses in both sides of the text (translated and untranslated)
            if dst_open != dst_close and src_open == src_close:
                df.at[i, "Flag"] = "R_Parenth"
                continue
            words_dstlang = str(row["dstlang_content"]).split()
            # starters, enders and contains
            for badstarts in startswith_redflag:
                if str(row["dstlang_content"]).startswith(badstarts):
                    df.at[i, "Flag"] = "R_BadStart"
                    continue
            for badends in endswith_redflag:
                if str(row["dstlang_content"]).endswith(badends):
                    df.at[i, "Flag"] = "R_BadEnd"
                    continue
            for badcontent in contain_orangeflag:
                if badcontent in str(row["dstlang_content"]):
                    df.at[i, "Flag"] = "MR_BadContain"
                    continue
            if len(words_dstlang) > 1:
                # ...and if starting word is exactly the same as ending word
                if words_dstlang[0] == words_dstlang[-1]:
                    df.at[i, "Flag"] = "R_WordAlign"
                    continue
                elif (
                    words_dstlang[0] in bothstartend_word_redflag
                    or words_dstlang[-1] in bothstartend_word_redflag
                ):
                    # flagged bad words
                    df.at[i, "Flag"] = "R_BadWords"
                    continue
            elif len(words_dstlang) == 1:
                # flagged exactly bad words
                if words_dstlang[0] in exact_redflag:
                    df.at[i, "Flag"] = "R_BadAlign"
                    continue
            # Now, check if Maybe Red (MR)
            # ...if translation contains punctuations while original does not
            if len(str(row["dstlang_content"])) < 4:
                df.at[i, "Flag"] = "MR_TooShort"
                continue
            # starters
            for badstart in startswith_orangeflag:
                if str(row["dstlang_content"]).startswith(badstart):
                    df.at[i, "Flag"] = "MR_BadStart"
                    continue
            # exact words
            if len(words_dstlang) == 1:
                # flagged exactly bad words orange
                if words_dstlang[0] in exact_orangeflag:
                    df.at[i, "Flag"] = "MR_BadAlign"
                    continue
            if has_punctuation(str(row["dstlang_content"])) and not has_punctuation(
                str(row["srclang_content"])
            ):
                df.at[i, "Flag"] = "MR_Punctuation"
                continue
            # ...and if translation contains way more words compared to original
            words_srclang = str(row["srclang_content"]).split()
            if (
                len(words_dstlang) >= len(words_srclang) + 2
                and row["matchingstyle"] == "1tm"
            ):
                df.at[i, "Flag"] = "MR_tooManyWordsTranslation"
            elif (
                len(words_dstlang) + 2 >= len(words_srclang)
                and row["matchingstyle"] == "mt1"
            ):
                df.at[i, "Flag"] = "MR_PotentiallyMissedWords"
            elif (
                abs(len(words_dstlang) - len(words_srclang)) >= 3
                and row["matchingstyle"] == "mtm"
            ):
                df.at[i, "Flag"] = "MR_ComplexManyToMany"

    # And now, YellowFlag for those whose count is only one and not flagged
    df.loc[
        (df["Flag"] == "Empty")
        & (df["dstlang_content"].map(df["dstlang_content"].value_counts()) == 1),
        "Flag",
    ] = "MR_Suspicious"

    # Write the updated DataFrame to a new TSV file
    df.to_csv(output_tsv_path, sep="\t", encoding="UTF-8", index=False)


def is_nan(value):
    """Check if the flag is NaN in the csv - the method is counter-intuitive because
    the NaN detection in pd.DataFrame is tricky (pd.NA, np.nan, and df.isna aren't working)
    """
    try:
        return math.isnan(float(value)) # we convert to a float because the type is originally a string
    except ValueError:
        return False


def replace_ann_labels_given_df(doccol: DocumentCollection, df):
    """The function consists in applying the flags from the csv file to your annotations"""
    # dico saving the filename as key, and the list of tuples (start, end, new_tag)
    mondico = dict()
    print(df.head())
    for _, row in df.iterrows():
        flag = row["Flag"]
        if not is_nan(flag):
            fn = row["filename"]
            start, end = row["dstlang_start"], row["dstlang_end"]
            if fn not in mondico.keys():
                mondico[fn] = [(start, end, flag)]
            else:
                mondico[fn].append((start, end, flag))

    for doc in doccol.documents:
        if doc.filename_without_ext in mondico.keys():
            tuples = mondico[doc.filename_without_ext]
            for t in tuples:
                start, end, tag = t
                for ann in doc.annotation_collections[0].annotations:
                    if isinstance(ann, EntityAnnotation):
                        if ann.label != "pathologie":
                            continue
                        if (
                            ann.fragments[0].start == start
                            and ann.fragments[0].end == end
                        ):
                            if tag:
                                ann.set_label(tag)
                            break
                else:
                    print(
                        f"issue with file {doc.filename_without_ext}, {start}:{end} and flag {tag}."
                    )
