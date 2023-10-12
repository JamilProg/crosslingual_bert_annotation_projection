# Clinical French Annotated Resource Produced through Crosslingual BERT-Based Annotation Projection

## Description

This is the source code to produce the translated versions of an annotated dataset in another language.

For this, you need to have, as an input:

- The .txt files, and their associated .ann annotation file (Brat standoff format: https://brat.nlplab.org/standoff.html) in a SOURCE language.
- The corresponding .txt files in a DESTINATION language.

You will get as an output:

- The generated .ann files in the DESTINATION language.

## Goal of the repository

- Provides tools to produce translated data (a corpora with txt and eventually ann files) from one language to another.

### Content

- Step 1: `translate_ann_files()` consists in automatically producing annotation files. You can do this by running `pdm run anntranslate`.
NOTE: `translate_ann_files()` also gives you a csv file which reports all annotation projections that has been performed.

- Step 2: `postprocess_csv()` consists in identifying bad annotation projections in the csv file (by flagging them), depending on your initial analysis of bad annotations.
Warning: the code is very resource and language-dependant, you must change it depending on your dataset.

- Step 3: `replace_ann_labels_given_xlsx()` consists in applying the flags in your annotation labels. This way, you can manually refine the annotations on Brat by going through the flags present in the annotated dataset.

## Quick Start

- Put your folder containing .txt and .ann files (source language) in the data/input folder.

- Put your folder containing .txt (destination language) in the data/input folder.

- Update the paths (data_path_es, data_path_fr) in src/app/\_\_init\_\_.py:translate_ann_files.

- Run `pdm run anntranslate`. Warning: the running time can be long, depending on the size of your dataset.

## Credits

This repository uses two additional packages whose code comes from:

- **bert-word-alignment**: https://github.com/andreabac3/Word_Alignment_BERT

- **labse-sentence-alignment**: https://github.com/bfsujason/bertalign
