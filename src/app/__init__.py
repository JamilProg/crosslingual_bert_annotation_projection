from silver_standard_factory import (
    translate_all_files,
    post_process_tsv,
    replace_ann_labels_given_df,
)
from utils_typings.annotations import (
    EntityAnnotation,
)
from utils_io.file_read import read_document_collection_from_folder
from utils_io.ann_write import write_ann_files_in_folder
from os.path import join
import pandas as pd


def translate_ann_files() -> None:
    """Step 1: produce automatic annotation files"""
    data_path_es = join("data", "input", "DISTEMIST-ES")
    data_path_fr = join("data", "input", "DISTEMIST-FR")
    # data_path_es = join("data", "input", "CANTEMIST-ES")
    # data_path_fr = join("data", "input", "CANTEMIST-FR")
    output_path = join("data", "output", "ann-fr")
    csv_output = join("data", "output", "ann-fr", "ann_to_fix.csv")
    translate_all_files(
        src_lang_txtann_folder=data_path_es,
        dst_lang_txt_folder=data_path_fr,
        dst_lang_ann_folder_output=output_path,
        csv_output=csv_output,
        srclang="es",
        tgtlang="fr",
    )


def postprocess_csv() -> None:
    """Step 2: find bad annotation projections in the csv (by flagging them),
    depending on the initial analysis of bad annotations.
    Warning: the code is very resource and language-dependant."""
    post_process_tsv(
        input_tsv_path=join("data", "input", "ann_to_fix.tsv"),
        output_tsv_path=join("data", "input", "ann_to_fix_v1.tsv"),
    )
    # In your side, you must open as Excel file to manually fill the 'FLAG' column to identify bad projections


def replace_ann_labels_given_xlsx(doc_col_path="", excel_path=""):
    """Step 3: after having (manually) found your bad annotations, this code consists in
    applying the flags in your annotation labels. This way, you can manually refine the annotations
    by going through the flags present in the annotated dataset."""
    if doc_col_path == "":
        doc_col_path = join("data", "input", "DISTEMIST-FR-TXT-AND-ANN")
    if excel_path == "":
        excel_path = join("data", "input", "ann_to_fix_final.xlsx")
    df = pd.read_excel(excel_path)
    doc_col = read_document_collection_from_folder(
        doc_col_path,
        no_duplicates_ann=False,
        sort_ann=False,
        renumerotize_ann=False,
        grammar_check_ann=True,
    )
    print(doc_col)
    doc_col.replace_annotation_labels("expression_complete", "pathologie")
    doc_col.replace_annotation_labels("expression_a_corriger", "pathologie")
    replace_ann_labels_given_df(doc_col, df)
    doc_col.stats_annotation_types(verbose=True)
    doc_col.stats_labels_given_annot_type(
        descendant_type=EntityAnnotation, verbose=True
    )
    write_ann_files_in_folder(
        doc_collection=doc_col,
        path=join("data", "distemist", "DISTEMIST-FR-TOANNOTATE"),
    )
