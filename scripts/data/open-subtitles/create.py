import json
import numpy as np
import shutil
from tqdm import tqdm
from datasets import load_dataset
from pathlib import Path

LANGUAGES = [
    "af",
    "ar",
    "bg",
    "bn",
    "br",
    "bs",
    "ca",
    "cs",
    "da",
    "de",
    "el",
    "en",
    "eo",
    "es",
    "et",
    "eu",
    "fa",
    "fi",
    "fr",
    "gl",
    "he",
    "hi",
    "hr",
    "hu",
    "hy",
    "id",
    "is",
    "it",
    "ja",
    "ka",
    "kk",
    "ko",
    "lt",
    "lv",
    "mk",
    "ml",
    "ms",
    "nl",
    "no",
    "pl",
    "pt",
    "ro",
    "ru",
    "si",
    "sk",
    "sl",
    "sq",
    "sr",
    "sv",
    "ta",
    "te",
    "th",
    "tl",
    "tr",
    "uk",
    "ur",
    "vi",
    "zh",
    "pt_br",
    "ze_en",
    "ze_zh",
    "zh_cn",
    "zh_tw"
]

bottom500 = ['ka-ml', 'br-sr', 'bg-br', 'kk-lv', 'br-sk', 'br-fi', 'eu-ze_zh', 'kk-nl', 'kk-vi', 'ja-kk', 'br-sv', 'kk-zh_cn', 'kk-ms', 'br-et', 'br-hu', 'eo-kk', 'br-tr', 'ko-tl', 'te-zh_tw', 'br-hr', 'br-nl', 'ka-si', 'br-cs', 'br-is', 'br-ro', 'br-de', 'et-kk', 'fr-hy', 'br-no', 'is-ko', 'br-da', 'br-en', 'eo-lt', 'is-ze_zh', 'eu-ko', 'br-it', 'br-id', 'eu-zh_cn', 'is-ja', 'br-sl', 'br-gl', 'br-pt_br', 'br-es', 'br-pt', 'is-th', 'fa-is', 'br-ca', 'eu-ka', 'is-zh_cn', 'eu-ur', 'id-kk', 'br-sq', 'eu-ja', 'uk-ur', 'is-zh_tw', 'ka-ko', 'eu-zh_tw', 'eu-th', 'eu-is', 'is-tl', 'br-eo', 'eo-ze_zh', 'eu-te', 'ar-kk', 'eo-lv', 'ko-ze_zh', 'ml-ze_zh', 'is-lt', 'br-fr', 'ko-te', 'kk-sl', 'eu-fa', 'eo-ko', 'ka-ze_en', 'eo-eu', 'ta-zh_tw', 'eu-lv', 'ko-lv', 'lt-tl', 'eu-si', 'hy-ru', 'ar-is', 'eu-lt', 'eu-tl', 'eu-uk', 'ka-ze_zh', 'si-ze_zh', 'el-is', 'bn-is', 'ko-ze_en', 'eo-si', 'cs-kk', 'is-uk', 'eu-ze_en', 'ta-ze_zh', 'is-pl', 'is-mk', 'eu-ta', 'ko-lt', 'is-lv', 'fa-ko', 'bn-ko', 'hi-is', 'bn-ze_zh', 'bn-eu', 'bn-ja', 'is-ml', 'eu-ru', 'ko-ta', 'is-vi', 'ja-tl', 'eu-mk', 'eu-he', 'ka-zh_tw', 'ka-zh_cn', 'si-tl', 'is-kk', 'eu-fi', 'fi-ko', 'is-ur', 'ka-th', 'ko-ur', 'eo-ja', 'he-is', 'is-tr', 'ka-ur', 'et-ko', 'eu-vi', 'is-sk', 'gl-is', 'fr-is', 'is-sq', 'hu-is', 'fr-kk', 'eu-sq', 'is-ru', 'ja-ka', 'fi-tl', 'ka-lv', 'fi-is', 'is-si', 'ar-ko', 'ko-sl', 'ar-eu', 'ko-si', 'bg-is', 'eu-hu', 'ko-sv', 'bn-hu', 'kk-ro', 'eu-hi', 'ka-ms', 'ko-th', 'ko-sr', 'ko-mk', 'fi-kk', 'ka-vi', 'eu-ml', 'ko-ml', 'de-ko', 'fa-ze_zh', 'eu-sk', 'is-sl', 'et-is', 'eo-is', 'is-sr', 'is-ze_en', 'kk-pt_br', 'hr-hy', 'kk-pl', 'ja-ta', 'is-ms', 'hi-ze_en', 'is-ro', 'ko-zh_cn', 'el-eu', 'ka-pl', 'ka-sq', 'eu-sl', 'fa-ka', 'ko-no', 'si-ze_en', 'ko-uk', 'ja-ze_zh', 'hu-ko', 'kk-no', 'eu-pl', 'is-pt_br', 'bn-lv', 'tl-zh_cn', 'is-nl', 'he-ko', 'ko-sq', 'ta-th', 'lt-ta', 'da-ko', 'ca-is', 'is-ta', 'bn-fi', 'ja-ml', 'lv-si', 'eu-sv', 'ja-te', 'bn-ur', 'bn-ca', 'bs-ko', 'bs-is', 'eu-sr', 'ko-vi', 'ko-zh_tw', 'et-tl', 'kk-tr', 'eo-vi', 'is-it', 'ja-ko', 'eo-et', 'id-is', 'bn-et', 'bs-eu', 'bn-lt', 'tl-uk', 'bn-zh_tw', 'da-eu', 'el-ko', 'no-tl', 'ko-sk', 'is-pt', 'hu-kk', 'si-zh_tw', 'si-te', 'ka-ru', 'lt-ml', 'af-ja', 'bg-eu', 'eo-th', 'cs-is', 'pl-ze_zh', 'el-kk', 'kk-sv', 'ka-nl', 'ko-pl', 'bg-ko', 'ka-pt_br', 'et-eu', 'tl-zh_tw', 'ka-pt', 'id-ko', 'fi-ze_zh', 'he-kk', 'ka-tr', 'hr-ko', 'ka-sk', 'eu-ms', 'ka-no', 'de-eu', 'af-fa', 'ko-ru', 'hr-is', 'eu-it', 'ko-ro', 'cs-eu', 'hr-kk', 'lv-te', 'ka-lt', 'eu-tr', 'eu-no', 'ml-zh_cn', 'ko-ms', 'tl-vi', 'is-no', 'ja-si', 'kk-sr', 'ko-tr', 'et-ta', 'fr-ko', 'ml-zh_tw', 'af-hi', 'eu-id', 'eo-ms', 'ka-sl', 'sk-tl', 'cs-ko', 'eu-nl', 'fa-ja', 'eo-zh_tw', 'is-sv', 'eo-hu', 'bg-kk', 'ko-pt', 'sr-tl', 'ka-ro', 'hu-hy', 'hu-ta', 'kk-ru', 'lt-te', 'ta-zh_cn', 'ka-sv', 'eo-fi', 'eu-pt_br', 'bn-tl', 'da-is', 'lt-si', 'fa-ta', 'ka-sr', 'bn-uk', 'sv-tl', 'et-te', 'eo-zh_cn', 'ko-pt_br', 'et-ml', 'eo-ml', 'ko-nl', 'es-is', 'fi-ta', 'eu-fr', 'es-ko', 'bn-th', 'it-ko', 'ca-ko', 'th-ze_zh', 'ml-th', 'bn-pl', 'it-kk', 'lv-ta', 'si-zh_cn', 'hu-ml', 'hu-ka', 'eu-ro', 'es-kk', 'bn-zh_cn', 'lv-ze_zh', 'gl-ko', 'sq-ze_zh', 'te-zh_cn', 'fa-zh_tw', 'ja-ze_en', 'fi-si', 'fa-te', 'sl-ze_zh', 'ja-lv', 'af-uk', 'hi-zh_tw', 'si-th', 'bn-el', 'fr-ka', 'ar-ze_zh', 'fa-si', 'eu-hr', 'de-is', 'bs-tl', 'et-ze_zh', 'af-vi', 'ca-ze_zh', 'bn-sk', 'ro-ze_zh', 'hu-te', 'eo-he', 'ml-pl', 'el-ka', 'hi-ze_zh', 'en-ko', 'el-ze_zh', 'te-tr', 'fa-lv', 'si-vi', 'kk-pt', 'bn-fa', 'lv-zh_tw', 'ar-ka', 'bn-vi', 'bn-sl', 'ms-ze_zh', 'ca-ml', 'ru-ze_zh', 'ja-lt', 'lt-ze_zh', 'fi-ml', 'uk-ze_zh', 'en-is', 'et-ka', 'bg-tl', 'et-si', 'fi-te', 'cs-tl', 'eo-sk', 'hu-ze_zh', 'hr-ze_zh', 'bg-ze_zh', 'ja-sl', 'ml-sl', 'vi-ze_zh', 'hu-tl', 'fa-tl', 'da-kk', 'fa-ml', 'te-vi', 'mk-te', 'sl-ta', 'sr-ze_zh', 'lv-ze_en', 'da-tl', 'ml-sk', 'fa-zh_cn', 'gl-ka', 'si-ta', 'ta-tr', 'eo-sl', 'gl-ml', 'ml-vi', 'eo-no', 'th-tl', 'ca-eu', 'eu-pt', 'bn-da', 'no-ze_zh', 'af-zh_cn', 'fa-ze_en', 'id-ka', 'da-ka', 'af-et', 'si-sk', 'ja-ur', 'ja-sq', 'bs-ka', 'fi-ka', 'fa-fi', 'tr-ze_zh', 'sk-ze_zh', 'bn-he', 'et-ja', 'ta-vi', 'eo-uk', 'bs-ze_zh', 'hu-si', 'eo-fa', 'bn-ze_en', 'th-ze_en', 'de-ze_zh', 'si-sv', 'bg-te', 'fr-ze_zh', 'bn-gl', 'bn-mk', 'ml-sv', 'af-bg', 'id-ze_zh', 'ja-sr', 'sq-zh_tw', 'sl-tl', 'el-te', 'es-ka', 'de-kk', 'lv-ml', 'ru-tl', 'it-ka', 'si-sl', 'ml-uk', 'pl-ta', 'de-ka', 'da-ze_zh', 'ar-tl', 'eo-pl', 'en-eu', 'ur-zh_tw', 'el-eo', 'sv-ze_zh', 'hr-ka', 'bn-tr', 'sk-ta', 'bn-ro', 'gl-ze_zh', 'af-eo', 'nl-ze_zh', 'he-tl', 'fa-vi', 'ja-th', 'bs-ta', 'fa-hu', 'eo-tr', 'bn-no', 'bn-cs', 'ja-no', 'cs-ka', 'hi-ko', 'bn-sr', 'bs-ja', 'ar-ja', 'ml-ze_en', 'bg-ta', 'it-ze_zh', 'af-lv', 'fa-lt', 'bn-sv', 'eo-sr', 'si-uk', 'ml-tr', 'ja-sk', 'ja-vi', 'gl-lv', 'gl-zh_tw']

def process_ds(l1, l2, ds, size:int, rng: np.random.Generator):
    n = len(ds['train'])
    size = min(n, size)
    idx = rng.choice(range(n), size, replace=False)
    new_ds = ds['train'].select(idx)['translation']
    new_ds = list(map(lambda x: {'sentence1': x[l1], 'sentence2': x[l2]}, new_ds))

    with open(f'open-subtitles/data/{l1}_{l2}.jsonl','w') as of:
        for line in new_ds:
            json.dump(line,of)
            of.write('\n')

def generate_config():
    s = "---\nconfigs:\n"
    for l1 in tqdm(LANGUAGES):
        for l2 in LANGUAGES:
            file_path = f"open-subtitles/data/{l1}-{l2}.jsonl"
            f = Path(file_path)
            if f.is_file():
                s += f"- config_name: {f'{l1}-{l2}'}\n  data_files: \"{file_path}\"\n"
    s += "---\n"
    with open('open-subtitles/README.md', 'w') as f:
        f.write(s)

def generate_lang_list():
    langs = []
    for l1 in tqdm(LANGUAGES):
        for l2 in LANGUAGES:
            file_path = f"open-subtitles/data/{l1}-{l2}.jsonl"
            f = Path(file_path)
            if f.is_file():
                langs.append(f'{l1}-{l2}')
    print(langs)

def generate_open_subtitles_250():
    s = "---\nconfigs:\n"
    langs = []
    for x in tqdm(bottom500[:250]):
        file_path = f"data/{x}.jsonl" 
        src = f"open-subtitles/{file_path}"
        dest = f"open-subtitles-250/{file_path}"

        if Path(src).is_file():
            with open(src, 'r') as file:
                lines = [next(file) for _ in range(512)] 

            with open(dest, 'w') as out_file:
                out_file.writelines(lines)

            s += f"- config_name: {f'{x}'}\n  data_files: \"{file_path}\"\n"
            langs.append(x)
    s += "---\n"
    print(langs)
    with open('open-subtitles-250/README.md', 'w') as f:
        f.write(s)

def stats():
    sum = 0
    n = 0
    for x in tqdm(bottom500[:250]):
        file_path = f"data/{x}.jsonl" 
        src = f"open-subtitles-250/{file_path}"
        if Path(src).is_file():

            with open(src, 'r') as fin:
                d = list(fin)
                for r in d:
                    r = json.loads(r)
                    sum += len(r['sentence1'])
                    sum += len(r['sentence2'])
                    n += 2
    print(sum / n)


def main():
    rng = np.random.default_rng(28042000)
    for l1 in LANGUAGES[8:]:
        for l2 in LANGUAGES:
            f = Path(f"data/{l1}_{l2}.jsonl")
            if f.is_file():
                print(f"{l1}_{l2} already exists, skipping")
                continue
            try:
                d = load_dataset("open_subtitles", lang1=l1, lang2=l2)
                process_ds(l1, l2, d, 1000, rng)
                print(f"{l1}_{l2} created")
            except FileNotFoundError:
                print(f"File not found for pair {l1}_{l2}")
            except Exception as e:
                print(f"Error {e} for pair {l1}_{l2}")

if __name__ == '__main__':
    stats()
