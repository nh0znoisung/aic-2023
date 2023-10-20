# pip install googletrans==4.0.0rc1

from googletrans import Translator
import os
import glob
import shutil
import argparse

translate = Translator()
parser = argparse.ArgumentParser()
parser.add_argument('--org', help='Queries folder. Renamed if folder_name = queries/', required=True, type=str)
parser.add_argument('--dest', default='queries', help='Queries after translation.')



def read_content(path: str):
    content = None
    with open(path, "r") as file:
        content = file.read()
    print(f'>> Read from path {path} successfully!!!')
    print('Content: ', content)
    return content

def write_content(path: str, content: str):
    with open(path, 'w') as file:
        file.write(content)
    print(f'>> Write to path {path} successfully!!!')

def translate_one(text: str):
    result = translate.translate(text, src='vi', dest='en')
    print('>> From: ', result.origin)
    print('>> To: ', result.text)
    print('---')

    return result.text

def translate_list(list_text: list):
    translations = translate.translate(list_text, src='vi', dest='en')

    for translation in translations:
        print(translation.origin, ' -> ', translation.text)

    result = [translation.text for translation in translations]
    return result

def change_path(file: str, origin_query_folder: str, translate_query_folder: str):
    os.makedirs(translate_query_folder, exist_ok=True)

    new_file = file.replace(origin_query_folder, translate_query_folder)
    return new_file

def clear_folder(folder_path: str):
    shutil.rmtree(folder_path)

def get_opts():
    args = parser.parse_args()

    origin_query_folder = args.org
    translate_query_folder = args.dest
    print('Origin Query folder: ', origin_query_folder)
    print('Translated Query folder: ', translate_query_folder)

    return origin_query_folder, translate_query_folder

def run():
    origin_query_folder, translate_query_folder = get_opts()

    clear_folder(origin_query_folder)
    list_file = glob.glob(os.path.join(origin_query_folder, '*.txt')) # list_path: 'dir/file.txt'
    print(list_file)
    list_content = [(change_path(file, origin_query_folder, translate_query_folder), translate_one(read_content(file))) for file in list_file]
    print(list_content)
    for path, content in list_content:
        write_content(path, content)
    print(f'--- Write to folder {translate_query_folder} done !!!')


# origin_query_folder = 'queries-p1'
# translate_query_folder = 'queries'
# origin_query_folder, translate_query_folder

run()

