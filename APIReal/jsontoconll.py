# -*- coding: utf-8 -*-
import sys
import re
import os
import json
from io import StringIO

sys.path.append(os.path.join(os.path.dirname(__file__), 'mylib'))
sys.path.append('.')
# import mytokenizer

api_list = ['absl', 'apiclient', 'appdirs', 'bcrypt', 'boto', \
            'click', 'colorama', 'cryptography', 'docker', 'flask', \
            'futures', 'google auth httplib2', 'google protobuf', 'ipaddress', 'mock', \
            'more itertools', 'packaging', 'pandas', 'pluggy', 'prometheus client', \
            'prompt toolkit', 'requests', 'websocket']

multi_token_api = set(['google', 'auth', 'httplib2',
                    'protobuf', 'more', 'itertools', 
                    'prometheus', 'client', 'prompt',
                    'toolkit'])

def text_to_conll(f):
    """Convert plain text into CoNLL format."""
    lines = []
    content = list(f)
    for s in content:
        nonspace_token_seen = False
        tweet = s[0]

        if s[1].lower().strip() == "lib":
            label = "B-API"
        else:
            label = "O"
        p = re.sub(r"[^a-zA-Z0-9]"," ",tweet).strip()

        indiv = p.split(" ")
        indiv = list(filter(None, indiv))

        if label == "O":
            for i, t in enumerate(indiv):
                if not t.isspace():
                    lines.append([t, 'O'])
        else:
            count_multi_tokens = 0
            found_api = False
            for i, t in enumerate(indiv):
                if not t.isspace():
                    if t.lower() in api_list:
                        lines.append([t, 'B-API'])
                        found_api = True
                    else:
                        lines.append([t, 'O'])
                        for multi_token in multi_token_api:
                            if multi_token in t.lower():
                                count_multi_tokens += 1

            if not found_api and count_multi_tokens >= 2:
                for i, t in enumerate(indiv):
                    if not t.isspace():
                        for multi_token in multi_token_api:
                            if multi_token in t.lower():
                                lines.append([t, 'B-API'])
                                found_api = True
                        else:
                            lines.append([t, 'O'])
            
            # single-token, but it is substring
            if not found_api:
                for i, t in enumerate(indiv):
                    if not t.isspace():
                        for api in api_list:
                            if api in t.lower():
                                lines.append([t, 'B-API'])
                                found_api = True                    
                            else:
                                lines.append([t, 'O'])
        nonspace_token_seen = True

        # sentences delimited by empty lines
        lines.append(["[]", 'O'])
        if nonspace_token_seen:
            lines.append([])

    lines = [[l[0], l[1]] if l else l for l in lines]
    return StringIO('\n'.join(('\t'.join(l) for l in lines)))


def handle_cross():
    base_folder = '../data/cross'
    output_folder = './api_recog/cross'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    folders = os.listdir(base_folder)
    fol_counter = 1

    for fol in folders:
        fol_path = base_folder + "/" + str(fol_counter) + "/training.json"

        print("Fold: " + str(fol_counter))

        with open(fol_path) as json_file:
            data = json.load(json_file)
            lines = text_to_conll(data)
            name = "training" + str(fol_counter) + ".conll"

            name_path = output_folder + "/" + name
            with open(name_path, 'wt') as of:
                of.write(''.join(lines))
                of.write('\n')
        fol_counter = fol_counter + 1
    
    fol_counter = 1
    for fol in folders:
        fol_path = base_folder + "/" + str(fol_counter) + "/test.json"

        print("Fold: " + str(fol_counter))

        with open(fol_path) as json_file:
            data = json.load(json_file)
            lines = text_to_conll(data)
            name = "test" + str(fol_counter) + ".conll"

            name_path = output_folder + "/" + name
            with open(name_path, 'wt') as of:
                of.write(''.join(lines))
                of.write('\n')
        fol_counter = fol_counter + 1


def handle_stratified():
    base_folder = '../data/stratified'
    output_folder = './api_recog/stratified'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    folders = os.listdir(base_folder)
    fol_counter = 1

    for fol in folders:
        fol_path = base_folder + "/" + str(fol_counter) + "/training.json"

        print("Fold: " + str(fol_counter))

        with open(fol_path) as json_file:
            data = json.load(json_file)
            lines = text_to_conll(data)
            name = "training" + str(fol_counter) + ".conll"

            name_path = output_folder + "/" + name
            with open(name_path, 'wt') as of:
                of.write(''.join(lines))
                of.write('\n')
        fol_counter = fol_counter + 1
    
    fol_counter = 1
    for fol in folders:
        fol_path = base_folder + "/" + str(fol_counter) + "/test.json"

        print("Fold: " + str(fol_counter))

        with open(fol_path) as json_file:
            data = json.load(json_file)
            lines = text_to_conll(data)
            name = "test" + str(fol_counter) + ".conll"

            name_path = output_folder + "/" + name
            with open(name_path, 'wt') as of:
                of.write(''.join(lines))
                of.write('\n')
        fol_counter = fol_counter + 1


def handle_within():
    lib_names = ['boto', 'docker', 'flask', 'mock', 'pandas']

    for lib in lib_names:
        base_folder = '../data/within/{}'.format(lib)
        output_folder = './api_recog/within/{}'.format(lib)

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        folders = os.listdir(base_folder)
        fol_counter = 1

        for fol in folders:
            fol_path = base_folder + "/" + str(fol_counter) + "/training.json"

            print("Fold: " + str(fol_counter))

            with open(fol_path) as json_file:
                data = json.load(json_file)
                lines = text_to_conll(data)
                name = "training" + str(fol_counter) + ".conll"

                name_path = output_folder + "/" + name
                with open(name_path, 'wt') as of:
                    of.write(''.join(lines))
                    of.write('\n')
            fol_counter = fol_counter + 1
        
        fol_counter = 1
        for fol in folders:
            fol_path = base_folder + "/" + str(fol_counter) + "/test.json"

            print("Fold: " + str(fol_counter))

            with open(fol_path) as json_file:
                data = json.load(json_file)
                lines = text_to_conll(data)
                name = "test" + str(fol_counter) + ".conll"

                name_path = output_folder + "/" + name
                with open(name_path, 'wt') as of:
                    of.write(''.join(lines))
                    of.write('\n')
            fol_counter = fol_counter + 1


if __name__ == '__main__':
    handle_stratified()
    handle_cross()
    handle_within()