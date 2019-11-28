from unidecode import unidecode
import streamlit as st
from glob import glob
from pathlib import Path

import functions

st.title('InCredible result inspection')
st.sidebar.title('Info')
st.sidebar.info('This is a reproduction of the experiments of the paper '
    '[Explaining Credibility in News Articles using Cross-Referencing](https://ears2018.github.io/ears18-bountouridis.pdf) '
    'based on the authors\' [demo](http://fairnews.ewi.tudelft.nl/InCredible/) '
    'and [source code](https://github.com/dbountouridis/InCredible). ')
st.sidebar.info('The code for this demo is [here](https://github.com/MartinoMensio/InCredible)')
st.sidebar.title('Instructions')
st.sidebar.info('Select a document clique to load: each one represents a different story')
st.sidebar.info('Select a main source to see the corresponding article')

doc_cliques_input_file = st.text_input('Document cliques input file:', 'Data/dataset.json')
output_path = st.text_input('Path where the computation results are:', 'temp/cliques_GA/')

document_cliques = functions.readJsonFile(doc_cliques_input_file)
document_cliques_ids = list(document_cliques.keys())

default = '0.7996630192184154-7-6180'
chosen_doc_clique_id = st.selectbox('Document clique to load:', document_cliques_ids, document_cliques_ids.index(default))
chosen_doc_clique = document_cliques[chosen_doc_clique_id]

clique_outputs = functions.readJsonFile(Path(output_path) / f'cliques_{chosen_doc_clique_id}.json')
st.text(f'Clique {chosen_doc_clique_id} selected')

st.text('Titles:\n\n'+'\n'.join([f'{p}--> {t}' for p,t in zip(chosen_doc_clique['publications'], chosen_doc_clique['sentences'])]))
available_outlets = [el for el in chosen_doc_clique['publications']]
if 'Reuters' in available_outlets:
    selected_outlet = st.selectbox('Main source:', available_outlets, available_outlets.index('Reuters') )
else:
    selected_outlet = st.selectbox('Main source:', available_outlets)
selected_outlet_idx = available_outlets.index(selected_outlet)
st.text(f'Selected outlet: {selected_outlet}')

print('\n\n\n')

main_text = chosen_doc_clique['contents'][selected_outlet_idx]
main_text = unidecode(main_text)


main_text_fragments = [{'text': main_text, 'start': 0, 'end': len(main_text) + 1, 'colour': 'white'}]
# print(main_text)
for clique_key, clique in clique_outputs.items():
    if selected_outlet in clique['publications']:
        # print(sorted([(el['start'], el['end']) for el in main_text_fragments]))
        # this clique is corroborated
        selected_idx = clique['publications'].index(selected_outlet)
        # iterate on all of them, because the publications field is not sorted in the same order
        for selected_idx in range(len(clique['sentences'])):
            piece_to_find = clique['sentences'][selected_idx]
            piece_to_find = unidecode(piece_to_find)
            start_idx = main_text.find(piece_to_find)
            if start_idx != -1:
                break
        #print('main_text', main_text)
        if start_idx == -1:
            print('not found any sentences:', clique['sentences'])
            #raise ValueError(piece_to_find)
            continue
        # print(selected_outlet_idx, selected_outlet, available_outlets)
        end_idx = start_idx + len(piece_to_find) + 1

        parent_fragment_matches = [el for el in main_text_fragments if (el['start']<=start_idx and el['end']>=end_idx)]
        # print(start_idx, end_idx, len(parent_fragment_matches))
        if len(parent_fragment_matches) != 1:
            # this is a fragment that is wider than other fragments already identified, ignore it
            # print('len(parent_fragment_matches):', len(parent_fragment_matches), f'in {start_idx}:{end_idx}', sorted([(el['start'], el['end']) for el in main_text_fragments]))
            # print(piece_to_find, parent_fragment_matches)
            continue
        parent_fragment = parent_fragment_matches.pop()

        main_text_fragments.remove(parent_fragment)

        parent_start = parent_fragment['start']
        parent_end = parent_fragment['end']
        if start_idx > parent_start:
            # not starting at the beginning of the parent
            main_text_fragments.append({'text': main_text[parent_start:start_idx], 'start': parent_start, 'end': start_idx, 'colour': 'white'})
        main_text_fragments.append({'text': piece_to_find, 'start': start_idx, 'end': end_idx, 'colour': 'lightblue', 'clique': clique})
        if end_idx < parent_end:
            # not ending at the end of the parent
            main_text_fragments.append({'text': main_text[end_idx-1:parent_end], 'start': end_idx-1, 'end': parent_end, 'colour': 'white'})
    else:
        # this is omitted
        pass
    #print('fragments:', [f"{el['start']}:{el['end']}" for el in main_text_fragments])

sorted_fragments = sorted(main_text_fragments, key=lambda el: el['start'])

show_other_sentences = st.checkbox('Show sentences from others', False)

def give_colour(el):
    result = f'<span style="background-color:{el["colour"]}">{el["text"]}</span>'
    if 'clique' in el:
        # print(el['clique'])
        others = [el for el in zip(el["clique"]["publications"], el['clique']['sentences'])]
        others = ''.join([f'<li>{p}: <div style="background-color:yellow">{s}</div></li>' for p,s in others])
        # print(others)
        if show_other_sentences:
            result = result + f'<ul style="background-color:grey;">{others}</ul>'
    return result

markdown = ''.join([give_colour(el) for el in sorted_fragments])
st.markdown(markdown, unsafe_allow_html=True)
