import os

import matplotlib
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F

from faceted_domain_encoder import FacetedDomainEncoder
from faceted_domain_encoder.util.linalg import CategoryDistance
from faceted_domain_encoder.util.plotting import CATEGORY_PALLETTE

st.markdown('# Faceted Domain Encoder - Explorer 🔍')


@st.cache(allow_output_mutation=True)
def load_model(path):
    model_path = os.path.join(path, 'model', 'model.ckpt')
    return FacetedDomainEncoder.load_from_checkpoint(checkpoint_path=model_path)


@st.cache()
def load_documents(path):
    return pd.read_csv(path, sep='\n', header=None, names=['text'])


@st.cache(allow_output_mutation=True)
def load_embeddings(documents):
    return model.embed(documents)


def get_category_knns(embeddings, document_id, category_id, k=10):
    category_distance = CategoryDistance(512, 16)
    category_distances = category_distance(embeddings[document_id], embeddings)[:, category_id]
    sort_keys = torch.argsort(category_distances)
    return sort_keys[1:k + 1]


def get_knns(embeddings, document_id, k=10):
    x = embeddings[document_id]
    distances = 1 - F.cosine_similarity(x, embeddings, dim=-1)
    sort_keys = torch.argsort(distances)
    return sort_keys[1:k + 1]


def display_entity(model, doc, filter_category=None):
    indices, categories, length = model.processor(doc)
    markdown = ''

    for index, category in zip(indices, categories):
        if index == 0:
            continue

        category = category.item()
        token = model.processor.vocabulary.tokens[index]
        color = None

        if filter_category is not None:
            if filter_category == category:
                color = CATEGORY_PALLETTE[category]
        else:
            if category != -1:
                color = CATEGORY_PALLETTE[category]

        markdown += f'<span style="color:{color}"><b>{token}</b></span> ' if color else f'{token} '

    st.write(markdown, unsafe_allow_html=True)


def display_attention(model, document, filter_category=None):
    indices, categories, length = model.processor(document)

    batch_indices = indices.unsqueeze(0)
    batch_categories = categories.unsqueeze(0)
    batch_lengths = torch.tensor([length], dtype=torch.long)

    x, attention_weights = model.forward_batch(batch_indices, batch_categories, batch_lengths, attention=True)
    attention = attention_weights[0, :, :length].detach().cpu()

    markdown = ''

    for i, (index, category) in enumerate(zip(indices, categories)):
        token_category = filter_category

        if index == 0:
            continue

        if filter_category is None:
            token_category = attention[:, i].argmax()
            token_category = token_category.item()

        token = model.processor.vocabulary.tokens[index]
        weight = (attention[token_category][i] / attention[token_category].max()).item()
        color = CATEGORY_PALLETTE[token_category]
        rgb = matplotlib.colors.to_rgb(color)
        rgb = tuple([weight * c for c in rgb])
        color = matplotlib.colors.to_hex(rgb)
        font_weight = ((weight * 1000) // 100) * 100

        markdown += f'<span style="color:{color}; font-weight: {font_weight}">{token}</span> ' if color else f'{token} '

    st.write(markdown, unsafe_allow_html=True)


def modify_magnitude(embeddings, categories, document_id):
    embeddings = embeddings.view(-1, 16, 32)
    magnitudes = torch.tensor([embeddings[document_id, c].norm().item() for c in range(16)])
    st.sidebar.markdown(f'### Adjust Document Embedding')

    for category in categories:
        category_id = category2index[category]
        magnitude = magnitudes[category_id].item()
        new_magnitude = st.sidebar.slider(category, value=magnitude, min_value=0.001)

        embeddings[document_id, category_id] /= magnitude
        embeddings[document_id, category_id] *= new_magnitude

    return embeddings.view(-1, 512)


model = load_model('./outputs/ohsumed_classification-lstm-category_attention-document')
model.eval()
categories = model.hparams.graph.categories
category2index = {c: i for i, c in enumerate(categories)}

df = load_documents('./data/experiment/classification/ohsumed/test.txt')
embeddings = load_embeddings(df.text.values)

document_id = st.selectbox('Select document', list(range(100)))
document = df.iloc[document_id].text
indices, document_categories, length = model.processor(document)

categories = list(model.hparams.graph.categories)
display_categories = [categories[c.item()] for c in document_categories.unique() if c.item() != -1]
category = st.selectbox('Select category', ['All'] + display_categories)
color_by = st.selectbox('Color by', ['Entity', 'Attention'])

embeddings = modify_magnitude(embeddings, display_categories, document_id)

if category == 'All':
    neighbor_ids = get_knns(embeddings, document_id)
    category_id = None
else:
    category_id = category2index[category]
    neighbor_ids = get_category_knns(embeddings, document_id, category_id)

st.markdown(f'#### Document {document_id}')

if color_by == 'Entity':
    display_entity(model, document, filter_category=category_id)
else:
    display_attention(model, document, filter_category=category_id)

st.markdown('### Top 10 Neighboring Documents')
for neighbor_id in neighbor_ids:
    neighbor = df.iloc[neighbor_id.item()].text
    st.markdown(f'#### Document {neighbor_id.item()}')

    if color_by == 'Entity':
        display_entity(model, neighbor, filter_category=category_id)
    else:
        display_attention(model, neighbor, filter_category=category_id)
