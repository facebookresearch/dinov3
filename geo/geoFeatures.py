from omegaconf import OmegaConf
from model.fossilVL import FossilVL
import torch
import os 
from argparse import ArgumentParser
from omegaconf import OmegaConf
import os
import torch
import json
import matplotlib.pyplot as plt
from glob import glob
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
import torch


LABEL_MAP = {'classification': ['Constituintes Principais', 'Constituintes Secundários', 'Gênese', 'Tamanho do Elemento', 'Comp. Atual do Elemento', 'Acessórios', 'Núcleo do Esferulito'],
             'composition': ['litologia_microscopica'],
             'texture': ['Estrutura/Textura', 'Granulação <2 mm', 'Granulação modal principal (mm)', 'Granulação secundária (mm)', 'Seleção', 'Empacotamento', 'Arranjo', 'Matriz', 'Tipo de Matriz', 'Matriz (Dunham 1962)', 'Tipo de Laminação', 'Laminação Caracterizada Por', 'Proporção Cascalho/Areia/Lama', 'Tipo Contato entre Partic.', 'Tamanho do Cristal', 'Integridade das Conchas', 'Orientação das Conchas'],
             'porosity': ['Tipo(s) de Poro(s)', 'Estimativa Visual', 'Tam. Modal do(s) Poro(s)'],
             'diagenesis': ['Eventos Diagenéticos', 'Cimento', 'Espaço Interconstituintes']}

def normalizeObj(obj):
    return tuple(
        (key, tuple(sorted(value)) if isinstance(value, list) else value)
        for key, value in sorted(obj.items())
    )


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', help='path to model output folder', required=True)
    parser.add_argument('--data',  help='file to load', required=True, choices=['test', 'train'])
    parser.add_argument('--n', default=None, help='number of captions to generate', type=int)
    parser.add_argument('--ckpt', choices=['best', 'last'], help='checkpoint to load', required=True)
    parser.add_argument('--campo', default=None, choices=['classification', 'texture', 'porosity', 'diagenesis', 'composition'])
    parser.add_argument('--size', type=int, default=512)
    args = parser.parse_args()
  
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    conf = OmegaConf.load(os.path.join(args.model, 'config.yaml'))
    model = FossilVL(conf)
    model.to(torch.float32)

    if not hasattr(model.decoder.model, "peft_config") and conf.decoder.apply_lora:
        model.decoder.apply_lora(conf)
    
    print('loading model: {}'.format(os.path.join(args.model, f'{args.ckpt}_checkpoint.pt')))
    ckpt = torch.load(os.path.join(args.model, f'{args.ckpt}_checkpoint.pt'), map_location=torch.device('cpu'))

    model.load_state_dict(ckpt)
    model.to(device)

    # load data
    data = pd.read_csv(args.file)
    data = data.dropna(subset=[args.campo])
    data = data[data['user_group'].isin([2, 3])]
    data = data[data['size'] >= args.size]
    images = []
    labels = []
    groups = data.groupby('slide_id')

    for group, values in groups:
        images.append(values['image_id'].to_list()[0])
        labels.append(values['labels'].to_list()[0])    

    categories = []

    images = images[:args.n]
    labels = labels[:args.n]

    # create category mapping
    for e in labels:
        e = e.replace('"', "cramunhao").replace("'", '"').replace("cramunhao", "'")
        e = json.loads(e)

        filtered_labels = {}
        for key, value in e.items():
            if key in LABEL_MAP[args.campo]:
                filtered_labels[key] = value 

        if not filtered_labels:
            filtered_labels['no info'] = True

        labels.append(filtered_labels)
        categories.append(filtered_labels)

    data = pd.DataFrame({'labels': categories})
    data['labels'] = data['labels'].map(normalizeObj)
    _, categories  =  pd.factorize(data['labels'] )

    
    with torch.no_grad():
        image_tensors = model.encoder.get_image_tensors(images, size=args.size)
        features = model.encoder(image_tensors, return_grid=model.use_grid)

    print(features.shape)
    tsne = TSNE(n_components=2, learning_rate='auto', metric='cosine', method='exact', random_state=42)
    x = tsne.fit_transform(features)

    df = pd.DataFrame(x, columns=['t-SNE 1', 't-SNE 2'])
    df['Class'] = [categories.get_loc(e) for e in labels]

    k = len(df['Class'].drop_duplicates())

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df, 
        x='t-SNE 1', 
        y='t-SNE 2', 
        # s=25,
        hue='Class',       # Color by this column
        palette='colorblind', # Choose a color palette
        alpha=0.5,
        legend=False,
    )
    plt.title(f'TSNE dino classes={k}')
    plt.savefig(f'dino TSNE.png')
    plt.clf()

    