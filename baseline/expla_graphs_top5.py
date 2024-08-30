from baseline.expla_graphs import BM25, E5Base, sentence_transformer_models
from sentence_transformers import SentenceTransformer
import pandas as pd
import os
import warnings
from tqdm import tqdm

warnings.simplefilter("ignore")

if __name__ == '__main__':

    PATH = '/home/ubuntu/G-Retriever/dataset/expla_graphs'
    prompt = 'Question: Do argument 1 and argument 2 support or counter each other? Answer in one word in the form of \'support\' or \'counter\'.\n\nAnswer:'
    all_texts = pd.read_csv(f'{PATH}/train_dev.tsv', sep='\t')
    
    for topk in [3,5,10]:
        # os.makedirs(f'/home/ubuntu/G-Retriever/baseline/BM25/expla_graphs/top{topk}', exist_ok=True) #'baseline/BM25/top3
        # for model in ['sentence-transformers/all-MiniLM-L12-v2', 'sentence-transformers/LaBSE','nthakur/mcontriever-base-msmarco']:
        #     os.makedirs(f'/home/ubuntu/G-Retriever/baseline/{model}/expla_graphs/top{topk}', exist_ok=True)
        #os.makedirs(f'/home/ubuntu/G-Retriever/baseline/E5Base/expla_graphs/top{topk}', exist_ok=True)
        for index in tqdm(range(230,250)):
            text = all_texts.iloc[index]
            question = f'Argument 1: {text.arg1}\nArgument 2: {text.arg2}\n{prompt}'
            nodes = pd.read_csv(f'{PATH}/nodes/{index}.csv')
            edges = pd.read_csv(f'{PATH}/edges/{index}.csv')
            nodes=nodes['node_attr'].astype(str).tolist()
            edges=edges['edge_attr'].astype(str).tolist()

            # if index <30:
            #     prompts=BM25(question, nodes, edges, topk) 
            #     with open(f'/home/ubuntu/G-Retriever/baseline/BM25/expla_graphs/top{topk}/{index}.txt', 'w', encoding='utf-8') as file:
            #         file.write(prompts)
            
            for model_name in ['sentence-transformers/all-MiniLM-L12-v2', 'sentence-transformers/LaBSE','nthakur/mcontriever-base-msmarco']:
                model = SentenceTransformer(model_name)
        
                prompts=sentence_transformer_models(question, nodes,edges, topk,model)
                with open(f'/home/ubuntu/G-Retriever/baseline/{model}/expla_graphs/top{topk}/{index}.txt', 'w', encoding='utf-8') as file:
                    file.write(prompts)

            # prompts=E5Base(question,nodes,edges,topk)
            # with open(f'/home/ubuntu/G-Retriever/baseline/E5Base/expla_graphs/top{topk}/{index}.txt', 'w', encoding='utf-8') as file:
            #         file.write(prompts)