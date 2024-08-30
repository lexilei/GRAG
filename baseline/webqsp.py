
import datasets
from tqdm import tqdm
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
from torch import Tensor
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import os
import warnings
warnings.simplefilter("ignore")
from tqdm import tqdm

def sentence_transformer_models(query, nodes,edges, topk,model):
    

    query_emb = model.encode(query)

    if len(nodes)>topk:
        nodes_emb = model.encode(nodes)
        nodes_scores = torch.tensor(query_emb @ nodes_emb.T)
        _, topk_nodes_indices = torch.topk(nodes_scores, topk, largest=True)
        selected_nodes = " ".join([nodes[idx] for idx in topk_nodes_indices])
    else:
        selected_nodes = " ".join(nodes)
    
    if len(edges)>topk:
        edges_emb = model.encode(edges)
        edges_scores = torch.tensor(query_emb @ edges_emb.T)
        _, topk_edges_indices = torch.topk(edges_scores, topk, largest=True)
        selected_edges = " ".join([edges[idx] for idx in topk_edges_indices])
    else:
        selected_edges = " ".join(edges)
    answer= selected_nodes + " "+ selected_edges 

    return answer

def sentence():
    #'sentence-transformers/all-MiniLM-L12-v2', 'sentence-transformers/LaBSE',
    for model_name in ['nthakur/mcontriever-base-msmarco']:
        model = SentenceTransformer(model_name)
        #3,5,10,15,
        for topk in tqdm([20]):
            for index in tqdm(range(4100,len(dataset))):
                data = dataset[index]
                question = f'Question: {data["question"]}\nAnswer: '
        
                nodes = pd.read_csv(f'{PATH}/nodes/{index}.csv')
                edges = pd.read_csv(f'{PATH}/edges/{index}.csv')
                nodes=nodes['node_attr'].astype(str).tolist()
                edges=edges['edge_attr'].astype(str).tolist()

                # prompts=BM25(question, nodes, edges, topk) 
                
                # with open(f'/home/ubuntu/G-Retriever/baseline/BM25/{data_type}/top{topk}/{index}.txt', 'w', encoding='utf-8') as file:
                #     file.write(prompts)
                
                prompts=sentence_transformer_models(question, nodes,edges, topk,model)
                with open(f'/home/ubuntu/G-Retriever/baseline/{model_name}/{data_type}/top{topk}/{index}.txt', 'w', encoding='utf-8') as file:
                    file.write(prompts)

def bm():
    PATH = '/home/ubuntu/G-Retriever/dataset/webqsp'
    dataset = datasets.load_dataset("rmanluo/RoG-webqsp")
    dataset = datasets.concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])
    data_type='webqsp'
    print("finished loading dataset") 
    tokenizer = AutoTokenizer.from_pretrained('facebook/spar-wiki-bm25-lexmodel-query-encoder')
    query_encoder = AutoModel.from_pretrained('facebook/spar-wiki-bm25-lexmodel-query-encoder')
    context_encoder = AutoModel.from_pretrained('facebook/spar-wiki-bm25-lexmodel-context-encoder')
    print("finished loading model") 
    def BM25(query, nodes,edges, topk):
        query_input = tokenizer(query, padding=True, truncation=True, return_tensors='pt')
        query_emb = query_encoder(**query_input).last_hidden_state[:, 0, :]

        nodes_input = tokenizer(nodes, padding=True, truncation=True, return_tensors='pt')
        nodes_emb = context_encoder(**nodes_input).last_hidden_state[:, 0, :]
        nodes_scores = query_emb @ nodes_emb.T
        _, topk_nodes_indices = torch.topk(nodes_scores, topk, largest=True)
        selected_nodes = " ".join([nodes[idx] for idx in topk_nodes_indices[0]])

        edges_input = tokenizer(edges, padding=True, truncation=True, return_tensors='pt')
        edges_emb = context_encoder(**edges_input).last_hidden_state[:, 0, :]
        edges_scores = query_emb @ edges_emb.T
        _, topk_edges_indices = torch.topk(edges_scores, topk, largest=True)
        selected_edges = " ".join([edges[idx] for idx in topk_edges_indices[0]])

        answer = selected_nodes + " " + selected_edges 
        return answer
    
    # for topk in [3]:
    #     for index in tqdm(range(3,len(dataset))):
    #         data = dataset[index]
    #         question = f'Question: {data["question"]}\nAnswer: '
    #         nodes = pd.read_csv(f'{PATH}/nodes/{index}.csv')
    #         edges = pd.read_csv(f'{PATH}/edges/{index}.csv')
    #         nodes=nodes['node_attr'].astype(str).tolist()
    #         edges=edges['edge_attr'].astype(str).tolist()
    #         prompts=BM25(question, nodes, edges, topk) 
            
    #         with open(f'/home/ubuntu/G-Retriever/baseline/BM25/{data_type}/top{topk}/{index}.txt', 'w', encoding='utf-8') as file:
    #             file.write(prompts)

    print("checkpoint 1") 
    topk=3
    index=3
    data = dataset[index]
    question = f'Question: {data["question"]}\nAnswer: '
    nodes = pd.read_csv(f'{PATH}/nodes/{index}.csv')
    edges = pd.read_csv(f'{PATH}/edges/{index}.csv')
    nodes=nodes['node_attr'].astype(str).tolist()
    edges=edges['edge_attr'].astype(str).tolist()
    prompts=BM25(question, nodes, edges, topk) 
    print("now writing file") 
    with open(f'/home/ubuntu/G-Retriever/baseline/BM25/{data_type}/top{topk}/{index}.txt', 'w', encoding='utf-8') as file:
        file.write(prompts)

    

def make():
    # make dir
    for topk in [3,5,10,15,20]:
        os.makedirs(f'/home/ubuntu/G-Retriever/baseline/BM25/webqsp/top{topk}', exist_ok=True) #'baseline/BM25/top3
        for model in ['sentence-transformers/all-MiniLM-L12-v2', 'sentence-transformers/LaBSE','nthakur/mcontriever-base-msmarco']:
            os.makedirs(f'/home/ubuntu/G-Retriever/baseline/{model}/webqsp/top{topk}', exist_ok=True)
        os.makedirs(f'/home/ubuntu/G-Retriever/baseline/E5Base/webqsp/top{topk}', exist_ok=True)

if __name__ == '__main__':
    bm()
    # PATH = '/home/ubuntu/G-Retriever/dataset/webqsp'
    # dataset = datasets.load_dataset("rmanluo/RoG-webqsp")
    # dataset = datasets.concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])
    # data_type='webqsp'
    
    
    #load model
    # E5_tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-base')
    # E5_model = AutoModel.from_pretrained('intfloat/multilingual-e5-base')
    # for topk in tqdm([3,5,10,15,20]):
    #     for index in tqdm(range(len(dataset))):
    #         data = dataset[index]
    #         question = f'Question: {data["question"]}\nAnswer: '
    #         nodes = pd.read_csv(f'{PATH}/nodes/{index}.csv')
    #         edges = pd.read_csv(f'{PATH}/edges/{index}.csv')
    #         nodes=nodes['node_attr'].astype(str).tolist()
    #         edges=edges['edge_attr'].astype(str).tolist()
            
            # prompts=E5Base(question,nodes,edges,topk,E5_tokenizer,E5_model)
            # with open(f'/home/ubuntu/G-Retriever/baseline/E5Base/{data_type}/top{topk}/{index}.txt', 'w', encoding='utf-8') as file:
            #         file.write(prompts)


            