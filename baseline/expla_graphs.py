from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
from torch import Tensor
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import pandas as pd
import os
from tqdm import tqdm
import warnings
warnings.simplefilter("ignore")

E5_tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-base')
E5_model = AutoModel.from_pretrained('intfloat/multilingual-e5-base')

def BM25(query, nodes,edges, topk):
    tokenizer = AutoTokenizer.from_pretrained('facebook/spar-wiki-bm25-lexmodel-query-encoder')
    query_encoder = AutoModel.from_pretrained('facebook/spar-wiki-bm25-lexmodel-query-encoder')
    context_encoder = AutoModel.from_pretrained('facebook/spar-wiki-bm25-lexmodel-context-encoder')

    query_input = tokenizer(query, padding=True, truncation=True, return_tensors='pt')
    query_emb = query_encoder(**query_input).last_hidden_state[:, 0, :]
    if len(nodes)>topk:
        nodes_input = tokenizer(nodes, padding=True, truncation=True, return_tensors='pt')
        nodes_emb = context_encoder(**nodes_input).last_hidden_state[:, 0, :]
        nodes_scores = query_emb @ nodes_emb.T
        _, topk_nodes_indices = torch.topk(nodes_scores, topk, largest=True)
        selected_nodes = " ".join([nodes[idx] for idx in topk_nodes_indices[0]])
    else:
        selected_nodes = " ".join(nodes)
    if len(edges)>topk:
        edges_input = tokenizer(edges, padding=True, truncation=True, return_tensors='pt')
        edges_emb = context_encoder(**edges_input).last_hidden_state[:, 0, :]
        edges_scores = query_emb @ edges_emb.T
        _, topk_edges_indices = torch.topk(edges_scores, topk, largest=True)
        selected_edges = " ".join([edges[idx] for idx in topk_edges_indices[0]])
    else:
        selected_edges = " ".join(edges)
    answer = selected_nodes + " " + selected_edges 
    return answer

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

def E5Base(query,nodes,edges,topk,E5_tokenizer,E5_model):
    def average_pool(last_hidden_states: Tensor,
                    attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    
    
    if len(nodes)>topk:
        nodes_texts = combined_list = [f"query: {query}" ] + [f"passage: {n}" for n in nodes]
        nodes_dict = E5_tokenizer(nodes_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
        nodes_outputs =E5_model(**nodes_dict)
        nodes_embeddings = average_pool(nodes_outputs.last_hidden_state, nodes_dict['attention_mask'])
        nodes_embeddings = F.normalize(nodes_embeddings, p=2, dim=1)
        nodes_scores = (nodes_embeddings[:1] @ nodes_embeddings[1:].T) * 100
        _, topk_nodes_indices = torch.topk(nodes_scores, topk, largest=True)
        selected_nodes = " ".join([nodes[idx] for idx in topk_nodes_indices[0]])
    else:
        selected_nodes = " ".join(nodes)
    
    if len(edges)>topk:
        edges_texts = combined_list = [f"query: {query}" ] + [f"passage: {e}" for e in edges]
        edges_dict = E5_tokenizer(edges_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
        edges_outputs = E5_model(**edges_dict)
        edges_embeddings = average_pool(edges_outputs.last_hidden_state, edges_dict['attention_mask'])
        edges_embeddings = F.normalize(edges_embeddings, p=2, dim=1)
        edges_scores = (edges_embeddings[:1] @ edges_embeddings[1:].T) * 100
        _, topk_edges_indices = torch.topk(edges_scores, topk, largest=True)
        selected_edges = " ".join([edges[idx] for idx in topk_edges_indices[0]])
    else:
        selected_edges = " ".join(edges)
    answer= selected_nodes + " "+ selected_edges 
    return answer

if __name__ == '__main__':
    PATH = '/home/ubuntu/G-Retriever/dataset/expla_graphs'
    prompt = 'Question: Do argument 1 and argument 2 support or counter each other? Answer in one word in the form of \'support\' or \'counter\'.\n\nAnswer:'
    all_texts = pd.read_csv(f'{PATH}/train_dev.tsv', sep='\t')
    
    # for topk in [3,5,10]:
    #     os.makedirs(f'/home/ubuntu/G-Retriever/baseline/BM25/expla_graphs/top{topk}', exist_ok=True) #'baseline/BM25/top3
    #     for model in ['sentence-transformers/all-MiniLM-L12-v2', 'sentence-transformers/LaBSE','nthakur/mcontriever-base-msmarco']:
    #         os.makedirs(f'/home/ubuntu/G-Retriever/baseline/{model}/expla_graphs/top{topk}', exist_ok=True)
    #     os.makedirs(f'/home/ubuntu/G-Retriever/baseline/E5Base/expla_graphs/top{topk}', exist_ok=True)
    
    for model_name in ['sentence-transformers/all-MiniLM-L12-v2', 'sentence-transformers/LaBSE','nthakur/mcontriever-base-msmarco']:

        model = SentenceTransformer(model_name)
        for topk in [3,5,10]:
            for index in tqdm(range(224,230)):
                text = all_texts.iloc[index]
                question = f'Argument 1: {text.arg1}\nArgument 2: {text.arg2}\n{prompt}'
                nodes = pd.read_csv(f'{PATH}/nodes/{index}.csv')
                edges = pd.read_csv(f'{PATH}/edges/{index}.csv')
                nodes=nodes['node_attr'].astype(str).tolist()
                edges=edges['edge_attr'].astype(str).tolist()

            # prompts=BM25(question, nodes, edges, topk) 
            # with open(f'/home/ubuntu/G-Retriever/baseline/BM25/expla_graphs/top{topk}/{index}.txt', 'w', encoding='utf-8') as file:
            #     file.write(prompts)
                prompts=sentence_transformer_models(question, nodes,edges, topk,model)
                with open(f'/home/ubuntu/G-Retriever/baseline/{model_name}/expla_graphs/top{topk}/{index}.txt', 'w', encoding='utf-8') as file:
                    file.write(prompts)

            
        

    # for index in tqdm(range(130,len(all_texts))):
    #     text = all_texts.iloc[index]
    #     question = f'Argument 1: {text.arg1}\nArgument 2: {text.arg2}\n{prompt}'
    #     nodes = pd.read_csv(f'{PATH}/nodes/{index}.csv')
    #     edges = pd.read_csv(f'{PATH}/edges/{index}.csv')
    #     nodes=nodes['node_attr'].astype(str).tolist()
    #     edges=edges['edge_attr'].astype(str).tolist()

    #     prompts=BM25(question, nodes, edges, topk) 
    #     with open(f'/home/ubuntu/G-Retriever/baseline/BM25/expla_graphs/top{topk}/{index}.txt', 'w', encoding='utf-8') as file:
    #         file.write(prompts)
    #     prompts=E5Base(question,nodes,edges,topk)
    #     with open(f'/home/ubuntu/G-Retriever/baseline/E5Base/expla_graphs/top{topk}/{index}.txt', 'w', encoding='utf-8') as file:
    #                 file.write(prompts)