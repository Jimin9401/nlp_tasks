import torch.nn as nn
import torch
import torch.nn.functional as F

def attention(query,context):
    scores=torch.matmul(query,context.transpose(1,2))

    return scores

class WordEmbedder(nn.Module):
    def __init__(self,d_vector,vocab_size):
        super(WordEmbedder,self).__init__()
        self.embedding=nn.Embedding(num_embeddings=vocab_size,embedding_dim=d_vector)

    def forward(self,x):
        # [batch, seq_lens]

        return self.embedding(x)         # [batch, seq_lens,embedding]



class Encoder(nn.Module):
    def __init__(self,hidden_dim,word_dim,Bidirectional=True):
        super(Encoder,self).__init__()
        self.encoder=nn.LSTM(bidirectional=Bidirectional,hidden_size=hidden_dim, input_size=word_dim)

    def forward(self,query,context):
        query,_=self.encoder.forward(query)
        context,_= self.encoder.forward(context)


        return query,context


class AttentionFlow(nn.Module):
    def __init__(self):
        super(AttentionFlow, self).__init__()

    def forward(self, query, context):
        # [batch, query_lens, d_h]
        # [batch, context_lens, d_h]

        context_lens = context.size(1)

        # energy=[batch,query_lens,context_lens]

        #scores= [batch, context, query)
        scores = attention(query, context).transpose(1, 2)

        c2q_attn = F.softmax(scores, dim=2)

        c2q_cxt = torch.matmul(c2q_attn, query)

        q2c_attn = F.softmax(torch.max(scores, dim=2)[0].unsqueeze(1), dim=2)

        q2c_cxt = torch.matmul(q2c_attn, context)


        q2c_cxt = q2c_cxt.expand(-1, context_lens, -1)

        # c2q_attn== softmax at each query_lens [ batch, query_lens, context_lens]
        # c2q=[batch,context_lens,d_hidden]

        x=torch.cat([context,c2q_cxt,context*c2q_cxt,context*q2c_cxt],dim=-1)
        return x

class ModelingLayer(nn.Module):
    def __init__(self,d_vector,bidirectional=True):
        super(ModelingLayer,self).__init__()

        self.modeling1=nn.LSTM(input_size=d_vector*8,hidden_size=d_vector,bidirectional=bidirectional)
        self.modeling2=nn.LSTM(input_size=d_vector*2,hidden_size=d_vector,bidirectional=bidirectional)

    def forward(self, g):

        #x= [batch, c_lens, dim*8]

        g,_=self.modeling1.forward(g)

        m,_=self.modeling2.forward(g)


        return m



class OutputLayer(nn.Module):
    def __init__(self,d_vector):
        super(OutputLayer,self).__init__()
        self.start_g=nn.Linear(in_features=d_vector*8,out_features=1)
        self.start_m=nn.Linear(in_features=d_vector*2,out_features=1)

        self.end_g=nn.Linear(in_features=d_vector*8,out_features=1)
        self.end_m=nn.Linear(in_features=d_vector*2,out_features=1)

        self.out_lstm=nn.LSTM(input_size=d_vector*2,hidden_size=d_vector,bidirectional=True)


    def forward(self,g,m):

        # concat_input= [batch, context_lens, dim*8]

        p1=(self.start_g(g)+self.start_m(m)).squeeze()

        m2=self.out_lstm(m)[0]

        p2=(self.end_g(g)+self.end_m(m2)).squeeze()



        return p1,p2



class BIDAF(nn.Module):
    def __init__(self,embedder,encoder,attention_flow,modeling_layer,output_layer):
        super(BIDAF,self).__init__()
        self.embedder=embedder
        self.encoder=encoder
        self.attention_flow=attention_flow
        self.modeling_layer=modeling_layer
        self.output_layer=output_layer

    def forward(self,q,c):

        q=self.embedder(q)
        c=self.embedder(c)

        q,c=self.encoder(q,c)
        g=self.attention_flow(q,c)
        m=self.modeling_layer(g)

        p1,p2=self.output_layer(g,m)

        return p1,p2