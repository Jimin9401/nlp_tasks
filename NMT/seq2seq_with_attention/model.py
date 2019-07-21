
import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderGRU(nn.Module):
    def __init__(self, emb_dim, bidirectional, vocab_size):
        super().__init__()

        self.bidirectional = bidirectional
        self.embed = nn.Embedding(embedding_dim=emb_dim, padding_idx=0, num_embeddings=vocab_size)
        self.gru = nn.GRU(hidden_size=emb_dim, input_size=emb_dim, bidirectional=bidirectional)
        self.fc = nn.Linear(in_features=emb_dim * 2, out_features=emb_dim)

    def forward(self, src):
        batch_size = src.shape[0]
        src_length = src.shape[1]
        # src= (batch, src_length)

        src_embed = self.embed(src)
        # src_embed= (batch, src_length, embedd_dim)

        src_embed = src_embed.transpose(0, 1)
        # src_embed=(src_len, batch, embedd_dim)

        outputs, hidden = self.gru(src_embed)

        # outputs (src_length,batch,dim)
        # hidden (batch,1,dim)
        # cell (batch,1,dim)

        outputs = outputs.transpose(0, 1)

        if self.bidirectional == True:
            hidden = torch.tanh(self.fc(torch.cat((hidden[-1, :, :], hidden[-2, :, :]), dim=1)))

            hidden = hidden.unsqueeze(1)
        #            cell=cell.unsqueeze(1)

        #        else:
        #            hidden=hidden.squeeze(0)
        #            cell=cell.squeeze(0)

        # final!

        # outputs (batch,src,dim)
        # hidden (batch,dim)
        # cell (batch,dim)

        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, emb_dim,padding_idx):
        super().__init__()

        self.linear_out = nn.Linear(emb_dim * 2, emb_dim)

        self.padding_idx=padding_idx
    def forward(self, previous_hidden, contexts,src_batch):
        src_len = contexts.shape[1]


        weight_mask=torch.zeros(src_batch.reshape([-1,1]).size(0)).view(src_batch.size(0),src_batch.size(1))


        mask=(src_batch==self.padding_idx)


        # previous_hidden (batch, dim*2 )  if bidirection
        # context (batch, seq_lens,dim*2 )  if bidirection

        # previous_hidden=previous_hidden.unsqueeze(1)

        # dim*2 => dim
        contexts = self.linear_out(contexts)

        ## (batch , 1,dim )  * (batch, dim, seq_lens) =(batch, 1, seq_lens)

        energy=torch.bmm(previous_hidden,contexts.transpose(1,2))

        energy=energy.squeeze(1)

        energy[mask]=float('-inf')

        energy=energy.unsqueeze(1)



        attn_score = torch.softmax(energy,dim=2)

        ## (batch, 1, seq_lens) * (batch,seq_lens,dim)

        weighted = torch.bmm(attn_score, contexts)

        # attn_score(batch, 1, sec_lens)

        # weighted ( batch ,1 , dim)

        return attn_score, weighted

class DecoderGRU(nn.Module):
    def __init__(self, emb_dim, n_class, attention):
        super().__init__()

        # field 선언.. static 변수처럼 사용~

        self.n_class = n_class

        self.embed = nn.Embedding(embedding_dim=emb_dim, num_embeddings=n_class)

        self.out = nn.Linear(emb_dim, n_class)

        self.attention = attention

        self.gru = nn.GRU(emb_dim * 2, emb_dim)

    def forward(self, trg_input, hidden, contexts,src_input):
        # contexts=[batch,sec_lens,embed_dim*2]

        # hidden=[batch, 1,embed_dim*2]

        # trg_input =[batch_size]

        trg_input = trg_input.unsqueeze(0)
        # trg_input =[1, batch_size]

        trg_embed = self.embed(trg_input)

        # trg_embed=[1,batch_size ,embed_dim]

        a, weighted = self.attention(hidden, contexts,src_input)

        concat_input = torch.cat((trg_embed.transpose(0, 1), weighted), dim=2)
        ## embed + weight= 900

        _, hidden = self.gru(concat_input.transpose(0, 1), hidden.transpose(0, 1))

        hidden = hidden.transpose(0, 1)

        #        output=self.out(torch.cat((hidden,weighted,trg_embed.transpose(0,1)),dim=2))  embed + hidden + weighted
        # output=self.out(torch.cat((hidden,weighted),dim=2)) hidden _ weighted
        output = self.out(hidden)  # hidden

        output = output.squeeze(1)

        return output, hidden




class Seq2Seq_a(nn.Module):
    def __init__(self, encoder, decoder, device,word2idx):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.word2idx=word2idx

    def forward(self, src, trg, teacher_force):

        # src= (batch, src_lens )

        # trg= (batch, trg_lens)

        batch_size = src.shape[0]
        seq_lens = trg.shape[1]

        n_class = self.decoder.n_class

        contexts, encoder_state = self.encoder(src)

        trg = trg.transpose(0, 1)
        output = trg[0, :]

        output_scores = torch.zeros(seq_lens, batch_size, len(self.word2idx)).to(self.device)

        next_input = trg[0]

        hidden = encoder_state

        predict_list = []

        if (teacher_force):
            for t in range(1, seq_lens):
                output, hidden = self.decoder.forward(trg_input=next_input, hidden=hidden, contexts=contexts,src_input=src)

                output_scores[t] = output
                y_true = trg[t]
                ##                y_predict=output.data.max(1)[1]
                next_input = y_true

#            print("=========1 mini-batch=============\n")

            return F.log_softmax(output_scores, dim=1).permute(1,0,2)

        else:
            for t in range(1, 10):
                output, hidden = self.decoder.forward(trg_input=next_input, hidden=hidden, contexts=contexts)

                y_predict = output.data.max(1)[1]

                next_input = y_predict

                y_predict = output.data.max(1)[1].item()

                predict_list.append(y_predict)

                if y_predict == self.word2idx["[EOS]"]:
                    break

            return predict_list