import torch
import torch.nn.functional as F
from functorch.einops import rearrange
from torch import nn


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        # query [B, 1, H]
        # keys [B, T, H]
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))  # [B, T, 1]
        scores2 = rearrange(scores, 'B T 1 -> B 1 T')
        weights = F.softmax(scores2, dim=-1)  # [B, 1, T]
        context = weights @ keys  # [B, 1, H]
        assert context.shape[0] == query.shape[0]
        return context, weights


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, BOS_token, block_size, dropout_p=0.1, ):
        super(AttnDecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)
        self.BOS_token = BOS_token
        self.block_size = block_size

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        # encoder_outputs: [B, T, H]
        # encoder_hidden: [1, B, H]
        # target_tensor: [B, T+1]
        device = encoder_outputs.device
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(self.BOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(self.block_size):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1)  # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)  # [B, T, V]
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)  # [B, T, V]
        attentions = torch.cat(attentions, dim=1)  # [B, T, T]

        return decoder_outputs, decoder_hidden, attentions

    def forward_step(self, input, hidden, encoder_outputs):
        # input: [B, 1]
        # hidden: [1, B, H]
        # encoder_output: [B, T, H]
        embedded = self.dropout(self.embedding(input))  # [B, 1, H]
        query = hidden.permute(1, 0, 2)  # [B, 1, H]
        context, attn_weights = self.attention(query, encoder_outputs)  # [B,1,H], [B,1,T]
        assert embedded.shape[0] == input.shape[0]
        input_gru = torch.cat((embedded, context), dim=2)  # [B, 1, 2H]

        output, hidden = self.gru(input_gru, hidden)  # [B, 1, H], [1, B, H]
        output = self.out(output)  # [B, 1, V]

        return output, hidden, attn_weights
