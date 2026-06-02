# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


### DRQN: input is sequence of past responses only.
### Response embedding has 3 entries: index 0 = wrong, 1 = correct, 2 = start-of-episode.
### At decision step t, the LSTM has consumed [START, o_0, ..., o_{t-1}].
class DRQN(nn.Module):
    def __init__(self, action_space, embed_dim, lstm_hidden, dropout_rate):
        super(DRQN, self).__init__()
        self.embed = nn.Embedding(3, embed_dim)  # 0=wrong, 1=correct, 2=start
        self.lstm = nn.LSTM(embed_dim, lstm_hidden, batch_first=True)
        self.out = nn.Linear(lstm_hidden, action_space)
        self.dropout = nn.Dropout(dropout_rate)
        self.action_space = action_space
        self.embed_dim = embed_dim
        self.lstm_hidden = lstm_hidden

    def forward(self, resps, hidden=None):
        # resps: (batch, seq_len) long, values in {0, 1, 2}
        x = self.embed(resps)
        x = self.dropout(x)
        out, hidden = self.lstm(x, hidden)
        out = self.dropout(out)
        return self.out(out), hidden

    def init_hidden(self, batch_size=1):
        h = torch.zeros(1, batch_size, self.lstm_hidden, device=device)
        c = torch.zeros(1, batch_size, self.lstm_hidden, device=device)
        return (h, c)

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if "weight" in name:
                        nn.init.kaiming_normal_(param)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.1)


### ADRQN (Zhu et al., 2018, arXiv:1704.07978): input is action-observation pairs.
### At decision step t, the LSTM has consumed [(START,0), (a_0,o_0), ..., (a_{t-1},o_{t-1})].
### The action (item index) is embedded, then concatenated with the observation (response 0/1).
class ADRQN(nn.Module):
    def __init__(self, action_space, embed_dim, lstm_hidden, dropout_rate):
        super(ADRQN, self).__init__()
        # action_space + 1 embeddings: indices 0..action_space-1 for items, action_space for START
        self.embed = nn.Embedding(action_space + 1, embed_dim)
        self.lstm = nn.LSTM(embed_dim + 1, lstm_hidden, batch_first=True)
        self.out = nn.Linear(lstm_hidden, action_space)
        self.dropout = nn.Dropout(dropout_rate)
        self.action_space = action_space
        self.embed_dim = embed_dim
        self.lstm_hidden = lstm_hidden

    def forward(self, items, resps, hidden=None):
        # items: (batch, seq_len) long  — item indices (or START_TOKEN)
        # resps: (batch, seq_len) long  — responses 0/1 (or dummy 0 for START)
        e = self.embed(items)
        r = resps.unsqueeze(-1).float()
        x = torch.cat([e, r], dim=-1)
        x = self.dropout(x)
        out, hidden = self.lstm(x, hidden)
        out = self.dropout(out)
        return self.out(out), hidden

    def init_hidden(self, batch_size=1):
        h = torch.zeros(1, batch_size, self.lstm_hidden, device=device)
        c = torch.zeros(1, batch_size, self.lstm_hidden, device=device)
        return (h, c)

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if "weight" in name:
                        nn.init.kaiming_normal_(param)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.1)


### DDRQN (Foerster et al., 2016): action history and observation history are processed by
### separate LSTMs (decoupled). Their outputs are concatenated to produce Q values.
###
###   lstm_item : embed(a_0), embed(a_1), ..., embed(a_{t-1})   — item (action) stream
###   lstm_resp : o_0, o_1, ..., o_{t-1}                        — response (observation) stream
###
### hidden state is a nested tuple: ((h_item, c_item), (h_resp, c_resp))
class DDRQN(nn.Module):
    def __init__(self, action_space, embed_dim, lstm_hidden, dropout_rate):
        super(DDRQN, self).__init__()
        # action_space + 1 embeddings: indices 0..action_space-1 for items, action_space for START
        self.embed = nn.Embedding(action_space + 1, embed_dim)
        self.lstm_item = nn.LSTM(embed_dim, lstm_hidden, batch_first=True)
        self.lstm_resp = nn.LSTM(1, lstm_hidden, batch_first=True)
        self.out = nn.Linear(lstm_hidden * 2, action_space)
        self.dropout = nn.Dropout(dropout_rate)
        self.action_space = action_space
        self.embed_dim = embed_dim
        self.lstm_hidden = lstm_hidden

    def forward(self, items, resps, hidden=None):
        # items: (batch, seq_len) long
        # resps: (batch, seq_len) long
        # hidden: ((h_item, c_item), (h_resp, c_resp)) or None
        if hidden is None:
            hidden_item, hidden_resp = None, None
        else:
            hidden_item, hidden_resp = hidden

        e = self.dropout(self.embed(items))
        r = resps.unsqueeze(-1).float()

        out_item, hidden_item = self.lstm_item(e, hidden_item)
        out_resp, hidden_resp = self.lstm_resp(r, hidden_resp)

        out = torch.cat([self.dropout(out_item), self.dropout(out_resp)], dim=-1)
        return self.out(out), (hidden_item, hidden_resp)

    def init_hidden(self, batch_size=1):
        zeros = lambda: torch.zeros(1, batch_size, self.lstm_hidden, device=device)
        return ((zeros(), zeros()), (zeros(), zeros()))

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if "weight" in name:
                        nn.init.kaiming_normal_(param)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.1)


### ADRQN_Params: ADRQN-style but with IRT parameters (a, b, c) instead of item ID embedding.
### At each step t the LSTM receives [a_{t-1}, b_{t-1}, c_{t-1}, r_{t-1}] (4-dim float).
### START token is the zero vector [0, 0, 0, 0].
### Single LSTM processes item-response pairs jointly, allowing the network to directly learn
### the IRT relationship between item difficulty and response.
class ADRQN_Params(nn.Module):
    def __init__(self, action_space, lstm_hidden, dropout_rate):
        super(ADRQN_Params, self).__init__()
        self.lstm = nn.LSTM(4, lstm_hidden, batch_first=True)  # 3 IRT params + 1 response
        self.out = nn.Linear(lstm_hidden, action_space)
        self.dropout = nn.Dropout(dropout_rate)
        self.action_space = action_space
        self.lstm_hidden = lstm_hidden

    def forward(self, item_params, resps, hidden=None):
        # item_params: (batch, seq_len, 3) float — (a, b, c); zeros for START
        # resps:       (batch, seq_len)    long  — 0/1 responses; 0 for START
        r = resps.unsqueeze(-1).float()
        x = torch.cat([item_params.float(), r], dim=-1)  # (batch, seq_len, 4)
        x = self.dropout(x)
        out, hidden = self.lstm(x, hidden)
        out = self.dropout(out)
        return self.out(out), hidden

    def init_hidden(self, batch_size=1):
        h = torch.zeros(1, batch_size, self.lstm_hidden, device=device)
        c = torch.zeros(1, batch_size, self.lstm_hidden, device=device)
        return (h, c)

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if "weight" in name:
                        nn.init.kaiming_normal_(param)


### DDRQN_Params: item stream uses IRT parameters (a, b, c) directly instead of ID embeddings.
###
###   lstm_item : [a_0,b_0,c_0], [a_1,b_1,c_1], ...   — item parameter stream (3-dim float)
###   lstm_resp : o_0, o_1, ...                         — response (observation) stream
###
### START token is represented by a zero vector [0, 0, 0].
### hidden state is a nested tuple: ((h_item, c_item), (h_resp, c_resp))
class DDRQN_Params(nn.Module):
    def __init__(self, action_space, lstm_hidden, dropout_rate):
        super(DDRQN_Params, self).__init__()
        self.lstm_item = nn.LSTM(3, lstm_hidden, batch_first=True)
        self.lstm_resp = nn.LSTM(1, lstm_hidden, batch_first=True)
        self.out = nn.Linear(lstm_hidden * 2, action_space)
        self.dropout = nn.Dropout(dropout_rate)
        self.action_space = action_space
        self.lstm_hidden = lstm_hidden

    def forward(self, item_params, resps, hidden=None):
        # item_params: (batch, seq_len, 3) float — (a, b, c) per step; zeros for START
        # resps:       (batch, seq_len)    long  — 0/1 responses; 0 for START
        # hidden: ((h_item, c_item), (h_resp, c_resp)) or None
        if hidden is None:
            hidden_item, hidden_resp = None, None
        else:
            hidden_item, hidden_resp = hidden

        p = self.dropout(item_params.float())
        r = resps.unsqueeze(-1).float()

        out_item, hidden_item = self.lstm_item(p, hidden_item)
        out_resp, hidden_resp = self.lstm_resp(r, hidden_resp)

        out = torch.cat([self.dropout(out_item), self.dropout(out_resp)], dim=-1)
        return self.out(out), (hidden_item, hidden_resp)

    def init_hidden(self, batch_size=1):
        zeros = lambda: torch.zeros(1, batch_size, self.lstm_hidden, device=device)
        return ((zeros(), zeros()), (zeros(), zeros()))

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if "weight" in name:
                        nn.init.kaiming_normal_(param)
