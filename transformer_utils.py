import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, input_dim, query_dim, key_dim, value_dim):
        super(SelfAttention, self).__init__()
        assert (query_dim == key_dim)
        self.query_dim = query_dim
        self.input_dim = input_dim

        self.W_query = nn.Linear(input_dim, query_dim)
        self.W_key = nn.Linear(input_dim, key_dim)
        self.W_value = nn.Linear(input_dim, value_dim)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        scaling_factor = torch.sqrt(torch.tensor(self.query_dim, dtype=torch.float32))
        output = torch.bmm(self.softmax(torch.bmm(queries, keys.transpose(1, 2)) / scaling_factor), values)

        return output


class LayerNorm(nn.Module):
    def __init__(self, input_dim, eps=1e-5):
        super().__init__()
        assert isinstance(input_dim, int)

        self.input_dim = input_dim
        self.eps = eps

        self.w = nn.Parameter(torch.ones(input_dim))
        self.b = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x: torch.Tensor):
        assert x.shape[-1] == self.input_dim

        normalized_tensor = (x - (x.mean(dim=-1, keepdim=True))) / torch.sqrt((x.var(dim=-1, keepdim=True, unbiased=True)) + self.eps)
        return self.w.view(1, 1, -1) * normalized_tensor + self.b.view(1, 1, -1)

