import torch.nn as nn
import torch
import math
import copy

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class SelfAttention(nn.Module):
    def __init__(self, args):
        super(SelfAttention, self).__init__()
        if args.hidden_sz % args.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (args.hidden_sz, args.num_attention_heads))
        self.num_attention_heads = args.num_attention_heads
        self.attention_head_size = int(args.hidden_sz / args.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(args.hidden_sz, self.all_head_size)
        self.key = nn.Linear(args.hidden_sz, self.all_head_size)
        self.value = nn.Linear(args.hidden_sz, self.all_head_size)

        self.dropout = nn.Dropout(args.dropout)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, query, key, value):
        mixed_query_layer = self.query(query)
        mixed_key_layer = self.key(key)
        mixed_value_layer = self.value(value)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer



class SelfOutput(nn.Module):
    def __init__(self, args):
        super(SelfOutput, self).__init__()
        self.dense = nn.Linear(args.hidden_sz, args.hidden_sz)
        self.LayerNorm = LayerNorm(args.hidden_sz, eps=1e-12)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class Attention(nn.Module):
    def __init__(self, args):
        super(Attention, self).__init__()

        self.self = SelfAttention(args)
        self.output = SelfOutput(args)

    def forward(self, query, key, value):
        self_output = self.self(query, key, value)
        attention_output = self.output(self_output, query)
        return attention_output


class Intermediate(nn.Module):
    def __init__(self, args):
        super(Intermediate, self).__init__()
        self.dense = nn.Linear(args.hidden_sz, args.intermediate_size)
        self.intermediate_act_fn = nn.GELU()

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class Output(nn.Module):
    def __init__(self, args):
        super(Output, self).__init__()
        self.dense = nn.Linear(args.intermediate_size, args.hidden_sz)
        self.LayerNorm = LayerNorm(args.hidden_sz, eps=1e-12)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class Layer(nn.Module):
    def __init__(self, args):
        super(Layer, self).__init__()
        self.attention = Attention(args)
        self.intermediate = Intermediate(args)
        self.output = Output(args)

    def forward(self, query, key, value):
        attention_output = self.attention(query, key, value)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class Mlp(nn.Module):
    def __init__(self, in_channel, mlp_channel):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(in_channel, mlp_channel)
        self.fc2 = nn.Linear(mlp_channel, in_channel)
        self.act_fn = nn.GELU()
        self.dropout = nn.Dropout(0.2)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Crosslayer_stage1(nn.Module):
    def __init__(self, args):
        super(Crosslayer_stage1, self).__init__()

        self.shared_to_ehr = Layer(args)
        self.shared_to_cxr = Layer(args)
        self.ehr = Layer(args)
        self.cxr = Layer(args)

    def forward(self, shared_feat, ehr_feat, cxr_feat):
        shared_ehr = self.shared_to_ehr(query=shared_feat, key=ehr_feat, value=ehr_feat)
        shared_cxr = self.shared_to_cxr(query=shared_feat, key=cxr_feat, value=cxr_feat)
        ehr_feat = self.ehr(query=shared_ehr, key=shared_ehr, value=shared_ehr)
        cxr_feat = self.cxr(query=shared_cxr, key=shared_cxr, value=shared_cxr)

        return ehr_feat, cxr_feat

class Crosslayer_stage2(nn.Module):
    def __init__(self, args):
        super(Crosslayer_stage2, self).__init__()

        self.shared_to_ehr = Layer(args)
        self.shared_to_cxr = Layer(args)
        self.ehr = Layer(args)
        self.cxr = Layer(args)

    def forward(self, ehr_feat, cxr_feat):
        ehr_feat = self.shared_to_ehr(query=ehr_feat, key=cxr_feat, value=cxr_feat)
        cxr_feat = self.shared_to_cxr(query=cxr_feat, key=ehr_feat, value=ehr_feat)
        ehr_feat = self.ehr(query=ehr_feat, key=ehr_feat, value=ehr_feat)
        cxr_feat = self.cxr(query=cxr_feat, key=cxr_feat, value=cxr_feat)

        return ehr_feat, cxr_feat

class CrossFusion(nn.Module):
    def __init__(self, args):
        super(CrossFusion, self).__init__()
        self.layer_stage1 = Crosslayer_stage1(args)
        self.layer_stage2 = Crosslayer_stage2(args)

        self.fusion = nn.Sequential(
            nn.Linear(args.hidden_sz * 2, args.hidden_sz),
            nn.ReLU(),
            nn.Linear(args.hidden_sz, args.hidden_sz)
        )

        self.classifier = nn.Sequential(
            nn.Linear(args.hidden_sz, args.num_classes),
            nn.Sigmoid()
        )

        # self.ehr_classifier = nn.Sequential(
        #     nn.Linear(args.hidden_sz, args.hidden_sz),
        #     nn.ReLU(),
        #     nn.Linear(args.hidden_sz, args.num_classes),
        #     nn.Sigmoid()
        # )
        #
        # self.cxr_classifier = nn.Sequential(
        #     nn.Linear(args.hidden_sz, args.hidden_sz),
        #     nn.ReLU(),
        #     nn.Linear(args.hidden_sz, args.num_classes),
        #     nn.Sigmoid()
        # )


    def forward(self, shared_feat, ehr_feat, cxr_feat):
        ehr_feat, cxr_feat = self.layer_stage1(shared_feat, ehr_feat, cxr_feat)
        ehr_feat, cxr_feat = self.layer_stage2(ehr_feat, cxr_feat)

        IB_ehr_feat = ehr_feat.mean(1)
        IB_cxr_feat = cxr_feat.mean(1)

        fusion_feat = torch.cat([ehr_feat, cxr_feat], dim=2)
        fusion_feat = self.fusion(fusion_feat)
        fusion_feat = fusion_feat.mean(1)
        pred = self.classifier(fusion_feat)

        # ehr_pred = self.ehr_classifier(IB_ehr_feat)
        # cxr_pred = self.cxr_classifier(IB_cxr_feat)

        # return pred, ehr_pred, cxr_pred, fusion_feat, IB_ehr_feat, IB_cxr_feat
        return pred, fusion_feat, IB_ehr_feat, IB_cxr_feat

# if __name__ == '__main__':
#     import argparse
#     import numpy as np
#     parser = argparse.ArgumentParser(description='arguments')
#     parser.add_argument('--hidden_sz', default=256, type=int)
#     parser.add_argument('--num_attention_heads', type=int, default=8)
#     parser.add_argument('--intermediate_size', default=256, type=int)
#     parser.add_argument('--num_classes', default=25, type=int)
#     parser.add_argument('--dropout', type=float, default=0.1)
#     args = parser.parse_args()
#
#     model = CrossFusion(args)
#     shared = torch.rand((2, 50, 256))
#     ehr = torch.rand((2, 1, 256))
#     cxr = torch.rand((2, 49, 256))
#
#     pred, fusion_feat, IB_ehr_feat, IB_cxr_feat = model(shared, ehr, cxr)
#     print(pred.shape)
#     print(fusion_feat.shape)
#     print(IB_ehr_feat.shape)
#     print(IB_cxr_feat.shape)