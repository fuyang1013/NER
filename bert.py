"""

"""
###############################################################################
# Bert的精简实现，帮助深入理解模型
# 功能特点：
# 1. 轻量：基于Pytorch实现，移除对transformers的依赖
# 2. 兼容性：可以和transformers pytorch model进行逐层参数拷贝
# 备忘：
# 1. 在前期实现阶段，为了更方便和transformer版本的BERT作对比，所有的命名尽量和
# transformers中的名称错开
###############################################################################
import math
import pdb
import torch
import torch.nn as nn
from torch.nn.modules.loss import CrossEntropyLoss
from log import logger


class BertParam:
    vocab_size = 21128  # 词典大小
    pad_token_id = 0
    type_vocab_size = 2  # 句子类型的embedding词典大小，token_type_id有2种

    dropout_prob = 0.1  # 目前所有dropout层的dropout rate都一样
    layer_norm_eps = 1e-12  # layer normalization
    initializer_range = 0.02

    num_hidden_layers = 12  # 总共12层transformer encoder
    num_attention_heads = 12  # 总共12个注意力head

    hidden_size = 768  # 嵌入
    intermediate_size = 3072
    max_position_embeddings = 512

    pooler_num_fc_layers = 3
    pooler_size_per_head = 128


class BertEncoderLayer(nn.Module):
    """
    Bert中的Transformer Encoder层
    """

    def __init__(self, config: BertParam):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError('num_attention_heads必须整除hidden size!')

        """ 
        BertAttention中的BertSelfAttention 
        """
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // self.num_attention_heads
        # 所有head的head_size总和
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.self_attention_dropout = nn.Dropout(config.dropout_prob)

        """ 
        BertAttention中的BertSelfOutput
        """
        self.self_output_dense = nn.Linear(
            config.hidden_size, config.hidden_size)
        self.self_output_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.self_output_dropout = nn.Dropout(config.dropout_prob)

        """
        BertIntermediate
        """
        self.intermediate_dense = nn.Linear(
            config.hidden_size, config.intermediate_size)

        """
        BertOutput
        """
        self.output_dense = nn.Linear(
            config.intermediate_size, config.hidden_size)
        self.output_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.output_dropout = nn.Dropout(config.dropout_prob)

    def transpose_for_scores(self, x):
        # [B, L] + [12, 64] => [B, L, 12, 64]
        # new_x_shape = x.size()[
        #     :-1] + (self.num_attention_heads, self.attention_head_size)
        # x = x.view(*new_x_shape)
        x = x.reshape([x.size(0), x.size(1), self.num_attention_heads, self.attention_head_size])
        return x.permute(0, 2, 1, 3)  # [B, 12, L, 64]

    def forward(self, hidden_states, attention_mask):
        """
        暂时不考虑head_mask，不考虑修剪heads，也不返回past_key_value
        """

        """
        SelfAttention 自注意力模块
        """

        # [B, L, hidden_size] -> [B, L, all_head_size]
        #                     -> [B, num_attention_heads, L, head_size]
        query_output = self.query(hidden_states)
        query_layer = self.transpose_for_scores(query_output)
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # [B, num_heads, L, L]
        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / \
            math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # 将注意力分数归一化成概率值，[B, 12, L, L]
        attention_probs = torch.softmax(attention_scores, dim=-1)
        # （译）可能会有点奇怪，但是原始的Transformer论文就是这么干的
        attention_probs = self.self_attention_dropout(attention_probs)

        # [B, 12, L, 64]
        context_layer = torch.matmul(attention_probs, value_layer)
        # [B, 12, L, 64] => [B, L, 12, 64]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # [B, L] + [768] => [B, L, 768] 
        new_context_layer_shape = context_layer.size()[
            :-2] + (self.all_head_size,)
        # [B, L, 768]
        # context_layer = context_layer.view(*new_context_layer_shape)
        context_layer = context_layer.reshape(new_context_layer_shape)

        """
        BertSelfOutput
        """
        context_layer = self.self_output_dense(context_layer)
        context_layer = self.self_output_dropout(context_layer)
        attention_output = self.self_output_layer_norm(
            context_layer + hidden_states)

        """
        BertIntermediate
        原始的transformers用了chunk机制，但是实际上，config.chunk_size = 0，等价于没用
        """
        intermediate_output = self.intermediate_dense(attention_output)
        intermediate_output = nn.functional.gelu(intermediate_output)

        """
        BertOutput
        """
        layer_output = self.output_dense(intermediate_output)
        layer_output = self.output_dropout(layer_output)
        layer_output = self.output_layer_norm(layer_output + attention_output)

        return layer_output


class Bert(nn.Module):
    """
    大致分为三个部分:
    1. embedding层: 由三个embeddings相加后，经过layernorm和dropout
    2. encoder层:
        i. 自注意力层: 包含query/key/value
        ii. 前向传播层
    3. pooler层
    """
    def __init__(self, config: BertParam):
        super().__init__()
        self.config = config
        # input embeddings
        self.token_embeddings = nn.Embedding(
            config.vocab_size,  config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout_prob)
        # encoder
        self.encoder = nn.ModuleList(BertEncoderLayer(
            config) for _ in range(config.num_hidden_layers))
        # pooler
        self.pooler_dense = nn.Linear(config.hidden_size, config.hidden_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_ids, attention_mask, token_type_ids, position_ids):
        """
        input_ids和attention_mask不能为空
        token_type_ids = torch.zeros_like(input_ids, dtype=torch.long)
        position_ids = torch.arange(args.eval_max_seq_len).unsqueeze(dim=0).repeat([input_ids.size(0), 1])
        """

        token_emb = self.token_embeddings(input_ids)
        token_type_emb = self.token_type_embeddings(token_type_ids)
        position_emb = self.position_embeddings(position_ids)

        embeddings = token_emb + token_type_emb + position_emb
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        # transformers中最底层的计算
        attention_mask = (1.0 - attention_mask) * -10000.0  # [B, L]
        attention_mask = attention_mask.unsqueeze(
            dim=1).unsqueeze(dim=1)  # [B, 1, 1, L]

        hidden_states = embeddings
        for layer_module in self.encoder:
            layer_outputs = layer_module(hidden_states, attention_mask)
            hidden_states = layer_outputs

        pooler_output = self.pooler_dense(hidden_states[:, 0])  # 把[CLS]处的token输入到pooler中去
        pooler_output = torch.tanh(pooler_output)

        return {'last_hidden_state': hidden_states, 'pooler': pooler_output}

    def load_params(self, src):
        self.token_embeddings.load_state_dict(
            src.embeddings.word_embeddings.state_dict())
        self.position_embeddings.load_state_dict(
            src.embeddings.position_embeddings.state_dict())
        self.token_type_embeddings.load_state_dict(
            src.embeddings.token_type_embeddings.state_dict())
        self.LayerNorm.load_state_dict(src.embeddings.LayerNorm.state_dict())
        self.dropout.load_state_dict(src.embeddings.dropout.state_dict())
        self.pooler_dense.load_state_dict(src.pooler.dense.state_dict())

        for i in range(12):
            src_layer = src.encoder.layer[i]
            dst_layer = self.encoder[i]

            dst_layer.query.load_state_dict(
                src_layer.attention.self.query.state_dict())
            dst_layer.key.load_state_dict(
                src_layer.attention.self.key.state_dict())
            dst_layer.value.load_state_dict(
                src_layer.attention.self.value.state_dict())
            dst_layer.self_attention_dropout.load_state_dict(
                src_layer.attention.self.dropout.state_dict())
            dst_layer.self_output_dense.load_state_dict(
                src_layer.attention.output.dense.state_dict())
            dst_layer.self_output_layer_norm.load_state_dict(
                src_layer.attention.output.LayerNorm.state_dict())
            dst_layer.self_output_dropout.load_state_dict(
                src_layer.attention.output.dropout.state_dict())
            dst_layer.intermediate_dense.load_state_dict(
                src_layer.intermediate.dense.state_dict())
            dst_layer.output_dense.load_state_dict(
                src_layer.output.dense.state_dict())
            dst_layer.output_layer_norm.load_state_dict(
                src_layer.output.LayerNorm.state_dict())
            dst_layer.output_dropout.load_state_dict(
                src_layer.output.dropout.state_dict())


class BertPretrainModel(nn.Module):
    def __init__(self, config:BertParam) -> None:
        super().__init__()
        self.vocab_size = config.vocab_size
        self.bert = Bert(config)
        """
        BertPredictionHeadTransform
        """
        self.transform_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        """
        token prediction decoder
        """
        self.mlm_dense = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.mlm_dense_bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.mlm_dense.bias = self.mlm_dense_bias

        """
        pooler prediction
        """
        self.seq_relationship = nn.Linear(config.hidden_size, 2)
        
        self.apply(self._init_weights)
    
    def forward(self, input_ids, attention_mask, token_type_ids, position_ids):
        sequence_output, pooler_output = self.bert(input_ids, attention_mask, token_type_ids, position_ids)

        # [B, L, D] -> [B, L, D]
        hidden_states = self.transform_dense(sequence_output)
        hidden_states = nn.functional.gelu(hidden_states)
        hidden_states = self.transform_layernorm(hidden_states)

        # [B, L, D] -> [B, L, V]
        mlm_logits = self.mlm_dense(hidden_states)

        # [B, D] -> [B, 2]
        nsp_logits = self.seq_relationship(pooler_output)

        return mlm_logits, nsp_logits


    def loss_fn(self, mlm_logits, nsp_logits, labels, next_sentence_label, loss_fn=None):
        """
        labels: id，取值为{-100, 0, 1, ..., vocab_size-1}，-100会被忽视，其余id（被mask的token）会被取出算loss
        next_sentence_label: 0表示为下一句，1表示为随机句子
        """
        loss_fn = nn.CrossEntropyLoss()
        mlm_loss = loss_fn(mlm_logits.view(-1, self.vocab_size), labels.view(-1))
        nsp_loss = loss_fn(nsp_logits.view(-1, 2), next_sentence_label.view(-1))
        total_loss = mlm_loss + nsp_loss
        return total_loss

    def load_params(self, src):
        self.transform_dense.load_state_dict(src.cls.predictions.transform.dense.state_dict())
        self.transform_layernorm.load_state_dict(src.cls.predictions.transform.LayerNorm.state_dict())
        self.mlm_dense.load_state_dict(src.cls.predictions.decoder.state_dict())
        self.seq_relationship.load_state_dict(src.cls.seq_relationship.state_dict())
        self.bert.load_params(src.bert)


def test_bert():
    from transformers import BertModel, BertConfig, BertTokenizerFast

    torch.manual_seed(42)
    tokenizer = BertTokenizerFast.from_pretrained('../bert-base-chinese')
    config = BertConfig.from_pretrained('../bert-base-chinese')
    model = BertModel.from_pretrained('../bert-base-chinese')

    # print(model)
    # assert 0

    # print(config.hidden_act)
    # assert 0


    bert = Bert(BertParam())
    # bert.load_params(model)

    # print(bert)
    # print(model)
    # assert 0

    # bert: 102267648
    total_params = 0
    for k, v in bert.named_parameters():
        print(k, v.size())
        total_params += v.numel()
    print(total_params)
    assert 0

    bert.eval()
    # model.eval()

    sentences = ['我爱中国', '你是谁']
    be = tokenizer.batch_encode_plus(
        sentences, max_length=128, truncation=True, padding='max_length')
    # print(f'be:{be}')

    input_ids = torch.LongTensor(be['input_ids'])
    attention_mask = torch.LongTensor(be['attention_mask'])

    # outputs = model(input_ids,
    #                 attention_mask=attention_mask,
    #                 return_dict=True,
    #                 output_hidden_states=True,
    #                 output_attentions=True)

    # print(outputs.keys())

    """
    返回注意力
    outputs['attentions']返回一个长度为12的元组，每个元素为[batch_size, num_heads, max_seq_len, max_seq_len]的张量
    """
    # print(outputs['last_hidden_state'])
    # print(outputs['pooler_output'])
    logger.info('start...')
    last_hidden_states, pooler_output = bert(input_ids, attention_mask=attention_mask)
    logger.info('end...')
    # print(last_hidden_states)
    # print(pooler_output)


def test_bert_pretrain():
    from transformers import BertForPreTraining, BertTokenizerFast, BertConfig
    
    # config = BertConfig.from_pretrained('../bert-base-chinese')
    # model = BertForPreTraining(config=config)
    # for k, v in model.named_parameters():
    #     print(k, v.size())
    # assert 0

    model = BertForPreTraining.from_pretrained('../bert-base-chinese')
    # model = BertForPreTraining(config=config)
    for k, v in model.named_parameters():
        print(k, v.size())
    assert 0


    tokenizer = BertTokenizerFast.from_pretrained('../bert-base-chinese')
    model.eval()
    # print(model.cls.predictions.transform.dense)
    text = '为了避免工作猝死，我们最好不要卷。'
    res = tokenizer.batch_encode_plus([text])
    
    input_ids = res['input_ids']
    input_ids = torch.LongTensor(input_ids)
    mask_word = '卷。'
    start = text.index(mask_word)
    for i in range(start+1, start+1+len(mask_word)):
        input_ids[0, i] = 103
    
    print('input:', tokenizer.decode(input_ids[0]))

    # input_ids = torch.LongTensor([[101, 704, 1744, 4638, 7674, 6963, 3221, 1266, 776, 102]])
    token_type_ids = torch.LongTensor(res['token_type_ids'])
    attention_mask = torch.LongTensor(res['attention_mask'])
    position_ids = torch.arange(input_ids.size(1), dtype=torch.long).reshape(
        [1, input_ids.size(1)]).repeat([input_ids.size(0), 1])
    
    print(position_ids.size())

    f = lambda x: tokenizer.decode(result['prediction_logits'][0][x].argmax())
    with torch.no_grad():
        result = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids)
        print('output:', tokenizer.decode(result['prediction_logits'][0].argmax(dim=-1)[1:-1]))

    # modell = BertPretrainModel(BertParam())
    # modell.load_params(model)


# test_bert()
test_bert_pretrain()