from transformers import BertModel
from ark_nlp.nn.base.bert import BertForTokenClassification
from ark_nlp.nn.layer.global_pointer_block import GlobalPointer, EfficientGlobalPointer
from ark_nlp.nn.base.nezha import NeZhaModel
from ark_nlp.nn.configuration.configuration_nezha import NeZhaConfig
import torch

class GlobalPointerBert(BertForTokenClassification):
    """
    GlobalPointer + Bert 的命名实体模型

    Args:
        config: 模型的配置对象
        bert_trained (:obj:`bool`, optional): 预训练模型的参数是否可训练

    Reference:
        [1] https://www.kexue.fm/archives/8373
    """  # noqa: ignore flake8"

    def __init__(
        self,
        config,
        encoder_trained=True,
        head_size=64
    ):
        super(GlobalPointerBert, self).__init__(config)

        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        #print('===== NeZhaModel =====')
        #self.bert = NeZhaModel(config)

        for param in self.bert.parameters():
            param.requires_grad = encoder_trained

        word_enhance_dim=4
        embedding_dim=300
        self.global_pointer = GlobalPointer(
            self.num_labels,
            head_size,
            config.hidden_size + word_enhance_dim*embedding_dim
        )

        self.init_weights()

        #soft_lexicon
        import pickle
        embedding_cache_path = '/kaggle/working/vocab_data/lexicon_embeddings.txt'
        #embedding_cache_path = '/opt/kelvin/python/knowledge_graph/ai_contest/gaiic2022/baseline/baseline/vocab_data/lexicon_embeddings.txt'
        lexicon_embeddings = pickle.load(open(embedding_cache_path, 'rb'))
        #np_emb = torch.from_numpy(lexicon_embeddings)
        self.lexicon_embedding_layer = torch.nn.Embedding.from_pretrained(torch.from_numpy(lexicon_embeddings))
        print('===== soft lexicon init done =====')

    def soft_lexicon(self, ids, weights, sequence_output, max_seq_len=128, word_enhance_dim=4, max_lexicon_len=4, embedding_dim=300):
        ids_tensor = ids #torch.tensor([ids], dtype=torch.long)
        weights_tensor = weights #torch.tensor([weights], dtype=torch.long)
        emb_tensor = self.lexicon_embedding_layer(ids_tensor)
        weight_ep_tensor = torch.unsqueeze(weights_tensor, dim=-1)
        wh_embedding = emb_tensor * weight_ep_tensor
        wh_embedding = torch.reshape(wh_embedding, (-1, max_seq_len, word_enhance_dim, max_lexicon_len, embedding_dim))
        wh_embedding = torch.sum(wh_embedding, dim=3)
        wh_embedding = torch.reshape(wh_embedding, (-1, max_seq_len, int(word_enhance_dim * embedding_dim)))
         
        dropout = torch.nn.Dropout(p=0.5)
        wh_embedding = dropout(wh_embedding)
        lexicon_output = torch.cat([wh_embedding, sequence_output], dim=-1)
        return lexicon_output

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        **kwargs
    ):
        #print('====================================')
        soft_ids = kwargs['soft_ids']
        soft_weights = kwargs['soft_weights']
        #print('input_ids.shape: {}, soft_ids.shape: {}, soft_weights.shape: {}'.format(input_ids.shape, soft_ids.shape, soft_weights.shape))
        #print('input_ids: ', input_ids[0])
        #print('soft_ids: ', soft_ids[0])
        #print('soft_weights: ', soft_weights[0])
        #import sys
        #sys.exit(0)
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
            output_hidden_states=True
        ).hidden_states

        sequence_output = outputs[-1]
        '''
        #nezha
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        sequence_output = outputs[0]
        '''

        lexicon_output = self.soft_lexicon(soft_ids, soft_weights, sequence_output)
        lexicon_output = lexicon_output.to(torch.float32)
        #logits = self.global_pointer(sequence_output, mask=attention_mask)
        logits = self.global_pointer(lexicon_output, mask=attention_mask)

        return logits

class EfficientGlobalPointerBert(BertForTokenClassification):
    """
    EfficientGlobalPointer + Bert 的命名实体模型

    Args:
        config: 模型的配置对象
        bert_trained (:obj:`bool`, optional): 预训练模型的参数是否可训练

    Reference:
        [1] https://www.kexue.fm/archives/8877
        [2] https://github.com/powerycy/Efficient-GlobalPointer
    """  # noqa: ignore flake8"

    def __init__(
        self,
        config,
        encoder_trained=True,
        head_size=64
    ):
        super(EfficientGlobalPointerBert, self).__init__(config)

        self.num_labels = config.num_labels

        self.bert = BertModel(config)

        for param in self.bert.parameters():
            param.requires_grad = encoder_trained

        self.efficient_global_pointer = EfficientGlobalPointer(
            self.num_labels,
            head_size,
            config.hidden_size
        )

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        **kwargs
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
            output_hidden_states=True
        ).hidden_states

        sequence_output = outputs[-1]

        logits = self.efficient_global_pointer(sequence_output, mask=attention_mask)

        return logits
