import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from paddlenlp.transformers import PretrainedModel, register_base_model
from paddlenlp.transformers.model_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    SequenceClassifierOutput,
    TokenClassifierOutput,
    MultipleChoiceModelOutput,
    MaskedLMOutput,
    CausalLMOutputWithCrossAttentions,
    ModelOutput,
)
from paddlenlp.transformers import RobertaPretrainedModel

@dataclass # 输出的数据结构
class QuestionAnsweringModelOutput(ModelOutput):
    """
    Base class for outputs of question answering models.

    Args:
        loss (`paddle.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        start_logits (`paddle.Tensor` of shape `(batch_size, sequence_length)`):
            Span-start scores (before SoftMax).
        end_logits (`paddle.Tensor` of shape `(batch_size, sequence_length)`):
            Span-end scores (before SoftMax).
        hidden_states (`tuple(paddle.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `paddle.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(paddle.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `paddle.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[paddle.Tensor] = None
    start_logits: paddle.Tensor = None
    end_logits: paddle.Tensor = None
    span_logits: paddle.Tensor = None # 添加一个span_logits
    hidden_states: Optional[Tuple[paddle.Tensor]] = None
    attentions: Optional[Tuple[paddle.Tensor]] = None

class biaffine(nn.Layer):
    def __init__(self, in_size, out_size, bias_x=True, bias_y=True):
        super().__init__()
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.out_size = out_size
        x = paddle.randn([in_size + int(bias_x),out_size,in_size + int(bias_y)])
        self.U = paddle.create_parameter(shape=x.shape,
                        dtype=str(x.numpy().dtype),
                        default_initializer=paddle.nn.initializer.Assign(x))
        # self.U1 = self.U.view(size=(in_size + int(bias_x),-1))
        #U.shape = [in_size,out_size,in_size]  
    def forward(self, x, y):
        if self.bias_x:
            x = paddle.concat([x, paddle.ones_like(x[..., :1])], axis=-1)
        if self.bias_y:
            y = paddle.concat([y, paddle.ones_like(y[..., :1])], axis=-1)
        
        """
        batch_size,seq_len,hidden=x.shape
        bilinar_mapping=paddle.matmul(x,self.U)
        bilinar_mapping=bilinar_mapping.view(size=(batch_size,seq_len*self.out_size,hidden))
        y=paddle.transpose(y,dim0=1,dim1=2)
        bilinar_mapping=paddle.matmul(bilinar_mapping,y)
        bilinar_mapping=bilinar_mapping.view(size=(batch_size,seq_len,self.out_size,seq_len))
        bilinar_mapping=paddle.transpose(bilinar_mapping,dim0=2,dim1=3)
        """
        bilinar_mapping = paddle.einsum('bxi,ioj,byj->bxyo', x, self.U, y)
        return bilinar_mapping

class RobertaForQuestionAnswering(RobertaPretrainedModel):
    r"""
    Roberta Model with a linear layer on top of the hidden-states output to compute `span_start_logits`
     and `span_end_logits`, designed for question-answering tasks like SQuAD.

    Args:
        roberta (:class:`RobertaModel`):
            An instance of RobertaModel.
    """

    def __init__(self, roberta):
        super(RobertaForQuestionAnswering, self).__init__()
        self.roberta = roberta  # allow roberta to be config
        
        hidden_size = self.roberta.config["hidden_size"]
        
        self.start_layer = paddle.nn.Sequential(paddle.nn.Linear(in_features=hidden_size, out_features=128),
                                            paddle.nn.ReLU())
        self.end_layer = paddle.nn.Sequential(paddle.nn.Linear(in_features=hidden_size, out_features=128),
                                            paddle.nn.ReLU())
        self.biaffine_layer = biaffine(128,1)
        
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None,
                start_positions=None,
                end_positions=None,
                output_hidden_states=False,
                output_attentions=False,
                return_dict=False):
        r"""
        Args:
            input_ids (Tensor):
                See :class:`RobertaModel`.
            token_type_ids (Tensor, optional):
                See :class:`RobertaModel`.
            position_ids (Tensor, optional):
                See :class:`RobertaModel`.
            attention_mask (Tensor, optional):
                See :class:`RobertaModel`.
            start_positions (Tensor of shape `(batch_size,)`, optional):
                Labels for position (index) of the start of the labelled span for computing the token classification loss.
                Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
                are not taken into account for computing the loss.
            end_positions (Tensor of shape `(batch_size,)`, optional):
                Labels for position (index) of the end of the labelled span for computing the token classification loss.
                Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
                are not taken into account for computing the loss.
            output_hidden_states (bool, optional):
                Whether to return the hidden states of all layers.
                Defaults to `False`.
            output_attentions (bool, optional):
                Whether to return the attentions tensors of all attention layers.
                Defaults to `False`.
            return_dict (bool, optional):
                Whether to return a :class:`~paddlenlp.transformers.model_outputs.QuestionAnsweringModelOutput` object. If
                `False`, the output will be a tuple of tensors. Defaults to `False`.

        Returns:
            An instance of :class:`~paddlenlp.transformers.model_outputs.QuestionAnsweringModelOutput` if `return_dict=True`.
            Otherwise it returns a tuple of tensors corresponding to ordered and
            not None (depending on the input arguments) fields of :class:`~paddlenlp.transformers.model_outputs.QuestionAnsweringModelOutput`.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import RobertaForSequenceClassification, RobertaTokenizer

                tokenizer = RobertaTokenizer.from_pretrained('roberta-wwm-ext')
                model = RobertaForSequenceClassification.from_pretrained('roberta-wwm-ext')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                logits = model(**inputs)

        """
        outputs = self.roberta(input_ids,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               attention_mask=attention_mask,
                               output_attentions=output_attentions,
                               output_hidden_states=output_hidden_states,
                               return_dict=return_dict)

        sequence_output = outputs[0] # 首先经过bert编码层
        
        #TODO 加LSTM层和不加有什么影响？
        
        start_logits = self.start_layer(sequence_output) # shape [6,512,128]
        end_logits = self.end_layer(sequence_output) 

        span_logits = self.biaffine_layer(start_logits,end_logits)
        span_logits = span_logits.contiguous()
        # span_logits = self.relu(span_logits)
        # span_logits = self.logits_layer(span_logits)

        # span_prob = paddle.nn.functional.sigmoid(span_logits)
        # 计算损失时用logits就可以


        total_loss = None
        # 计算损失 [INFO] span_prob shape is [6, 512, 512, 1], start_positions shape is [6]
        span_logits=span_logits.squeeze(-1) # [6, 512, 512]
        span_logits=span_logits.reshape([span_logits.shape[0],-1])
        
        if start_positions is not None and end_positions is not None: # 计算损失
            # If we are on multi-GPU, split add a dimension
            if start_positions.ndim > 1:
                start_positions = start_positions.squeeze(-1)
            if start_positions.ndim > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = paddle.shape(start_logits)[1] # 获取sequence_length = 512
            start_positions = start_positions.clip(0, ignored_index)
            end_positions = end_positions.clip(0, ignored_index) # 将end_position限制在[0,sequence_length]的范围内
            
            # 计算损失
            gold_logits=paddle.zeros([span_logits.shape[0]],dtype='int64')
            seq_len=int(ignored_index)
            for i in range(start_positions.shape[0]):
                gold_logits[i]=int(start_positions[i])*seq_len+int(end_positions[i])
            
            loss_fct = paddle.nn.CrossEntropyLoss(ignore_index=ignored_index)
            
            # print("[INFO] span_logits shape is {}, gold_logits shape is {}".format(span_logits.shape,gold_logits.shape))
            total_loss = loss_fct(span_logits, gold_logits)
            
            
            # # 更改计算损失的方法为二分类交叉熵损失，效果会不会更好？理论上应该是等价的？ 已验证：结果毫无变化，一模一样
            # gold_logits=paddle.zeros(span_logits.shape,dtype='float32')
            # seq_len=int(ignored_index)
            # for i in range(start_positions.shape[0]):
            #     gold_logits[i][int(start_positions[i])*seq_len+int(end_positions[i])]=1
            
            # loss_fct = paddle.nn.BCEWithLogitsLoss()
            
            # total_loss = loss_fct(span_logits, gold_logits)            
            
        
        # 返回部分  
        if not return_dict:
            output = (start_logits, end_logits, span_logits) + outputs[2:]
            return ((total_loss, ) +
                    output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            span_logits=span_logits, # 这里把张量压缩成 [6, 512*512] 效率应该会高一些
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )