1. linear warm_up 不一样
```
from paddlenlp.transformers.optimization import LinearDecayWithWarmup
scheduler=LinearDecayWithWarmup(learning_rate,total_steps=total_steps,warmup=0)
```
2. paddle.seed(int seed)
不用torch.manual_seed()和torch.cuda.manual_seed()
3. cuda:1->gpu:1
4. return_attention_mask默认为Flase，需要显式指出为True
5. module 'paddle' has no attribute 'LongTensor'，只能用to_tensor()
6. optimizer = AdamW(model.parameters(), learning_rate=scheduler)
7. AdamW的第一个参数是学习率
8. 没有to(device), 或许可以试试cuda(device_id)
9. zero_grad -> clear_grad
10. 需要指出outputs = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels, return_dict=True)中return_dict=True
11. 没有 paddle.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
需要
```
    clip = paddle.nn.ClipGradByNorm(clip_norm=1.0)
    optimizer = AdamW(parameters=model.parameters(), learning_rate=scheduler,grad_clip=clip)
```
12. paddle只需要一个全局的paddle.set_device(args.device)，不需要to(device)等等 
13. 读取模型参数不同
```
    model.load_state_dict(torch.load(best_model_dir)) # 读取模型参数
    替换为：
    load_layer_state_dict = paddle.load(best_model_dir) # 读取模型参数
    model.set_state_dict(load_layer_state_dict) # 加载模型参数
```
14. 问题：在paddle中如何实现torch.nn.utils.rnn.pack_padded_sequence和torch.nn.utils.rnn.pad_packed_sequence这两个API？
答复：目前paddle中没有和上述两个API完全对应的实现。关于torch中这两个API的详细介绍可以参考知乎上的文章 pack_padded_sequence 和 pad_packed_sequence : pack_padded_sequence的功能是将mini-batch数据进行压缩，压缩掉无效的填充值，然后输入RNN网络中；pad_packed_sequence则是把RNN网络输出的压紧的序列再填充回来，便于进行后续的处理。 在paddle中，大家可以在GRU、LSTM等RNN网络中输入含有填充值的mini-batch数据的同时传入对应的sequence_length参数实现上述等价功能，具体用法可以参考 RNN 。
这个地方paddle处理的挺好，比torch方便
15. 没有.gt()，可用.greater_than()代替，但是参数不能为单个数字，可以先将单个数字拓展为张量
16. paddle.sort()的返回值只有一个，torch.sort()返回两个，一个sorted，一个indices
17. LSTM:不需要batch_first=True
18. LSTM:direction='bidirectional'
19. nn.Parameter用paddle.create_parameter代替
20. paddle最捞的一点，没有parameter group
pytorch可以为模型不同层的参数组设置不同的学习率
paddle可以使用paramAttr指定参数的学习率，实际训练的学习率等于参数的学习率乘以optimizer的全局学习率
21. paddle的tensor没有long类型，有：
'bool'，'float16'，'float32'，'float64'，'uint8'，'int8'，'int16'，'int32'，'int64'
22. paddle没有view，直接reshape就好
23. paddle的randn、reshape等函数只能输入list、tuple、variable，不能可变参数reshape(1,3)这样用
24. gather: 
PyTorch：索引(index)的维度数和输入(input)的维度数一致，索引(index)的形状大小要小于等于输入(input)的形状大小。
PaddlePaddle：索引(index)的秩有且只能等于1。
https://github.com/PaddlePaddle/X2Paddle/blob/develop/docs/pytorch_project_convertor/API_docs/ops/torch.gather.md
25. 非常坑，pytorch可以直接用BertModel加载Roberta，但是paddle必须使用RobertaModel, RobertaForQuestionAnswering, RobertaTokenizer
26. https://blog.csdn.net/zhqh100/article/details/124410399?spm=1001.2101.3001.6661.1&utm_medium=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-124410399-blog-122741583.t5_download_comparev1&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-124410399-blog-122741583.t5_download_comparev1&utm_relevant_index=1 非常坑 ubuntu22.04的gcc不支持老版的paddle(小于2.4)，但是新版的paddle又和x2paddle不兼容，所以最好回退到ubuntu20.04
