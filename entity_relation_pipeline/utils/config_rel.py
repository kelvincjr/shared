# -*- coding: utf-8 -*-

# @Author  : xmh
# @Time    : 2021/3/11 16:53
# @File    : config_rel.py

"""
file description:：

"""
import torch

if torch.cuda.is_available():
    USE_CUDA = True
    print("USE_CUDA....")
else:
    USE_CUDA = False


class ConfigRel:
    def __init__(self,
                 lr=0.001,
                 epochs=100,
                 vocab_size=16116,  # 22000,
                 embedding_dim=100,
                 hidden_dim_lstm=64,
                 num_layers=3,
                 batch_size=16,
                 layer_size=64,
                 token_type_dim=8
                 ):
        self.lr = lr
        self.epochs = epochs
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim_lstm = hidden_dim_lstm
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.layer_size = layer_size
        self.token_type_dim = token_type_dim
        self.relations = ["N", '防止', '配合', '指挥', '建立', '指导', '参与', '实施', '开展', '协调', '组织', '组建', '启动', '安置', '履行', '配备', '发布', '落实', '公开', '成立', '收集', '培训', '支援', '掌握', '关注', '更新', '修订', '报送', '承办', '领导', '成员', '调度', '储备', '管理', '补充', '督促', '健全', '处理', '调查', '分析', '控制', '评估', '核定', '引导', '会同', '监管', '维护', '普及', '监测', '通报', '跟踪', '研判', '完善', '协助', '提交', '解除', '完成', '总结', '报告', '提出', '动用', '批准', '抢修', '担任', '安排', '放行', '疏散', '设立', '保障', '配置', '加强', '增强', '保证', '接受', '提供', '宣布', '赔偿', '安抚', '制定', '确保', '采取', '获取', '通知', '疏导', '消除', '利用', '调整', '主持', '核实', '赶赴', '维持', '派遣', '保护', '拨打', '上报', '起草', '研究', '参加', '承担', '监督', '召开', '确定', '预测', '终止', '沟通', '设置', '宣传', '检查', '供应', '提高', '深入', '命令', '构建', '装备', '检测', '发现', '联系', '给予', '统计', '整理', '做好', '保持', '了解', '封存', '调集', '考虑', '反思', '敦促', '避免', '排查', '进行', '撤离', '修复', '汇总', '公布', '部署', '下达', '动员', '调动', '登记', '观察', '处置', '注视', '打击', '明确', '提示', '督导', '汇报', '进入', '执行']
        self.num_relations = len(self.relations)
        self.token_types_origin = ['机构', '灾害', '人物', '方案', '工作', '措施', '职责', '资源', '信息', '文件', '物资', '情况', '产品', '制度', '知识', '报告', '原因', '经验', '建议', '任务', '责任', '问题', '职务', '秩序', '区域', '设施', '能力', '捐赠', '支援', '预案', '交通', '意见', '行为', '电话', '会议', '标准', '计划', '响应', '行动', '命令', '决策', '公告', '支持', '机制', '疾病', '演练', '必需品', '意识', '资金', '指令', '政策', '部署', '体系', '事件', '影响', '状态', '物品', '事故', '建筑', '通知', '效果', '培训', '方法', '程序', '原则', '犯罪', '趋势', '分工']
        self.token_types = self.get_token_types()
        self.num_token_type = len(self.token_types)
        self.vocab_file = '../data/vocab.txt'
        self.max_seq_length = 256
        self.num_sample = 200000#1480 
        
        self.dropout_embedding = 0.1  # 从0.2到0.1
        self.dropout_lstm = 0.1
        self.dropout_lstm_output = 0.9
        self.dropout_head = 0.9  # 只更改这个参数 0.9到0.5
        self.dropout_ner = 0.8
        self.use_dropout = True
        self.threshold_rel = 0.95  # 从0.7到0.95
        self.teach_rate = 0.2
        self.ner_checkpoint_path = '../models/rel_cls/'
        self.pretrained = False
        self.pad_token_id = 0
        self.rel_num = 500

        self.pos_dim = 32
    
    def get_token_types(self):
        token_type_bio = []
        for token_type in self.token_types_origin:
            token_type_bio.append('B-' + token_type)
            token_type_bio.append('I-' + token_type)
        token_type_bio.append('O')
        
        return token_type_bio

