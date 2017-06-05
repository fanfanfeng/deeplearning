label_list = ['IT', '体育', '健康', '军事', '招聘', '教育', '文化', '旅游', '财经']
data_label_dict = {
    'C000008': '财经',
    'C000010': 'IT',
    'C000013': '健康',
    'C000014': '体育',
    'C000016': '旅游',
    'C000020': '教育',
    'C000022': '招聘',
    'C000023': '文化',
    'C000024': '军事'
}


data_path_origin  = r'data\sogou_classification_data'
data_path_jieba = r'data\sogou_classificaiton_data_jieba'

word2Vect_path = r'model\sogou_classification\word2vec'
model_name = "wordVec"

lstm_model_save_path='model\sogou_classification\lstm'
