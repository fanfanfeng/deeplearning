__author__ = 'fanfan'
import numpy as np
import tensorflow as tf
from setting import  defaultPath,tv_classfication
from src.tv_classfication import lstm_model
import os,time
from gensim.models import Word2Vec
from common import data_convert

def change_gensim_mode2array():
    model_path = tv_classfication.word2vec_path
    word2vec_model = Word2Vec.load(model_path)
    array_list = []
    for i in word2vec_model.wv.index2word:
        array_list.append(word2vec_model.wv[i])
    return np.array(array_list)

def load_data(max_sentence_length = None):
    """
        从本地文件读取搜狗分类数据集
    """
    read_dir_path = os.path.join(defaultPath.PROJECT_DIRECTORY,tv_classfication.tv_data_path)
    label_dir_list = os.listdir(read_dir_path)
    x_raw = []
    y = []
    label2index_dict = {l.strip(): i for i,l in enumerate(tv_classfication.label_list)}

    for label_dir in label_dir_list:
        if label_dir == 'thu_jieba.txt':
            continue
        label_dir_path = os.path.join(read_dir_path,label_dir)
        label_index = label2index_dict[label_dir]
        label_item = np.zeros(len(tv_classfication.label_list),np.float32)
        label_item[label_index] = 1
        label_file_list = os.listdir(label_dir_path)
        for label_file in label_file_list:
            if label_file.endswith(".csv"):
                continue
            with open(os.path.join(label_dir_path,label_file),'rb') as reader:
                i = 0
                for line in reader:
                    i +=1
                    text = line.decode('utf-8').replace('\n','').replace('\r','').strip()
                    x_raw.append(text)
                    y.append(label_item)

                    if i>3000:
                        break
        if not max_sentence_length:
            max_sentence_length = max([len(item.split(" ") for item in x_raw)])
        x = []

        model_path = tv_classfication.word2vec_path
        word2vec_model = Word2Vec.load(model_path)
        text_converter = data_convert.SimpleTextConverter(word2vec_model,max_sentence_length,None)

        for sentence,sentence_leng in text_converter.transform_to_ids(x_raw):
            x.append(sentence)
    return np.array(x),np.array(y),max_sentence_length


def train():
    #设置参数
    #num_classes,分类的类别
    tf.flags.DEFINE_integer('num_classes',7,'class num')
    #embdding_dim,每个词表表示成向量的长度
    tf.flags.DEFINE_integer('embedding_dim',256,'Dimensionality of character embedding')
    #hidden_layer_num,隐层书,默认为3
    tf.flags.DEFINE_integer('hidden_layer_num',3,"LSTM hidden layer num")
    #hidden_neural_size,隐层单元数，默认为256
    tf.flags.DEFINE_integer('hidden_neural_size',256,"LSTM hidden neural size")
    #dropout_keep_prob,保留一个神经元的概率，这个概率只在训练的时候用到，默认0.5
    tf.flags.DEFINE_float('dropout_keep_prob',0.8,"Dropout keep probability")
    #batch_size,每批读入样本的数量
    tf.flags.DEFINE_integer('batch_size',300,'Batch Size')
    #max_sentence_length ,文本最大长度
    tf.flags.DEFINE_integer('max_sentence_length',80,'max sentence length')
    #initial_learning_rate,初始的学习率，默认为0.01
    tf.flags.DEFINE_float('initial_learning_rate',0.01,'init learning rate')
    #min_learning_rate,学习率的最小值，默认为0.0001
    tf.flags.DEFINE_float('min_learning_rate',0.00001,'min learning rate')
    #decay_rate，学习率衰减率
    tf.flags.DEFINE_float('decay_rate',0.3,'the learning rate decay')
    #decay_step,学习率衰减步长,默认为1000
    tf.flags.DEFINE_integer('decay_step',800,"Steps after which learning rate decays")
    #init_scale，参数随机初始化最大值，默认为0.1
    tf.flags.DEFINE_float('init_scale',0.1,'init scale')
    #max_grad_norm，梯度最大值，超过则截断，默认为5
    tf.flags.DEFINE_integer('max_grad_norm',5,"max_grad_norm")
    #num_epochs，每次训练读取的数据随机的次数
    tf.flags.DEFINE_integer('num_epochs',300,"Number of trainning epochs")
    #valid_num，训练数据中，用于验证数据的数量
    tf.flags.DEFINE_integer('valid_num',2000,'num of validation')
    #show_every，每次固定迭代次数以后，输出结果
    tf.flags.DEFINE_integer('show_every',10,"Show train results after this many steps")
    #valid_every,在每个固定迭代次数之后，在验证数据上评估模型
    tf.flags.DEFINE_integer('valid_every',100,"Evaluate model on dev set after this many steps")
    #checkpoint_every,在每个固定迭代次数之后，保存模型，默认为100
    tf.flags.DEFINE_integer('checkpoint_every',100,'Save model after this many steps')
    #out_dir,在每个固定迭代次数之后，保存模型
    #tf.flags.DEFINE_string('out_dir',"",'the path of save model')
    #allow_soft_placement,设置为true，如果设备不存在，允许tf自动分配设备
    tf.flags.DEFINE_boolean('allow_soft_placement',True,"Allow device soft device placement")


    FLAGS = tf.flags.FLAGS
    FLAGS._parse_flags()
    print("\n Paramerters:")
    for attr,value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(),value))

    class Config(object):
        hidden_neural_size = FLAGS.hidden_neural_size
        embedding_dim = FLAGS.embedding_dim
        hidden_layer_num = FLAGS.hidden_layer_num
        num_classes = FLAGS.num_classes
        dropout_keep_prob = FLAGS.dropout_keep_prob
        initial_learning_rate = FLAGS.initial_learning_rate
        min_learning_rate = FLAGS.min_learning_rate
        decay_rate = FLAGS.decay_rate
        decay_step = FLAGS.decay_step
        batch_size = FLAGS.batch_size
        max_grad_norm = FLAGS.max_grad_norm



    #2.数据准备
    #2.1加载数据
    print("Loading data ....")
    x,y,max_sentence_length = load_data(FLAGS.max_sentence_length)
    print(len(x))

    #2.1获取word2vec
    #model_path = os.path.join(defaultPath.PROJECT_DIRECTORY, sogou_classfication.word2Vect_path)
    model_save_path = tv_classfication.word2vec_path
    word2vec_model = Word2Vec.load(model_save_path)

    #2.3随机数据
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]
    #分割训练集和测试机，用于验证
    valid_sample_index = FLAGS.valid_num
    x_valid,x_train= x_shuffled[:valid_sample_index],x_shuffled[valid_sample_index:]
    y_valid,y_train= y_shuffled[:valid_sample_index],y_shuffled[valid_sample_index:]

    print("Vocabulary Size:{:d}".format(len(word2vec_model.wv.vocab) + 1))
    print("Train/Valid split: {:d}/{:d}".format(len(y_train),len(y_valid)))


    #用于训练时的参数
    config = Config()
    config.num_step = max_sentence_length
    config.vocabulary_size = len(word2vec_model.wv.vocab) + 1
    config.w2v = change_gensim_mode2array()


    #用于验证时的参数
    valid_config = Config()
    valid_config.vocabulary_size = len(word2vec_model.wv.vocab) + 1
    valid_config.num_step = max_sentence_length
    valid_config.dropout_keep_prob = 1.0
    valid_config.w2v = change_gensim_mode2array()


    print("begin training")
    graph = tf.Graph()
    with graph.as_default(),tf.Session() as sess:
        initializer = tf.random_uniform_initializer(-1 * FLAGS.init_scale, 1 * FLAGS.init_scale)
        with tf.variable_scope("model",initializer=initializer):
            model = lstm_model.LSTM(config)


        timestamp = str(int(time.time()))
        out_dir = "saveModel1/"#os.path.join(defaultPath.PROJECT_DIRECTORY,sogou_classfication.lstm_model_save_path)
        print("Model save path {}\n".format(out_dir))
        #train_summary_writer = tf.summary.FileWriter(os.path.join(out_dir,"summaries","train"),sess.graph)
        #valid_summary_writer = tf.summary.FileWriter(os.path.join(out_dir,"summaries","valid"),sess.graph)

        checkpoint_dir = os.path.join(out_dir,"saveModel1/")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables())
        sess.run(tf.global_variables_initializer())


        step = 0
        for num_epoch in range(FLAGS.num_epochs):
            training_batches = batch_iter(list(zip(x_train,y_train)),FLAGS.batch_size)
            print("epoch {}".format(num_epoch + 1))
            start_time = time.time()

            for training_batch in training_batches:
                #step +=1
                x_batch,y_batch = zip(*training_batch)

                step,_,_ = train_step(sess,model,x_batch,y_batch,config)
                if step % FLAGS.show_every == 0:
                    step_time = (time.time() - start_time) /FLAGS.show_every
                    examples_per_sec = FLAGS.batch_size/ step_time
                    _,train_loss,train_accuracy = train_step(sess,model,x_batch,y_batch,config)
                    learning_rate = model.learning_rate.eval()
                    print("Train epchp {},step {},lr {:g},loss {:g},acc {:g},step-time {:g},example/sec {:g}".format(
                        num_epoch +1,step,learning_rate,train_loss,train_accuracy,step_time,examples_per_sec
                    ))

                #验证测试集
                if step % FLAGS.valid_every == 0:
                    learning_rate = model.learning_rate.eval()
                    valid_loss,valid_accuracy = eval_step(sess,model,x_valid,y_valid,valid_config)
                    print("Valid, step {},lr {:g},loss {:g},acc {:g}".format(
                        step,learning_rate,valid_loss,valid_accuracy
                    ))

                #保存模型
                if step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess,checkpoint_dir,step)
                    print("Saved model checkpint to {}".format(path))






def train_step(session,model,input_x,input_y,config,summary_writer = None):
    """
        单一的训练步骤，定义一个函数用于模型评价，更新批量数据和跟新模型参数
    """
    feed_dict = {}
    feed_dict[model.input_x] = input_x
    feed_dict[model.input_y] = input_y
    feed_dict[model.dropout_keep_prob] = config.dropout_keep_prob
    fetches = [model.train_op,model.global_step,model.learning_rate,model.loss,model.accuracy]
    _,global_step,learning_rate_val,loss_val,accuracy_val = session.run(fetches,feed_dict)
    if summary_writer:
       # summary_writer.add_summary(summary,global_step)
        pass

    return global_step,loss_val,accuracy_val

def eval_step(session,model,input_x,input_y,valid_config,summary_writer=None):
    """
    在验证集上验证模型
    """

    feed_dict = {}
    feed_dict[model.input_x] = input_x
    feed_dict[model.input_y] = input_y
    feed_dict[model.dropout_keep_prob] = valid_config.dropout_keep_prob
    fetches = [model.global_step,model.loss,model.accuracy]
    global_step,loss_val,accuracy_val = session.run(fetches,feed_dict)
    if summary_writer:
        #summary_writer.add_summary(summary,global_step)
        pass

    return loss_val,accuracy_val


def batch_iter(data,batch_size,shuffle=True):
    """
    为数据集生成批迭代器
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    #Shuffle the data at each epoch
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffle_data = data[shuffle_indices]
    else:
        shuffle_data = data

    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size,data_size)
        if start_index < end_index:
            yield shuffle_data[start_index:end_index]
        else:
            raise StopIteration


if __name__ == '__main__':

    train()