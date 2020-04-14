import tensorflow as tf
import numpy as np
import re


class ATLSTM:
    def __init__(self, train_set, val_set, dev_set, seq_length, target_length, glove_embeddings, embedding_size, n_hidden,
                 n_class, vocab_size, batch_size, learning_rate, epochs, e_print, output_dir):
        self.train_set = train_set
        self.dev_set = dev_set
        self.val_set = val_set
        self.seq_length = seq_length
        self.target_length= target_length
        self.glove_embeddings = glove_embeddings
        self.embedding_size = embedding_size
        self.n_hidden= n_hidden
        self.n_class = n_class
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.e_print = e_print
        self.output_dir = output_dir

    def generate_batch(self):
        batches = []
        batch_number = len(self.train_set)//self.batch_size
        for number in range(batch_number):
            batch = self.train_set[number*self.batch_size:(number+1)*self.batch_size]
            inputs_batch, targets_batch, labels_batch = [], [], []
            for (input, target, label) in batch:
                inputs_batch.append(input)
                targets_batch.append(target)
                labels_batch.append(label)
            batches.append((inputs_batch, targets_batch, labels_batch))
        return batches

    def model(self):
        with tf.name_scope("inputs"):
            X = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.seq_length], name="input") #[batch_size, seq_length]
            TX = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.target_length], name="target-input") #[batch_size, seq_length]
            Y = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.n_class], name="label") # [batch_size, n_class]

        with tf.variable_scope("embeddings"):
            glove = tf.constant_initializer(self.glove_embeddings)
            embeddings = tf.get_variable(name="glove-embeddings", initializer=glove, shape=[self.vocab_size, self.embedding_size],trainable=False)
            w_embeddings = tf.nn.embedding_lookup(embeddings, X) # [batch_size, seq_length, embedding_size]
            t_embeddings = tf.nn.embedding_lookup(embeddings, TX) #[batch_size, target_length, embedding_size]
            v_a = tf.reduce_mean(t_embeddings, 1)
            v_a = tf.reshape(v_a, [-1, 1, self.embedding_size]) #[batch_size, 1, embedding_size]

        with tf.variable_scope("weights"):
            eN = tf.ones([self.batch_size, self.seq_length, self.embedding_size]) #[batch_size, seq_length, embedding_size]
            W = tf.Variable(tf.random_uniform([self.n_hidden+self.embedding_size, self.n_hidden+self.embedding_size], -0.01, 0.01, tf.float32)) #[n_hidden+embedding_size, n_hidden+embedding_size]
            w = tf.Variable(tf.random_uniform([self.n_hidden+self.embedding_size, 1], -0.01, 0.01, tf.float32)) #[n_hidden+embedding_size, 1]
            Wx = tf.Variable(tf.random_uniform([self.n_hidden, self.n_hidden], -0.01, 0.01, tf.float32)) #[n_hidden, n_hidden]
            Wp = tf.Variable(tf.random_uniform([self.n_hidden, self.n_hidden], -0.01, 0.01, tf.float32)) #[n_hidden, n_hidden]
            Ws = tf.Variable(tf.random_uniform([self.n_hidden, self.n_class], -0.01, 0.01, tf.float32)) #[n_hidden, n_class]
            bs = tf.Variable(tf.random_uniform([self.n_class], -0.01, 0.01, tf.float32)) #[n_class]

        with tf.variable_scope("lstm"):
            cell = tf.nn.rnn_cell.LSTMCell(self.n_hidden)
            outputs, output_states = tf.nn.dynamic_rnn(cell, w_embeddings, dtype=tf.float32) # [batch_size, seq_length, n_hidden], [batch_size, n_hidden]

        with tf.variable_scope("attention"):
            v_a = eN * v_a #[batch_size, seq_length, embedding_size]
            H = tf.concat([outputs, v_a], 2) #[batch_size, seq_length, n_hidden+embedding_size]
            M = tf.tanh(tf.matmul(H, W)) #[batch_size, seq_length, n_hidden+embedding_size]
            attn = tf.reshape(tf.matmul(M, w), [-1, 1, self.seq_length]) # [batch_size, 1, seq_length]
            alpha = tf.nn.softmax(attn, -1) #[batch_size, 1, seq_leng]
            r = tf.matmul(alpha, outputs) #[batch_size, 1, n_hidden]
            r = r[-1]
            h_aster = tf.tanh(tf.matmul(r, Wp) + tf.matmul(output_states[0], Wx)) #[batch_size, n_hidden]

        with tf.variable_scope("softmax"):
            ffn = tf.nn.xw_plus_b(h_aster, Ws, bs) #[batch_size, n_class]
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=ffn, labels=Y))+0.001*tf.nn.l2_loss(Ws)

        with tf.variable_scope("predicts"):
            predict = tf.cast(tf.arg_max(tf.nn.softmax(ffn), 1), dtype=tf.int32, name="output")

        return X, TX, Y, cost, predict


    def train(self):
        batches = self.generate_batch()
        X, TX, Y, cost, predict = self.model()

        optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(cost)
        initializer = tf.global_variables_initializer()

        with tf.Session() as session:
            session.run(initializer)

            total_loss = 0
            for epoch in range(self.epochs):
                for (input_batch, t_batch, label_batch) in batches:
                    _, loss = session.run([optimizer, cost], feed_dict={X: input_batch, TX: t_batch, Y: label_batch})
                    total_loss += loss

                if (epoch+1) % self.e_print == 0:
                    averg_loss = total_loss/float(len(batches))
                    print("Epoch:", "%4d"%(epoch+1), "cost=", "{:.6f}".format(averg_loss))

                    acount_dev = 0
                    for (input_dev, t_dev, label_dev) in self.dev_set:
                        input_dev_batch = []
                        t_dev_batch = []
                        input_dev_batch.append(input_dev)
                        t_dev_batch.append(t_dev)
                        pred_dev = session.run([predict], feed_dict={X: input_dev_batch, TX:t_dev_batch})
                        result_dev = pred_dev[0][0]
                        if result_dev == label_dev[0]:
                            acount_dev += 1
                    accuracy_dev = float(acount_dev)/float(len(self.dev_set))
                    print("Epoch:", "%4d"%(epoch+1), "accuracy on dev set:", "{:.6f}".format(accuracy_dev))

            acount_val = 0
            for(input_val, t_val, label_val) in self.val_set:
                input_val_batch = []
                t_val_batch = []
                input_val_batch.append(input_val)
                t_val_batch.append(t_val)
                pred_val = session.run([predict], feed_dict={X:input_val_batch, TX: t_val_batch})
                result_val = pred_val[0][0]
                if result_val == label_val[0]:
                    acount_val += 1
            accuracy_val = float(acount_val)/float(len(self.val_set))
            print("Epoch:", "%4d"%(epoch+1), "accuracy on val set:", "{:.6f}".format(accuracy_val))


            builder = tf.saved_model.builder.SavedModelBuilder(self.output_dir)
            signature = tf.saved_model.signature_def_utils.build_signature_def(inputs={"inputs": tf.saved_model.utils.build_tensor_info(X), "targets":tf.saved_model.utils.build_tensor_info(TX)},
                                                                           outputs={"outputs":tf.saved_model.utils.build_tensor_info(predict)})
            builder.add_meta_graph_and_variables(session, tags=[tf.saved_model.tag_constants.SERVING], signature_def_map={tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature},
                                             strip_default_attrs=True)
            builder.save()


def train_model(data_dir, glove_dir, output_dir):

    with open(data_dir, "r") as inputs:
        data = inputs.readlines()

    with open(glove_dir, "r") as gloves:
        glove_data = gloves.readlines()

    print("building embeddings ...")
    vocab = []
    vectors = []
    for line in glove_data:
        tokens = line.strip().split(" ")
        vocab.append(tokens[0])
        vectors.append(np.asarray([float(val) for val in tokens[1:]]))
    vocab.insert(0, "<PAD>")
    vectors.insert(0, np.random.randn(50))
    vocab.append("<UNK>")
    vectors.append(np.random.randn(50))
    embeddings = np.asarray(vectors)
    dictionary = {w:i for i, w in enumerate(vocab)}

    def index_formatter(tokens, dictionary, vocab):
        input_batch = []
        for tok in tokens:
            if tok not in vocab:
                input_batch.append(dictionary["<UNK>"])
            else:
                input_batch.append(dictionary[tok])
        if len(tokens) <= 128:
            for i in range(len(tokens), 128):
                input_batch.append(dictionary["<PAD>"])
        elif len(tokens) > 128:
            input_batch = input_batch[:128]
        return input_batch

    print("building dataset ... ")
    dataset = []
    for line in data:
        tokens = line.strip().split("\t")
        input_tokens = re.findall("[\w']+|[^A-Za-z0-9\\s]", tokens[0])
        t_tokens = re.findall("[\w']+|[^A-Za-z0-9\\s]", tokens[2])
        inputs = index_formatter(input_tokens, dictionary, vocab)
        t_inputs = index_formatter(t_tokens, dictionary, vocab)
        label = np.eye(6)[int(tokens[2])]
        dataset.append((np.asarray(inputs), np.asarray(t_inputs), np.asarray(label)))

    train_set = dataset[:int(len(dataset)*0.8)]
    dev_set = dataset[int(len(dataset)*0.8): int(len(dataset)*0.9)]
    val_set = dataset[int(len(dataset)*0.9):]

    model = ATLSTM(train_set, val_set, dev_set, 128, 128, embeddings, 50, 128, 6, len(vocab), 1, 0.01, 1, 5, output_dir)

    print("start training ....")
    model.train()

data_dir = "/home/bingxin/Downloads/tdata/sentiment/test/test_at_inputs.tsv"
glove_dir = "/home/bingxin/Downloads/tdata/sentiment/test/glove.6B.50d.txt"
output_dir = "/home/bingxin/Downloads/tdata/sentiment/test/tmp3"

train_model(data_dir, glove_dir, output_dir)

