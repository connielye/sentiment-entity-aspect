import tensorflow as tf
import numpy as np
import re


class TCLSTM:
    def __init__(self, train_set, dev_set, val_set, seq_left_length, seq_right_length, target_length, n_hidden,
                 n_class, glove_embeddings, embedding_size, vocab_size, batch_size, learning_rate, epochs, output_dir):
        self.train_set = train_set
        self.dev_set = dev_set
        self.val_set = val_set
        self.seq_left_length = seq_left_length
        self.seq_right_length = seq_right_length
        self.target_length = target_length
        self.n_hidden = n_hidden
        self.n_class = n_class
        self.glove_embeddings = glove_embeddings
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.output_dir = output_dir


    def generate_batch(self):
        batches = []
        batch_number = len(self.train_set)//self.batch_size
        for number in range(batch_number):
            batch = self.train_set[number*self.batch_size:(number+1)*self.batch_size]
            l_input_batch, r_input_batch, t_input_batch, label_batch = [], [], [], []
            for (l_input, r_input, t_input, label) in batch:
                l_input_batch.append(l_input)
                r_input_batch.append(r_input)
                t_input_batch.append(t_input)
                label_batch.append(label)
            batches.append((l_input_batch, r_input_batch, t_input_batch, label_batch))
        return batches


    def model(self):
        with tf.name_scope("inputs"):
            LX = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.seq_left_length], name="left-input")
            RX = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.seq_right_length], name="right-input")
            TX = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.target_length], name="target-input")
            Y = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.n_class], name="labels")

        with tf.variable_scope("embeddings"):
            glove_intial = tf.constant_initializer(self.glove_embeddings)
            embeddings = tf.get_variable(name="glove-embeddings", shape=[self.vocab_size, self.embedding_size], initializer=glove_intial, trainable=False)
            lc_embeddings = tf.nn.embedding_lookup(embeddings, LX)
            rc_embeddings = tf.nn.embedding_lookup(embeddings, RX)
            t_embeddings = tf.reduce_mean(tf.nn.embedding_lookup(embeddings, TX), 1)
            t_embeddings = tf.reshape(t_embeddings, [-1, 1, self.embedding_size])
            l_embeddings = tf.concat([lc_embeddings, t_embeddings], 1)
            r_embeddings = tf.concat([rc_embeddings, t_embeddings], 1)

        with tf.variable_scope("l_lstm"):
            l_cell = tf.nn.rnn_cell.LSTMCell(self.n_hidden)
            l_output, _ = tf.nn.dynamic_rnn(l_cell, l_embeddings, dtype=tf.float32)

        with tf.variable_scope("r_lstm"):
            r_cell = tf.nn.rnn_cell.LSTMCell(self.n_hidden)
            r_output, _ = tf.nn.dynamic_rnn(r_cell, r_embeddings, dtype=tf.float32)

        with tf.variable_scope("cost"):
            outputs = tf.concat([l_output, r_output], -1)
            outputs = tf.transpose(outputs, [1, 0, 2])
            outputs = outputs[-1]
            W = tf.Variable(tf.random_normal([2*self.n_hidden, self.n_class]))
            b = tf.Variable(tf.random_normal([self.n_class]))
            ffn = tf.nn.xw_plus_b(outputs, W, b)
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=ffn))

        with tf.variable_scope("predicts"):
            predict = tf.cast(tf.arg_max(tf.nn.softmax(ffn), 1), dtype=tf.int32, name="output")

        return LX, RX, TX, Y, cost, predict

    def train(self):
        batches = self.generate_batch()
        LX, RX, TX, Y, cost, predict = self.model()

        optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(cost)
        initializer = tf.global_variables_initializer()

        with tf.Session() as session:
            session.run(initializer)

            total_loss = 0
            for epoch in range(self.epochs):
                for (left_inputs, right_inputs, target_inputs, labels) in batches:
                    _, loss = session.run([optimizer, cost], feed_dict={LX: left_inputs, RX: right_inputs, TX: target_inputs, Y: labels})
                    total_loss+= loss

                if (epoch+1) % 10 == 0:
                    average_loss = total_loss/float(len(batches))
                    print("Epoch:", "%04d"%(epoch+1), "cost=", "{:.6f}".format(average_loss))

                    acount = 0
                    for(dev_left_input, dev_right_input, dev_target_input, dev_label) in self.dev_set:
                        test_l = []
                        test_r = []
                        test_t = []
                        test_l.append(dev_left_input)
                        test_r.append(dev_right_input)
                        test_t.append(dev_target_input)
                        pred = session.run([predict], feed_dict={LX:test_l, RX:test_r, TX: test_t})
                        result = pred[0][0]
                        if result == dev_label:
                            acount += 1
                    accuracy = float(acount)/float(len(self.dev_set))
                    print("Epoch:", "%04d"%(epoch+1), "accuracy=", "{:.6f}".format(accuracy))

            val_a_count = 0
            for(val_left_input, val_right_input, val_target_input, val_label) in self.val_set:
                test_l_val = []
                test_r_val = []
                test_t_val = []
                test_l_val.append(val_left_input)
                test_r_val.append(val_right_input)
                test_t_val.append(val_target_input)
                pred_val = session.run([predict], feed_dict={LX:test_l_val, RX:test_r_val, TX:test_t_val})
                result = pred_val[0][0]
                if result == val_label[0]:
                    val_a_count += 1
            accuracy_val = float(val_a_count)/float(len(self.val_set))
            print("Epoch:", "%04d"%(epoch+1), "accuracy=", "{:.6f}".format(accuracy_val))


            builder = tf.saved_model.builder.SavedModelBuilder(self.output_dir)
            signature = tf.saved_model.signature_def_utils.build_signature_def(inputs={"left-input": tf.saved_model.utils.build_tensor_info(LX), "right-input":tf.saved_model.utils.build_tensor_info(RX),
                                                                                       "target-input":tf.saved_model.utils.build_tensor_info(TX)},
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
        l_tokens = re.findall("[\w']+|[^A-Za-z0-9\\s]", tokens[0])
        r_tokens = re.findall("[\w']+|[^A-Za-z0-9\\s]", tokens[1])
        t_tokens = re.findall("[\w']+|[^A-Za-z0-9\\s]", tokens[2])
        l_inputs = index_formatter(l_tokens, dictionary, vocab)
        r_inputs = index_formatter(r_tokens, dictionary, vocab)
        t_inputs = index_formatter(t_tokens, dictionary, vocab)
        label = np.eye(6)[int(tokens[3])]
        dataset.append((np.asarray(l_inputs), np.asarray(r_inputs), np.asarray(t_inputs), np.asarray(label)))

    train_set = dataset[:int(len(dataset)*0.8)]
    dev_set = dataset[int(len(dataset)*0.8): int(len(dataset)*0.9)]
    val_set = dataset[int(len(dataset)*0.9):]

    model = TCLSTM(train_set, dev_set, val_set, 128, 128, 128, 128, 6, embeddings, 50, len(vocab), 1, 0.01, 1, output_dir)

    print("start training ....")
    model.train()

data_dir = "/home/bingxin/Downloads/tdata/sentiment/test/test_inputs.tsv"
glove_dir = "/home/bingxin/Downloads/tdata/sentiment/test/glove.6B.50d.txt"
output_dir = "/home/bingxin/Downloads/tdata/sentiment/test/tmp2"

train_model(data_dir, glove_dir, output_dir)
