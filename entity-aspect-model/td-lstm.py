import tensorflow as tf
import numpy as np
import re


class DTLSTM:
    def __init__(self, train_set, dev_set, val_set, seq_left_length, seq_right_length, n_hidden, n_class, batch_size, glove_embeddings, vocab_size, embedding_size, learning_rate, epochs, output_dir):
        self.train_set = train_set
        self.dev_set = dev_set
        self.val_set = val_set
        self.seq_left_length = seq_left_length
        self.seq_right_length = seq_right_length
        self.n_hidden = n_hidden
        self.n_class = n_class
        self.glove_embeddings = glove_embeddings
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.output_dir = output_dir


    def generate_batch(self):
        batches = []
        batch_number = len(self.train_set)//self.batch_size
        for number in range(batch_number):
            batch = self.train_set[number*self.batch_size:(number+1)*self.batch_size]
            l_input_batch, r_input_batch, targets_batch = [], [],[]
            for (l_input, r_input, target) in batch:
                l_input_batch.append(l_input)
                r_input_batch.append(r_input)
                targets_batch.append(target)
            batches.append((l_input_batch, r_input_batch, targets_batch))
        return batches


    def model(self):

        with tf.name_scope("inputs"):
            LX = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.seq_left_length], name="left-input")
            RX = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.seq_right_length], name="right-input")
            Y = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.n_class], name="target")

        with tf.variable_scope("embeddings"):
            glove_intializer = tf.constant_initializer(self.glove_embeddings)
            w_embeddings = tf.get_variable(name="glove-embeddings", shape=[self.vocab_size, self.embedding_size], dtype=tf.float32, initializer=glove_intializer, trainable=False)
            l_embs = tf.nn.embedding_lookup(w_embeddings, LX)
            r_embs = tf.nn.embedding_lookup(w_embeddings, RX)

        with tf.variable_scope("weights"):
            W = tf.Variable(tf.random_normal([2*self.n_hidden, self.n_class]))
            b = tf.Variable(tf.random_normal([self.n_class]))

        with tf.variable_scope("r_lstm"):
            l_cell = tf.nn.rnn_cell.LSTMCell(self.n_hidden)
            l_outputs, _ = tf.nn.dynamic_rnn(l_cell, l_embs, dtype=tf.float32)

        with tf.variable_scope("l_lstm"):
            r_cell = tf.nn.rnn_cell.LSTMCell(self.n_hidden)
            r_outputs, _ = tf.nn.dynamic_rnn(r_cell, r_embs, dtype=tf.float32)

        with tf.variable_scope("cost"):
            outputs = tf.concat([l_outputs, r_outputs], -1)
            outputs = tf.transpose(outputs, [1, 0, 2])
            outputs = outputs[-1]
            ffn = tf.nn.xw_plus_b(outputs, W, b)
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=ffn))

        with tf.variable_scope("predicts"):
            predict = tf.cast(tf.arg_max(tf.nn.softmax(ffn), 1), tf.int32, name="output")

        return LX, RX, Y, cost, predict

    def train(self):
        batches = self.generate_batch()
        LX, RX, Y, cost, predict = self.model()
        optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(cost)
        initializer = tf.global_variables_initializer()

        with tf.Session() as session:
            session.run(initializer)

            for epoch in range(self.epochs):
                total_loss = 0
                for (left_inputs, right_inputs, targets) in batches:
                    _, loss = session.run([optimizer, cost], feed_dict={LX: left_inputs, RX:right_inputs, Y:targets})
                    total_loss += loss

                if (epoch+1)%10 == 0:
                    average_loss = total_loss/float(len(batches))
                    print("Epochs:", "%04d"%(epoch+1), "cost=", "{:.6f}".format(average_loss))

                    acount = 0
                    for (left_input, right_input, target) in self.dev_set:
                        l_test = []
                        r_test = []
                        l_test.append(left_input)
                        r_test.append(right_input)
                        pred = session.run([predict], feed_dict={LX: l_test, RX:r_test})
                        result = pred[0][0]
                        if result == target:
                            acount += 1
                    accuracy = float(float(acount)/float(len(self.dev_set)))
                    print("Epoch:", "%04d"%(epoch+1), "accuracy=", "{:.6f}".format(accuracy))

            a_val_count = 0
            for(l_input, r_input, target_val) in self.val_set:
                l_test_val = []
                r_test_val = []
                l_test_val.append(l_input)
                r_test_val.append(r_input)
                pred_val = session.run([predict], feed_dict={LX:l_test_val, RX:r_test_val})
                result_val = pred_val[0][0]
                if result_val == target_val[0]:
                    a_val_count += 1
            accuracy_val = float(float(a_val_count)/float(len(self.val_set)))
            print("Epoch:", "%04d"%(epoch+1), "accuracy=", "{:.6f}".format(accuracy_val))

            builder = tf.saved_model.builder.SavedModelBuilder(self.output_dir)
            signature = tf.saved_model.signature_def_utils.build_signature_def(inputs={"left-input": tf.saved_model.utils.build_tensor_info(LX), "right-input":tf.saved_model.utils.build_tensor_info(RX)},
                                                                               outputs={"outputs":tf.saved_model.utils.build_tensor_info(predict)})
            builder.add_meta_graph_and_variables(session, tags=[tf.saved_model.tag_constants.SERVING], signature_def_map={tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature},
                                                 strip_default_attrs=True)
            builder.save()

def train_model(data_dir, glove_dir, output_dir):
    with open(data_dir, "r") as data_input:
        data = data_input.readlines()

    with open(glove_dir, "r") as glove_input:
        glove_data = glove_input.readlines()


    tf.compat.v1.logging.info("start importing embeddings...")
    print("start importing embeddings...")

    vectors = []
    vocabs = []
    for line in glove_data:
        tokens = line.strip().split(" ")
        vocabs.append(tokens[0])
        vectors.append(np.asarray(tokens[1:]))
    vectors.insert(0, np.random.randn(50))
    vocabs.insert(0, "<PAD>")
    vectors.append(np.random.randn(50))
    vocabs.append("<UNK>")
    embeddings = np.asarray(vectors)

    tf.compat.v1.logging.info("start building dataset...")
    print("start building dataset...")

    vocab_dict = {w:i for i, w in enumerate(vocabs)}


    def index_formatter(tokens, vocabs, vocab_dict):
        input_batch = []
        for tok in tokens:
            if tok in vocabs:
                input_batch.append(vocab_dict[tok])
            else:
                input_batch.append(vocab_dict["<UNK>"])
        if len(tokens) < 128:
            for i in range(len(tokens), 128):
                input_batch.append(0)
        elif len(tokens) > 128:
            input_batch = input_batch[: 128]
        return input_batch

    dataset = []
    for line in data:
        tokens = line.strip().split("\t")
        l_toks = re.findall("[\w']+|[^A-Za-z0-9\\s]", tokens[0].strip())
        l_input = index_formatter(l_toks, vocabs, vocab_dict)
        r_tokes = re.findall("[\w']+|[^A-Za-z0-9\\s]", tokens[1].strip())
        r_input = index_formatter(r_tokes, vocabs, vocab_dict)
        senti = np.eye(6)[int(tokens[3])]
        dataset.append((np.asarray(l_input), np.asarray(r_input), senti))

    train_set = dataset[:int(len(dataset)*0.8)]
    dev_set = dataset[int(len(dataset)*0.8):int(len(dataset)*0.9)]
    val_set = dataset[int(len(dataset)*0.9):]

    tf.compat.v1.logging.info("start training...")
    print("start training...")
    model = DTLSTM(train_set, dev_set, val_set, 128, 128, 128, 6, 1, embeddings, len(vocabs), 50, 0.01, 1, output_dir)
    model.train()


data_dir = "/home/bingxin/Downloads/tdata/sentiment/test/test_inputs.tsv"
glove_dir = "/home/bingxin/Downloads/tdata/sentiment/test/glove.6B.50d.txt"
output_dir = "/home/bingxin/Downloads/tdata/sentiment/test/tmp"

train_model(data_dir, glove_dir, output_dir)
