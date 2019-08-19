# Taken from  https://www.kaggle.com/fnands/1-mpnn
import tensorflow as tf


class Message_Passer_NNM(tf.keras.layers.Layer):
    def __init__(self, node_dim):
        super(Message_Passer_NNM, self).__init__()
        self.node_dim = node_dim
        self.nn = tf.keras.layers.Dense(units=self.node_dim * self.node_dim, activation=tf.nn.relu)

    def call(self, node_j, edge_ij):

        # Embed the edge as a matrix
        A = self.nn(edge_ij)

        # Reshape so matrix mult can be done
        A = tf.reshape(A, [-1, self.node_dim, self.node_dim])
        node_j = tf.reshape(node_j, [-1, self.node_dim, 1])

        # Multiply edge matrix by node and shape into message list
        messages = tf.linalg.matmul(A, node_j)
        messages = tf.reshape(messages, [-1, tf.shape(edge_ij)[1], self.node_dim])

        return messages


class Message_Agg(tf.keras.layers.Layer):
    def __init__(self):
        super(Message_Agg, self).__init__()

    def call(self, messages):
        return tf.math.reduce_sum(messages, 2)


class Update_Func_GRU(tf.keras.layers.Layer):
    def __init__(self, state_dim):
        super(Update_Func_GRU, self).__init__()
        self.concat_layer = tf.keras.layers.Concatenate(axis=1)
        self.GRU = tf.keras.layers.GRU(state_dim)

    def call(self, old_state, agg_messages):

        # Remember node dim
        n_nodes = tf.shape(old_state)[1]
        node_dim = tf.shape(old_state)[2]

        # Reshape so GRU can be applied, concat so old_state and messages are in sequence
        old_state = tf.reshape(old_state, [-1, 1, tf.shape(old_state)[-1]])
        agg_messages = tf.reshape(agg_messages, [-1, 1, tf.shape(agg_messages)[-1]])
        concat = self.concat_layer([old_state, agg_messages])

        # Apply GRU and then reshape so it can be returned
        activation = self.GRU(concat)
        activation = tf.reshape(activation, [-1, n_nodes, node_dim])

        return activation


# Define the final output layer
class Edge_Regressor(tf.keras.layers.Layer):
    def __init__(self, intermediate_dim):
        super(Edge_Regressor, self).__init__()
        self.concat_layer = tf.keras.layers.Concatenate()
        self.hidden_layer_1 = tf.keras.layers.Dense(units=intermediate_dim, activation=tf.nn.relu)
        self.hidden_layer_2 = tf.keras.layers.Dense(units=intermediate_dim, activation=tf.nn.relu)
        self.output_layer = tf.keras.layers.Dense(units=1, activation=None)

    def call(self, nodes, edges):

        # Remember node dims
        n_nodes = tf.shape(nodes)[1]
        node_dim = tf.shape(nodes)[2]

        # Tile and reshape to match edges
        state_i = tf.reshape(tf.tile(nodes, [1, 1, n_nodes]), [-1, n_nodes * n_nodes, node_dim])
        state_j = tf.tile(nodes, [1, n_nodes, 1])

        # concat edges and nodes and apply MLP
        concat = self.concat_layer([state_i, edges, state_j])
        activation_1 = self.hidden_layer_1(concat)
        activation_2 = self.hidden_layer_2(activation_1)

        return self.output_layer(activation_2)


# Define a single message passing layer
class MP_Layer(tf.keras.layers.Layer):
    def __init__(self, state_dim):
        super(MP_Layer, self).__init__(self)
        self.message_passers = Message_Passer_NNM(node_dim=state_dim)
        self.message_aggs = Message_Agg()
        self.update_functions = Update_Func_GRU(state_dim=state_dim)

        self.state_dim = state_dim

    def call(self, nodes, edges, mask):

        n_nodes = tf.shape(nodes)[1]
        node_dim = tf.shape(nodes)[2]

        state_j = tf.tile(nodes, [1, n_nodes, 1])

        messages = self.message_passers(state_j, edges)

        # Do this to ignore messages from non-existant nodes
        masked = tf.math.multiply(messages, mask)

        masked = tf.reshape(masked, [tf.shape(messages)[0], n_nodes, n_nodes, node_dim])

        agg_m = self.message_aggs(masked)

        updated_nodes = self.update_functions(nodes, agg_m)

        nodes_out = updated_nodes
        # Batch norm seems not to work.
        #nodes_out = self.batch_norm(updated_nodes)

        return nodes_out


class MPNN(tf.keras.Model):
    def __init__(self, out_int_dim, state_dim, T):
        super(MPNN, self).__init__(self)
        self.T = T
        self.embed = tf.keras.layers.Dense(units=state_dim, activation=tf.nn.relu)
        self.MP = MP_Layer(state_dim)
        self.edge_regressor = Edge_Regressor(out_int_dim)

    def call(self, inputs={'nod_input': None, 'adj_input': None}):

        nodes = tf.cast(inputs['nod_input'], tf.float32)
        edges = tf.cast(inputs['adj_input'], tf.float32)

        # Get distances, and create mask wherever 0 (i.e. non-existant nodes)
        # This also masks node self-interactions...
        # This assumes distance is last
        len_edges = tf.shape(edges)[-1]

        _, x = tf.split(edges, [len_edges - 1, 1], 2)
        mask = tf.where(tf.equal(x, 0), x, tf.ones_like(x))

        # Embed node to be of the chosen node dimension (you can also just pad)
        nodes = self.embed(nodes)

        # Run the T message passing steps
        for mp in range(self.T):
            nodes = self.MP(nodes, edges, mask)

        # Regress the output values
        con_edges = self.edge_regressor(nodes, edges)

        return con_edges
