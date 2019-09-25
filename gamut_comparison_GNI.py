import numpy as np
from utils import *
import os

'''In this blotto game, '''
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

# ------------------hypers------------------- #
batch_size = 100
game_dimension = 10
epoch_num = 2000
inner_loop = 1
player_num = 4
network_input_dimension = 20
# ------------------------------------------- #

class Quadratic(object):
    def __init__(self, sess):
        self.model_name = 'Quadratic'
        self.player_num = player_num
        self.epoch_num = epoch_num
        self.game_if_set = 0
        self.sess = sess
        self.inner_loop = inner_loop
        self.player_network_dict = {}
        self.sdim = network_input_dimension
        self.batch_size = batch_size
        self.Lambda = tf.placeholder(dtype=tf.float32)

    def game_setting(self, game_matrix_1, game_matrix_2, game_matrix_3, game_matrix_4):
        if self.game_if_set == 0:
            self.game_matrix_1 = tf.convert_to_tensor(game_matrix_1, dtype = tf.float32)
            self.game_matrix_2 = tf.convert_to_tensor(game_matrix_2, dtype = tf.float32)
            self.game_matrix_3 = tf.convert_to_tensor(game_matrix_3, dtype=tf.float32)
            self.game_matrix_4 = tf.convert_to_tensor(game_matrix_4, dtype=tf.float32)
            self.game_if_set = 1
            print('[*] Game Matrix Setting Complete.')
        else:
            print('[!!!] Detected Double Setting. Setting has been paused.')

    def build_network_pure(self):
        self.player1_action_pure_ = tf.Variable(initial_value = tf.random_uniform(shape = [game_dimension], minval = -0.2, maxval = 0.2), name = 'pure_p1')
        self.player2_action_pure_ = tf.Variable(initial_value=tf.random_uniform(shape=[game_dimension], minval=-0.2, maxval=0.2), name = 'pure_p2')
        self.player3_action_pure_ = tf.Variable(initial_value=tf.random_uniform(shape=[game_dimension], minval=-0.2, maxval=0.2), name='pure_p3')
        self.player4_action_pure_ = tf.Variable(initial_value=tf.random_uniform(shape=[game_dimension], minval=-0.2, maxval=0.2), name='pure_p4')

        self.player1_action_pure = tf.nn.tanh(self.player1_action_pure_)
        self.player2_action_pure = tf.nn.tanh(self.player2_action_pure_)
        self.player3_action_pure = tf.nn.tanh(self.player3_action_pure_)
        self.player4_action_pure = tf.nn.tanh(self.player4_action_pure_)

        player_action_pure = tf.expand_dims(tf.concat([self.player1_action_pure, self.player2_action_pure, self.player3_action_pure, self.player4_action_pure], axis = 0), axis = 0)

        self.expected_utility1_pure = tf.reduce_mean(tf.matmul(tf.matmul(player_action_pure, self.game_matrix_1),
                                                                  tf.transpose(player_action_pure, perm = [1,0])))
        self.expected_utility2_pure = tf.reduce_mean(tf.matmul(tf.matmul(player_action_pure, self.game_matrix_2),
                                                                  tf.transpose(player_action_pure, perm = [1,0])))
        self.expected_utility3_pure = tf.reduce_mean(tf.matmul(tf.matmul(player_action_pure, self.game_matrix_3),
                                                               tf.transpose(player_action_pure, perm=[1, 0])))
        self.expected_utility4_pure = tf.reduce_mean(tf.matmul(tf.matmul(player_action_pure, self.game_matrix_4),
                                                               tf.transpose(player_action_pure, perm=[1, 0])))

        self.player1_action_pure_local = tf.nn.softmax(self.player1_action_pure_ - self.Lambda * tf.gradients(self.expected_utility1_pure, self.player1_action_pure_)[0])
        self.player2_action_pure_local = tf.nn.softmax(self.player2_action_pure_ - self.Lambda * tf.gradients(self.expected_utility2_pure,self.player2_action_pure_)[0])
        self.player3_action_pure_local = tf.nn.softmax(self.player3_action_pure_ - self.Lambda * tf.gradients(self.expected_utility3_pure,self.player3_action_pure_)[0])
        self.player4_action_pure_local = tf.nn.softmax(self.player4_action_pure_ - self.Lambda * tf.gradients(self.expected_utility4_pure,self.player4_action_pure_)[0])

        player_action_pure_1 = tf.expand_dims(tf.concat([self.player1_action_pure_local, self.player2_action_pure, self.player3_action_pure, self.player4_action_pure],axis = 0), axis=0)
        player_action_pure_2 = tf.expand_dims(tf.concat([self.player1_action_pure, self.player2_action_pure_local, self.player3_action_pure, self.player4_action_pure], axis = 0), axis=0)
        player_action_pure_3 = tf.expand_dims(tf.concat([self.player1_action_pure, self.player2_action_pure, self.player3_action_pure_local,
             self.player4_action_pure], axis=0), axis=0)
        player_action_pure_4 = tf.expand_dims(tf.concat([self.player1_action_pure, self.player2_action_pure, self.player3_action_pure,
             self.player4_action_pure_local], axis=0), axis=0)


        self.local_expected_utility1_pure = tf.reduce_mean(tf.matmul(tf.matmul(player_action_pure_1, self.game_matrix_1),
                                                                  tf.transpose(player_action_pure_1, perm = [1,0])))
        self.local_expected_utility2_pure = tf.reduce_mean(tf.matmul(tf.matmul(player_action_pure_2, self.game_matrix_2),
                                                                  tf.transpose(player_action_pure_2, perm = [1,0])))
        self.local_expected_utility3_pure = tf.reduce_mean(
            tf.matmul(tf.matmul(player_action_pure_3, self.game_matrix_3),
                      tf.transpose(player_action_pure_3, perm=[1, 0])))
        self.local_expected_utility4_pure = tf.reduce_mean(
            tf.matmul(tf.matmul(player_action_pure_4, self.game_matrix_4),
                      tf.transpose(player_action_pure_4, perm=[1, 0])))
        self.loss_function_pure = tf.abs(self.expected_utility1_pure + self.expected_utility2_pure + self.expected_utility3_pure + self.expected_utility4_pure
                                         - self.local_expected_utility1_pure - self.local_expected_utility2_pure - self.local_expected_utility3_pure - self.local_expected_utility4_pure)

    def player_network(self, scope):
        if scope == 'player_number_1':
            self.b7_player1 = tf.placeholder(dtype = tf.float32, name = scope + '_b7')
        elif scope == 'player_number_2':
            self.b7_player2 = tf.placeholder(dtype=tf.float32, name = scope + '_b7')
        elif scope == 'player_number_3':
            self.b7_player3 = tf.placeholder(dtype=tf.float32, name = scope + '_b7')
        else:
            self.b7_player4 = tf.placeholder(dtype=tf.float32, name = scope + '_b7')
        random_sample = tf.random_uniform(shape = [self.batch_size, self.sdim], minval = -1.0, maxval = 1.0)
        # self.random_sample = tf.placeholder(shape = [self.batch_size, self.sdim], dtype = tf.float32, name = scope + '_input')
        self.W1 = tf.zeros(shape = [self.sdim, 2 * self.sdim], name = scope + '_W1')
        self.b1 = tf.zeros(shape = [2 * self.sdim], name = scope + '_b1')
        hidden_layer_1 = tf.nn.tanh(tf.matmul(random_sample, self.W1) + tf.tile(tf.expand_dims(self.b1, axis = 0), [self.batch_size, 1]))
        self.W2 = tf.zeros(shape = [2 * self.sdim, 4 * self.sdim], name = scope + '_W2')
        self.b2 = tf.zeros(shape = [4 * self.sdim], name = scope + '_b2')
        hidden_layer_2 = tf.nn.tanh(tf.matmul(hidden_layer_1, self.W2) + tf.tile(tf.expand_dims(self.b2, axis = 0), [self.batch_size, 1]))
        self.W3 = tf.zeros(shape = [4 * self.sdim, 16 * self.sdim], name = scope + '_W3')
        self.b3 = tf.zeros(shape = [16 * self.sdim], name = scope + '_b3')
        hidden_layer_3 = tf.nn.tanh(tf.matmul(hidden_layer_2, self.W3) + tf.tile(tf.expand_dims(self.b3, axis = 0), [self.batch_size, 1]))
        self.W4 = tf.zeros(shape = [16 * self.sdim, 16 * self.sdim], name = scope + '_W4')
        self.b4 = tf.zeros(shape = [16 * self.sdim], name = scope + '_b4')
        hidden_layer_4 = tf.nn.relu(tf.matmul(hidden_layer_3, self.W4) + tf.tile(tf.expand_dims(self.b4, axis = 0), [self.batch_size, 1]))
        self.W5 = tf.zeros(shape = [16 * self.sdim, 4 * self.sdim], name = scope + '_W5')
        self.b5 = tf.zeros(shape = [4 * self.sdim], name = scope + '_b5')
        hidden_layer_5 = tf.nn.tanh(tf.matmul(hidden_layer_4, self.W5) + tf.tile(tf.expand_dims(self.b5, axis = 0), [self.batch_size, 1]))
        self.W6 = tf.zeros(shape = [4 * self.sdim, 2 * self.sdim], name = scope + '_W6')
        self.b6 = tf.zeros(shape = [2 * self.sdim], name = scope + '_b6')
        hidden_layer_6 = tf.nn.tanh(tf.matmul(hidden_layer_5, self.W6) + tf.tile(tf.expand_dims(self.b6, axis = 0), [self.batch_size, 1]))
        self.W7 = tf.zeros(shape = [2 * self.sdim, game_dimension], name = scope + '_W7')
        if scope == 'player_number_1':
            self.b7 = self.b7_player1
        elif scope == 'player_number_2':
            self.b7 = self.b7_player2
        elif scope == 'player_number_3':
            self.b7 = self.b7_player3
        else:
            self.b7 = self.b7_player4
        hidden_layer_7 = tf.matmul(hidden_layer_6, self.W7) + tf.tile(tf.expand_dims(self.b7, axis = 0), [self.batch_size, 1])

        output_layer = tf.nn.softmax(hidden_layer_7, axis = 1)

        return output_layer

    def player_network_setting(self):
        self.player_network_dict0 = self.player_network(scope = 'player_number_' + str(1))
        self.player_network_dict1 = self.player_network(scope = 'player_number_' + str(2))
        self.player_network_dict2 = self.player_network(scope='player_number_' + str(3))
        self.player_network_dict3 = self.player_network(scope='player_number_' + str(4))

    def player_network_local(self, scope):
        if scope == 'player_number_1':
            L = self.expected_utility1
        elif scope == 'player_number_2':
            L = self.expected_utility2
        elif scope == 'player_number_3':
            L = self.expected_utility3
        else:
            L = self.expected_utility4
        with tf.name_scope(scope):
            random_sample = tf.random_uniform(shape = [self.batch_size, self.sdim], minval = -1.0, maxval = 1.0)
            W1_ = self.W1 - self.Lambda * tf.gradients(L, self.W1)[0]
            b1_ = self.b1 - self.Lambda * tf.gradients(L, self.b1)[0]
            hidden_layer_1 = tf.nn.tanh(tf.matmul(random_sample, W1_) + tf.tile(tf.expand_dims(b1_, axis = 0), [self.batch_size, 1]))
            W2_ = self.W2 - self.Lambda * tf.gradients(L, self.W2)[0]
            b2_ = self.b2 - self.Lambda * tf.gradients(L, self.b2)[0]
            hidden_layer_2 = tf.nn.tanh(tf.matmul(hidden_layer_1, W2_) + tf.tile(tf.expand_dims(b2_, axis = 0), [self.batch_size, 1]))
            W3_ = self.W3 - self.Lambda * tf.gradients(L, self.W3)[0]
            b3_ = self.b3 - self.Lambda * tf.gradients(L, self.b3)[0]
            hidden_layer_3 = tf.nn.tanh(tf.matmul(hidden_layer_2, W3_) + tf.tile(tf.expand_dims(b3_, axis = 0), [self.batch_size, 1]))
            W4_ = self.W4 - self.Lambda * tf.gradients(L, self.W4)[0]
            b4_ = self.b4 - self.Lambda * tf.gradients(L, self.b4)[0]
            hidden_layer_4 = tf.nn.relu(tf.matmul(hidden_layer_3, W4_) + tf.tile(tf.expand_dims(b4_, axis = 0), [self.batch_size, 1]))
            W5_ = self.W5 - self.Lambda * tf.gradients(L, self.W5)[0]
            b5_ = self.b5 - self.Lambda * tf.gradients(L, self.b5)[0]
            hidden_layer_5 = tf.nn.tanh(tf.matmul(hidden_layer_4, W5_) + tf.tile(tf.expand_dims(b5_, axis = 0), [self.batch_size, 1]))
            W6_ = self.W6 - self.Lambda * tf.gradients(L, self.W6)[0]
            b6_ = self.b6 - self.Lambda * tf.gradients(L, self.b6)[0]
            hidden_layer_6 = tf.nn.tanh(tf.matmul(hidden_layer_5, W6_) + tf.tile(tf.expand_dims(b6_, axis = 0), [self.batch_size, 1]))
            W7_ = self.W7 - self.Lambda * tf.gradients(L, self.W7)[0]
            b7_ = self.b7 - self.Lambda * tf.gradients(L, self.b7)[0]
            hidden_layer_7 = tf.nn.tanh(tf.matmul(hidden_layer_6, W7_) + tf.tile(tf.expand_dims(b7_, axis = 0), [self.batch_size, 1]))

            output_layer = tf.nn.softmax(hidden_layer_7, axis=1)

        return output_layer


    def build_model(self):
        player1_action_batch = self.player_network_dict0
        player2_action_batch = self.player_network_dict1
        player3_action_batch = self.player_network_dict2
        player4_action_batch = self.player_network_dict3

        player_action_batch = tf.concat([player1_action_batch, player2_action_batch, player3_action_batch, player4_action_batch], axis = 1)

        self.expected_utility1 = tf.reduce_mean(tf.diag_part(tf.matmul(tf.matmul(player_action_batch, self.game_matrix_1),
                                                                  tf.transpose(player_action_batch, perm = [1,0]))))
        self.expected_utility2 = tf.reduce_mean(tf.diag_part(tf.matmul(tf.matmul(player_action_batch, self.game_matrix_2),
                                                                  tf.transpose(player_action_batch, perm = [1,0]))))
        self.expected_utility3 = tf.reduce_mean(tf.diag_part(tf.matmul(tf.matmul(player_action_batch, self.game_matrix_3),
                                   tf.transpose(player_action_batch, perm=[1, 0]))))
        self.expected_utility4 = tf.reduce_mean(tf.diag_part(tf.matmul(tf.matmul(player_action_batch, self.game_matrix_4),
                                   tf.transpose(player_action_batch, perm=[1, 0]))))

        self.player1_action_local = self.player_network_local(scope='player_number_1')
        self.player2_action_local = self.player_network_local(scope='player_number_2')
        self.player3_action_local = self.player_network_local(scope='player_number_3')
        self.player4_action_local = self.player_network_local(scope='player_number_4')

        local_player1_batch = tf.concat([self.player1_action_local, player2_action_batch, player3_action_batch, player4_action_batch], axis = 1)
        local_player2_batch = tf.concat([player1_action_batch, self.player2_action_local, player3_action_batch, player4_action_batch], axis=1)
        local_player3_batch = tf.concat([player1_action_batch, player2_action_batch, self.player3_action_local, player4_action_batch], axis=1)
        local_player4_batch = tf.concat([player1_action_batch, player2_action_batch, player3_action_batch, self.player4_action_local], axis=1)

        self.local_expected_utility1 = tf.reduce_mean(tf.diag_part(tf.matmul(tf.matmul(local_player1_batch, self.game_matrix_1),
                                                                  tf.transpose(local_player1_batch, perm = [1,0]))))
        self.local_expected_utility2 = tf.reduce_mean(tf.diag_part(tf.matmul(tf.matmul(local_player2_batch, self.game_matrix_2),
                                                                  tf.transpose(local_player2_batch, perm = [1,0]))))
        self.local_expected_utility3 = tf.reduce_mean(tf.diag_part(tf.matmul(tf.matmul(local_player3_batch, self.game_matrix_3),
                                                                  tf.transpose(local_player3_batch, perm=[1, 0]))))
        self.local_expected_utility4 = tf.reduce_mean(tf.diag_part(tf.matmul(tf.matmul(local_player4_batch, self.game_matrix_4),
                                                                  tf.transpose(local_player4_batch, perm=[1, 0]))))

        self.loss_function = tf.abs(self.expected_utility1 + self.expected_utility2 + self.expected_utility3 + self.expected_utility4
                                    - self.local_expected_utility1 - self.local_expected_utility2 - self.local_expected_utility3 -self.local_expected_utility4)




    def optimize_model_pure(self):
        self.file_out = open('GNI_gamut_10.txt', 'w')
        self.optimizer_pure = tf.train.MomentumOptimizer(1e-2,0.9)
        self.minimize_pure = self.optimizer_pure.minimize(self.loss_function_pure, var_list=tf.trainable_variables(scope = 'pure'))
        tf.global_variables_initializer().run()

        Lambda = 1e-3
        for epoch in range(self.epoch_num):
            self.sess.run(self.minimize_pure, feed_dict = {self.Lambda: Lambda})
            loss = self.sess.run(self.loss_function_pure, feed_dict = {self.Lambda: Lambda})
            player1_pure_ = self.sess.run(self.player1_action_pure_)
            player2_pure_ = self.sess.run(self.player2_action_pure_)
            player3_pure_ = self.sess.run(self.player3_action_pure_)
            player4_pure_ = self.sess.run(self.player4_action_pure_)
            mixed_loss = self.sess.run(self.loss_function, feed_dict = {self.Lambda:Lambda, self.b7_player1:player1_pure_, self.b7_player2:player2_pure_, self.b7_player3:player3_pure_, self.b7_player4:player4_pure_})
            print('------------------ epoch '+str(epoch)+' has finished ---------------')
            print('loss:' + str(loss))
            self.file_out.write(str(mixed_loss) + '\n')
            print('mixed_loss:' + str(mixed_loss))


def main():
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # ------------game setting------------- #
        quadratic = Quadratic(sess)
        # ------------------------------------- #

        game_matrix_1 = np.random.rand(4 * game_dimension, 4 * game_dimension)
        game_matrix_2 = np.random.rand(4 * game_dimension, 4 * game_dimension)
        game_matrix_3 = np.random.rand(4 * game_dimension, 4 * game_dimension)
        game_matrix_4 = np.random.rand(4 * game_dimension, 4 * game_dimension)
        quadratic.game_setting(game_matrix_1, game_matrix_2, game_matrix_3, game_matrix_4)


        quadratic.build_network_pure()
        quadratic.player_network_setting()
        quadratic.build_model()
        quadratic.optimize_model_pure()


if __name__ == '__main__':
    main()

    # train = optimizer.minimize(loss, var_list=tf.trainable_variables())