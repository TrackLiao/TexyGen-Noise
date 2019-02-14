import json
from time import time

from models.Gan import Gan
from models.seqgan.SeqganDataLoader import DataLoader, DisDataloader
from models.seqgan.SeqganDiscriminator import Discriminator
from models.seqgan.SeqganGenerator import Generator
from models.seqgan.SeqganReward import Reward
from utils.metrics.Cfg import Cfg
from utils.metrics.EmbSim import EmbSim
from utils.metrics.Nll import Nll
from utils.metrics.Bleu import Bleu
from utils.oracle.OracleCfg import OracleCfg
from utils.oracle.OracleLstm import OracleLstm
from utils.text_process import *
from utils.utils import *


class Seqgan(Gan):
    def __init__(self, oracle=None):
        super().__init__()
        # you can change parameters, generator here
        self.vocab_size = 20
        self.emb_dim = 32
        self.hidden_dim = 32
        self.sequence_length = 20
        self.filter_size = [2, 3]
        self.num_filters = [100, 200]
        self.l2_reg_lambda = 0.2
        self.dropout_keep_prob = 0.75
        self.batch_size = 64
        self.generate_num = 128
        self.start_token = 0

        self.oracle_file = 'save/oracle.txt'
        self.fake_file = 'save/fake.txt'
        self.generator_file = 'save/generator.txt'
        self.test_file = 'save/test_file.txt'
        self.real_data = 'save/real_data.txt'
        self.noise_data = 'save/noise_data.txt'



    def init_metric(self):
        nll = Nll(data_loader=self.oracle_data_loader, rnn=self.oracle, sess=self.sess)
        self.add_metric(nll)

        inll = Nll(data_loader=self.gen_data_loader, rnn=self.generator, sess=self.sess)
        inll.set_name('nll-test')
        self.add_metric(inll)

        from utils.metrics.DocEmbSim import DocEmbSim
        docsim = DocEmbSim(oracle_file=self.oracle_file, generator_file=self.generator_file, num_vocabulary=self.vocab_size)
        self.add_metric(docsim)

    def train_discriminator(self):
        generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
        self.dis_data_loader.load_train_data(self.oracle_file, self.generator_file)
        for _ in range(3):
            self.dis_data_loader.next_batch()
            x_batch, y_batch = self.dis_data_loader.next_batch()
            feed = {
                self.discriminator.input_x: x_batch,
                self.discriminator.input_y: y_batch,
            }
            loss,_ = self.sess.run([self.discriminator.d_loss, self.discriminator.train_op], feed)
            print(loss)

    def evaluate(self):
        generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
        if self.oracle_data_loader is not None:
            self.oracle_data_loader.create_batches(self.generator_file)
        if self.log is not None:
            if self.epoch == 0 or self.epoch == 1:
                self.log.write('epochs, ')
                for metric in self.metrics:
                    self.log.write(metric.get_name() + ',')
                self.log.write('\n')
            scores = super().evaluate()
            for score in scores:
                self.log.write(str(score) + ',')
            self.log.write('\n')
            return scores
        return super().evaluate()

    def init_oracle_trainng(self, oracle=None):
        if oracle is None:
            oracle = OracleLstm(num_vocabulary=self.vocab_size, batch_size=self.batch_size, emb_dim=self.emb_dim,
                                hidden_dim=self.hidden_dim, sequence_length=self.sequence_length,
                                start_token=self.start_token)
        self.set_oracle(oracle)

        generator = Generator(num_vocabulary=self.vocab_size, batch_size=self.batch_size, emb_dim=self.emb_dim,
                              hidden_dim=self.hidden_dim, sequence_length=self.sequence_length,
                              start_token=self.start_token)
        self.set_generator(generator)

        discriminator = Discriminator(sequence_length=self.sequence_length, num_classes=2, vocab_size=self.vocab_size,
                                      emd_dim=self.emb_dim, filter_sizes=self.filter_size, num_filters=self.num_filters,
                                      l2_reg_lambda=self.l2_reg_lambda)
        self.set_discriminator(discriminator)

        gen_dataloader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length)
        oracle_dataloader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length)
        dis_dataloader = DisDataloader(batch_size=self.batch_size, seq_length=self.sequence_length)

        self.set_data_loader(gen_loader=gen_dataloader, dis_loader=dis_dataloader, oracle_loader=oracle_dataloader)

    def train_oracle(self):
        self.init_oracle_trainng()
        self.init_metric()
        self.sess.run(tf.global_variables_initializer())

        self.pre_epoch_num = 80
        self.adversarial_epoch_num = 100
        self.log = open('experiment-log-seqgan.csv', 'w')
        generate_samples(self.sess, self.oracle, self.batch_size, self.generate_num, self.oracle_file)
        generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
        self.gen_data_loader.create_batches(self.oracle_file)
        self.oracle_data_loader.create_batches(self.generator_file)

        print('start pre-train generator:')
        for epoch in range(self.pre_epoch_num):
            start = time()
            loss = pre_train_epoch(self.sess, self.generator, self.gen_data_loader)
            end = time()
            print('epoch:' + str(self.epoch) + '\t time:' + str(end - start))
            self.add_epoch()
            if epoch % 5 == 0:
                generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
                self.evaluate()

        print('start pre-train discriminator:')
        self.reset_epoch()
        for epoch in range(self.pre_epoch_num):
            print('epoch:' + str(epoch))
            self.train_discriminator()

        self.reset_epoch()
        print('adversarial training:')
        self.reward = Reward(self.generator, .8)
        for epoch in range(self.adversarial_epoch_num):
            # print('epoch:' + str(epoch))
            start = time()
            for index in range(1):
                samples = self.generator.generate(self.sess)
                rewards = self.reward.get_reward(self.sess, samples, 16, self.discriminator)
                feed = {
                    self.generator.x: samples,
                    self.generator.rewards: rewards
                }
                _ = self.sess.run(self.generator.g_updates, feed_dict=feed)
            end = time()
            self.add_epoch()
            print('epoch:' + str(self.epoch) + '\t time:' + str(end - start))
            if epoch % 5 == 0 or epoch == self.adversarial_epoch_num - 1:
                generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
                self.evaluate()

            self.reward.update_params()
            for _ in range(15):
                self.train_discriminator()

    def init_cfg_training(self, grammar=None):
        oracle = OracleCfg(sequence_length=self.sequence_length, cfg_grammar=grammar)
        self.set_oracle(oracle)
        self.oracle.generate_oracle()
        self.vocab_size = self.oracle.vocab_size + 1

        generator = Generator(num_vocabulary=self.vocab_size, batch_size=self.batch_size, emb_dim=self.emb_dim,
                              hidden_dim=self.hidden_dim, sequence_length=self.sequence_length,
                              start_token=self.start_token)
        self.set_generator(generator)

        discriminator = Discriminator(sequence_length=self.sequence_length, num_classes=2, vocab_size=self.vocab_size,
                                      emd_dim=self.emb_dim, filter_sizes=self.filter_size, num_filters=self.num_filters,
                                      l2_reg_lambda=self.l2_reg_lambda)
        self.set_discriminator(discriminator)

        gen_dataloader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length)
        oracle_dataloader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length)
        dis_dataloader = DisDataloader(batch_size=self.batch_size, seq_length=self.sequence_length)
        self.set_data_loader(gen_loader=gen_dataloader, dis_loader=dis_dataloader, oracle_loader=oracle_dataloader)
        return oracle.wi_dict, oracle.iw_dict

    def init_cfg_metric(self, grammar=None):
        cfg = Cfg(test_file=self.test_file, cfg_grammar=grammar)
        self.add_metric(cfg)

    def train_cfg(self):
        cfg_grammar = """
          S -> S PLUS x | S SUB x |  S PROD x | S DIV x | x | '(' S ')'
          PLUS -> '+'
          SUB -> '-'
          PROD -> '*'
          DIV -> '/'
          x -> 'x' | 'y'
        """

        wi_dict_loc, iw_dict_loc = self.init_cfg_training(cfg_grammar)
        with open(iw_dict_loc, 'r') as file:
            iw_dict = json.load(file)

        def get_cfg_test_file(dict=iw_dict):
            with open(self.generator_file, 'r') as file:
                codes = get_tokenlized(self.generator_file)
            with open(self.test_file, 'w') as outfile:
                outfile.write(code_to_text(codes=codes, dictionary=dict))

        self.init_cfg_metric(grammar=cfg_grammar)
        self.sess.run(tf.global_variables_initializer())

        self.pre_epoch_num = 80
        self.adversarial_epoch_num = 100
        self.log = open('experiment-log-seqgan-cfg.csv', 'w')
        generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
        self.gen_data_loader.create_batches(self.oracle_file)
        self.oracle_data_loader.create_batches(self.generator_file)
        print('start pre-train generator:')
        for epoch in range(self.pre_epoch_num):
            start = time()
            loss = pre_train_epoch(self.sess, self.generator, self.gen_data_loader)
            end = time()
            print('epoch:' + str(self.epoch) + '\t time:' + str(end - start))
            self.add_epoch()
            if epoch % 5 == 0:
                generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
                get_cfg_test_file()
                self.evaluate()

        print('start pre-train discriminator:')
        self.reset_epoch()
        for epoch in range(self.pre_epoch_num * 3):
            print('epoch:' + str(epoch))
            self.train_discriminator()

        self.reset_epoch()
        print('adversarial training:')
        self.reward = Reward(self.generator, .8)
        for epoch in range(self.adversarial_epoch_num):
            # print('epoch:' + str(epoch))
            start = time()
            for index in range(1):
                samples = self.generator.generate(self.sess)
                rewards = self.reward.get_reward(self.sess, samples, 16, self.discriminator)
                feed = {
                    self.generator.x: samples,
                    self.generator.rewards: rewards
                }
                loss, _ = self.sess.run([self.generator.g_loss, self.generator.g_updates], feed_dict=feed)
                print(loss)
            end = time()
            self.add_epoch()
            print('epoch:' + str(self.epoch) + '\t time:' + str(end - start))
            if epoch % 5 == 0 or epoch == self.adversarial_epoch_num - 1:
                generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
                get_cfg_test_file()
                self.evaluate()

            self.reward.update_params()
            for _ in range(15):
                self.train_discriminator()
        return

    # start from here
    def init_real_trainng(self, data_loc=None, fake_loc=None):
        from utils.text_process import text_precess, text_to_code
        from utils.text_process import get_tokenlized, get_word_list, get_dict
        print("data_loc : ", data_loc)
        print("fake_loc : ", fake_loc)
        self.real_data = data_loc
        # if data_loc is None:
        #     data_loc = 'data/image_coco.txt'
        # if fake_loc is None:
        #     fake_loc = 'data/image_coco.txt'
        self.sequence_length, self.vocab_size = text_precess(data_loc)
        #for fake data
        self.sequence_length_fake, self.vocab_size_fake = text_precess_fake(fake_loc)

        #assign fake size to real might need to modified
        # self.sequence_length = self.sequence_length_fake
        self.vocab_size = self.vocab_size_fake



        #set generator

        #this line has been modified 
        generator = Generator(num_vocabulary=self.vocab_size_fake, batch_size=self.batch_size, emb_dim=self.emb_dim,
                              hidden_dim=self.hidden_dim, sequence_length=self.sequence_length,
                              start_token=self.start_token)
        self.set_generator(generator)

        discriminator = Discriminator(sequence_length=self.sequence_length, num_classes=2, vocab_size=self.vocab_size,
                                      emd_dim=self.emb_dim, filter_sizes=self.filter_size, num_filters=self.num_filters,
                                      l2_reg_lambda=self.l2_reg_lambda)
        self.set_discriminator(discriminator)

        #this line has been modified 
        gen_dataloader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length)
        oracle_dataloader = None
        dis_dataloader = DisDataloader(batch_size=self.batch_size, seq_length=self.sequence_length)

        self.set_data_loader(gen_loader=gen_dataloader, dis_loader=dis_dataloader, oracle_loader=oracle_dataloader)
        tokens = get_tokenlized(data_loc)
        word_set = get_word_list(tokens)
        [word_index_dict, index_word_dict] = get_dict(word_set)

        #generate fake_file
        tokens_fake = get_tokenlized(fake_loc)
        word_set_fake = get_word_list(tokens_fake)
        [word_index_dict_fake, index_word_dict_fake] = get_dict(word_set_fake)

        with open(self.fake_file, 'w') as outfile:
            outfile.write(text_to_code(tokens_fake, word_index_dict_fake, self.sequence_length_fake))

        #oracle_file is the encoded real text
        #use fake file tokens
        with open(self.oracle_file, 'w') as outfile:
            # outfile.write(text_to_code(tokens, word_index_dict, self.sequence_length))
            outfile.write(text_to_code(tokens_fake, word_index_dict_fake, self.sequence_length))

        return word_index_dict, index_word_dict, word_index_dict_fake, index_word_dict_fake

    def init_real_metric(self):
        from utils.metrics.DocEmbSim import DocEmbSim
        docsim = DocEmbSim(oracle_file=self.oracle_file, generator_file=self.generator_file, num_vocabulary=self.vocab_size)
        self.add_metric(docsim)

        inll = Nll(data_loader=self.gen_data_loader, rnn=self.generator, sess=self.sess)
        inll.set_name('nll-test')
        self.add_metric(inll)

        nll = Nll(data_loader=self.oracle_data_loader, rnn=self.oracle, sess=self.sess)
        self.add_metric(nll)

        # bleu_2 = Bleu(test_text = self.test_file , real_text = self.real_data , gram = 2 )
        # bleu_2.set_name('BLEU-2')
        # self.add_metric(bleu_2)

        # bleu_3 = Bleu(test_text = self.test_file , real_text = self.real_data , gram = 2 )
        # bleu_3.set_name('BLEU-3')
        # self.add_metric(bleu_3)

        # bleu_4 = Bleu(test_text = self.test_file , real_text = self.real_data , gram = 4 )
        # bleu_4.set_name('BLEU-4')
        # self.add_metric(bleu_4)

        # bleu_5 = Bleu(test_text = self.test_file , real_text = self.real_data , gram = 5 )
        # bleu_5.set_name('BLEU-5')
        # self.add_metric(bleu_5)

    def train_real(self, data_loc=None, fake_loc=None):
        from utils.text_process import code_to_text
        from utils.text_process import get_tokenlized
        wi_dict, iw_dict , wi_dict_fake, iw_dict_fake = self.init_real_trainng(data_loc, fake_loc)
        self.init_real_metric()



        # def get_real_test_file(dict=iw_dict):
        def get_real_test_file(dict=iw_dict_fake):
            with open(self.generator_file, 'r') as file:
                codes = get_tokenlized(self.generator_file)
            with open(self.test_file, 'w') as outfile:
                outfile.write(code_to_text(codes=codes, dictionary=dict))

        ##readable oracle
        #with open(self.oracle_file, 'r') as file:
        #    codes = get_tokenlized(self.generator_file)
        #with open(self.real_data, 'w') as outfile:
        #    outfile.write(code_to_text(codes=codes, dictionary=iw_dict))

        
        ##readable noise
        #with open(self.fake_file, 'r') as file:
        #    codes = get_tokenlized(self.generator_file)
        #with open(self.noise_data, 'w') as outfile:
        #    outfile.write(code_to_text(codes=codes, dictionary=iw_dict_fake))
        
        self.sess.run(tf.global_variables_initializer())

        self.pre_epoch_num = 80
        self.adversarial_epoch_num = 100
        self.log = open('experiment-log-seqgan-real.csv', 'w')
        generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
        # load file for generator here
        self.gen_data_loader.create_batches(self.fake_file)

        print('start pre-train generator:')
        for epoch in range(self.pre_epoch_num):
            start = time()
            loss = pre_train_epoch(self.sess, self.generator, self.gen_data_loader)
            end = time()
            print('epoch:' + str(self.epoch) + '\t time:' + str(end - start))
            self.add_epoch()
            if epoch % 5 == 0 or epoch == self.pre_epoch_num - 1:
                generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
                get_real_test_file()
                self.evaluate()

        print('start pre-train discriminator:')
        self.reset_epoch()
        for epoch in range(self.pre_epoch_num):
            print('epoch:' + str(epoch))
            self.train_discriminator()

        self.reset_epoch()
        print('adversarial training:')
        self.reward = Reward(self.generator, .8)
        for epoch in range(self.adversarial_epoch_num):
            # print('epoch:' + str(epoch))
            start = time()
            for index in range(1):
                samples = self.generator.generate(self.sess)
                rewards = self.reward.get_reward(self.sess, samples, 16, self.discriminator)
                feed = {
                    self.generator.x: samples,
                    self.generator.rewards: rewards
                }
                loss, _ = self.sess.run([self.generator.g_loss, self.generator.g_updates], feed_dict=feed)
                print(loss)
            end = time()
            self.add_epoch()
            print('epoch:' + str(self.epoch) + '\t time:' + str(end - start))
            if epoch % 5 == 0 or epoch == self.adversarial_epoch_num - 1:
            # if epoch == self.adversarial_epoch_num - 1:
                generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
                get_real_test_file()
                self.evaluate()

            self.reward.update_params()
            for _ in range(15):
                self.train_discriminator()


