import tensorflow as tf
from networks import MatchModel
from utils import dataSet, Trainer
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default='-1')
    parser.add_argument('--datain', nargs='?', default='interview')
    parser.add_argument('--dataout', default='interview')
    parser.add_argument('--emb_dim', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    # word2vec arguments
    parser.add_argument('--w2v', type=int, default=0)
    parser.add_argument('--window_size', type=int, default=5)
    parser.add_argument('--w2v_ep', type=int, default=1)
    parser.add_argument('--w2v_lr', type=int, default=0.025)
    parser.add_argument('--min_count', type=int, default=5)
    # model arguments
    parser.add_argument('--doc_len', type=int, default=25)
    parser.add_argument('--sent_len', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--n_epoch', type=int, default=0)
    return parser.parse_args()


if __name__ == '__main__':
    # 参数接收器
    args = parse_args()

    # 显卡占用
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    # 预训练word to vector
    # if args.w2v:
    #     word2vec = networks.Word2Vec(
    #         input_file_name='{}.all'.format(args.datain),
    #         output_file_name='./data/{}.word_emb'.format(args.dataout),
    #         emb_dimension=args.emb_dim,
    #         iteration=args.w2v_ep,
    #         initial_lr=args.w2v_lr,
    #         min_count=args.min_count
    #     )
    #     word2vec.train()

    # 读取训练
    # word_dict, embs = dataSet.load_word_emb(
    #     './data/{}.word_emb'.format(args.dataout),
    #     args.emb_dim
    # )

    dataSet.data_split(args.datain, args.dataout, frac=0.1)

    word_dict = dataSet.build_dict('data/interview.train', 5)

    train_data = dataSet.data_generator(
        fp='./data/{}.train'.format(args.dataout),
        word_dict=word_dict,
        doc_len=args.doc_len,
        sent_len=args.sent_len,
        batch_size=args.batch_size
    )

    test_data = dataSet.data_generator(
        fp='./data/{}.test'.format(args.dataout),
        word_dict=word_dict,
        doc_len=args.doc_len,
        sent_len=args.sent_len,
        batch_size=args.batch_size
    )

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(
        graph=tf.Graph(),
        config=config
    ) as sess:
        model = MatchModel.MatchModel(
            n_word=len(word_dict),
            emb_dim=args.emb_dim,
            doc_len=args.doc_len,
            sent_len=args.sent_len,
            # emb_pretrain=embs
        )

        writer = tf.summary.FileWriter("board", sess.graph)

        Trainer.train(
            sess=sess,
            model=model,
            writer=writer,
            train_generator=train_data,
            test_generator=test_data,
            lr=args.lr,
            n_epoch=args.n_epoch,
        )

        constant_graph = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ['mlp/predict'])

        with tf.gfile.FastGFile('./data/' + 'model.pb', mode='wb') as f:
            f.write(constant_graph.SerializeToString())

    writer.close()


