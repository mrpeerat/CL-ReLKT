import argparse

def arg_convert():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-corpus', help='Corpus [MLQA,MKQA,XQUAD]')
    parser.add_argument('-corpus_mode', help='Corpus mode [train, test, dev]')
    parser.add_argument('-context_mode', help='doc or para')
    parser.add_argument('-top_start', help='top_start', type=int)
    parser.add_argument('-top_end', help='top_end', type=int)
    parser.add_argument('-batch_size', help='batch_size', type=int)
    parser.add_argument('-num_epoch', help='#epoch', type=int)
    parser.add_argument('-use_mode', help='small or large')
    parser.add_argument('-gpu_device', help='gpu device number')
    parser.add_argument('-margin', help='margin of triplet loss', type=float)
    parser.add_argument('-hard', help='#hard', type=int)
    parser.add_argument('-semi_hard', help='#semi-hard', type=int)
    parser.add_argument('-hard_samples', help='#hard', type=int)
    parser.add_argument('-semi_hard_samples', help='#hard', type=int)
    parser.add_argument('-batch_update', help='batch update', type=int)
    parser.add_argument('-loss_mode', help='Loss: cos or euc')
    parser.add_argument('-hard_update', help='Update only hard negative?')
    parser.add_argument('-warup_steps', help='Warmup steps with BM25 pair', type=int)
    parser.add_argument('-languages', help='language split with _')
    parser.add_argument('-semi_hard_update', help='Update only semi-hard negative?')
    parser.add_argument('-eval_mode', help='fc or none')
    parser.add_argument('-clean', help='True or False')
    parser.add_argument('-replace', help='True or False')
    parser.add_argument('-fc_dimension', help='int of #dimenion of fc', type=int)
    parser.add_argument('-dropout_rate', help='floating point of dropout rate', type=float)
    parser.add_argument('-teacher', help='teacher model')
    parser.add_argument('-learning_rate', help='learning rate')
    parser.add_argument('-mse_factor', help='factor of mse loss', type=int)
    parser.add_argument('-mse_factor_q', help='factor of mse loss (question)')
    parser.add_argument('-mse_factor_d', help='factor of mse loss (doc)',type=float)
    parser.add_argument('-mse_factor_qd', help='factor of mse loss (q-d)')
    parser.add_argument('-shuffle', help='True or False')
    parser.add_argument('-hidden_layer', help='True or False', type=int)
    parser.add_argument('-trip_loss_weight', help='floating point of weight loss', type=float)
    parser.add_argument('-classification_loss_weight', help='floating point of weight loss', type=float)
    parser.add_argument('-freeze', help='freeze USE student weight', type=int)
    parser.add_argument('-ranking_weight', help='weight of cosine loss')
    parser.add_argument('-weight_1', help='weight of term1', type=float)
    parser.add_argument('-weight_2', help='weight of term2', type=float)
    parser.add_argument('-weight_3', help='weight of term3', type=float)
    parser.add_argument('-weight_4', help='weight of term4', type=float)
    parser.add_argument('-instance_queue', help='queue size')
    args = parser.parse_args()
    return args
