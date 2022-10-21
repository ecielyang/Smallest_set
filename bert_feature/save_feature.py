from __future__ import absolute_import, division, print_function
from bert_util import *

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--l2",
                        default=None,
                        type=float,
                        required=True)
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--task",
                        default=None,
                        type=str,
                        required=True,
                        help="Sentiment analysis or natural language inference? (SA or NLI)")
    parser.add_argument("--learning_rate",
                        default=None,
                        required=True,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=None,
                        required=True,
                        type=float,
                        help="Total number of training epochs to perform.")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--trained_model_dir",
                        default="",
                        type=str,
                        help="Where is the fine-tuned (with the cloze-style LM objective) BERT model?")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test",
                        action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--freeze_bert',
                        action='store_true',
                        help="Whether to freeze BERT")
    parser.add_argument('--full_bert',
                        action='store_true',
                        help="Whether to use full BERT")
    parser.add_argument('--num_train_samples',
                        type=int,
                        default=-1,
                        help="-1 for full train set, otherwise please specify")
    args = parser.parse_args()

    device = torch.device("cpu")
    n_gpu = torch.cuda.device_count()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval and not args.do_test:
        raise ValueError("At least one of `do_train` or `do_eval` or `do_test` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        logger.info("WARNING: Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)


    processor = Sst2Processor()

    label_list = processor.get_labels()

    num_labels = len(label_list)

    print("set max_len 1000---------------------------------------")
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    # Prepare training data
    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        if args.task == "essay":
            train_examples = processor.get_train_examples(args.data_dir, args.num_train_samples)

        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size) * args.num_train_epochs

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(PYTORCH_PRETRAINED_BERT_CACHE,
                                                                   'distributed_{}'.format(-1))

    model = MyBertForSequenceClassification.from_pretrained(args.bert_model, cache_dir=cache_dir, num_labels=num_labels)

    if args.fp16:
        model.half()
    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer[-2:]], 'weight_decay': args.l2},
    ]

    if args.fp16:
        raise ValueError("Not sure if FP16 precision works yet.")
    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)
    print("learning rate", args.learning_rate)
    if args.do_train:
        global_step = 0
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  learning rate = %d", args.learning_rate)
        logger.info("  num_train_epochs = %d", args.num_train_epochs)
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_id = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_id)
        train_sampler = SequentialSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=1)

        model.eval()
        #         model.eval() # train in eval mode to avoid dropout

        idx = 0
        train_feature_save = np.zeros((11678, 768))
        train_label_save = np.zeros((11678, 1))
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            label_ids = label_ids[:, None].type(torch.FloatTensor).to(device)
            feature= model(input_ids, segment_ids, input_mask, label_ids).detach().numpy()
            print("input_ids", idx, "label", label_ids)
            id = idx
            train_feature_save[id, :] = np.reshape(feature, (768))
            train_label_save[id] = label_ids[0]
            idx += 1
        np.save("train_feature_save.npy", train_feature_save)
        np.save("train_label_save.npy", train_label_save)



        if args.do_test:
            test_examples = processor.get_dev_examples(args.data_dir)

            test_features = convert_examples_to_features(
                test_examples, label_list, args.max_seq_length, tokenizer)
            logger.info("***** Running final test *****")
            logger.info("  Num examples = %d", len(test_examples))
            logger.info("  Batch size = %d", args.eval_batch_size)
            all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
            all_label_id = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
            all_guid = torch.tensor([f.guid for f in test_features], dtype=torch.long)
            test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_id, all_guid)
            # Run prediction for full data
            test_sampler = SequentialSampler(test_data)
            test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)

            model.eval()
            test_loss, test_accuracy = 0, 0
            nb_test_steps, nb_test_examples = 0, 0
            wrong_list = []

            test_feature_save = np.zeros((1298, 768))
            test_label_save = np.zeros((1298, 1))
            idx = 0
            for input_ids, input_mask, segment_ids, label_ids, guids in tqdm(test_dataloader, desc="Testing"):
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                label_ids = label_ids

                with torch.no_grad():
                    tmp_test_loss = model(input_ids, segment_ids, input_mask, label_ids)
                    test_fe = model(input_ids, segment_ids, input_mask).detach().numpy()
                print("id", idx, "label", label_ids[0])
                test_feature_save[idx, :] = np.reshape(test_fe, (768))
                test_label_save[idx] = label_ids[0]
                idx += 1

            np.save("test_feature_save", test_feature_save)
            np.save("test_label_save", test_label_save)

#         pickle.dump(wrong_list, open(os.path.join(args.output_dir, "wrong_pred_guid.txt"), "wb"))

if __name__ == "__main__":
    main()
