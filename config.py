import torch


class Config:
    device = torch.device("mps")
    MAX_SEQ = 100
    EMBED_DIMS = 512
    ENC_HEADS = DEC_HEADS = 8
    NUM_ENCODER = NUM_DECODER = 4
    BATCH_SIZE = 32
    TRAIN_FILE = "./dataset/train.csv"
    TEST_FILE = "./dataset/example_test.csv"
    TOTAL_EXE = 13523
    TOTAL_CAT = 10000
    TOTAL_ANS = 2
