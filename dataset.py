from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from config import Config
from sklearn.model_selection import train_test_split
import gc
import torch

DATASET_SIZE = 90e6
# DATASET_SIZE = 10000


class DKTDataset(Dataset):
    def __init__(self, samples, max_seq):
        super().__init__()
        self.samples = samples
        self.max_seq = max_seq
        self.data = []
        # groupby를 해서
        # index => 샘플의 모든 인덱스의 리스트
        for id in self.samples.index:
            # 한 사용자에 대한 transaction 배열
            # content_id, answered_correctly, prior_question_elapsed_time, task_container_id
            exe_ids, answers, categories = self.samples[id]

            if len(exe_ids) > max_seq:
                # 100 문제가 넘으면 100문제 짜리 입력으로 데이터셋을 나눔
                # ex) 210 문제를 풀면 => 100문제 짜리 데이터 2개 + 10문제 짜리 데이터 1개
                # // 는 몫 연산
                for l in range((len(exe_ids)+max_seq-1)//max_seq):
                    self.data.append(
                        (exe_ids[l:l+max_seq], answers[l:l+max_seq], categories[l:l+max_seq]))
            # 50 문제 에서 100 문제 사이 일 경우 그대로 사용
            elif len(exe_ids) < self.max_seq and len(exe_ids) > 50:
                self.data.append((exe_ids, answers, categories))
            else:
                continue

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question_ids, answers, exe_category = self.data[idx]
        seq_len = len(question_ids)

        exe_ids = np.zeros(self.max_seq, dtype=int)
        ans = np.zeros(self.max_seq, dtype=int)
        exe_cat = np.zeros(self.max_seq, dtype=int)

        # 100개 미만의 데이터셋은
        # 끝에서 부터 채우고, 앞에는 zero padding
        if seq_len < self.max_seq:
            exe_ids[-seq_len:] = question_ids
            ans[-seq_len:] = answers
            exe_cat[-seq_len:] = exe_category
        # 아닐 경우
        # 시퀀스의 마지막 100개를 채움
        else:
            exe_ids[:] = question_ids[-self.max_seq:]
            ans[:] = answers[-self.max_seq:]
            exe_cat[:] = exe_category[-self.max_seq:]

        return exe_ids, exe_cat, ans, ans


def get_dataloaders():
    # 데이터 로딩 시 메모리 사용량을 줄이기 위해 데이터타입 명시
    dtypes = {'timestamp': 'int64', 'user_id': 'int32', 'content_id': 'int16',
              'answered_correctly': 'int8', "content_type_id": "int8",
              "task_container_id": "int16"}
    print("loading csv.....")
    # 1 : timestamp(lag time), 2: user_id, 3: content_id(문제 id), 4: content_type_id,
    # 5: task_container_id(문제 set id), 7: answered_correctly, 8: prior_question_elapsed_time(elapsed time 평균)
    train_df = pd.read_csv(Config.TRAIN_FILE, usecols=[
                           1, 2, 3, 4, 5, 7], dtype=dtypes, nrows=DATASET_SIZE)  # 9000만개 90e6
    print("shape of dataframe :", train_df.shape)
    # train_df = train_df.sample(frac=0.01)

    # content_type_id
    # 0: 질문 , 1: 강의
    # 질문에 대한 interaction만 추출
    train_df = train_df[train_df.content_type_id == 0]
    train_df = train_df.sort_values(
        ["timestamp"], ascending=True).reset_index(drop=True)

    # 중복을 제외한 고유값의 수를 반환
    n_skills = train_df.content_id.nunique()
    print("no. of skills :", n_skills)
    print("shape after exlusion:", train_df.shape)

    print("Grouping users...")
    # user_id 별로 데이터 그룹화
    group = train_df[["user_id", "content_id", "answered_correctly",  "task_container_id"]]\
        .groupby("user_id")\
        .apply(lambda r: (r.content_id.values, r.answered_correctly.values, r.task_container_id.values))
    # group에 저장하고, train_df 메모리에서 삭제
    del train_df
    gc.collect()

    # train set / validation set 분리
    print("splitting")
    train, val = train_test_split(group, test_size=0.2)
    print("train size: ", train.shape, "validation size: ", val.shape)

    # DKT를 위한 데이터 셋으로 전처리
    # RNN Max sequence를 100으로 설정
    # 인풋 형태 => user_id 별로, content_id, answered_correctly, prior_question_elapsed_time, task_container_id
    train_dataset = DKTDataset(train, max_seq=Config.MAX_SEQ)
    val_dataset = DKTDataset(val, max_seq=Config.MAX_SEQ)

    # DataLoader로 변환
    train_loader = DataLoader(train_dataset,
                              batch_size=Config.BATCH_SIZE,
                              num_workers=8,
                              persistent_workers=True,
                              shuffle=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=Config.BATCH_SIZE,
                            num_workers=8,
                            persistent_workers=True,
                            shuffle=False)

    # DataLoader로 변환 후 로드한 데이터 메모리에서 삭제
    del train_dataset, val_dataset
    gc.collect()
    return train_loader, val_loader
