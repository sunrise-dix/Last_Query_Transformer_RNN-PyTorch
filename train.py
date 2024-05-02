import pytorch_lightning as pl

from config import Config
from trainer import ModelTrainer
from dataset import get_dataloaders
from pytorch_lightning.callbacks import ModelCheckpoint

MAX_EPOCH = 10
TRIAL = 1

checkpoint_callback = ModelCheckpoint(
    dirpath=f"model/trial_{TRIAL}",  # 체크포인트 저장 디렉토리
    filename="{MAX_EPOCH}-{epoch}-{val_loss:.2f}",  # 파일명 포맷
    save_top_k=3,  # 가장 좋은 3개의 체크포인트만 저장
    monitor="val_loss",  # 모니터링할 메트릭
    mode="min",  # "min"은 val_loss를 최소화하는 체크포인트를 저장
)

if __name__ == '__main__':
    train_loader, val_loader = get_dataloaders()

    model = ModelTrainer(dim_model=Config.EMBED_DIMS, heads_en=Config.ENC_HEADS, total_ex=Config.TOTAL_EXE,
                         total_cat=Config.TOTAL_CAT, total_in=Config.TOTAL_ANS, seq_len=Config.MAX_SEQ, use_lstm=True)
    trainer = pl.Trainer(max_epochs=MAX_EPOCH, callbacks=[checkpoint_callback])

    trainer.fit(model=model, train_dataloaders=train_loader,
                val_dataloaders=[val_loader, ])
