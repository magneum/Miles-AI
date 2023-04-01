import logging
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback
from tqdm import tqdm
import colorlog

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
handler.setFormatter(
    colorlog.ColoredFormatter("%(log_color)s%(levelname)s:%(name)s:%(message)s")
)

logger.addHandler(handler)


class PrintMetricsCallback(Callback):
    def __init__(self, total_epochs, total_batches):
        super(PrintMetricsCallback, self).__init__()
        self.total_epochs = total_epochs
        self.total_batches = total_batches
        self.epoch_pbar = None
        self.batch_pbar = None

    def on_train_begin(self, logs=None):
        self.epoch_pbar = tqdm(total=self.total_epochs, unit="epoch", desc="Training")
        self.batch_pbar = tqdm(
            total=self.total_batches, unit="batch", desc="Training Batch"
        )

    def on_batch_end(self, batch, logs=None):
        self.batch_pbar.update(1)
        loss = logs["loss"]
        acc = logs["accuracy"]
        logger.info(colorlog.info(f"Batch {batch}: loss={loss:.4f}, acc={acc:.4f}"))

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_pbar.update(1)
        val_loss = logs["val_loss"]
        val_acc = logs["val_accuracy"]
        val_metrics = self.get_metrics(logs["val_pred"], logs["val_true"])
        logger.info(
            colorlog.info(
                f"Epoch {epoch+1}/{self.total_epochs}: "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, "
                f"val_metrics={val_metrics}"
            )
        )
        self.batch_pbar.n = 0
        self.batch_pbar.last_print_n = 0

    def on_train_end(self, logs=None):
        self.epoch_pbar.close()
        self.batch_pbar.close()

    def get_metrics(self, y_pred, y_true):
        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_true, axis=1)
        metrics = {}
        metrics["precision"] = K.eval(
            K.mean(K.cast(K.equal(y_true, y_pred), dtype="float32"))
        )
        metrics["recall"] = K.eval(
            K.mean(K.cast(K.equal(y_true, y_pred), dtype="float32"))
        )
        metrics["f1_score"] = (
            2
            * metrics["precision"]
            * metrics["recall"]
            / (metrics["precision"] + metrics["recall"] + K.epsilon())
        )
        return metrics
