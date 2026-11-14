from config import LEARNING_RATE
from data import load_splits
from models import build_cnn
from train import reset, train

X_train, X_val, X_test, y_train, y_val, y_test = load_splits()

reset()

model = build_cnn(augment=False)

_, metrics = train("cnn_noaug",model,X_train,y_train,X_val,y_val,LEARNING_RATE)

print(metrics)

