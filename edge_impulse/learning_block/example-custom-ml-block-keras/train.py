import argparse, os, sys, random, logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from conversion import save_saved_model
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, roc_auc_score

# Lower TensorFlow log levels
tf.get_logger().setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set random seeds for repeatable results
RANDOM_SEED = 3
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Load files
parser = argparse.ArgumentParser(description='Running custom Keras models in Edge Impulse')
parser.add_argument('--data-directory', type=str, required=True)
parser.add_argument('--epochs', type=int, required=True)
parser.add_argument('--learning-rate', type=float, required=True)
parser.add_argument('--out-directory', type=str, required=True)

args, unknown = parser.parse_known_args()

if not os.path.exists(args.out_directory):
    os.mkdir(args.out_directory)

# grab train/test set and convert into TF Dataset
X_train = np.load(os.path.join(args.data_directory, 'X_split_train.npy'), mmap_mode='r')
Y_train = np.load(os.path.join(args.data_directory, 'Y_split_train.npy'))
X_test = np.load(os.path.join(args.data_directory, 'X_split_test.npy'), mmap_mode='r')
Y_test = np.load(os.path.join(args.data_directory, 'Y_split_test.npy'))

classes = Y_train.shape[1]
MODEL_INPUT_SHAPE = X_train.shape[1:]

# print GPU/CPU info
print('Training on:', 'gpu' if len(tf.config.list_physical_devices('GPU')) > 0 else 'cpu')
print('')

# callbacks
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True
)
callbacks = [early_stop]

# --- K-Fold setup ---
Kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
fold = 0
aucs = []

# Convert one-hot encoded labels to integer class labels
y_labels = np.argmax(Y_train, axis=1)

# --- Cross-validation loop ---
for train_idx, val_idx in Kfold.split(X_train, y_labels):
    X_tr, y_tr = X_train[train_idx], Y_train[train_idx]
    X_val, y_val = X_train[val_idx], Y_train[val_idx]

    model = Sequential([
        Dense(64, activation='relu', activity_regularizer=tf.keras.regularizers.l1(1e-5)),
        BatchNormalization(),
        Dropout(0.1),
        Dense(128, activation='relu', activity_regularizer=tf.keras.regularizers.l1(1e-5)),
        BatchNormalization(),
        Dropout(0.1),
        Dense(64, activation='relu', activity_regularizer=tf.keras.regularizers.l1(1e-5)),
        BatchNormalization(),
        Dropout(0.1),
        Dense(32, activation='relu'),
        Dense(classes, activation='softmax' if classes > 2 else 'sigmoid', name='y_pred')
    ])

    loss_fn = 'categorical_crossentropy' if classes > 2 else 'binary_crossentropy'

    opt = Adam(learning_rate=args.learning_rate)
    model.compile(loss=loss_fn, optimizer=opt, metrics=['accuracy', tf.keras.metrics.AUC()])

    model.fit(X_tr, y_tr, epochs=args.epochs, validation_data=(X_val, y_val),
              verbose=2, callbacks=callbacks)

    # --- Evaluate ---
    pred_prob = model.predict(X_val)

    if classes > 2:
        pred = np.argmax(pred_prob, axis=1)
        true = np.argmax(y_val, axis=1)
        auc_score = roc_auc_score(y_val, pred_prob, multi_class='ovr')
    else:
        # Handle binary classification robustly
        true = y_val.flatten()
        pred = (pred_prob > 0.5).astype(int).flatten()
        auc_score = roc_auc_score(true, pred_prob.reshape(-1, 1))

    acc_score = accuracy_score(true, pred)

    print(f"======= Fold {fold} ========")
    print(f"Accuracy: {acc_score:.4f}, AUC: {auc_score:.4f}")
    aucs.append(auc_score)
    fold += 1

oof_auc = np.mean(aucs)
print(f"\n✅ Out-of-fold AUC score: {oof_auc:.4f}")

print('\nTraining network OK\n')

save_saved_model(model, args.out_directory)
