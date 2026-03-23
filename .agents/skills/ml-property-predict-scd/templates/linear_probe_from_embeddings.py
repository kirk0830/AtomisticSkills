import json
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


EMBEDDINGS_PATH = "scd_embeddings.npz"
TEST_SIZE = 0.1
VAL_SIZE = 0.1
RANDOM_SEED = 1
ALPHA = 1.0


def main():
    data = np.load(EMBEDDINGS_PATH, allow_pickle=True)
    x = data["mol_emb"]
    y = data["y"]

    if y.ndim == 1:
        y = y.reshape(-1, 1)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    val_fraction = VAL_SIZE / (1.0 - TEST_SIZE)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=val_fraction, random_state=RANDOM_SEED
    )

    model = Ridge(alpha=ALPHA)
    model.fit(x_train, y_train)

    splits = {
        "train": (x_train, y_train),
        "val": (x_val, y_val),
        "test": (x_test, y_test),
    }

    metrics = {}
    for split, (x_split, y_split) in splits.items():
        pred = model.predict(x_split)
        metrics[split] = {
            "mae": float(mean_absolute_error(y_split, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_split, pred))),
            "r2": float(r2_score(y_split, pred)),
        }

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
