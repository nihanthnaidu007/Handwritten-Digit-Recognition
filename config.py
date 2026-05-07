from pathlib import Path

# Root
ROOT        = Path(__file__).parent
STEPS_DIR   = ROOT / "pipeline"

# Output folders (auto-created on import)
OUTPUTS_DIR = ROOT / "outputs"
DATA_DIR    = OUTPUTS_DIR / "data"
MODELS_DIR  = OUTPUTS_DIR / "models"
PLOTS_DIR   = OUTPUTS_DIR / "plots"
LOGS_DIR    = OUTPUTS_DIR / "logs"
REPORTS_DIR = OUTPUTS_DIR / "reports"

for _d in [DATA_DIR, MODELS_DIR, PLOTS_DIR, LOGS_DIR, REPORTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# Datasets: mnist, emnist (digits split), usps -- all auto-downloaded in step 01
DATASETS = ["mnist", "emnist", "usps"]

# Per-dataset path helpers
def data_file(ds):    return DATA_DIR    / f"{ds}.npz"
def flat_file(ds):    return DATA_DIR    / f"{ds}_flat.npz"
def cnn_file(ds):     return DATA_DIR    / f"{ds}_cnn.npz"
def lr_model(ds):     return MODELS_DIR  / f"logistic_regression_{ds}.pkl"
def knn_model(ds):    return MODELS_DIR  / f"knn_{ds}.pkl"
def rfc_model(ds):    return MODELS_DIR  / f"rfc_{ds}.pkl"
def svm_model(ds):    return MODELS_DIR  / f"svm_{ds}.pkl"
def cnn_base(ds):     return MODELS_DIR  / f"cnn_baseline_{ds}.pt"
def cnn_imp(ds):      return MODELS_DIR  / f"cnn_improved_{ds}.pt"
def lr_report(ds):    return REPORTS_DIR / f"logistic_regression_{ds}.json"
def knn_report(ds):   return REPORTS_DIR / f"knn_{ds}.json"
def rfc_report(ds):   return REPORTS_DIR / f"rfc_{ds}.json"
def svm_report(ds):   return REPORTS_DIR / f"svm_{ds}.json"
def cnn_b_report(ds): return REPORTS_DIR / f"cnn_baseline_{ds}.json"
def cnn_i_report(ds): return REPORTS_DIR / f"cnn_improved_{ds}.json"

# Backward-compatible aliases (MNIST)
DATA_FILE    = data_file("mnist")
FLAT_FILE    = flat_file("mnist")
CNN_FILE     = cnn_file("mnist")
LR_MODEL     = lr_model("mnist")
KNN_MODEL    = knn_model("mnist")
RFC_MODEL    = rfc_model("mnist")
SVM_MODEL    = svm_model("mnist")
CNN_BASE     = cnn_base("mnist")
CNN_IMP      = cnn_imp("mnist")
LR_REPORT    = lr_report("mnist")
KNN_REPORT   = knn_report("mnist")
RFC_REPORT   = rfc_report("mnist")
SVM_REPORT   = svm_report("mnist")
CNN_B_REPORT = cnn_b_report("mnist")
CNN_I_REPORT = cnn_i_report("mnist")

# Shared reports
BV_REPORT  = REPORTS_DIR / "bias_variance.json"
CMP_REPORT = REPORTS_DIR / "comparison.json"

# Step registry
STEPS = {
    1:  "step_01_data",
    2:  "step_02_logistic_regression",
    3:  "step_03_knn",
    4:  "step_04_rfc",
    5:  "step_05_svm",
    6:  "step_06_cnn_baseline",
    7:  "step_07_cnn_improved",
    8:  "step_08_bias_variance",
    9:  "step_09_comparison",
    10: "step_10_visualisations",
    11: "step_11_demo",
}

# Hyperparameters
LR_PARAMS  = dict(solver="lbfgs", max_iter=1000, C=1.0, random_state=42)
KNN_PARAMS = dict(n_neighbors=5, algorithm="auto", n_jobs=-1)
RFC_PARAMS = dict(n_estimators=100, n_jobs=-1, random_state=42)
SVM_PARAMS = dict(gamma=0.1, kernel="poly")

CNN_BASE_PARAMS = dict(batch_size=256, epochs=20, lr=1e-3, random_state=13)

CNN_IMP_PARAMS = dict(
    batch_size=256,
    epochs=30,
    lr=1e-3,
    random_state=13,
    dropout_conv=0.25,
    dropout_dense=0.4,
    weight_decay=1e-3,
    patience=5,
    val_split=0.2,
)

# Sample caps: KNN/SVM are slow on large EMNIST. None = use all.
CLASSICAL_SAMPLE_CAP = {
    "lr":  {"mnist": None,   "emnist": None,   "usps": None},
    "knn": {"mnist": None,   "emnist": 20000,  "usps": None},
    "rfc": {"mnist": None,   "emnist": 60000,  "usps": None},
    "svm": {"mnist": None,   "emnist": 10000,  "usps": None},
}

# Bias-variance sweep values
BV_KNN_K_VALUES  = [1, 3, 5, 7, 10, 15, 20]
BV_RFC_TREES     = [10, 25, 50, 100, 200]
BV_LR_C_VALUES   = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
BV_SAMPLE_SIZE   = 10000

# Plot style
PLOT_DPI    = 150
PLOT_STYLE  = "seaborn-v0_8-whitegrid"
