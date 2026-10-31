import numpy as np
import os
import pandas as pd

DATASET_NAMES = {
    'CheXpert5x200': ('CheXpert5x200_train_convirt.npz', 'CheXpert5x200_test_convirt.npz'),
    'FIVES':         ('FIVES_train_flair.npz',           'FIVES_test_flair.npz'),
    'MESSIDOR':      ('MESSIDOR_train_flair.npz',        'MESSIDOR_test_flair.npz'),
    'MMAC':        ('MMAC_A_train_flair.npz',          'MMAC_A_test_flair.npz'),
    'NCT':           ('NCT_train_conch.npz',             'NCT_test_conch.npz'),
    'SICAPv2':       ('SICAPv2_conch.npz',               'SICAPv2_test_conch.npz'),
    'Skin':          ('Skin_train_conch.npz',            'Skin_test_conch.npz'),
    'covid':         ('covid_train_convirt.npz',         'covid_test_convirt.npz'),
    'nihlt':         ('nihlt_train_convirt.npz',         'nihlt_test_convirt.npz'),
    #'mBRSET':        ('mBRSET_train_flair.npz',          'mBRSET_test_flair.npz'),
}

# List of datasets in each modality.
MODALITY_DATASETS = {
    'Radiology':       ['CheXpert5x200', 'covid', 'nihlt'],
    'Ophtalmology':    ['FIVES', 'MESSIDOR', 'MMAC'],#, 'mBRSET'],
    'Histology': ['NCT', 'SICAPv2', 'Skin'],
}

# Reverse lookup: dataset -> modality.
DATASET_MODALITY = {ds: mod for mod, dss in MODALITY_DATASETS.items() for ds in dss}

# Class names per dataset
# Sources: FCA repo dataframes at FCA/local_data/dataframes/{cxr,fundus,histology}/*.csv.
CLASS_NAMES = {
    # CXR (CONVIRT encoder)
    'CheXpert5x200': ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion'],
    'covid':         ['COVID', 'Normal', 'Pneumonia', 'Lung Opacity'],
    'nihlt': [
        'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
        'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
        'Nodule', 'Pleural Thickening', 'Pneumonia', 'Pneumothorax',
        'Pneumoperitoneum', 'Pneumomediastinum', 'Subcutaneous Emphysema',
        'Tortuous Aorta', 'Calcification of the Aorta',
    ],
    # Fundus (FLAIR encoder)
    'FIVES':    ['normal', 'diabetic retinopathy', 'glaucoma', 'age related macular degeneration'],
    'MESSIDOR': [
        'no diabetic retinopathy', 'mild diabetic retinopathy',
        'moderate diabetic retinopathy', 'severe diabetic retinopathy',
        'proliferative diabetic retinopathy',
    ],
    'MMAC': [
        'no retinal lesion', 'tessellated fundus',
        'diffuse chorioretinal atrophy', 'patchy chorioretinal atrophy',
        'macular atrophy',
    ],
    'mBRSET': [
        'no diabetic retinopathy', 'mild diabetic retinopathy',
        'moderate diabetic retinopathy', 'severe diabetic retinopathy',
        'proliferative diabetic retinopathy',
    ],
    # Histology (CONCH encoder)
    'NCT': [
        'Adipose', 'Background', 'Debris', 'Lymphocytes', 'Mucus',
        'Smooth muscle', 'Normal colon mucosa', 'Cancer-associated stroma',
        'Colorectal adenocarcinoma epithelium',
    ],
    'SICAPv2': ['NC', 'G3', 'G4', 'G5'],
    'Skin': [
        'Necrosis', 'Skeletal muscle', 'Eccrine sweat glands', 'Vessels',
        'Elastosis', 'Chondral tissue', 'Hair follicle', 'Epidermis',
        'Nerves', 'Subcutis', 'Dermis', 'Sebaceous glands',
        'Squamous-cell carcinoma', 'Melanoma in-situ', 'Basal-cell carcinoma', 'Naevus',
    ],
}
CLASS_NAMES['MMAC_A'] = CLASS_NAMES['MMAC']   # old key (results cached before the rename)

# External test sets (no train split): embeddings file and the dataset whose classes/prototypes they share.
EXTERNAL_TEST_SETS = {
    'ZURICH':  {'file': 'ZURICH_test_conch.npz',  'source': 'SICAPv2'},
    'GERTYCH': {'file': 'GERTYCH_test_conch.npz', 'source': 'SICAPv2'},
    'DeepDRiD': {'file': 'DeepDRiD_test_flair.npz', 'source': 'MESSIDOR'},
}

# Device-shift variants (apps/extract_embeddings_device_shift.py): <embeddings>/<corruption>_L<level>/
CORRUPTIONS = ['blur', 'noise']
CORRUPTION_LEVELS = [1, 2, 3, 4, 5]


def external_test_sets(source):
    """Names of the external test sets that share classes / prototypes with `source` (e.g. 'SICAPv2')."""
    return [name for name, info in EXTERNAL_TEST_SETS.items() if info['source'] == source]


def datasets_by_modality(modality):
    """Return the list of dataset names in the given modality ('Radiology', 'Ophtalmology', 'Histology')."""
    return list(MODALITY_DATASETS[modality])


def get_class_names(name):
    """Return the ordered list of class names for a dataset."""
    return list(CLASS_NAMES[name])


def load_dataset(name, path):
    """Train / test embeddings, labels, text prototypes and logit scale of one dataset (see DATASET_NAMES)."""
    train_f, test_f = DATASET_NAMES[name]
    tr = np.load(os.path.join(path, train_f), allow_pickle=True)
    te = np.load(os.path.join(path, test_f), allow_pickle=True)
    return dict(
        # Load pre-computed feature embeddings and labels for train and test
        x_train=tr['feats_ds'].astype(np.float32), y_train=tr['refs_ds'].astype(np.int64),
        x_test=te['feats_ds'].astype(np.float32),  y_test=te['refs_ds'].astype(np.int64),
        # Text prototypes and logit_scale are the same in train/test files; take from train.
        # Load pre-computed class prototypes using the VLM's text encoder and class descriptions
        text_protos=tr['initial_prototypes'].astype(np.float32), # CxF: one feature embedding with F features for each class C.
        logit_scale=float(tr['logit_scale']),  # This is a scalar temperature value to regulate the similarities. It is learned during pre-training.
        n_classes=int(tr['initial_prototypes'].shape[0]),
        modality=DATASET_MODALITY.get(name),
        class_names=CLASS_NAMES.get(name),
    )


def get_datasets_summary(path):
    """Number of classes, train / test images and smallest train class of every dataset."""
    rows = []
    for name in DATASET_NAMES:
        d = load_dataset(name, path)
        per_class = np.bincount(d['y_train'], minlength=d['n_classes'])
        rows.append({
            'dataset': name,
            '# classes': d['n_classes'],
            '# train instances': len(d['y_train']),
            '# test instances': len(d['y_test']),
            'min/class (train)': int(per_class.min()), # size of the smallest class in the training split — the class with the fewest labeled examples. 
            # Few-shot with k shots per class means you need to draw k distinct examples from every class. That is only possible if k ≤ min/class
        })

    return pd.DataFrame(rows).set_index('dataset')


def load_external_test_set(name, path):
    """Load an external test set (see EXTERNAL_TEST_SETS) as (x_test, y_test, text_protos, logit_scale)."""
    f = np.load(os.path.join(path, EXTERNAL_TEST_SETS[name]['file']))
    return (f['feats_ds'].astype(np.float32), f['refs_ds'].astype(np.int64),
            f['initial_prototypes'].astype(np.float32), float(f['logit_scale']))


def load_corrupted_test_set(name, variant, path):
    """Test embeddings of a device-shift variant (e.g. 'blur_L3') as (x_test, y_test),
    or None if that variant was not extracted."""
    f_path = os.path.join(path, variant, DATASET_NAMES[name][1])
    if not os.path.exists(f_path):
        return None
    f = np.load(f_path)
    return f['feats_ds'].astype(np.float32), f['refs_ds'].astype(np.int64)
