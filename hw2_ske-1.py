import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, roc_curve, auc, accuracy_score

'''
General Instructions:

1. Do not use any additional libraries. Your code will be tested in a pre-built environment with only 
the library above available.

2. You are expected to fill in the skeleton code precisely as per provided. On top of skeleton code given,
you may write whatever deemed necessary to complete the assignment. For example, you may define additional 
default arguments, class parameters, or methods to help you complete the assignment.

3. Some initial steps or definition are given, aiming to help you getting started. As long as you follow 
the argument and return type, you are free to change them as you see fit.

4. Your code should be free of compilation errors. Compilation errors will result in 0 marks.
'''

# Initialize the global model variable
my_best_model = XGBClassifier()


'''
Problem A-1: Data Preprocessing and EDA
'''
class DataLoader:
    '''
    This class will be used to load the data and perform initial data processing.
    '''
    def __init__(self, data_root: str, random_state: int):
        '''
        Initialize the DataLoader class with the data_root path.
        '''
        self.random_state = random_state
        np.random.seed(self.random_state)
        
        # Try to read CSV, if it fails, use empty DataFrame
        try:
            self.data = pd.read_csv(data_root, sep=';')
        except:
            self.data = pd.DataFrame()
        
        self.data_train = None
        self.data_valid = None

        # Make call to defined class methods
        self.data_prep()
        self.data_split()

    def data_split(self) -> None:
        '''
        Split the training data into train/valid datasets on the ratio of 80/20.
        '''
        if self.data is None or len(self.data) == 0:
            self.data_train = pd.DataFrame()
            self.data_valid = pd.DataFrame()
            return

        shuffled_data = self.data.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        split_marker = int(len(shuffled_data) * 0.8)
        self.data_train = shuffled_data[:split_marker]
        self.data_valid = shuffled_data[split_marker:]

    def data_prep(self) -> None:
        '''
        Drop any rows with missing values and map categorical variables to numeric values.
        Move the target column 'y' to the last position after encoding.
        '''
        if self.data is None or len(self.data) == 0:
            return
    
        # Drop missing values
        self.data = self.data.dropna()
        
        # Encode all categorical columns
        for col in self.data.columns:
            if self.data[col].dtype == 'object':
                unique_vals = sorted(self.data[col].unique())
                
                # For target column 'y', ensure it's mapped to 0 and 1
                if col == 'y' and len(unique_vals) == 2:
                    mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
                else:
                    mapping = {val: idx for idx, val in enumerate(unique_vals)}
                
                self.data[col] = self.data[col].map(mapping)
                
        # Move 'y' column to the end for reliable extraction
        if 'y' in self.data.columns:
            y_col = self.data.pop('y')
            self.data['y'] = y_col

    def extract_features_and_label(self, data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        '''
        Extract features and labels from train/valid/test data.
        Assumes 'y' is the last column due to the final step in data_prep().
        '''
        if data is None or len(data) == 0:
            n_features = len(self.data.columns) - 1 if self.data is not None and len(self.data.columns) > 0 else 1
            return np.empty((0, n_features), dtype=np.float64), np.array([], dtype=np.int64)
        
        data_work = data.copy()
        
        # Use positional indexing
        if data_work.shape[1] > 1:
            X = data_work.iloc[:,:-1].values.astype(np.float64)
            y = data_work.iloc[:,-1].values.astype(np.int64)
            return X, y
        elif data_work.shape[1] == 1 and 'y' in data_work.columns:
            y = data_work.iloc[:,-1].values.astype(np.int64)
            return np.empty((len(y), 0), dtype=np.float64), y
        else:
            return data_work.values.astype(np.float64), np.array([], dtype=np.int64)


'''
Problem A-2: Classification Tree Implementation
'''
class ClassificationTree:
    '''
    Simple classification tree from scratch.
    '''
    class Node:
        '''
        A data structure to represent a node in the tree.
        '''
        def __init__(self, split=None, left=None, right=None, prediction=None):
            self.split = split
            self.left = left
            self.right = right
            self.prediction = prediction 

        def is_leaf(self):
            return self.prediction is not None

    def __init__(self, random_state: int, max_depth: int = 8):
        self.random_state = random_state
        np.random.seed(self.random_state)
        self.max_depth = max_depth
        self.tree_root = None

    def split_crit(self, y: np.ndarray) -> float:
        '''
        Implement the impurity measure (Gini impurity).
        '''
        if len(y) == 0:
            return 0.0
        y = y.astype(int) 
        unique_y, counts = np.unique(y, return_counts=True)
        prob = counts / len(y)
        gini_impurity = 1.0 - np.sum(prob**2)
        return gini_impurity
        
    def build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0):
        '''
        Build the tree recursively. Returns the root node.
        '''
        y = y.astype(int)
        
        if len(y) == 0:
            return self.Node(prediction=0)
        
        majority_prediction = int(np.bincount(y).argmax())
        
        if len(np.unique(y)) == 1:
            return self.Node(prediction=majority_prediction)
        
        if depth >= self.max_depth:
            return self.Node(prediction=majority_prediction)
        
        if X.shape[1] == 0 or X.shape[0] == 0:
            return self.Node(prediction=majority_prediction)
        
        best_split = self.search_best_split(X, y)
        
        if best_split is None:
            return self.Node(prediction=majority_prediction)
        
        feature, splitval = best_split
        
        left_mask = X[:, feature] <= splitval
        right_mask = ~left_mask
        
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return self.Node(prediction=majority_prediction)
        
        left = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self.build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return self.Node(split=best_split, left=left, right=right)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        '''
        Fit the classification tree to the training data.
        '''
        X = X.astype(np.float64)
        y = y.astype(np.int64)
        self.tree_root = self.build_tree(X, y, depth=0)

    def search_best_split(self, X: np.ndarray, y: np.ndarray):
        '''
        Search for the best split.
        '''
        best_gain = -1e-8
        best_split = None 
        current_imp = self.split_crit(y)
    
        for features in range(X.shape[1]):
            feature_values = X[:, features]
            unique_vals = np.unique(feature_values)
            
            if len(unique_vals) < 2:
                continue

            for i in range(len(unique_vals) - 1):
                split_value = (unique_vals[i] + unique_vals[i+1]) / 2.0
                
                left_mask = feature_values <= split_value
                right_mask = ~left_mask
                
                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                if n_left == 0 or n_right == 0:
                    continue
                
                left_imp = self.split_crit(y[left_mask])
                right_imp = self.split_crit(y[right_mask])
                
                n = len(y)
                weighted_imp = (n_left/n)*left_imp + (n_right/n)*right_imp
                
                gain = current_imp - weighted_imp 
                
                if gain > best_gain:
                    best_gain = gain
                    best_split = (features, split_value)
        
        return best_split

    def predict(self, X: np.ndarray) -> np.ndarray:
        '''
        Predict the class labels for input data X.
        '''
        if X is None or len(X) == 0:
            return np.array([], dtype=np.int64)

        if self.tree_root is None:
            return np.zeros(len(X), dtype=np.int64) 
        
        def traverse(node, x):
            if node.is_leaf():
                return node.prediction
            
            feature, splitval = node.split
            if x[feature] <= splitval:
                return traverse(node.left, x)
            else:
                return traverse(node.right, x)

        preds = np.array([traverse(self.tree_root, x) for x in X], dtype=np.int64)
        return preds


def train_XGBoost() -> dict:
    '''
    Train XGBoost with L1 regularization using bootstrap validation (100 iterations)
    '''
    loader = DataLoader(data_root="bank.csv", random_state=42)
    X_train, y_train = loader.extract_features_and_label(loader.data_train)
    X_valid, y_valid = loader.extract_features_and_label(loader.data_valid)
    
    y_train = y_train.astype(int)
    y_valid = y_valid.astype(int)

    positive_count = np.sum(y_train == 1)
    negative_count = np.sum(y_train == 0)
    scale_pos_weight = negative_count / positive_count if positive_count > 0 else 1.0
    
    # Alpha values as specified in assignment
    alpha_vals = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]
    
    # Store F1 scores for each alpha across bootstrap iterations
    alpha_f1_scores = {alpha: [] for alpha in alpha_vals}
    
    # Bootstrap validation loop
    for i in range(100):
        indices = np.random.choice(len(X_train), size=len(X_train), replace=True)
        X_boot = X_train[indices]
        y_boot = y_train[indices]

        if len(np.unique(y_boot)) < 2:
            continue
        
        pos_boot = np.sum(y_boot == 1)
        neg_boot = np.sum(y_boot == 0)
        scale_boot = neg_boot / pos_boot if pos_boot > 0 else 1.0
                
        # Test each alpha value
        for alpha in alpha_vals:
            model = XGBClassifier(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.1,
                reg_alpha=alpha,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_boot,
                objective='binary:logistic',
                eval_metric='logloss',
            
            )
            
            model.fit(X_boot, y_boot, verbose=False)
            
            y_pred = model.predict(X_valid)
            f1 = f1_score(y_valid, y_pred, average='binary', zero_division=0)
            alpha_f1_scores[alpha].append(f1)

    # Find best alpha
    avg_f1_scores = {}
    for alpha in alpha_vals:
        if len(alpha_f1_scores[alpha]) > 0:
            avg_f1_scores[alpha] = np.mean(alpha_f1_scores[alpha])
        else:
            avg_f1_scores[alpha] = 0.0
    
    best_alpha = max(avg_f1_scores, key=avg_f1_scores.get)
    
    # Train final model
    best_model = XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.1,
        reg_alpha=best_alpha,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        objective='binary:logistic',
        eval_metric='logloss',
        
    )
    best_model.fit(X_train, y_train)

    y_train_pred = best_model.predict(X_train)
    y_valid_pred = best_model.predict(X_valid)
    train_acc = np.mean(y_train_pred == y_train)
    valid_acc = np.mean(y_valid_pred == y_valid)
    f1_train = f1_score(y_train, y_train_pred, average='binary', zero_division=0)
    f1_valid = f1_score(y_valid, y_valid_pred, average='binary', zero_division=0)
    
    y_valid_probs = best_model.predict_proba(X_valid)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_valid, y_valid_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

    global my_best_model
    my_best_model = best_model

    return {
        'train_accuracy': train_acc,
        'valid_accuracy': valid_acc,
        'train_f1': f1_train,
        'valid_f1': f1_valid,
        'best_alpha': best_alpha,
        'auc': roc_auc,
        'model': best_model
    }


if __name__ == "__main__":
    # You can test with any random_state you want locally
    test_random_state = 42  # Change this to test different splits
    
    print("="*70)
    print("TESTING HOMEWORK 2 IMPLEMENTATION")
    print("="*70)

    # Test 1: DataLoader
    print("\n[TEST 1] DataLoader")
    print("-"*70)
    loader = DataLoader(data_root="bank.csv", random_state=42)
    X_train, y_train = loader.extract_features_and_label(loader.data_train)
    X_valid, y_valid = loader.extract_features_and_label(loader.data_valid)
    print(f"✅ Data loaded: Train={X_train.shape}, Valid={X_valid.shape}")
    print(f"   Train labels: {np.unique(y_train, return_counts=True)}")
    print(f"   Valid labels: {np.unique(y_valid, return_counts=True)}")

    # Test 2: Classification Tree
    print("\n[TEST 2] Classification Tree")
    print("-"*70)
    tree = ClassificationTree(random_state=test_random_state, max_depth=8)
    tree.fit(X_train, y_train)
    y_pred_tree = tree.predict(X_valid)
    f1_tree = f1_score(y_valid, y_pred_tree, average='binary', zero_division=0)
    acc_tree = accuracy_score(y_valid, y_pred_tree)
    print(f"✅ Tree F1: {f1_tree:.4f}, Accuracy: {acc_tree:.4f}")

    # Test 3: XGBoost
    print("\n[TEST 3] XGBoost")
    print("-"*70)
    results = train_XGBoost()
    print(f"✅ XGBoost Valid F1: {results['valid_f1']:.4f}")
    print(f"   Best Alpha: {results['best_alpha']}")
    print(f"   AUC: {results['auc']:.4f}")

    print("\n" + "="*70)
    print("ALL TESTS COMPLETE")
    print("="*70)
