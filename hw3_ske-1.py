import numpy as np
import pandas as pd
from sklearn.svm import SVC

'''
Problem: University Admission Classification using SVMs

Instructions:
1. Do not use any additional libraries. Your code will be tested in a pre-built environment with only 
   the library specified in question instruction available. Importing additional libraries will result in 
   compilation errors and you will lose marks.

2. Fill in the skeleton code precisely as provided. You may define additional 
   default arguments or helper functions if necessary, but ensure the input/output format matches.
'''
class DataLoader:
    '''
    Put your call to class methods in the __init__ method. Autograder will call your __init__ method only. 
    '''
    
    def __init__(self, data_path: str):
        """
        Initialize data processor with paths to train dataset. You need to have train and validation sets processed.
        
        Args:
            data_path: absolute path to your data file
        """
        self.train_data = pd.DataFrame()
        self.val_data = pd.DataFrame()

           # Load data
        data = pd.read_csv(data_path)
        
        # Remove any rows with missing values
        data = data.dropna()
        
        # Create binary label
        data = self.create_binary_label(data)
        
        # Drop unnecessary columns
        data = data.drop(['Serial No.', 'Chance of Admit'], axis=1, errors='ignore')
        
        # Split 80-20
        #copying the same implimentation from HW2
        # Shuffle data first
        shuffled_data = data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Calculate split point
        split_marker = int(len(shuffled_data) * 0.8)
        
        # Split into train and validation
        self.train_data = shuffled_data[:split_marker]
        self.val_data = shuffled_data[split_marker:]
       
    
    def create_binary_label(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Create a binary label for the training data.
        '''
    
        if 'label' in df.columns:
            return df
        
        # use the median like we did in HW1 logistic regression
        median = df['Chance of Admit'].median()
        df['label'] = (df['Chance of Admit'] > median).astype(int)
        return df

class SVMTrainer:
    def __init__(self):
        pass

    def train(self, X_train: np.ndarray, y_train: np.ndarray, kernel: str, **kwargs) -> SVC:
        '''
        Train the SVM model with the given kernel and parameters.

        Parameters:
            X_train: Training features
            y_train: Training labels
            kernel: Kernel type
            **kwargs: Additional arguments you may use
        Returns:
            SVC: Trained sklearn.svm.SVC model
        '''
        model = SVC(kernel=kernel, **kwargs)
        model.fit(X_train, y_train)
        return model
    
    def get_support_vectors(self,model: SVC) -> np.ndarray:
        '''
        Get the support vectors from the trained SVM model.
        '''
        return model.support_vectors_
    
'''
Initialize my_best_model with the best model you found.
'''
my_best_model = SVC(kernel='rbf', C=10, gamma=0.1)  

if __name__ == "__main__":
    print("Hello, World!")
    print("="*80)
    print("HW3 QUESTION 3 - COMPREHENSIVE TEST SUITE")
    print("="*80)

    # ==============================================================================
    # TEST SUITE FOR QUESTION 3(a): Data Pre-processing
    # ==============================================================================

    print("\n" + "="*80)
    print("QUESTION 3(a): DATA PRE-PROCESSING TESTS")
    print("="*80)

    print("\n[TEST 1] Binary Label Creation (3a.1)")
    print("-"*80)
    print("Requirement: Create binary label based on 'Chance of Admit' column")
    print("             Assign 1 if value > median, otherwise 0")

    loader = DataLoader('data-2.csv')

    # Verify label exists
    assert 'label' in loader.train_data.columns, "❌ Label column missing!"
    assert 'label' in loader.val_data.columns, "❌ Label column missing in validation!"

    # Verify label is binary (0 and 1 only)
    train_labels = loader.train_data['label'].unique()
    val_labels = loader.val_data['label'].unique()
    assert set(train_labels).issubset({0, 1}), "❌ Labels are not binary!"
    assert set(val_labels).issubset({0, 1}), "❌ Validation labels are not binary!"

    print("✅ Binary label created successfully")
    print(f"   Train labels: {sorted(train_labels)}")
    print(f"   Label distribution - 0: {sum(loader.train_data['label']==0)}, "
          f"1: {sum(loader.train_data['label']==1)}")

    print("\n[TEST 2] Data Pre-processing (3a.2)")
    print("-"*80)
    print("Requirement: Apply appropriate pre-processing to data")

    # Check that unnecessary columns are removed
    assert 'Serial No.' not in loader.train_data.columns, "❌ Serial No. should be removed!"
    assert 'Chance of Admit' not in loader.train_data.columns, "❌ Chance of Admit should be removed!"

    # Check that feature columns are present
    required_features = ['GRE Score', 'TOEFL Score', 'University Rating', 
                         'SOP', 'LOR', 'CGPA', 'Research']
    for feat in required_features:
        assert feat in loader.train_data.columns, f"❌ Feature {feat} missing!"

    print("✅ Data preprocessing successful")
    print(f"   Features present: {len(loader.train_data.columns)-1}")  # -1 for label
    print(f"   Available features: {[col for col in loader.train_data.columns if col != 'label']}")

    print("\n[TEST 3] Train/Validation Split (3a.3)")
    print("-"*80)
    print("Requirement: 80-20 split, no data leakage")

    total_samples = len(loader.train_data) + len(loader.val_data)
    train_ratio = len(loader.train_data) / total_samples
    val_ratio = len(loader.val_data) / total_samples

    assert 0.75 <= train_ratio <= 0.85, f"❌ Train ratio {train_ratio:.2f} not ~0.80!"
    assert 0.15 <= val_ratio <= 0.25, f"❌ Val ratio {val_ratio:.2f} not ~0.20!"

    print("✅ 80-20 split successful")
    print(f"   Train: {len(loader.train_data)} samples ({train_ratio:.1%})")
    print(f"   Val:   {len(loader.val_data)} samples ({val_ratio:.1%})")
    print(f"   Total: {total_samples} samples")


    # ==============================================================================
    # TEST SUITE FOR QUESTION 3(b): Model Initialization
    # ==============================================================================

    print("\n" + "="*80)
    print("QUESTION 3(b): MODEL INITIALIZATION TESTS")
    print("="*80)

    print("\n[TEST 4] Initialize 3 Different SVM Models")
    print("-"*80)
    print("Requirement: Linear, RBF, and Polynomial (degree 3) kernels")

    trainer = SVMTrainer()

    # Get sample data for testing
    X_sample = loader.train_data[['CGPA', 'SOP']].values[:100]
    y_sample = loader.train_data['label'].values[:100]

    # Test 1: Linear kernel
    model_linear = trainer.train(X_sample, y_sample, kernel='linear')
    assert model_linear.kernel == 'linear', "❌ Linear kernel not set correctly!"
    print("✅ Linear kernel model initialized")

    # Test 2: RBF kernel
    model_rbf = trainer.train(X_sample, y_sample, kernel='rbf')
    assert model_rbf.kernel == 'rbf', "❌ RBF kernel not set correctly!"
    print("✅ RBF kernel model initialized")

    # Test 3: Polynomial kernel (degree 3)
    model_poly = trainer.train(X_sample, y_sample, kernel='poly', degree=3)
    assert model_poly.kernel == 'poly', "❌ Poly kernel not set correctly!"
    assert model_poly.degree == 3, "❌ Polynomial degree not set to 3!"
    print("✅ Polynomial (degree 3) kernel model initialized")


    # ==============================================================================
    # TEST SUITE FOR QUESTION 3(c): Feature Selection and Model Training
    # ==============================================================================

    print("\n" + "="*80)
    print("QUESTION 3(c): FEATURE SELECTION AND MODEL TRAINING TESTS")
    print("="*80)

    print("\n[TEST 5] Train Models with Required Feature Combinations")
    print("-"*80)
    print("Requirement: Train each SVM with:")
    print("  • CGPA and SOP")
    print("  • CGPA and GRE Score")
    print("  • SOP and LOR")
    print("  • LOR and GRE Score")

    feature_combinations = [
        ['CGPA', 'SOP'],
        ['CGPA', 'GRE Score'],
        ['SOP', 'LOR'],
        ['LOR', 'GRE Score']
    ]

    kernels = ['linear', 'rbf', 'poly']
    results = []

    for features in feature_combinations:
        # Verify features exist
        for feat in features:
            assert feat in loader.train_data.columns, f"❌ Feature {feat} not in data!"
        
        # Extract features
        X_train = loader.train_data[features].values
        y_train = loader.train_data['label'].values
        X_val = loader.val_data[features].values
        y_val = loader.val_data['label'].values
        
        for kernel in kernels:
            # Train model
            if kernel == 'poly':
                model = trainer.train(X_train, y_train, kernel=kernel, degree=3)
            else:
                model = trainer.train(X_train, y_train, kernel=kernel)
            
            # Verify model is trained
            assert hasattr(model, 'support_vectors_'), f"❌ Model not fitted for {features} + {kernel}!"
            
            # Calculate accuracy
            train_acc = model.score(X_train, y_train)
            val_acc = model.score(X_val, y_val)
            
            results.append({
                'features': f"{features[0]} + {features[1]}",
                'kernel': kernel,
                'train_acc': train_acc,
                'val_acc': val_acc
            })

    print("✅ All 12 models trained successfully (3 kernels × 4 feature pairs)")
    print(f"   Total combinations tested: {len(results)}")

    # Display results
    print("\nResults Summary:")
    print("-"*80)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))


    # ==============================================================================
    # TEST SUITE FOR QUESTION 3(d): Support Vectors
    # ==============================================================================

    print("\n" + "="*80)
    print("QUESTION 3(d): SUPPORT VECTORS TESTS")
    print("="*80)

    print("\n[TEST 6] Identify Support Vectors")
    print("-"*80)
    print("Requirement: Identify support vectors for each model and feature combination")

    X_train = loader.train_data[['CGPA', 'SOP']].values
    y_train = loader.train_data['label'].values

    for kernel in kernels:
        if kernel == 'poly':
            model = trainer.train(X_train, y_train, kernel=kernel, degree=3)
        else:
            model = trainer.train(X_train, y_train, kernel=kernel)
        
        # Get support vectors
        support_vecs = trainer.get_support_vectors(model)
        
        # Verify support vectors exist
        assert support_vecs is not None, f"❌ No support vectors for {kernel}!"
        assert len(support_vecs) > 0, f"❌ Empty support vectors for {kernel}!"
        assert support_vecs.shape[1] == 2, f"❌ Support vectors wrong dimension for {kernel}!"
        
        print(f"✅ {kernel:8s} kernel: {len(support_vecs):3d} support vectors identified")


    # ==============================================================================
    # TEST SUITE FOR QUESTION 3(e): Result Visualization
    # ==============================================================================

    print("\n" + "="*80)
    print("QUESTION 3(e): RESULT VISUALIZATION TESTS")
    print("="*80)

    print("\n[TEST 7] Visualize Predictions")
    print("-"*80)
    print("Requirement: Visualize predictions for each kernel-feature combination")
    print("             Use color coding for labels")

    # Test that we can create visualizations (we'll create one example)
    X_train = loader.train_data[['CGPA', 'SOP']].values
    y_train = loader.train_data['label'].values

    model = trainer.train(X_train, y_train, kernel='linear')

    # Test prediction functionality
    y_pred = model.predict(X_train)
    assert len(y_pred) == len(y_train), "❌ Prediction length mismatch!"
    assert set(y_pred).issubset({0, 1}), "❌ Predictions not binary!"

    print("✅ Visualization capability verified")
    print(f"   Can predict on training data")
    print(f"   Predictions are binary (0/1)")
    print(f"   Ready to create decision boundary plots")
    print("\n   NOTE: Actual visualizations should be created and included in report")


    # ==============================================================================
    # TEST SUITE FOR QUESTION 3(f): Result Analysis
    # ==============================================================================

    print("\n" + "="*80)
    print("QUESTION 3(f): RESULT ANALYSIS TESTS")
    print("="*80)

    print("\n[TEST 8] Find Best Model Combination")
    print("-"*80)
    print("Requirement: Determine best feature-kernel combination")
    print("             Aim for 0.83 accuracy on test set")

    # Find best from our results
    best_result = max(results, key=lambda x: x['val_acc'])
    print(f"✅ Best validation performance identified:")
    print(f"   Features: {best_result['features']}")
    print(f"   Kernel: {best_result['kernel']}")
    print(f"   Train Accuracy: {best_result['train_acc']:.4f}")
    print(f"   Val Accuracy: {best_result['val_acc']:.4f}")

    print("\n[TEST 9] my_best_model Variable")
    print("-"*80)
    print("Requirement: Initialize my_best_model with best configuration")

    # Check that my_best_model exists
    assert my_best_model is not None, "❌ my_best_model is None!"
    assert isinstance(my_best_model, SVC), "❌ my_best_model is not an SVC instance!"

    print("✅ my_best_model variable exists")
    print(f"   Type: {type(my_best_model)}")
    print(f"   Current config: kernel={my_best_model.kernel}")
    print("\n   NOTE: Update my_best_model with your best hyperparameters after tuning!")


    # ==============================================================================
    # ADDITIONAL VALIDATION TESTS
    # ==============================================================================

    print("\n" + "="*80)
    print("ADDITIONAL VALIDATION TESTS")
    print("="*80)

    print("\n[TEST 10] Data Integrity")
    print("-"*80)

    # Check for no data leakage (features should not include label)
    X_train_full = loader.train_data.drop('label', axis=1).values
    assert X_train_full.shape[1] == 7, "❌ Wrong number of features (should be 7)!"
    print("✅ No data leakage - label not in features")

    # Check data types
    assert loader.train_data['label'].dtype in [np.int64, np.int32, int], "❌ Label not integer!"
    print("✅ Label is integer type")

    # Check for missing values
    assert not loader.train_data.isnull().any().any(), "❌ Missing values in train data!"
    assert not loader.val_data.isnull().any().any(), "❌ Missing values in val data!"
    print("✅ No missing values in processed data")

    print("\n[TEST 11] Model Training Capability")
    print("-"*80)

    # Test that models can be trained and make predictions
    X_train = loader.train_data[['CGPA', 'SOP']].values
    y_train = loader.train_data['label'].values
    X_val = loader.val_data[['CGPA', 'SOP']].values
    y_val = loader.val_data['label'].values

    model = trainer.train(X_train, y_train, kernel='rbf', C=1.0, gamma='scale')
    train_acc = model.score(X_train, y_train)
    val_acc = model.score(X_val, y_val)

    assert 0.0 <= train_acc <= 1.0, "❌ Train accuracy out of range!"
    assert 0.0 <= val_acc <= 1.0, "❌ Val accuracy out of range!"
    assert train_acc > 0.5, "❌ Train accuracy too low (worse than random)!"

    print("✅ Model trains and predicts correctly")
    print(f"   Train accuracy: {train_acc:.4f}")
    print(f"   Val accuracy: {val_acc:.4f}")


    # ==============================================================================
    # SUMMARY
    # ==============================================================================

    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    all_tests = [
        "Binary Label Creation (3a.1)",
        "Data Pre-processing (3a.2)",
        "Train/Validation Split (3a.3)",
        "Model Initialization - 3 Kernels (3b)",
        "Feature Selection & Training - 12 Models (3c)",
        "Support Vector Identification (3d)",
        "Visualization Capability (3e)",
        "Best Model Selection (3f)",
        "my_best_model Configuration (3f)",
        "Data Integrity Check",
        "Model Training Capability"
    ]

    print(f"\n✅ All {len(all_tests)} test categories PASSED!\n")
    for i, test in enumerate(all_tests, 1):
        print(f"   {i:2d}. {test}")

    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("✓ Skeleton code is complete and working")
    print("✓ All 12 model combinations tested")
    print("✓ Support vectors can be extracted")
    print()
    print("TODO:")
    print("☐ Hyperparameter tuning (try different C, gamma values)")
    print("☐ Create decision boundary visualizations for report")
    print("☐ Update my_best_model with best hyperparameters")
    print("☐ Aim for 0.83+ accuracy on test set")
    print("☐ Write analysis and discussion in LaTeX report")

    print("\n" + "="*80)
    print("TEST SUITE COMPLETE")
    print("="*80)
    
    # ==========================================================================
    # GENERATE PNG FILES FOR LATEX REPORT
    # ==========================================================================
    
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS FOR LATEX REPORT")
    print("="*80)
    
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from mpl_toolkits.mplot3d import Axes3D
    
    # Plot 1: Question 1 - Optimal Hyperplane (7 points dataset)
    print("\n[1/7] Creating q1_hyperplane.png...")
    blue_points = np.array([[3, 6], [2, 2], [4, 4], [1, 3]])
    red_points = np.array([[2, 0], [4, 2], [4, 0]])
    
    plt.figure(figsize=(8, 6))
    plt.scatter(blue_points[:, 0], blue_points[:, 1], c='blue', s=100, marker='o', 
                label='Blue', edgecolors='black', linewidth=1.5)
    plt.scatter(red_points[:, 0], red_points[:, 1], c='red', s=100, marker='s', 
                label='Red', edgecolors='black', linewidth=1.5)
    plt.xlabel('$X_1$', fontsize=12)
    plt.ylabel('$X_2$', fontsize=12)
    plt.title('Question 1: Data Points (Add Your Hyperplane)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 5)
    plt.ylim(-1, 7)
    plt.tight_layout()
    plt.savefig('q1_hyperplane.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Created: q1_hyperplane.png")
    
    # Plot 2: Question 2 - Original 2D Space (XOR problem)
    print("[2/7] Creating q2_original_space.png...")
    positive = np.array([[1, 1], [-1, -1]])
    negative = np.array([[1, -1], [-1, 1]])
    
    plt.figure(figsize=(7, 7))
    plt.scatter(positive[:, 0], positive[:, 1], c='green', s=150, marker='o', 
                label='Positive (+1)', edgecolors='black', linewidth=2)
    plt.scatter(negative[:, 0], negative[:, 1], c='purple', s=150, marker='X', 
                label='Negative (-1)', edgecolors='black', linewidth=2)
    plt.xlabel('$x_1$', fontsize=12)
    plt.ylabel('$x_2$', fontsize=12)
    plt.title('Question 2: Original 2D Space (Not Linearly Separable)', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linewidth=0.5)
    plt.axvline(x=0, color='k', linewidth=0.5)
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.tight_layout()
    plt.savefig('q2_original_space.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Created: q2_original_space.png")
    
    # Plot 3: Question 2 - Transformed 3D Space
    print("[3/7] Creating q2_transformed_space.png...")
    X_q2 = np.array([[1, 1], [-1, -1], [1, -1], [-1, 1]])
    y_q2 = np.array([1, 1, -1, -1])
    X_transformed = np.column_stack([X_q2, X_q2[:, 0] * X_q2[:, 1]])
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    positive_idx = y_q2 == 1
    negative_idx = y_q2 == -1
    ax.scatter(X_transformed[positive_idx, 0], X_transformed[positive_idx, 1], 
               X_transformed[positive_idx, 2], c='green', s=150, marker='o', 
               label='Positive (+1)', edgecolors='black', linewidth=2)
    ax.scatter(X_transformed[negative_idx, 0], X_transformed[negative_idx, 1], 
               X_transformed[negative_idx, 2], c='purple', s=150, marker='X', 
               label='Negative (-1)', edgecolors='black', linewidth=2)
    xx, yy = np.meshgrid(np.linspace(-1.5, 1.5, 10), np.linspace(-1.5, 1.5, 10))
    zz = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, alpha=0.3, color='yellow')
    ax.set_xlabel('$\\phi_1 = x_1$', fontsize=11)
    ax.set_ylabel('$\\phi_2 = x_2$', fontsize=11)
    ax.set_zlabel('$\\phi_3 = x_1 x_2$', fontsize=11)
    ax.set_title('Question 2: Transformed 3D Space (Linearly Separable)', 
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('q2_transformed_space.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Created: q2_transformed_space.png")
    
    # Plots 4-7: Question 3 - Decision Boundaries
    feature_combinations = [
        (['CGPA', 'SOP'], 'decision_boundaries_cgpa_sop.png'),
        (['CGPA', 'GRE Score'], 'decision_boundaries_cgpa_gre.png'),
        (['SOP', 'LOR'], 'decision_boundaries_sop_lor.png'),
        (['LOR', 'GRE Score'], 'decision_boundaries_lor_gre.png')
    ]
    
    kernels_list = [('linear', 'Linear'), ('rbf', 'RBF'), ('poly', 'Polynomial (degree=3)')]
    
    for idx, (features, filename) in enumerate(feature_combinations, start=4):
        print(f"[{idx}/7] Creating {filename}...")
        
        X_train = loader.train_data[features].values
        y_train = loader.train_data['label'].values
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for ax_idx, (kernel, kernel_name) in enumerate(kernels_list):
            ax = axes[ax_idx]
            
            if kernel == 'poly':
                model = trainer.train(X_train, y_train, kernel=kernel, degree=3)
            else:
                model = trainer.train(X_train, y_train, kernel=kernel)
            
            h = 0.02
            x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
            y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
            
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            ax.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(['#FFAAAA', '#AAAAFF']))
            ax.contour(xx, yy, Z, colors='black', linewidths=1, levels=[0.5])
            ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, 
                      cmap=ListedColormap(['red', 'blue']), edgecolor='black', s=30, alpha=0.7)
            
            support_vecs = trainer.get_support_vectors(model)
            ax.scatter(support_vecs[:, 0], support_vecs[:, 1], s=80, 
                      facecolors='none', edgecolors='green', linewidths=2)
            
            ax.set_xlabel(features[0], fontsize=11)
            ax.set_ylabel(features[1], fontsize=11)
            ax.set_title(f'{kernel_name}\\n(SVs: {len(support_vecs)})', 
                        fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Decision Boundaries: {features[0]} + {features[1]}', 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Created: {filename}")
    
    print("\n" + "="*80)
    print("ALL VISUALIZATIONS GENERATED!")
    print("="*80)
    print("\nGenerated files:")
    print("  1. q1_hyperplane.png")
    print("  2. q2_original_space.png")
    print("  3. q2_transformed_space.png")
    print("  4. decision_boundaries_cgpa_sop.png")
    print("  5. decision_boundaries_cgpa_gre.png")
    print("  6. decision_boundaries_sop_lor.png")
    print("  7. decision_boundaries_lor_gre.png")
    print("\nYou can now compile your LaTeX:")
    print("  pdflatex Joseph_DeLeonardis_HW3.tex")
    print("="*80)