import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from typing import Tuple, List

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

class DataProcessor:
    def __init__(self, data_root: str):
        """Initialize data processor with paths to train and test data.

        #comment to test if git is working
        
        Args:
            data_root: root path to data directory
        """
        #data_root is the directory the csvs live in
        #train_file and test_file are the arguments that will go into load_data
        self.data_root = data_root
        self.train_file = f"{data_root}/data_train_25s.csv"
        self.test_file = f"{data_root}/data_test_25s.csv"
        self.train_data = None
        self.test_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load training and test data from CSV files.
        
        Returns:
            Tuple containing training and test dataframes
        """

        self.train_data = pd.read_csv(self.train_file)
        self.test_data = pd.read_csv(self.test_file)
        return self.train_data, self.test_data
        
    def check_missing_values(self, data: pd.DataFrame) -> int:
        """Count number of missing values in dataset.
        
        Args:
            data: Input dataframe
        missing_values = 
            
        Returns:
            Number of missing values 
        """
        #I had to add an extra sum because isnull is looking for 
        #each column and when it prints it is giving several values
        #if you want the total sum of cell i.e values we need the 
        #extra .sum()
        missing_count = data.isnull().sum().sum()
        return missing_count
        
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove rows with missing values.
        
        Args:
            data: Input dataframe
            
        Returns:
            Cleaned dataframe
        """
        #that is the easiest way to do it use this pandas built in function
        #removes NaN,None,N/A and blank data
        return data.dropna()
        
    def extract_features_labels(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features and labels from dataframe, convert to numpy arrays.
    

        Args:
            data: Input dataframe
            
        Returns:
            Tuple of feature matrix X and label vector y
        """
        #Pandas notes for future assignments to help with data manipulation
        #negative numbers can denote starting at the end 
        #[:] all rows, [:,:] all rows all columns, [:,0] all rows column 0 only [0,:] Row all columns
        X = data.iloc[:,:-1] #all rows , all columns except the last
        y = data.iloc[:,-1]  #all rows, only the last column
        return  X,y
        
    
class LinearRegression:
    def __init__(self):
        """Initialize linear regression model.
        
        Args:
            learning_rate: Learning rate for gradient descent
            max_iter: Maximum number of iterations
            l2_lambda: L2 regularization strength
        """
        self.weights = None
        self.bias = None
        self.learning_rate = None
        self.max_iter = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> list[float]:
        """Train linear regression model.
        Args:
        X: Feature matrix
        y: Target vector
        Returns:
        List of loss values
        """
        if self.weights is None:
                self.weights = np.zeros(X.shape[1]) # make them all 0 to start shape is the columns of X
        if self.bias is None:
            self.bias = 0

        # TODO: Implement linear regression training
        #define some hyper parameters
        #Notes on parameter tuning learning rate of 1e-6 it exploded learning rate 1e-10 it was too small
        #1e-6 is the best so far for it would require a small amount of sample sizes to prevent it from exploding
        #MSE should be less than 71^2 == 5041 so 
        #these 2 parameters below were tweaked, I increased the learning rate slowly higher and higher until it diverged to see how far I could go
        #when I could no longer increase it to find improvements I started upping the max iterations
        #NOTE: there are other ways to define when to stop searching for a min like slope or compile time. 
        self.learning_rate = 0.095
        self.max_iter = 7000
        loss_values = []
        #get the number of samples
        n=len(y)  
        #normalize X to see if it gives a better loss
        X = (X - X.mean()) / X.std()

#The MSE is calculated in this for loop in order to tweak the hyperparameters
#our notes say MSE is how we most commonly define loss for gradient descent
#it is assumed the the criterion function is to analyze MSE for either closed form or specific values
#the fact is was in there was a bit confusing initially.

        #Perform gradient Descient 
        for i in range(self.max_iter):
            #get the predicted values of the target variable
            y_pred = self.bias + X @ self.weights
            #get the loss function
            #J = (1/n) * np.sum((y - y_pred)**2)

            # Compute the loss (Mean Squared Error)
            loss = (1/n) * np.sum((y - y_pred) ** 2)
            loss_values.append(loss)

            #get the gradients
            #grad_weights = (-2/n)*np.sum(y-y_pred) @ X
            # Compute the gradient for the weights
            grad_weights = (-2 / n) * X.T @ (y - y_pred)
            grad_bias = (-2/n)*np.sum(y-y_pred)

            #update the weights and bias
            self.weights -= self.learning_rate * grad_weights
            self.bias -= self.learning_rate * grad_bias

         

        return loss_values
        
        

                
    def fit_closed_form(self, X: np.ndarray, y: np.ndarray) -> list[float]:
        """Train linear regression model.
        #w = (X^T X)^(-1) X^T y)
        #y=W0+w1x1+....WnXn
        #Residuals is actual - pred which is most likely what this function is asking for
        #This was created initially and used to compute RSME values 
        #This is Mean Squared Error Value:  5052.274847749776
        #This is the Root Mean Squared Error Value:  71.0793559885694
        #The problem with that is we can't tweak hyperparameters to get it under 71 so gradient descent needs to be used


        
        
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            List of loss values
        """
        # TODO: Implement linear regression training
        X_with_bias = np.c_[np.ones(X.shape[0]), X] # this adds 1 column to create a bias ex. y=w0(1)+w1x1+w2x2....wnxn etc
        Xt = X_with_bias.T                      # Transpose of X with the included bias
        #@ is used for matrix mult not *
        weights_with_bias = np.linalg.inv(Xt @ X_with_bias) @ (Xt @ y)

        #split the bias and weights set them equal to the terms in the init class
        self.bias = weights_with_bias[0]
        self.weights = weights_with_bias[1:]

        #I think this is what we are looking for but I am not 100% sure
        y_pred = X_with_bias @ weights_with_bias
        #residuals is the individual losss at each point
        residuals = y - y_pred
        #we could want the absolute error but idk what gradescope wants that
        #absolute_errors = np.abs(y - y_pred)
        #return the residuals 
        return residuals.tolist()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with trained model.
        # y_pred = bias + X * weights
        # y_pred = w0 + w1x1 + w2x2 + ... + WnXn
        Args:
            X: Feature matrix
            
        Returns:
            Predicted values
        """
        y_pred = self.bias + X @ self.weights

        return y_pred

    def criterion(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate MSE loss.
        #MSE = (1/n) * Σ(y_true - y_pred)²
        #n is the number of samples 
        Args:
            y_true: True target values
            y_pred: Predicted values
            
        Returns:
            Loss value
        """
        n = len(y_pred) #either array would work
        #my original approach was going to be a for loop to represent Σ but numpy can multiply entire arrays, extremely useful 
        #in this case working with 1D arrays they should be the same size 
        MSE = (1/n) * np.sum((y_true - y_pred)**2)
        # TODO: Implement loss function
        return MSE

    def metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate RMSE.
        #RMSE = √(MSE) = √[(1/n) * Σ(y_true - y_pred)²]
        Args:
            y_true: True target values
            y_pred: Predicted values
            
        Returns:
            Metric value
        """
        # TODO: Implement RMSE calculation
        n = len(y_pred)
        #MSE = np.mean((y_true - y_pred)**2)
        MSE = (1/n) *np.sum((y_true - y_pred)**2)
        RMSE = np.sqrt(MSE)
        return RMSE

class LogisticRegression:
    def __init__(self):
        """Initialize logistic regression model.
        
        Args:
            learning_rate: Learning rate for gradient descent
            max_iter: Maximum number of iterations
        """
        self.weights = None
        self.bias = None
        self.learning_rate = None
        self.max_iter = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> list[float]:
        """Train logistic regression model with normalization and L2 regularization.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            List of loss values
        """
        # TODO: Implement logistic regression training
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Calculate prediction probabilities using normalized features.
        
        Args:
            X: Feature matrix
            
        Returns:
            Prediction probabilities
        """
        # TODO: Implement logistic regression prediction probabilities
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with trained model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted values
        """
        # TODO: Implement logistic regression prediction

    def criterion(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate BCE loss.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            
        Returns:
            Loss value
        """
        # TODO: Implement loss function
    
    def F1_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate F1 score with handling of edge cases.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            
        Returns:
            F1 score (between 0 and 1), or 0.0 for edge cases
        """
        # TODO: Implement F1 score calculation

    def label_binarize(self, y: np.ndarray) -> np.ndarray:
        """Binarize labels for binary classification.
        
        Args:
            y: Target vector
            
        Returns:
            Binarized labels
        """
        # TODO: Implement label binarization

    def get_auroc(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate AUROC score.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted probabilities
            
        Returns:
            AUROC score (between 0 and 1)
        """
        # TODO: Implement AUROC calculation

    def metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate AUROC.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            
        Returns:
            AUROC score
        """
        # TODO: Implement AUROC calculation

class ModelEvaluator:
    def __init__(self, n_splits: int = 5, random_state: int = 42):
        """Initialize evaluator with number of CV splits.
        
        Args:
            n_splits: Number of cross-validation folds
            random_state: Random state for reproducibility
        """
        self.n_splits = n_splits
        self.random_state = random_state
        self.kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        
    def cross_validation(self, model, X: np.ndarray, y: np.ndarray) -> List[float]:
        """Perform cross-validation
        
        Args:
            model: Model to be evaluated
            X: Feature matrix
            y: Target vector
            
        Returns:
            List of metric scores
        """
        # TODO: Implement cross-validation

if __name__ == "__main__":
    print("CSCE 633 Homework 1 test of functions")
    # #I put it in my own directory so that is why it is a dot when creating an instance of the class
    processor = DataProcessor(".")
    train_data, test_data = processor.load_data()
    print("Train data shape:", train_data.shape)
    print("Test data shape:", test_data.shape)
    missing_count_training = processor.check_missing_values(train_data)
    # print("Missing Values:", missing_count_training)
    # print("Missing Count Type:", type(missing_count_training))
    # print(processor.clean_data(train_data))
    clean_data_training = processor.clean_data(train_data)
    # #print the ones from the training set the test does not have a target variable
    #X_train_features, y_train_target_var = processor.extract_features_labels(clean_data_training)
    # print("X_train features are: ", X_train_features.values)
    # print("y_train target variable is: ", y_train_target_var.values)
    # print("Feature column names:", clean_data_training.columns[:-1].tolist())  # All except last  .tolist() converts pandas list to numpy array
    # print("Target column name:", clean_data_training.columns[-1])  # Last column only
    #model = LinearRegression()
    # #the features and target variables were extracted we can dot it with .values to get the numbers of those 
    #res_array = model.fit_closed_form(X_train_features.values, y_train_target_var.values)
    # print('This is a list of the residuals: ', res_array )
    #y_predict = model.predict(X_train_features.values)
    #MSE = model.criterion(y_train_target_var.values, y_predict)
    #RMSE = model.metric(y_train_target_var.values, y_predict)
    # print('This is the array of y_predict: ', y_predict)
    # print('This is Mean Squared Error Value: ', MSE )
    # print('This is the Root Mean Squared Error Value: ', RMSE)
    # print("Type of y_predict:", type(y_predict))
    # print("Type of MSE:", type(MSE))
    # print("Type of RMSE:", type(RMSE))
    #fit_array = model.fit(X_train_features.values, y_train_target_var.values)
    #print('This is a list of what the fit() does : ', fit_array )
    # 1. Feature histograms
   # 1. Feature histograms
    # clean_data_training.iloc[:,:-1].hist(bins=30, figsize=(15, 10))
    # plt.suptitle('Feature Histograms')
    # plt.tight_layout()
    # plt.savefig('histograms.png')
    # #plt.show()

    # # 2. Scatter plot (pick any two features)
    # plt.figure(figsize=(8, 6))
    # plt.scatter(clean_data_training['AH'], clean_data_training['RH'], alpha=0.6)
    # plt.xlabel('Absolute Humidity')
    # plt.ylabel('Relative Humidity')
    # plt.title(f"Correlation: {clean_data_training['AH'].corr(clean_data_training['RH']):.3f}")
    # plt.savefig('scatter.png')
    # #plt.show()

    # # 3. Correlation heatmap
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(clean_data_training.corr(), annot=True, cmap='coolwarm', center=0)
    # plt.title('Correlation Matrix')
    # plt.savefig('correlation.png')
    # #plt.show()

    # # 4. Train model and plot loss
    # model_plot = LinearRegression()
    # loss_values = model_plot.fit(X_train_features.values, y_train_target_var.values)

    # plt.figure(figsize=(8, 5))
    # plt.plot(loss_values)
    # plt.title('Training Loss')
    # plt.xlabel('Iteration')
    # plt.ylabel('MSE')
    # plt.savefig('loss.png')
    # plt.show()



    
    

    
    #PT08.S1(CO) is the target variable and that is why there is an extra column

