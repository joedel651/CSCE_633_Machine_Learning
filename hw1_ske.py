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
                self.weights = np.random.randn(X.shape[1]) * 0.01
                #self.weights = np.zeros(X.shape[1]) # make them all 0 to start shape is the columns of X
        if self.bias is None:
            self.bias = 0

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        rmse_scores = []

        # TODO: Implement linear regression training
        #define some hyper parameters
        #Notes on parameter tuning learning rate of 1e-6 it exploded learning rate 1e-10 it was too small
        #1e-6 is the best so far for it would require a small amount of sample sizes to prevent it from exploding
        #MSE should be less than 71^2 == 5041 so 
        #these 2 parameters below were tweaked, I increased the learning rate slowly higher and higher until it diverged to see how far I could go
        #when I could no longer increase it to find improvements I started upping the max iterations
        #NOTE: there are other ways to define when to stop searching for a min like slope or compile time. 
        #the best i can get with the traning set is about 71.07 with both gradient descent and closed form soltion
        self.learning_rate = 0.1
        self.max_iter = 100000
        loss_values = []
        #get the number of samples
        n=len(y)  
        #normalize X to see if it gives a better loss
        self.X_mean = X.mean(axis=0)  # Save the mean
        self.X_std = X.std(axis=0)    # Save the std
        X_normalized = (X - self.X_mean) / self.X_std
        

#The MSE is calculated in this for loop in order to tweak the hyperparameters
#our notes say MSE is how we most commonly define loss for gradient descent
#it is assumed the the criterion function is to analyze MSE for either closed form or specific values
#the fact is was in there was a bit confusing initially.

    

        #Perform gradient Descient 
        for i in range(self.max_iter):
            #get the predicted values of the target variable
            #y_pred = self.bias + X @ self.weights
            y_pred = np.dot(X_normalized, self.weights) + self.bias
            #get the loss function
            #J = (1/n) * np.sum((y - y_pred)**2)

            # Compute the loss (Mean Squared Error)
            #loss = (1/n) * np.sum((y - y_pred) ** 2)
            #l2 regularization
            #NOTE: both methods of loss return the same thing because i hit the floor for performance
            lambda_ = 0.001
            loss = (1/n) * np.sum((y - y_pred) ** 2) + lambda_ * np.sum(model.weights ** 2)
            loss_values.append(loss)

            #get the gradients
            #grad_weights = (-2/n)*np.sum(y-y_pred) @ X
            # Compute the gradient for the weights
            #grad_weights = (-2 / n) * X.T @ (y - y_pred)
            grad_weights = (-2 / n) * X_normalized.T @ (y - y_pred)

            grad_bias = (-2/n)*np.sum(y-y_pred)

            #update the weights and bias
            self.weights -= self.learning_rate * grad_weights
            self.bias -= self.learning_rate * grad_bias
        
        final_rmse = self.metric(y, np.dot((X - self.X_mean) / self.X_std, self.weights) + self.bias)
        print("Final RMSE:", final_rmse)


         

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
        #residuals = y - y_pred
        loss = self.metric(y, y_pred)
        #we could want the absolute error but idk what gradescope wants that
        #absolute_errors = np.abs(y - y_pred)
        #return the residuals 
        return [loss]
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with trained model.
        # y_pred = bias + X * weights
        # y_pred = w0 + w1x1 + w2x2 + ... + WnXn
        Args:
            X: Feature matrix
            
        Returns:
            Predicted values
        """
        X_normalized = (X - self.X_mean) / self.X_std  
        y_pred = self.bias + X_normalized @ self.weights  # Use normalized version
        #the non-normalized version was giving too high of values
        #y_pred = self.bias + X @ self.weights

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
        self.X_mean = None
        self.X_std = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> list[float]:
        """Train logistic regression model with normalization and L2 regularization.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            List of loss values
        """
        # TODO: Implement logistic regression training
        if self.weights is None:
            self.weights = np.random.randn(X.shape[1]) * 0.01
            #self.weights = np.zeros(X.shape[1]) # make them all 0 to start shape is the columns of X
        if self.bias is None:
            self.bias = 0


        #define some hyper parameters
        self.learning_rate = 0.1
        self.max_iter = 5000
        loss_values = []
        #get the number of samples
        n=len(y)  
        #normalize X to see if it gives a better loss
        self.X_mean = X.mean(axis=0)  # Save the mean
        self.X_std = X.std(axis=0)    # Save the std
        X_normalized = (X - self.X_mean) / self.X_std
        

    

        #Perform gradient Descient 
        for i in range(self.max_iter):
            #get the predicted values of the target variable
            #y_pred = self.bias + X @ self.weights
            #y_pred = np.dot(X_normalized, self.weights) + self.bias

            #get the loss function
            #J = (1/n) * np.sum((y - y_pred)**2)
            z = np.dot(X_normalized, self.weights) + self.bias
            z = np.clip(z, -250, 250)  # Prevent overflow
            #sigmoid is specific to logistic regression 
            y_pred = 1 / (1 + np.exp(-z))  # Sigmoid activation
            #l2 regularization to prevent overfitting
            #NOTE: both methods of loss return the same thing because i hit the floor for performance
            lambda_ = 0.001
            #most ML liberies use 1e-15, it doesn thave to be that value but it
            #prevents log(0)
            epsilon = 1e-15
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Prevent log(0)
            y_binary = (y > np.median(y)).astype(int)  
            loss = -(1/n) * np.sum(y_binary * np.log(y_pred) + (1 - y_binary) * np.log(1 - y_pred)) + lambda_ * np.sum(self.weights ** 2)
            loss_values.append(loss)

            #get the gradients take the l2 regularization into account
            grad_weights = (1/n) * X_normalized.T @ (y_pred - y_binary) + 2 * lambda_ * self.weights
            grad_bias = (1/n) * np.sum(y_pred - y_binary)

            #update the weights and bias
            self.weights -= self.learning_rate * grad_weights
            self.bias -= self.learning_rate * grad_bias
        



         

        return loss_values
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Calculate prediction probabilities using normalized features.
        
        Args:
            X: Feature matrix
            
        Returns:
            Prediction probabilities
        """
        # TODO: Implement logistic regression prediction probabilities
        X_normalized = (X - self.X_mean) / self.X_std  
        z = X_normalized @ self.weights + self.bias
        # basically the same but we need to take sigmoid into account
        # weights can be too before converging lets clip (just in case)
        z = np.clip(z, -250, 250)  # Prevent overflow
        y_pred = 1 / (1 + np.exp(-z))  # Sigmoid activation
        return y_pred
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with trained model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted values
        """
        # TODO: Implement logistic regression prediction
        #get the probabilities from the function above
        probabilities = self.predict_proba(X)
        #I am going to guess and pick a threshold if it is above that 
        #threshold we will consider it a prediciton
        predictions = (probabilities >= 0.5).astype(int)

    
        return predictions


    def criterion(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate BCE loss.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            
        Returns:
            Loss value
        """
        # TODO: Implement loss function
        #y_binary = (y > np.median(y)).astype(int) 
        #for the report this line needs to be changed to label 1 if greater than 1000
        y_binary = (y > 1000).astype(int) 

        #Add epsilon to prevent log(0)
        epsilon = 1e-15
        #1-epsilon is the upperbound, epsilon is the lowerbound
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        #calculate BCE without L2
        n = len(y_pred)
        BCE = bce = -(1/n) * np.sum(y_binary * np.log(y_pred) + (1 - y_binary) * np.log(1 - y_pred))
        return BCE 
    
    def F1_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate F1 score with handling of edge cases.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            
        Returns:
            F1 score (between 0 and 1), or 0.0 for edge cases
        """
        # TODO: Implement F1 score calculation
        #count True Positives False Positives etc 

        #Need to make the input variables binary first
        y_true_binary = (y_true > np.median(y_true)).astype(int)
        #We would have to do something different if it was coming from the 
        #Predict_proba function but think we can assume the input is not those
        #values
        y_pred_binary = (y_pred > np.median(y_pred)).astype(int)

        #True Positive
        TP = np.sum((y_true_binary == 1)& (y_pred_binary == 1))
        #True Negative
        TN = np.sum((y_true_binary == 0)& (y_pred_binary == 0))
        #False Positive
        FP = np.sum((y_true_binary == 0)& (y_pred_binary == 1))
        #False Negative 
        FN = np.sum((y_true_binary == 1)& (y_pred_binary == 0))

        #edge case so we do not divide by zero
        if TP == 0:
            return 0.0
        
        precision = (TP)/(TP+FP)
        recall = (TP)/(TP+FN)

        F1 = 2* (precision*recall)/(precision+recall)
        return F1


    def label_binarize(self, y: np.ndarray) -> np.ndarray:
        """Binarize labels for binary classification.
        
        Args:
            y: Target vector
            
        Returns:
            Binarized labels
        """
        # TODO: Implement label binarization
        #threshold comparison 
        threshold = np.median(y)
        return (y > threshold).astype(int)


    def get_auroc(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate AUROC score.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted probabilities
            
        Returns:
            AUROC score (between 0 and 1)
        """
        # TODO: Implement AUROC calculation
        #calculate the area under the curve
        #basically this is a way to predict where a point will end up in the confusion matrix
        #y_true_binary = self.label_binarize(y_true)
        y_true_binary = (y_true > 1000).astype(int)

        #sort in descending order
        sort_desc = np.argsort(-y_pred)
        y_pred_sorted = y_pred[sort_desc]
        y_true_sorted = y_true_binary[sort_desc]

        positive = np.sum(y_true_sorted)  #sum of positive entries 
        negative = len(y_true_sorted) - positive #all the entries not positive

        #make an edge case like the other function
        if positive == 0 or negative == 0:
            return 0.0 #in this case the AUROC would not be defined

        # Compute TPR and FPR 
        TPR = np.cumsum(y_true_sorted) / positive
        FPR = np.cumsum(1 - y_true_sorted) / negative

        #all this does is make sure that our 0s start at 0 to help get a better graph
        #(array,index,value)
        FPR = np.insert(FPR, 0, 0)
        TPR = np.insert(TPR, 0, 0)
        #trapz allows us to take the area under a curve
        auroc = np.trapz(TPR,FPR)

        return auroc


    def metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate AUROC.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            
        Returns:
            AUROC score
        """
        # TODO: Implement AUROC calculation
        #one liner but need to solve the one above
        return self.get_auroc(y_true, y_pred)

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
        #there is different methods of cross-validation 
        #leave one out or K fold would do kf is the most ideal and the library is linked
        scores = []
        for fold, (train_index, test_index) in enumerate(self.kf.split(x)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            #since we split the model differently we have to train it again
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = model.metric(y_test, y_pred)
            scores.append(score)

        return scores




if __name__ == "__main__":
    print("CSCE 633 Homework 1 test of functions")
    # #I put it in my own directory so that is why it is a dot when creating an instance of the class
    processor = DataProcessor(".")
    train_data, test_data = processor.load_data()
    print("Train data shape:", train_data.shape)
    print("Test data shape:", test_data.shape)
    missing_count_training_train = processor.check_missing_values(train_data)
    missing_count_training_test  = processor.check_missing_values(test_data)
    # print("Missing Values:", missing_count_training)
    # print("Missing Count Type:", type(missing_count_training))
    # print(processor.clean_data(train_data))
    clean_data_training = processor.clean_data(train_data)
    clean_data_test = processor.clean_data(test_data)
    X_train_features, y_train_target_var = processor.extract_features_labels(clean_data_training)
    X_test_features = clean_data_test  
    # print("X_train features are: ", X_train_features.values)
    # print("y_train target variable is: ", y_train_target_var.values)
    # print("Feature column names:", clean_data_training.columns[:-1].tolist())  # All except last  .tolist() converts pandas list to numpy array
    # print("Target column name:", clean_data_training.columns[-1])  # Last column only
    #model = LinearRegression()
    #model.fit_closed_form(X_train_features.values, y_train_target_var.values)
    #gradient_descent_problem = model.fit(X_train_features.values, y_train_target_var.values)
    # #the features and target variables were extracted we can dot it with .values to get the numbers of those 
    #closed = model.fit_closed_form(X_train_features.values, y_train_target_var.values)
    #print('The closed form RSME Is ', closed )
    #predict the model 
    #y_predict = model.predict(X_train_features.values)
    #print(y_predict)
    #MSE = model.criterion(y_train_target_var.values, y_predict)
    #RMSE = model.metric(y_train_target_var.values, y_predict)
    # print('This is the array of y_predict: ', y_predict)
    #print('This is Mean Squared Error Value: ', MSE )
    #print('This is the Root Mean Squared Error Value: ', RMSE)
    # print("Type of y_predict:", type(y_predict))
    #print("Type of MSE:", type(MSE))
    #print("Type of RMSE:", type(RMSE))


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

    # Find strongest correlations in the entire dataset
    # ah_rh_corr = clean_data_training['AH'].corr(clean_data_training['RH'])
    # print(f"Correlation between AH and RH: {ah_rh_corr:.4f}")
    #corr_matrix = clean_data_training.corr()
    #corr_matrix.to_csv("correlation_matrix.csv")
    # print(corr_matrix)
        # Clean and prepare test data


    #get the MSE 
    # mse = model.criterion(y_train_target_var.values, y_predict)
    # print(f"MSE: {mse:.2f}")

    
    
    # Plot loss
    # plt.figure(figsize=(8, 5))
    # plt.plot(gradient_descent_problem )
    # plt.title('Training Loss')
    # plt.xlabel('Iteration')
    # plt.ylabel('MSE')
    # plt.savefig('loss.png')
    # plt.show()

    # Now add the binary classification part:
    print("=== Binary Label Creation ===")
    print(f"Original PT08.S1(CO) range: {y_train_target_var.min():.2f} to {y_train_target_var.max():.2f}")

    # Create and train logistic regression model
    logistic_model = LogisticRegression()

    # Create binary labels using the 1000 threshold
    y_binary = logistic_model.label_binarize(y_train_target_var.values)
    print(f"Binary labels distribution:")
    print(f"  Label 0 (≤ 1000): {np.sum(y_binary == 0)} samples ({np.mean(y_binary == 0)*100:.1f}%)")
    print(f"  Label 1 (> 1000): {np.sum(y_binary == 1)} samples ({np.mean(y_binary == 1)*100:.1f}%)")

    # Train the logistic regression model
    print("\n=== Training Logistic Regression ===")
    loss_values = logistic_model.fit(X_train_features.values, y_train_target_var.values)
    #get the BCE
    print(f"Final BCE Loss: {loss_values[-1]:.4f}")

    # Plot the loss 
    plt.figure(figsize=(10, 6))
    plt.plot(loss_values)
    plt.title('Logistic Regression Training Loss (Binary Cross Entropy)')
    plt.xlabel('Iteration')
    plt.ylabel('BCE Loss')
    plt.grid(True)
    plt.savefig('logistic_loss.png', dpi=300, bbox_inches='tight')
    plt.show()

        # Make predictions on training data for evaluation
    print("\n=== Model Evaluation ===")
    y_train_proba = logistic_model.predict_proba(X_train_features.values)
    y_train_pred = logistic_model.predict(X_train_features.values)

    # Calculate metrics on training data
    f1_train = logistic_model.F1_score(y_train_target_var.values, y_train_proba)
    auroc_train = logistic_model.get_auroc(y_train_target_var.values, y_train_proba)

    print(f"Training F1 Score: {f1_train:.4f}")
    print(f"Training AUROC: {auroc_train:.4f}")
    plt.savefig('logistic_bce_loss.png', dpi=300, bbox_inches='tight')

    

    


    





    



    
    

    
    #PT08.S1(CO) is the target variable and that is why there is an extra column

