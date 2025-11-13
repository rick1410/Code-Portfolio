import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.optimize import minimize
from statsmodels.tools.numdiff import approx_hess1
from numpy.linalg import inv


class PoissonAdRegressionMCMC:
    """
    Class wrapper for your Assignment 3 code.
    All original functions are now methods (mostly static),
    with their internal code unchanged.
    """

    @staticmethod
    def handleData():
        
        #Import the data
        data = pd.read_excel(r'/Users/rickteeuwissen/Downloads/Assignment3Dataset.xlsx')
        return data

    @staticmethod
    def OLS(data):
        
        # Add a constant term to the independent variable
        X = sm.add_constant(data['x'])  
        
        # Fit the OLS regression model
        model = sm.OLS(data['y'], X).fit()  
    
        
        # Range of x values for the line
        x_values = pd.Series([data['x'].min(), data['x'].max()])  
        
        # Add constant term for prediction
        x_values_with_const = sm.add_constant(x_values)  
        
        # Predict corresponding y values
        y_values = model.predict(x_values_with_const) 
    
        return x_values,y_values

    @staticmethod
    def scatterPlot(data):
        
        # Scatter plot of 'y' against 'x'
        plt.figure(figsize=(8, 6))  
        plt.scatter(data['x'], data['y'], color='blue')  
        plt.title('Scatter Plot of Sales against Advertisement expenditure')  
        plt.xlabel('Advertisement expenditure')  
        plt.ylabel('Sales')  
        plt.show()  
        
        # Scatter plot of 'y' against 'x' with OLS regression
        plt.figure(figsize=(8, 6))  
        plt.scatter(data['x'], data['y'], color='blue')  
        plt.plot(PoissonAdRegressionMCMC.OLS(data)[0], PoissonAdRegressionMCMC.OLS(data)[1], color='red', label='OLS Regression')
        plt.title('Scatter Plot and OLS regression of Sales against Advertisement expenditure')  
        plt.xlabel('Advertisement expenditure')  
        plt.ylabel('Sales')  
        plt.show()  

    @staticmethod
    def loglikelihood(theta, data):
        beta_0 = theta[0]
        beta_1 = theta[1]
        loglikelihood = 0
        
        for i in range(len(data)):
            loglikelihood += data['y'][i] * (beta_0 + beta_1 * data['x'][i]) - np.exp(beta_0 + beta_1 * data['x'][i])
            
        return loglikelihood

    @staticmethod
    # Define the negative log-likelihood to minimize
    def neg_loglikelihood(theta, data):
        return -PoissonAdRegressionMCMC.loglikelihood(theta, data)

    @staticmethod
    def InverseHessianMLE(data,initial_theta):
        
       # Minimize the negative log-likelihood
        result = minimize(PoissonAdRegressionMCMC.neg_loglikelihood, initial_theta, args=(data,), method='BFGS')
        
        # Maximum likelihood estimates
        max_likelihood_estimates = result.x
    
        # Compute Hessian at maximum likelihood estimates
        hessian = approx_hess1(max_likelihood_estimates, PoissonAdRegressionMCMC.neg_loglikelihood, args=(data,))
        inverse_hessian = inv(hessian)
        
        return inverse_hessian,max_likelihood_estimates

    @staticmethod
    def serial_corrolation(data):
            auto_correlation = np.corrcoef(data[:-1], data[1:])[0, 1]
            return auto_correlation

    @staticmethod
    def randomWalkMetropolisHastings(data,ndraws,burn_in,initial_theta,maximum_likelihood_estimates,var):
        
        thetas = np.empty(ndraws, dtype = object)
        thetas[0] = initial_theta
        
        candidate_list = []
        accepted_beta_0 = []
        accepted_beta_1 = []
        
    
        for i in range(1,ndraws):
    
            candidate_theta = np.random.multivariate_normal(thetas[i - 1], var)
            candidate_list.append(candidate_theta)
            
            ratio = np.exp(PoissonAdRegressionMCMC.loglikelihood(candidate_theta,data) - PoissonAdRegressionMCMC.loglikelihood(thetas[i-1],data))
            
            alpha = min(ratio, 1)
            
            u = np.random.uniform()
            
            if u<= alpha:
    
                thetas[i]= candidate_theta
                if i > burn_in:
                    accepted_beta_0.append(candidate_theta[0])
                    accepted_beta_1.append(candidate_theta[1])
    
            else:
                thetas[i] = thetas[i - 1]
                
                
        candidate_list = candidate_list[burn_in:]
        thetas = thetas[burn_in:]
        
        #Extracting beta_0 and beta_1 from the candidate draws
        candidate_beta_0 = np.array([item[0] for item in candidate_list])
        candidate_beta_1 = np.array([item[1] for item in candidate_list])
        
        # Extracting beta_0 and beta_1 from the candidate draws
        beta_0 = np.array([item[0] for item in thetas])
        beta_1 = np.array([item[1] for item in thetas])
        
        
        acceptance_percentage_beta_0 = (len(accepted_beta_0) / ndraws) * 100
        acceptance_percentage_beta_1 = (len(accepted_beta_1) / ndraws) * 100
        
        corr_beta_0 = PoissonAdRegressionMCMC.serial_corrolation(beta_0)
        corr_beta_1 = PoissonAdRegressionMCMC.serial_corrolation(beta_1)
        
        mean_beta_0 = np.mean(beta_0)
        mean_beta_1 = np.mean(beta_1)
        
    
        # Calculate the 2.5% and 97.5% quantiles for beta_0
        beta_0_lower = np.percentile(beta_0, 2.5)
        beta_0_upper = np.percentile(beta_0, 97.5)
    
        # Calculate the 2.5% and 97.5% quantiles for beta_1
        beta_1_lower = np.percentile(beta_1, 2.5)
        beta_1_upper = np.percentile(beta_1, 97.5)
    
        # Construct the 95% posterior interval for beta_0 and beta_1
        beta_0_interval = (beta_0_lower, beta_0_upper)
        beta_1_interval = (beta_1_lower, beta_1_upper)
    
        
        print(f'The acceptance percentage  is: {np.round(acceptance_percentage_beta_0,3)} %')
        print(f'The first order serial correlation of accepted draws for beta 0 is: {np.round(corr_beta_0,3)}')
        print(f'The first order serial correlation of accepted draws for beta 1 is: {np.round(corr_beta_1,3)}')
        print(f' The mean of beta 1 is: {np.round(mean_beta_1,3)}')
        print(f'The mean of beta 0 is: {np.round(mean_beta_0,3)}')
        print(f'The value of the maximum likelihood estimators are: {maximum_likelihood_estimates}')
        print(f' The 95% posterior interval for beta 0 is: {beta_0_interval}')
        print(f' The 95% posterior interval for beta 1 is: {beta_1_interval}')
        
        # Trace plot for beta_0
        plt.figure(figsize=(8, 6))
        plt.plot(beta_0, color='blue')
        plt.xlabel('Iterations(without burn in)')
        plt.ylabel('Beta 0')
        plt.title('Trace Plot for Beta 0')
        plt.show()
        
        # Trace plot for beta_1
        plt.figure(figsize=(8, 6))
        plt.plot(beta_1, color='green')
        plt.xlabel('Iterations(without burn in)')
        plt.ylabel('Beta_1')
        plt.title('Trace Plot for accepted/ possibly repeated  Beta 1')
        plt.show()
        
        # Scatter plot of 'beta_0' against 'beta_1'
        plt.figure(figsize=(8, 6))  
        plt.scatter(beta_0,beta_1, color='blue')  
        plt.title('Scatter Plot of beta 1 against beta 0 accepted/possibly repeated draws')  
        plt.xlabel('beta 0')  
        plt.ylabel('beta 1')  
        plt.show()      
            
        # Scatter plot of 'beta_0' against 'beta_1'
        plt.figure(figsize=(8, 6))  
        plt.scatter(candidate_beta_0,candidate_beta_1, color='blue')  
        plt.title('Scatter Plot of beta 1 against beta 0 candidate draws')  
        plt.xlabel('beta 0')  
        plt.ylabel('beta 1')  
        plt.show()  
