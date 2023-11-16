# imports
from typing import Optional
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

class OPM():
    """
    OPM (Options Prediction Model) class that provides various methods for
    predicting greeks & implied volatility of options given arbitrary
    strike prices.

    This class will implement various machine learnings models that have been
    researched and optimized for the specific greek/IV that is being predicted.

    Methods of this class require the user to input an options chain that will
    be used as the baseline for prediction. 

    For production environments, this class should be used by:
        1. Creating an instance of the class
        2. Repeatedly calling the update_models() method with the latest options chain
           - This method will asynchronously update each of the models in the class, 
             storing them in-memory so they can be used on-demand by calling individual
             prediction methods.
             NOTE: This means that it is possible for a prediction to be made with outdated
                   options data. This is a tradeoff that is made for performance reasons.
        3. Calling the individual prediction methods, which use the in-memory models
    """

    def __init__(self, options_chain: Optional[dict]=None) -> None:
        """
        Initialize an OPM class.

        :param options_chain: Options chain that will be used to initialize the models.
        """
        self.iv_model_materials = {}
        self.delta_model_materials = {}

        if options_chain is not None:
            self.update_models(options_chain)
        

    def update_models(self, options_chain: dict) -> None:
        """
        Updates all of the models in the class with the latest options chain.

        options_chain (dict): Options chain that will be used to update the models.
        """

        strikes = np.array([x['strike_price'] for x in options_chain['call']])
        deltas = np.array([x['delta'] for x in options_chain['call']])
        self.strikes = strikes
        self.deltas = deltas
        self.ivs = np.array([x['iv'] for x in options_chain['call']])

        self.__update_delta_model()
        self.__update_iv_model()

    def predict_iv(self, strike_price: float) -> float:
        """
        Predicts the implied volatility of an option given a strike price and expiration date.

        strike_price (float): Strike price of the option.

        Returns: Predicted implied volatility of the option.
        """

        # First, check if the IV is already known
        if strike_price in self.strikes:
            return self.ivs[np.where(self.strikes == strike_price)][0]

        if len(self.iv_model_materials.keys()) < 1:
            raise Exception('IV model has not been initialized. Please call update_models() first.')
        
        if np.abs(strike_price - self.iv_model_materials["at_the_money_strike"]) < self.iv_model_materials["bandwidth"]:
            # within the "NTM" range

            x_poly = self.iv_model_materials["poly_near"].transform([[strike_price]])
            return self.iv_model_materials["regressor_near"].predict(x_poly)[0]
        
        else:
            # outside the "NTM" range

            x_poly = self.iv_model_materials["poly_other"].transform([[strike_price]])
            return self.iv_model_materials["regressor_other"].predict(x_poly)[0]

    def predict_delta(self, strike_price: float) -> float:
        """
        Predicts the delta of an option given a strike price and expiration date.

        strike_price (float): Strike price of the option.
        expiration_date (str): Expiration date of the option.

        Returns: Predicted delta of the option.
        """

        if strike_price in self.strikes:
            return self.deltas[np.where(self.strikes == strike_price)][0]

        if len(self.delta_model_materials.keys()) < 1:
            raise Exception('Delta model has not been initialized. Please call update_models() first.')
        
        x = self.delta_model_materials['scaler_X'].transform(np.array(strike_price).reshape(-1, 1))
        y_pred = self.delta_model_materials['svr'].predict(x)

        return self.delta_model_materials['scaler_Y'].inverse_transform(y_pred.reshape(-1, 1))[0][0]

    def get_iv_model_info(self) -> dict:
        """
        Returns a dictionary containing information about the IV model.
        """

        if len(self.iv_model_materials.keys()) < 1:
            raise Exception('IV model has not been initialized. Please call update_models() first.')
        
        predicted_ivs_near = self.iv_model_materials["regressor_near"].predict(self.iv_model_materials["poly_near"].transform(self.iv_model_materials["X_near"]))
        predicted_ivs_other = self.iv_model_materials["regressor_other"].predict(self.iv_model_materials["poly_other"].transform(self.iv_model_materials["X_other"]))

        # Calculate MSE for both regions
        mse_near = mean_squared_error(self.iv_model_materials["y_near"], predicted_ivs_near)
        rmse_near = np.sqrt(mse_near)
        mse_other = mean_squared_error(self.iv_model_materials["y_other"], predicted_ivs_other)
        rmse_other = np.sqrt(mse_other)

        return {'MSE_near': mse_near, 'RMSE_near': rmse_near, 'MSE_other': mse_other, 'RMSE_other': rmse_other}

    def get_delta_model_info(self) -> dict:
        """
        Returns a dictionary containing information about the delta model.
        """

        if len(self.delta_model_materials.keys()) < 1:
            raise Exception('Delta model has not been initialized. Please call update_models() first.')

        predicted_deltas = [self.predict_delta(s) for s in self.strikes]
        mse = mean_squared_error(self.deltas, predicted_deltas)
        rmse = np.sqrt(mse)
        
        return {'MSE': mse, 'RMSE': rmse}

    def __update_delta_model(self) -> None:

        # Scaling the data can often improve performance with SVR
        scaler_X = StandardScaler()
        scaler_Y = StandardScaler()
        X = scaler_X.fit_transform(self.strikes.reshape(-1, 1))
        y = scaler_Y.fit_transform(self.deltas.reshape(-1, 1))

        # Create and fit the SVR model
        svr = SVR(kernel='rbf', C=1e3, gamma=5)
        svr.fit(X, y.ravel())

        # Updating in-memory variables
        self.delta_model_materials['scaler_X'] = scaler_X
        self.delta_model_materials['scaler_Y'] = scaler_Y
        self.delta_model_materials['svr'] = svr

    def __update_iv_model(self) -> None:
        
        # Finding the strike for which the IV is the lowest
        at_the_money_strike = self.strikes[np.argmin(self.ivs)]
        bandwidth = int(at_the_money_strike * .05)  # 5% around atm

        near_the_money_mask = np.abs(self.strikes - at_the_money_strike) < bandwidth
        X_near = self.strikes[near_the_money_mask].reshape(-1, 1)
        y_near = self.ivs[near_the_money_mask]

        X_other = self.strikes[~near_the_money_mask].reshape(-1, 1)
        y_other = self.ivs[~near_the_money_mask]

        # Define polynomial degree for each region
        degree_near = 4

        # Fit polynomial regression for near-the-money data
        poly_near = PolynomialFeatures(degree=degree_near)
        X_poly_near = poly_near.fit_transform(X_near)
        regressor_near = LinearRegression().fit(X_poly_near, y_near)

        degree_other = 4

        # Fit polynomial regression for other data
        poly_other = PolynomialFeatures(degree=degree_other)
        X_poly_other = poly_other.fit_transform(X_other)
        regressor_other = LinearRegression().fit(X_poly_other, y_other)

        # Updating in-memory variables
        self.iv_model_materials['X_near'] = X_near
        self.iv_model_materials['y_near'] = y_near
        self.iv_model_materials['X_other'] = X_other
        self.iv_model_materials['y_other'] = y_other
        self.iv_model_materials['at_the_money_strike'] = at_the_money_strike
        self.iv_model_materials['bandwidth'] = bandwidth
        self.iv_model_materials['poly_near'] = poly_near
        self.iv_model_materials['regressor_near'] = regressor_near
        self.iv_model_materials['poly_other'] = poly_other
        self.iv_model_materials['regressor_other'] = regressor_other