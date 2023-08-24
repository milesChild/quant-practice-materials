# imports
from typing import Optional

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

    def __init__(self, ) -> None:
        self.iv_model = None
        self.delta_model = None
        pass

    def update_models(self, options_chain: dict) -> None:
        """
        Updates all of the models in the class with the latest options chain.

        options_chain (dict): Options chain that will be used to update the models.
        """
        pass

    def predict_iv(self, strike_price: float) -> float:
        """
        Predicts the implied volatility of an option given a strike price and expiration date.

        strike_price (float): Strike price of the option.

        Returns: Predicted implied volatility of the option.
        """
        pass

    def predict_delta(self, strike_price: float) -> float:
        """
        Predicts the delta of an option given a strike price and expiration date.

        strike_price (float): Strike price of the option.
        expiration_date (str): Expiration date of the option.

        Returns: Predicted delta of the option.
        """
        pass