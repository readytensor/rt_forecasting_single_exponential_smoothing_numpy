import os
import warnings
import joblib
import numpy as np
import pandas as pd
from typing import Union
from schema.data_schema import ForecastingSchema
from sklearn.exceptions import NotFittedError

from data_models.schema_validator import Frequency

warnings.filterwarnings("ignore")


PREDICTOR_FILE_NAME = "predictor.joblib"

def get_smoothing_factor(freq: str) -> float:
    """
    Calculates and returns a smoothing factor appropriate for the specified frequency
    of a time series. The smoothing factor is inversely related to the frequency,
    with more frequent data (e.g., secondly) getting a smaller factor, and less
    frequent data (e.g., yearly) getting a larger factor. 

    Args:
        freq (str): A string representation of the time series frequency, expected
                    to be one of the predefined frequencies (e.g., 'SECONDLY', 'MINUTELY',
                    'HOURLY', 'DAILY', 'WEEKLY', 'MONTHLY', 'QUARTERLY', 'YEARLY', 'OTHER').
                    These frequencies should correspond to a predefined enumeration or
                    set of constants that define different possible time series frequencies.

    Returns:
        float: The smoothing factor determined based on the specified frequency. The factor
               is a fraction, calculated as the reciprocal of a base number that varies
               according to the frequency. The base numbers are chosen to scale the smoothing
               factor in a way that is inversely proportional to the assumed volatility or
               the level of detail associated with each frequency.

    Raises:
        ValueError: If the provided frequency is not recognized as one of the predefined
                    frequencies, indicating that the caller has specified an invalid or
                    unsupported frequency.
    """
    if freq == str(Frequency.SECONDLY):
        return 1 / (120) # last 2 minutes
    elif freq == str(Frequency.MINUTELY):
        return 1 / (120) # last 2 hours
    elif freq == str(Frequency.HOURLY):
        return 1 / 48.0 # last 2 days
    elif freq == str(Frequency.DAILY):
        return 1 / 14. # last 2 weeks
    elif freq == str(Frequency.WEEKLY):
        return 1 / 8. # last 2 months
    elif freq == str(Frequency.MONTHLY):
        return 1 / 6. # last 2 quarters
    elif freq == str(Frequency.QUARTERLY):
        return 1 / 4. # last 1 year
    elif freq == str(Frequency.YEARLY):
        return 1 / 4. # last 4 years
    elif freq == str(Frequency.OTHER):
        return 1 / 20.
    else:
        raise ValueError(f"Invalid frequency: {freq}.")


class Forecaster:
    """A Single Expoential Smoothing Forecaster Model."""

    model_name = "Single Expoential Smoothing Forecaster"

    def __init__(
        self,
        data_schema: ForecastingSchema,
        history_forecast_ratio: int = None,
        init_period: int = 10,
        alpha: Union[float, str] = "auto",
        random_state: int = 0,
        **kwargs,
    ):
        """Construct a new Single Expoential Smoothing Forecaster

        Args:

            data_schema (ForecastingSchema):
                Schema of training data.

            history_forecast_ratio (int):
                Sets the history length depending on the forecast horizon.
                For example, if the forecast horizon is 20 and the history_forecast_ratio is 10,
                history length will be 20*10 = 200 samples.

            init_period (int):
                The number of initial observations to use for initializing the mean.
                Default is 10.

            alpha (float):
                Smoothing factor. Default is "auto". If "auto", the smoothing factor is
                calculated based on the frequency of the time series.
                Valid float values are between 0 and 1.

            random_state (int): Sets the underlying random seed at model initialization time.
        """
        self.data_schema = data_schema
        self.random_state = random_state
        self.init_period = int(init_period)
        if alpha == "auto":
            self.alpha = get_smoothing_factor(self.data_schema.frequency)
        else:
            alpha = float(alpha)
            if alpha < 0 or alpha > 1:
                raise ValueError("alpha must be between 0 and 1")
            self.alpha = alpha
        self._is_trained = False
        self.kwargs = kwargs
        self.history_length = None

        if history_forecast_ratio:
            self.history_length = (
                self.data_schema.forecast_length * history_forecast_ratio
            )

    def fit(
        self,
        history: pd.DataFrame,
        data_schema: ForecastingSchema,
    ) -> None:
        """Fit the Forecaster to the training data.
        A separate NaiveMean model is fit to each series that is contained
        in the data.

        Args:
            history (pandas.DataFrame): The features of the training data.
            data_schema (ForecastingSchema): The schema of the training data.

        """
        np.random.seed(0)
        groups_by_ids = history.groupby(data_schema.id_col)
        all_ids = list(groups_by_ids.groups.keys())
        all_series = [
            groups_by_ids.get_group(id_).drop(columns=data_schema.id_col)
            for id_ in all_ids
        ]

        self.means = {}

        for id, series in zip(all_ids, all_series):
            if self.history_length:
                series = series[-self.history_length :]
            model = self._fit_on_series(history=series, data_schema=data_schema)
            self.means[id] = model

        self.all_ids = all_ids
        self._is_trained = True
        self.data_schema = data_schema

    def _fit_on_series(self, history: pd.DataFrame, data_schema):
        """Fit model to given individual series of data using single exponential smoothing"""
        time_series = np.array(history[data_schema.target])
        
        # Initialize the smoothed value as the mean of the first `init_period` observations
        if len(time_series) < self.init_period:
            # If there aren't enough data points, fallback to the mean of available data
            smoothed_value = np.mean(time_series)
        else:
            smoothed_value = np.mean(time_series[:self.init_period])
        
        # Apply exponential smoothing to the rest of the data
        for observation in time_series[self.init_period:]:
            smoothed_value = self.alpha * observation + (1 - self.alpha) * smoothed_value
        
        return smoothed_value

    def predict(
        self, test_data: pd.DataFrame, prediction_col_name: str
    ) -> pd.DataFrame:
        """Make the forecast of given length.

        Args:
            test_data (pd.DataFrame): Given test input for forecasting.
            prediction_col_name (str): Name to give to prediction column.
        Returns:
            pd.DataFrame: The prediction dataframe.
        """
        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")

        groups_by_ids = test_data.groupby(self.data_schema.id_col)
        all_series = [
            groups_by_ids.get_group(id_).drop(columns=self.data_schema.id_col)
            for id_ in self.all_ids
        ]
        # forecast one series at a time
        all_forecasts = []
        for id_, series_df in zip(self.all_ids, all_series):
            forecast = self._predict_on_series(key_and_future_df=(id_, series_df))
            forecast.insert(0, self.data_schema.id_col, id_)
            all_forecasts.append(forecast)

        # concatenate all series' forecasts into a single dataframe
        all_forecasts = pd.concat(all_forecasts, axis=0, ignore_index=True)

        all_forecasts.rename(
            columns={self.data_schema.target: prediction_col_name}, inplace=True
        )
        return all_forecasts

    def _predict_on_series(self, key_and_future_df):
        """Make forecast on given individual series of data"""
        key, future_df = key_and_future_df

        if self.means.get(key) is not None:
            forecasts = np.full(len(future_df), self.means[key])
            future_df[self.data_schema.target] = forecasts

        else:
            # no model found - key wasnt found in history, so cant forecast for it.
            future_df = None

        return future_df

    def save(self, model_dir_path: str) -> None:
        """Save the Forecaster to disk.

        Args:
            model_dir_path (str): Dir path to which to save the model.
        """
        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")
        joblib.dump(self, os.path.join(model_dir_path, PREDICTOR_FILE_NAME))

    @classmethod
    def load(cls, model_dir_path: str) -> "Forecaster":
        """Load the Forecaster from disk.

        Args:
            model_dir_path (str): Dir path to the saved model.
        Returns:
            Forecaster: A new instance of the loaded Forecaster.
        """
        model = joblib.load(os.path.join(model_dir_path, PREDICTOR_FILE_NAME))
        return model

    def __str__(self):
        # sort params alphabetically for unit test to run successfully
        return f"Model name: {self.model_name}"


def train_predictor_model(
    history: pd.DataFrame,
    data_schema: ForecastingSchema,
    hyperparameters: dict,
) -> Forecaster:
    """
    Instantiate and train the predictor model.

    Args:
        history (pd.DataFrame): The training data inputs.
        data_schema (ForecastingSchema): Schema of the training data.
        hyperparameters (dict): Hyperparameters for the Forecaster.

    Returns:
        'Forecaster': The Forecaster model
    """

    model = Forecaster(
        data_schema=data_schema,
        **hyperparameters,
    )
    model.fit(
        history=history,
        data_schema=data_schema,
    )
    return model


def predict_with_model(
    model: Forecaster, test_data: pd.DataFrame, prediction_col_name: str
) -> pd.DataFrame:
    """
    Make forecast.

    Args:
        model (Forecaster): The Forecaster model.
        test_data (pd.DataFrame): The test input data for forecasting.
        prediction_col_name (int): Name to give to prediction column.

    Returns:
        pd.DataFrame: The forecast.
    """
    return model.predict(test_data, prediction_col_name)


def save_predictor_model(model: Forecaster, predictor_dir_path: str) -> None:
    """
    Save the Forecaster model to disk.

    Args:
        model (Forecaster): The Forecaster model to save.
        predictor_dir_path (str): Dir path to which to save the model.
    """
    if not os.path.exists(predictor_dir_path):
        os.makedirs(predictor_dir_path)
    model.save(predictor_dir_path)


def load_predictor_model(predictor_dir_path: str) -> Forecaster:
    """
    Load the Forecaster model from disk.

    Args:
        predictor_dir_path (str): Dir path where model is saved.

    Returns:
        Forecaster: A new instance of the loaded Forecaster model.
    """
    return Forecaster.load(predictor_dir_path)


def evaluate_predictor_model(
    model: Forecaster, x_test: pd.DataFrame, y_test: pd.Series
) -> float:
    """
    Evaluate the Forecaster model and return the accuracy.

    Args:
        model (Forecaster): The Forecaster model.
        x_test (pd.DataFrame): The features of the test data.
        y_test (pd.Series): The labels of the test data.

    Returns:
        float: The accuracy of the Forecaster model.
    """
    return model.evaluate(x_test, y_test)
