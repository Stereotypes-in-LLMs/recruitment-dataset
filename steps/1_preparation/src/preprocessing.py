import pandas as pd
import numpy as np
import logging
import uuid

from src.constants import SUPPORTED_LANGUAGES
from src.helpers import concurrent_processor
from src.lang_detector import lang_detection_func


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Preprocessing:
    """
    Preprocessing general class
    """
    def __init__(self, input_path: str, output_path: str) -> None:
        """
        Preprocessing class constructor

        Args:
            input_path (str): path to input dataset
            output_path (str): path to output dataset

        Returns:
            None
        """
        logging.info("Reading dataset...")
        self.dataset = pd.read_csv(input_path)
        self.output_path = output_path

    def drop_duplicates(self, dataset_type:str ) -> None:
        """"
        Drop duplicates in dataset
        
        Args:
            dataset_type (str): dataset type (jobs or candidates)
        
        Returns:
            None
        """
        # drop duplicates in candidates dataframe
        logging.info(f"Dropping duplicates in {dataset_type} dataframe...")
        self.dataset.drop_duplicates(inplace=True)

    def lang_detection(self, column_name: str) -> None:
        """"
        Detect language in column
        
        Args:
            column_name (str): column name
        
        Returns:
            None
        """
        logging.info(f"Detecting language in {column_name} column...")
        lang_detect_df = pd.DataFrame(concurrent_processor(self.dataset[column_name].unique(), lang_detection_func))
        lang_detect_df = lang_detect_df.rename(columns={
            'item': column_name,
            'lang_detect': f'{column_name}_lang'
        })
        self.dataset = self.dataset.merge(lang_detect_df, on=column_name, how='left')
        logging.info(f"Detecting language in {column_name} column finished.")

    def filter_supported_languages(self, column_name: str) -> None:
        """"
        Filter supported languages in column
        
        Args:
            column_name (str): column name
        
        Returns:
            None
        """
        logging.info(f"Filtering supported languages in {column_name} column...")
        self.dataset = self.dataset[self.dataset[f'{column_name}_lang'].isin(SUPPORTED_LANGUAGES)]

        # reset index
        self.dataset.reset_index(drop=True, inplace=True)
        logging.info(f"Filtering supported languages in {column_name} column finished.")
    
    def _create_id(self, unique_column: str) -> None:
        """
        Create ID column

        Args:
            unique_column (str): unique column name
        Returns:
            None
        """
        # hex id from CV column
        logging.info("Creating ID column...")
        namespace = uuid.NAMESPACE_URL
        self.dataset['id'] = self.dataset[unique_column].apply(lambda x: uuid.uuid5(namespace, x))
        logging.info("Creating ID column finished.")
    
    def save_dataset(self) -> None:
        """"
        Save dataset to output path
        """
        logging.info("Saving dataset...")
        self.dataset.to_csv(self.output_path, index=False)


class CandidatesPreprocessor(Preprocessing):
    """
    CandidatesPreprocessing class
    """

    def __init__(self, 
                 input_path: str, 
                 output_path: str,
                 processor_type: str = "candidates",
                 outlier_threshold: float = 0.05,
                 lang_detect_columns: list[str] = ["CV"],
                 id_component: str = 'CV') -> None:
        """
        CandidatesPreprocessing class constructor

        Args:
            input_path (str): path to input dataset
            output_path (str): path to output dataset
            processor_type (str): processor type
            outlier_threshold (float): outlier threshold
            lang_detect_columns (list[str]): columns to detect language
            id_component (str): id component name
        """
        super().__init__(input_path, output_path)

        self.processor_type = processor_type
        self.outlier_threshold = outlier_threshold
        self.lang_detect_columns = lang_detect_columns
        self.id_component = id_component

    def process(self):
        """"
        Preprocess candidates dataset
        """
        # drop duplicates in candidates dataframe
        self.drop_duplicates(self.processor_type)

        # preprocess POSITION column
        self._position_processing()

        # create CV column
        self.dataset['CV'] = self._create_cv_column()
        # drop CV duplicates
        self._drop_cv_duplicates()
        # drop CV outliers
        self._drop_cv_outliers()

        # lang detect and filter supported languages
        for column in self.lang_detect_columns:
            self.lang_detection(column)
            self.filter_supported_languages(column)

        # create ID column
        self._create_id(self.id_component)

    def _position_processing(self) -> None:
        """"
        Preprocess POSITION column and drop rows with empty POSITION
        """
        logging.info("Preprocessing POSITION column...")
        # clean all possible symbols from positions
        self.dataset['Position_cleaned'] = self.dataset['Position'].str.replace('[^a-zA-Zа-яА-Я ]', '', regex=True)
        self.dataset['Position_cleaned'] = self.dataset['Position_cleaned'].str.strip()

        # empty positions equal to None
        self.dataset['Position_cleaned'] = self.dataset['Position_cleaned'].replace('', None)

        # drop rows with empty positions
        self.dataset.dropna(subset=['Position_cleaned'], inplace=True)

        # drop cleaned position column
        self.dataset.drop(columns=['Position_cleaned'], inplace=True)

        # reset index
        self.dataset.reset_index(drop=True, inplace=True)
        logging.info("Preprocessing POSITION column finished.")
        
    def _create_cv_column(self) -> pd.Series:
        """
        Create CV column from "Highlights", "Moreinfo" and "Looking For" columns

        Returns:
            pd.Series: CV column
        """
        return self.dataset['Highlights'].fillna('') + '\n' + \
            self.dataset['Moreinfo'].fillna('') + '\n' + \
            self.dataset['Looking For'].fillna('')
    
    def _drop_cv_outliers(self) -> None:
        """
        Drop CV outliers

        Returns:
            None
        """
        logging.info("Dropping CV outliers...")
        # drop CV outliers
        self.dataset[self.dataset['CV'].str.len() < self.dataset['CV'].str.len().quantile(self.outlier_threshold)]
        # reset index
        self.dataset.reset_index(drop=True, inplace=True)
        logging.info("Dropping CV outliers finished.")

    def _drop_cv_duplicates(self) -> None:
        """
        Drop CV duplicates

        Returns:
            None
        """
        logging.info("Dropping CV duplicates...")
        # drop CV duplicates
        self.dataset.drop_duplicates(subset=['CV'], inplace=True, keep='first')
        # reset index
        self.dataset.reset_index(drop=True, inplace=True)
        logging.info("Dropping CV duplicates finished.")
