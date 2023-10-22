import json
import logging
import pandas as pd
from tqdm import tqdm


# logging config
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class GroupMatcher:
    """
    Group Matcher class
    """
    def __init__(self,
                df_candidates_path: str,
                df_jobs_path: str,
                output_path: str) -> None:
        """
        Group Matcher constructor
        
        Args:
            df_candidates_path (str): candidates file path
            df_jobs_path (str): jobs file path
            output_path (str): output path

        Returns:
            None
        """
        self.job_groups = dict()
        self.output_path = output_path
        self.df_candidates = pd.read_csv(df_candidates_path)
        self.df_jobs = pd.read_csv(df_jobs_path)

        # create experience years groups
        logging.info('Creating experience years groups...')
        self.df_candidates['year_exp'] = self.df_candidates['Experience Years'].map(
            self.experience_years_groups
        )

        self.df_jobs['Exp Years'] = self.df_jobs['Exp Years'].map(
            lambda x: 0 if x == 'no_exp' else x.replace('y','')
        ).astype(int)
        self.df_jobs['year_exp'] = self.df_jobs['Exp Years'].map(
            self.experience_years_groups
        )


    def process(self) -> None:
        """
        Match candidates with jobs for creating groups
        
        Returns:
            None
        """
        # possible job groups for each candidate
        logging.info('Matching candidates with jobs...')
        for cand_id, cand_position, year_exp, cand_lang in tqdm(zip(self.df_candidates['id'].tolist(),
                                                               self.df_candidates['Position'].tolist(),
                                                               self.df_candidates['year_exp'].tolist(),
                                                               self.df_candidates['CV_lang'].tolist())):
            # filter jobs by position, year_exp and lang
            df_jobs_filtered = self.df_jobs[(self.df_jobs['position'] == cand_position) &
                                            (self.df_jobs['year_exp'] == year_exp) &
                                            (self.df_jobs['Description_lang'] == cand_lang)]
            # get job ids
            job_ids = df_jobs_filtered['id'].tolist()

            # add job ids to job groups
            self.job_groups[cand_id] = job_ids

    def save(self) -> None:
        """
        Save job groups
        
        Returns:
            None
        """
        logging.info(f'Saving job groups to {self.output_path}')
        with open(self.output_path, 'w') as f:
            json.dump(self.job_groups, f)

    @staticmethod
    def experience_years_groups(year_exp: int) -> int:
        """
        Get experience years group

        Args:
            year_exp (int): experience years
        
        Returns:
            int: experience years group
        """
        if year_exp <= 2:
            return 'Elementary'
        elif year_exp <= 5:
            return 'Intermediate'
        else:
            return 'Advanced'
