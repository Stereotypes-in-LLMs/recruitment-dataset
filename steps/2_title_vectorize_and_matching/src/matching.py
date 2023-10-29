import json
import faiss
import logging
import numpy as np
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
            df_jobs_filtered = self.df_jobs[(self.df_jobs['Position'] == cand_position) &
                                            (self.df_jobs['year_exp'] == year_exp) &
                                            (self.df_jobs['Long Description_lang'] == cand_lang)]
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
        # https://magnet.me/guide/en/the-difference-between-junior-medior-and-senior/
        if year_exp <= 2:
            return 'Elementary'
        elif year_exp <= 5:
            return 'Intermediate'
        else:
            return 'Advanced'

     
class FinalMatching:
    """
    Final Matching class
    """
    def __init__(self,
                job_groups_folder_path: str,
                df_candidates_path: str,
                emb_candidates_folder_path: str,
                emb_jobs_folder_path: str,
                output_path: str,
                vec_dim: int = 1024) -> None:
        """
        Group Matcher constructor
        
        Args:
            job_groups_path (str): job groups file folder path
            df_candidates_path (str): candidates file path
            emb_candidates_folder_path (str): candidates embeddings folder path
            emb_jobs_folder_path (str): jobs embeddings folder path
            output_path (str): output path
            vec_dim (int, optional): embeddings dimension. Defaults to 1024.

        Returns:
            None
        """
        self.job_groups = {
            k:v 
            for k,v in json.load(open(f'{job_groups_folder_path}/groups.json', 'r')).items() 
            if len(v)!=0
        }
        self.output_path = output_path
        df_candidates = pd.read_csv(df_candidates_path)
        self.candidates_lang = df_candidates[['id', 'CV_lang']].set_index('id').to_dict()['CV_lang']

        self.vec_dim = vec_dim

        # load candidates embeddings
        self.cand_emb_id_map_en = {
            v: int(k) 
            for k,v in json.load(open(f'{emb_candidates_folder_path}/emb_id_map_en.json', 'r')).items()
        }
        self.cand_emb_id_map_uk = {
            v: int(k) 
            for k,v in json.load(open(f'{emb_candidates_folder_path}/emb_id_map_uk.json', 'r')).items()
        }
        self.cand_emb_en = faiss.read_index(f"{emb_candidates_folder_path}/emb_index_en.index")
        self.cand_emb_uk = faiss.read_index(f"{emb_candidates_folder_path}/emb_index_en.index")

        # load jobs embeddings
        self.job_emb_id_map_en = {
            v: int(k) 
            for k,v in json.load(open(f'{emb_jobs_folder_path}/emb_id_map_en.json', 'r')).items()
        }
        self.job_emb_id_map_uk = {
            v: int(k) 
            for k,v in json.load(open(f'{emb_jobs_folder_path}/emb_id_map_uk.json', 'r')).items()
        }
        self.job_emb_en = faiss.read_index(f"{emb_jobs_folder_path}/emb_index_en.index")
        self.job_emb_uk = faiss.read_index(f"{emb_jobs_folder_path}/emb_index_en.index")

    def process(self) -> None:
        """
        Match candidates with jobs for creating groups

        Returns:
            None
        """

        logging.info('Matching candidates with jobs...')
        for cand_id, job_ids in tqdm(self.job_groups.items()):

            if self.candidates_lang[cand_id] == 'uk':

                vec_cand = self.cand_emb_uk.index.reconstruct(self.cand_emb_id_map_uk[cand_id],
                                                              np.empty((1, 1024), dtype='float32')[0])
                
                jobs_list = [self.job_emb_id_map_uk[job_id] for job_id in job_ids]
                vec_jobs = self.job_emb_uk.index.reconstruct_batch(jobs_list,
                                                            np.empty((len(job_ids), 1024), dtype='float32'))
                
            elif self.candidates_lang[cand_id] == 'en':
                    
                vec_cand = self.cand_emb_en.index.reconstruct(self.cand_emb_id_map_en[cand_id],
                                                            np.empty((1, 1024), dtype='float32')[0])
                
                jobs_list = [self.job_emb_id_map_en[job_id] for job_id in job_ids]
                vec_jobs = self.job_emb_en.index.reconstruct_batch(jobs_list,
                                                            np.empty((len(job_ids), 1024), dtype='float32'))
                
            else:
                raise ValueError(f'Unknown language {self.candidates_lang[cand_id]}')
            
            # compute cosine similarity
            sim = self._cosine_similarity(vec_cand, vec_jobs)

            # filter jobs by similarity
            job_ids_filtered = self._jobs_filtering(job_ids, sim)

            # update job groups
            self.job_groups[cand_id] = job_ids_filtered
        logging.info('Matching candidates with jobs finished.')

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
    def _jobs_filtering(job_list: list,
                        sim_matrix: np.array,
                        threshold: float = 0.9) -> list:
        """
        Filter jobs by similarity

        Args:
            job_list (list): list of jobs
            sim_matrix (np.array): similarity matrix
            threshold (float, optional): similarity threshold. Defaults to 0.9.

        Returns:
            list: filtered jobs
        """
        # use threshold 0.9 because it will filter really different jobs to the candidate and keep similar
        ids = np.where(sim_matrix >= threshold)

        return np.array(job_list)[ids].tolist()

    @staticmethod
    def _cosine_similarity(vec1: np.array, 
                           vec2: np.array) -> float:
        """
        Compute cosine similarity between vector and array of vectors

        Args:
            vec1 (np.array): vector 1
            vec2 (np.array): array of vectors
        
        Returns:
            float: cosine similarity
        """
        return np.dot(vec1, vec2.T)/(np.linalg.norm(vec1)*np.linalg.norm(vec2, axis=1))             
