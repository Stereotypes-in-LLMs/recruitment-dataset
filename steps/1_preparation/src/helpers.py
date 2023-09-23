import logging
from tqdm import tqdm
from typing import Any 

import concurrent
from concurrent.futures.thread import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def concurrent_processor(items: list[Any], func: object) -> list[Any]:
    """"
    Apply function to each item in list of items in parallel

    Args:
        items (Any): list of items
        func (object): function to be applied to each item
    
    Returns:
        list(Any): list of processed items
    """
    processed_items = []
    with ThreadPoolExecutor() as executor:
        futures = []
        logging.info("Processing items...")
        for item in tqdm(items):
            future = executor.submit(func, item)
            futures.append(future)
        
        logging.info("Waiting for results...")
        with tqdm(total=len(items)) as pbar:
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                processed_items.append(result)
                pbar.update(1)
    logging.info("Concurrent processing finished!")
    return processed_items