import pandas as pd
import argparse
import json
import logging

def main(args):
    df = pd.read_csv('dataset/split_test.csv')
    logging.info(f"Sample a row from the dataframe with seed: {args.seed}")
    sampled_row = df.sample(n=1, random_state=args.seed)
    logging.info(f"Sampled row index: {sampled_row.index.values[0]}")
    sample_dict = sampled_row.to_dict(orient='records')[0]
    logging.info(f"Sampled row:\n{json.dumps(sample_dict, indent=4)}")
    logging.info(f"Save sample as json")
    with open('test_sample.json', 'w') as f:
        json.dump(sample_dict, f, indent=4)



# Define the command line arguments
parser = argparse.ArgumentParser(description='Sample a row from a pandas dataframe based on a seed point. If no seed is passed, it will default to None and sample randomly.')
parser.add_argument(
    '--seed',
    # nargs='?',
    type=int, 
    default=None,
    help='The seed point to use for sampling the row.', 
    )

if __name__ == "__main__":
    # Parse the command line arguments
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()
    main(args)


