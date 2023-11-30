import argparse
from bert_tm.utils import str2bool
from bert_tm.topic_modeling import TopicModeling
from bert_tm.preprocessing import PreProcessing
import pandas as pd
import time

import warnings
warnings.simplefilter('ignore')


def main(args, model_name: str = ""):
    start_time = time.time()
    input_data = None
    if args.input_path.endswith(".csv"):
        input_data = pd.read_csv(args.input_path)
    elif args.input_path.endswith(".pkl"):
        input_data = pd.read_pickle(args.input_path)

    print("Number of records: ", len(input_data))

    if args.clean_text:
        input_data = PreProcessing(input_data, text_column= args.text_column).clean_input
    
    if input_data is not None:
        topic_model_results, topic_model = TopicModeling(input_data,
                                                         output_path= args.output_path,
                                                         use_embedding=args.use_embeddings,
                                                         bert_model=args.bert_model,
                                                         text_column="clean_text" if args.clean_text else args.text_column,
                                                         embedding_column=args.embedding_column if args.use_embeddings else None,
                                                         batch_size=args.batch_size).run()
        if args.save_topic_model:
            topic_model.save(f"{args.output_path}/tm_{model_name}", serialization="pytorch")
    print(f"Topic modeling is finished in {round((time.time() - start_time)/60.0, 3)} minutes")
    return topic_model_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Topic modeling using BERTopic with BERT and OpenAI embeddings")
    parser.add_argument("--input_path", required = True, help = "Path to the input file, including the file name.")
    parser.add_argument("--output_path", required = True, help = "Path to the output folder.")
    parser.add_argument("--text_column", default = "TEXT")
    parser.add_argument("--embedding_column", default = "EMBEDDING")
    parser.add_argument("--use_embeddings", default = False, help = "Use pre-calculated embeddings or not.", type = str2bool)
    parser.add_argument("--bert_model", default = "all-MiniLM-L6-v2", help = "BERT default model to use with BERTopic")
    parser.add_argument("--clean_text", default = False, help = "Preprocess text and map acronyms", type = str2bool)
    parser.add_argument("--save_topic_model", default=True, type=str2bool)
    parser.add_argument("--batch_size", default=1000, help="Batch size for embedding generation.")

    args = parser.parse_args()

    topic_model_results = main(args, "local")

    topic_model_results.to_csv(f"{args.output_path}/topic_model_results.csv", index = None)