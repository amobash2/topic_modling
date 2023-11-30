import pandas as pd
import numpy as np
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance
from bertopic.backend import BaseEmbedder
from sentence_transformers import SentenceTransformer

class CustomEmbedder(BaseEmbedder):
    def __init__(self, embedding_model, batch_size):
        super().__init__()
        self.embedding_model = embedding_model
        self.batch_size = batch_size

    def embed(self, documents, verbose=False):
        embeddings = self.embedding_model.encode(documents, batch_size=self.batch_size, show_progress_bar=verbose)
        return embeddings 

class TopicModeling:
    def __init__(self,
                 input_data: pd.DataFrame(),
                 output_path: str,
                 text_column: str = "TEXT",
                 use_embedding: bool = False,
                 embedding_column: str = "EMBEDDING",
                 bert_model: str = "all-MiniLM-L6-v2",
                 batch_size: int = 1000):
        self._input_data = input_data
        self._output_path = output_path
        self._text_column = text_column
        self._use_embedding = use_embedding
        self._embedding_column = embedding_column
        self._bert_model = bert_model
        self._batch_size = batch_size

        # Add internal_id column to input data
        if "internal_id" not in list(self._input_data):
            print("Adding an internal_id column...")
            self._input_data["internal_id"] = self._input_data.index + 1

        self._input_data.dropna(inplace=True)

        # Clean dataframe from empty and numeric data
        self._input_data = self._input_data.loc[(self._input_data[self._text_column].str.replace("Chief compliant: ", "").str.strip() != "") & 
                                                (~self._input_data[self._text_column].str.replace("Chief compliant: ", "").str.isnumeric())]
        
        self._input_data[["internal_id", self._text_column]].to_csv(f"{self._output_path}/input_data.csv", index=None)

        self._text_data = list(self._input_data[self._text_column])
        self._embedding_data = list(self._input_data[self._embedding_column]) if self._embedding_column else []
        self._ids = list(self._input_data["internal_id"])

    def run(self):
        if self._use_embedding:
            topic_model_results = self._topic_modeling_with_embedding()
        else:
            topic_model_results = self._topic_modeling_with_model()
        return topic_model_results

    def _topic_modeling_with_embedding(self):
        print("Using pre-calculated embeddings...")
        representation_model = MaximalMarginalRelevance(diversity=0.2)
        topic_model = BERTopic(representation_model = representation_model)
        topics, _ = topic_model.fit_transform(self._text_data, np.array([np.array(e) for e in self._embedding_data]))
        # print(topic_model.get_topic_info())
        topic_results = topic_model.get_document_info(self._text_data)
        topic_results["internal_id"] = self._ids
        return topic_results, topic_model

    def _topic_modeling_with_model(self):
        print("Using provided BERT model...")

        embedding_model = SentenceTransformer(self._bert_model)
        custom_embedder = CustomEmbedder(embedding_model=embedding_model, batch_size = self._batch_size)

        representation_model = MaximalMarginalRelevance(diversity=0.2)
        topic_model = BERTopic(embedding_model = custom_embedder, representation_model = representation_model)
        topics, _ = topic_model.fit_transform(self._text_data)

        # print(topic_model.get_topic_info())
        topic_results = topic_model.get_document_info(self._text_data)
        topic_results["internal_id"] = self._ids
        return topic_results, topic_model