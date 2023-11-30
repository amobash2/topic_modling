import pandas as pd

class PreProcessing:
    def __init__(self,
                 input_data: pd.DataFrame(),
                 text_column: str = "TEXT"):
        self._input_data = input_data
        self._text_column = text_column
        self._clean_input = self._clean_data()

    @property
    def clean_input(self):
        return self._clean_input
    
    def _replace_chars(self, text: str, chars = [",", "/", "-"]):
        for c in chars:
            text = text.replace(c, " ")
        return text

    def _clean_row_data(self, df_row):
        return " ".join([a for a in self._replace_chars(df_row[self._text_column]).split()])

    def _clean_data(self):
        print("Cleaning text for the purpose of topic modeling...")
        self._input_data["clean_text"] = self._input_data.apply(lambda row: self._clean_row_data(row), axis = 1)
        return self._input_data
