import pandas as pd
import seaborn as sns


class Charts:
    
    @classmethod
    def correlation_table(cls, dataframe):
        corr = dataframe.corr()
        corr.style.set_precision(2)
        return corr.style.background_gradient()

    @classmethod
    def heatmap(cls, dataframe):
        corr = dataframe.corr()
        sns.heatmap(dataframe.corr(),
                    xticklabels=dataframe.corr().columns.values,
                    yticklabels=dataframe.corr().columns.values,
        )

    @classmethod
    def scatter_matrix(cls, dataframe):
        pd.plotting.scatter_matrix(dataframe, alpha=0.3, figsize=(14, 8), diagonal='kde')
