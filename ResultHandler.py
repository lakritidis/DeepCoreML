import pandas as pd


class ResultHandler:
    def __init__(self, results):
        self.r_ = results

    def to_df(self):
        df_columns = [
            'Classifier',
            'Sampler',
            'Accuracy_Mean',
            'Accuracy_Std',
            'Balanced_Accuracy_Mean',
            'Balanced_Accuracy_Std',
            'Sensitivity_Mean',
            'Sensitivity_Std',
            'Specificity_Mean',
            'Specificity_Std',
            'F1_Mean',
            'F1_Std',
            'Order'
        ]

        return pd.DataFrame(self.r_, columns=df_columns)

    def to_latex(self, columns='all'):
        df_temp = self.to_df()

        if columns == 'all':
            latex_code = df_temp.to_latex()
        else:
            latex_code = df_temp[columns].to_latex()

        return latex_code
