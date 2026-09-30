import os
import numpy as np
import pandas as pd
import json


class Wagenmakers:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.csv_path = os.path.join(self.base_dir, "wagenmakers.csv")
        self._df = pd.read_csv(self.csv_path)
        # remove participant 2 (not enough trials)
        # self._df = self.df[self.df["id"] != 2]
        # recode id's so that the coding is dense integer coding (easier to track for indexing arrays)
        self._df["id"] = self.df["id"].factorize()[0]

        # recode rt to pos when correct and neg when incorrect
        self._df["rt"] = np.where(
            self.df["stim_cat"] == self.df["response"], self.df["rt"], -self.df["rt"]
        )
        # binary coding for condition
        self._df["speed"] = np.where(self.df["condition"] == "speed", 1, 0)

        self._df_array = None

        # only select a subset of data
        self.n_blocks = 4
        self.n_trials_per_block = 96
        self.n_trials = self.n_blocks * self.n_trials_per_block

        conditions = [1, 0] * (self.n_blocks // 2)  # [1, 0, 1, 0] for 4 blocks
        self.conditions = np.repeat(conditions, self.n_trials_per_block)
        # 96*1, 96*0, 96*1, 96*0

    @property
    def df(self):
        return self._df

    @property
    def df_grouped(self):
        return self.df.groupby("id")

    @property
    def df_stan(self):
        def stanify(df):
            return dict(
                rt=list(df["rt"][: self.n_trials]),
                condition=list(df["speed"][: self.n_trials]),
                N=self.n_trials,
            )

        return self.df_grouped.apply(stanify)

    def write_stan(self):
        for id, df in self.df_stan.items():
            filename = "p" + str(id) + ".json"
            path = os.path.join(self.base_dir, "json", "empirical", filename)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                json.dump(df, f)

    @property
    def df_array(self):
        if self._df_array is None:
            df_array = np.zeros(
                (17, self.n_trials, 2)
            )  # (participants, trials, rt + condition)

            for subj, df in self.df_grouped:
                df_array[subj, : self.n_trials, 0] = df["rt"][: self.n_trials]
                df_array[subj, : self.n_trials, 1] = df["speed"][: self.n_trials]

            self._df_array = df_array

        return self._df_array

    @property
    def ids(self):
        return [f"p{id}" for id, _ in self.df_grouped]

    def as_conditions(self, data=None):
        if data is None:
            data = self.df_array
        data = np.asarray(data, dtype=np.float32)
        return {
            "rt": data[..., 0],
            "conditions": data[..., 1],
        }


wagenmakers = Wagenmakers()

if __name__ == "__main__":
    wagenmakers.write_stan()
