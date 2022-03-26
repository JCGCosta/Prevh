"""Prevh DataSet Plot"""

import matplotlib.pyplot as plt
import random as rd
from prevh.legacy._PrevhClassifierLegacy import PrevhClassifier

class PrevhPlot(PrevhClassifier):

    def __init__(self, df_dataset):
        super().__init__(df_dataset=df_dataset)

    def show(self, **kwargs):
        # kwargs
        figx = kwargs.get('figx', 10)
        figy = kwargs.get('figy', 10)

        def genrdhexcolor(num):
            colors = []
            for i in range(num):
                sc = "#%06x" % rd.randint(0, 0xFFFFFF)
                colors += [sc]
            return colors

        title = kwargs.get('title', "Data Set without normalization")
        colors = genrdhexcolor(len(self.posibleresults))

        if len(self.axisheader) == 2 or len(self.axisheader) == 3:
            fig = plt.figure(figsize=(figx, figy))
            fig.suptitle(title, fontsize=16)
            if len(self.axisheader) == 3:
                ax = fig.add_subplot(111, projection='3d')
                for i in range(len(self.rawdata)):
                    for c, r in enumerate(self.posibleresults):
                        if r == self.rawdata.iat[i, 3]:
                            ax.scatter(self.rawdata.iat[i, 0], self.rawdata.iat[i, 1], self.rawdata.iat[i, 2], zdir='z', c=colors[c], s=15)
                            break
            else:
                ax = fig.add_subplot(111)
                for i in range(len(self.rawdata)):
                    for c, r in enumerate(self.posibleresults):
                        if r == self.rawdata.iat[i, 2]:
                            ax.scatter(self.rawdata.iat[i, 0], self.rawdata.iat[i, 1], c=colors[c], s=15)
                            break
            plt.legend(self.posibleresults, labelcolor=colors, markerscale=0, handletextpad=-1.5, shadow=True)
            mng = plt.get_current_fig_manager()
            mng.window.state('zoomed')
            plt.show()
        else:
            raise TypeError("Impossible to plot with less then 2 or more then 3 dimensions.")