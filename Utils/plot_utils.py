import matplotlib.pyplot as plt
import numpy as np


class PlotUtils:
    @staticmethod
    def setup_plotting():
        plt.style.use('default')
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 12

    @staticmethod
    def save_plot(filename):
        plt.tight_layout()
        plt.savefig(f'plots/{filename}', dpi=300, bbox_inches='tight')

    @staticmethod
    def show_plot():
        plt.tight_layout()
        plt.show()
        plt.close()