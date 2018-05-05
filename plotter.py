import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class Plotter:
    def __init__(self,rows,cols):
        self.rows, self.cols = rows,cols
        #self.fig = plt.figure(figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
        [w,h] = plt.rcParams.get('figure.figsize')
        self.fig = plt.figure(figsize=(w,1.5*h))
        self.axes = {}
        for i in range(self.rows):
            for j in range(self.cols):
                self.axes[(i,j)] = self.fig.add_subplot(self.rows, self.cols, i*self.cols+j+1)

        plt.ion()
        self.fig.show()
        self.fig.canvas.draw()

    def update(self, x, y_list, plot_num):
        i = plot_num//self.cols
        j = plot_num%self.cols
        ax = self.axes[(i,j)]
        ax.clear()
        for y in y_list:
            ax.plot(x, y)
        self.fig.canvas.draw()