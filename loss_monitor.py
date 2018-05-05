from commons import *

#import stopwatch
from plotter import Plotter
from utils import StopWatch

class LossMonitor:
    def __init__(self, do_print, do_plot, batch_size, data_collector,
                    plot_every_batch=10, print_every_batch = 100):
        self.do_print = do_print
        self.do_plot = do_plot
        self.batch_size = batch_size
        self.data_collector = data_collector

        self.plot_every=batch_size*plot_every_batch
        self.print_every = print_every_batch
        if self.do_plot:
            self.plotter = Plotter(2,1)
        self.stop_watch = StopWatch()
        #self.stop_watch.start()
        #self.num_optimizer_updates = 0
        self.plot_running_loss = 0.0
        self.plot_losses, self.plot_avg_losses, self.plot_seen_samples = [],[],[]
        self.acc_loss, self.acc_points_reported = 0,0

    def epochStart(self, epoch):
        if not(self.do_print or self.do_plot):
            return
        if epoch != 0:
            self.data_collector.appendLosses(self.plot_losses,
                                            self.plot_seen_samples, 'running')
            self.acc_points_reported += len(self.plot_losses)
            self.acc_loss += np.sum(self.plot_losses)
        self.plot_losses = []
        self.plot_avg_losses = []
        self.plot_seen_samples = []
        self.running_loss = 0.0

        self.__print_thr, self.__plot_thr = self.print_every, self.plot_every

    def epochEnd(self, epoch, num_epochs):
        if self.do_print == False and epoch != 0 and epoch % 10 == 0:
            print('epoch #%d/%d \t %.2f, %.2f seconds elapsed' %
                  (epoch, num_epochs, self.stop_watch.lap(), self.stop_watch.getElapsed()))

    def optimizerUpdate(self, loss, epoch, iteration, num_seen_samples):
        if not(self.do_print or self.do_plot):
            return
        self.running_loss += loss.data[0]
        self.plot_running_loss += loss.data[0]
        if self.do_print and iteration >= self.__print_thr:
            self.__print_thr += self.print_every
            print('[%d, %5d] loss: %.3f \t %.2f, %.2f seconds elapsed' %
                    (epoch + 1, iteration*self.batch_size,
                    self.running_loss / (self.batch_size*self.print_every),
                    self.stop_watch.lap(), self.stop_watch.getElapsed()))
            self.running_loss = 0.0
        if num_seen_samples >= self.__plot_thr: # plot every 100 seen samples
            self.__plot_thr += self.plot_every
            self.plot_losses.append(self.plot_running_loss/self.plot_every)
            self.plot_avg_losses.append((np.sum(self.plot_losses)+self.acc_loss) /
                                        (len(self.plot_losses)+self.acc_points_reported))
            self.plot_seen_samples.append(num_seen_samples)
            self.plot_running_loss = 0.0
            if self.do_plot:
                self.plotter.update(self.plot_seen_samples,
                                    [self.plot_losses, self.plot_avg_losses], 0)
                self.plotter.update(self.plot_seen_samples[-50:],
                                [self.plot_losses[-50:]], 1)

    def trainingDone(self):
        if self.do_print or self.do_plot:
            self.data_collector.appendLosses(self.plot_losses,
                                             self.plot_seen_samples, 'running')