from commons import *

from sklearn.cluster import KMeans

from data_loader import DataClass
import evaluate
from loss_monitor import LossMonitor
from utils import *
from const import *

class Trainer:
    #methods
    def __init__(self, data_collector,
                 learning_rate,
                 use_hierarchy,
                 dynamic_hierarchy_flag,
                 parent_enc_update_interval = None,
                 hierarchy_update_interval = None,
                 hierarchy_update_method = None,
                 kmeans_num_clusters = None):
        #hierarchy_update_method = None/'greedy'/'kmeans

        self.data_collector = data_collector
        self.learning_rate = learning_rate   #The default value for leaning rate to be used in training. train function has an optional argument if a different learning rate was needed.
        self.use_hierarchy = use_hierarchy
        self.dynamic_hierarchy_flag = dynamic_hierarchy_flag
        self.hierarchy_update_interval = hierarchy_update_interval
        self.parent_enc_update_interval = parent_enc_update_interval

        if self.dynamic_hierarchy_flag:
            myassertList(hierarchy_update_method, [GREEDY_NAME, KMEANS_NAME])
            if hierarchy_update_method == GREEDY_NAME:
                self.__updateHierarchyFromBeta = self.__updateHierarchyGreedy
            elif hierarchy_update_method == KMEANS_NAME:
                self.__updateHierarchyFromBeta = self.__updateHierarchyKMeans
                myassert(kmeans_num_clusters != None,
                        'kmeans_num_clusters is not provided')
                myassert(kmeans_num_clusters > 0,
                        'kmeans_num_clusters is not valid (got {0})'.format(kmeans_num_clusters))
                self.kmeans_num_clusters = kmeans_num_clusters
        else:
            myassertList(hierarchy_update_method, [None])

    def train(self, model, data_class, lambda_1, lambda_2,
              num_epochs, learning_rate = None,
              do_print=True, do_plot=True):
        if learning_rate == None:
            learning_rate = self.learning_rate
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        
        model.train()
        criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                              momentum=0.9, nesterov=True)

        self.parent_list = list(data_class.parent_list)
        self.__refreshTheta()

        loss_monitor = LossMonitor(do_print, do_plot, data_class.batch_size,
                                    data_collector=self.data_collector)
        num_seen_samples, num_optimizer_updates = 0, 0
        for epoch in range(num_epochs):  # loop over the dataset multiple times
            print('------------------------------------ epoch #', epoch)
            loss_monitor.epochStart(epoch)
            if epoch % 50 == 0 and epoch != 0:
                learning_rate /= 10
                self.__adjustLearningRate(learning_rate)

            for i, data in enumerate(data_class.trainloader, 0):
                # get the inputs
                inputs, labels = data
                num_seen_samples += inputs.size(0)

                # wrap them in Variable
                inputs = Variable(inputs).cuda() if use_cuda else Variable(inputs)
                labels = Variable(labels).cuda() if use_cuda else Variable(labels)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels) + \
                        (lambda_2/2) * self.__hierarchicalRegularizer(model)
                loss.backward()
                self.optimizer.step()

                loss_monitor.optimizerUpdate(loss, epoch, i, num_seen_samples)

                num_optimizer_updates += 1
                if self.dynamic_hierarchy_flag and num_optimizer_updates % self.hierarchy_update_interval == 0:
                    # Update the hierarchy after each few iterations.
                    print('updating hierarchy ....')
                    self.__updateTheta(model, lambda_1, lambda_2)
                    self.__updateHierarchy(model)
                    self.__updateTheta(model, lambda_1, lambda_2)

                elif self.use_hierarchy and num_optimizer_updates % self.parent_enc_update_interval == 0:
                    self.__updateTheta(model, lambda_1, lambda_2)

            loss_monitor.epochEnd(epoch, num_epochs)
            self.data_collector.storeModel(model, 'ep'+str(epoch))

        print('Finished Training')
        loss_monitor.trainingDone()
        self.data_collector.storeModel(model, 'final')

    def hyperParamTuning(self, model, data_class, possible_vals_1, possible_vals_2):
        self.data_collector.storeModel(model, 'hyperparam')
        lambda_options = [(i,j) for i in possible_vals_1 for j in possible_vals_2]
        random.shuffle(lambda_options)
        hyper_param_res = {}
        for (lambda_1, lambda_2) in lambda_options:
            self.train(model, data_class, lambda_1, lambda_2,
                       num_epochs=100, do_print=False, do_plot=False)
            accuracy = evaluate.top1(model, data_class.validloader)
            print('Validation accuracy = %f \t (%f, %f)' % (accuracy, lambda_1, lambda_2))
            hyper_param_res[(lambda_1, lambda_2)] = accuracy
            model.load_state_dict(self.data_collector.loadModelStatedict('hyperparam'))
            self.data_collector.storeHyperparamLoss(hyper_param_res)

        self.data_collector.finalizeHyperparamLoss('results')
        max_accuracy = max(hyper_param_res.values())
        return [k for k in hyper_param_res if hyper_param_res[k] == max_accuracy][0]

    def getHierarchy(self):
        hierarchy = {}
        for idx, parent in enumerate(self.parent_list):
            if parent in hierarchy:
                hierarchy[parent].append(idx)
            else:
                hierarchy[parent] = [idx]
        return hierarchy
    
    def getNamedHierarchy(self, data_class):
        hierarchy = self.getHierarchy()
        named_hierarchy = {}
        for i, children in hierarchy.items():
            named_hierarchy[i] = [data_class.classes[i] for i in children]
            
        return named_hierarchy

    #private methods:
    def __hierarchicalRegularizer(self, model):
        l2_reg = None
        for name, param in model.named_parameters():
            if 'beta' not in name:
                if l2_reg is None:
                    l2_reg = param.pow(2).sum()
                else:
                    l2_reg = l2_reg + param.pow(2).sum()
        for name, param in model.beta.named_parameters():
            myassertList(name, ['weight', 'bias'])
            if(name == 'weight'):
                tensor = torch.FloatTensor(self.__parents_weight)
            elif(name == 'bias'):
                tensor = torch.FloatTensor(self.__parents_bias)

            var = Variable(tensor).cuda() if use_cuda else Variable(tensor)
            l2_reg = l2_reg + (param - var).pow(2).sum()
        return l2_reg

    def __refreshTheta(self):
        self.theta_weight = np.zeros((100, 2048))
        self.theta_bias = np.zeros(100)

        self.__updateParentsEnc()

    def __updateTheta(self, model, lambda_1, lambda_2):
        beta = self.__getBetaEnc(model)
        beta_weight = beta[:, 0:-1]
        beta_bias = beta[:, -1]

        hierarchy = self.getHierarchy()
        for parent, children in hierarchy.items():
            self.theta_weight[parent] = np.sum(beta_weight[children], axis=0) / (len(children)+lambda_1/lambda_2)
            self.theta_bias[parent] = np.sum(beta_bias[children], axis=0) / (len(children)+lambda_1/lambda_2)

        self.__updateParentsEnc()

    def __updateParentsEnc(self):
        self.__parents_weight = self.theta_weight[self.parent_list]
        self.__parents_bias = self.theta_bias[self.parent_list]

    def __updateHierarchy(self, model):
        self.__updateHierarchyFromBeta(self.__getBetaEnc(model))

    def __adjustLearningRate(lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def __getBetaEnc(self, model):
        for name, param in model.beta.named_parameters():
            myassertList(name, ['weight', 'bias'])
            beta_params = param.data.cpu().numpy()
            if(name == 'weight'):
                beta_weight = beta_params
            elif(name == 'bias'):
                beta_bias = beta_params
        return np.column_stack([beta_weight, beta_bias])

    def __updateHierarchyGreedy(self, beta):
        unique_parents = list(set(self.parent_list))
        num_classes = len(beta)
        theta = np.column_stack([self.theta_weight, self.theta_bias])

        for class_idx in range(num_classes):
            nearest_parent_idx = 0
            nearest_dist = float('Inf')
            class_enc = beta[class_idx]
            for parent_idx in unique_parents:
                parent_enc = theta[parent_idx]
                dist = np.inner(class_enc-parent_enc, class_enc-parent_enc)
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_parent_idx = parent_idx

            self.parent_list[class_idx] = nearest_parent_idx

    def __updateHierarchyKMeans(self, beta):
        kmeans = KMeans(n_clusters=self.kmeans_num_clusters).fit(beta)
        self.parent_list = list(kmeans.labels_)