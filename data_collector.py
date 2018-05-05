from commons import *
from utils import *

class DataCollector:
    def __init__(self, root_dir, experiment_name):
        self.losses_ext = '_losses.npy'
        self.num_seen_ext = '_numseen.npy'
        self.model_ext = '.pt'
        self.hyperparam_ext = '_hyperparam.npy'
        self.hierarchy_filename = 'final_hierarchy.npy'

        self.experiment_name = experiment_name
        self.root_dir = root_dir if (root_dir[-1] == '/') else root_dir+'/'
        self.losses_dir = self.root_dir + self.experiment_name + '/' + 'losses/'
        self.models_dir = self.root_dir + self.experiment_name + '/' + 'models/'

        checkMakeDir(self.root_dir)
        checkMakeDir(self.root_dir + self.experiment_name + '/')
        checkMakeDir(self.losses_dir)
        checkMakeDir(self.models_dir)
        wipeDir(self.losses_dir)
        wipeDir(self.models_dir)

    def getLossesDir(self):
        return self.losses_dir
    def getModelsDir(self):
        return self.models_dir

    def storeModel(self, model, state):
        torch.save(model.state_dict(), self.models_dir+state+self.model_ext)

    def storeLosses(self, losses, num_seen, state):
        np.save(self.losses_dir+state+self.losses_ext, losses)
        np.save(self.losses_dir+state+self.num_seen_ext, num_seen)

    def appendLosses(self, losses, num_seen, state):
        if doesFileExist(self.losses_dir+state+self.losses_ext):
            tmp = np.load(self.losses_dir+state+self.losses_ext).tolist()
            tmp.extend(losses)
            np.save(self.losses_dir+state+self.losses_ext, tmp)

            tmp = np.load(self.losses_dir+state+self.num_seen_ext).tolist()
            tmp.extend(num_seen)
            np.save(self.losses_dir+state+self.num_seen_ext, tmp)
        else:
            self.storeLosses(losses, num_seen, state)

    def storeHyperparamLoss(self, losses):
        np.save(self.losses_dir+self.hyperparam_ext, losses)

    def loadModelStatedict(self, state):
        return torch.load(self.models_dir+state+self.model_ext)

    def loadLosses(self, state):
        losses = np.load(self.losses_dir+state+self.losses_ext).tolist()
        num_seen = np.load(self.losses_dir+state+self.num_seen_ext).tolist()
        return [losses, num_seen]

    def finalizeLosses(self, state, dst_root):
        dst_root = dst_root if (dst_root[-1] == '/') else dst_root+'/'
        dst_dir = dst_root + self.experiment_name + '/' + 'losses/'

        checkMakeDir(dst_root)
        checkMakeDir(dst_root + self.experiment_name + '/')
        checkMakeDir(dst_dir)

        from shutil import copyfile
        copyfile(self.losses_dir+state+self.losses_ext, dst_dir+'final'+self.losses_ext)
        copyfile(self.losses_dir+state+self.num_seen_ext, dst_dir+'final'+self.num_seen_ext)

    def finalizeModel(self, state, dst_root):
        dst_root = dst_root if (dst_root[-1] == '/') else dst_root+'/'
        dst_dir = dst_root + self.experiment_name + '/' + 'models/'

        checkMakeDir(dst_root)
        checkMakeDir(dst_root + self.experiment_name + '/')
        checkMakeDir(dst_dir)

        from shutil import copyfile
        copyfile(self.models_dir+state+self.model_ext, dst_dir+'final_model'+self.model_ext)

    def finalizeHyperparamLoss(self, dst_root):
        dst_root = dst_root if (dst_root[-1] == '/') else dst_root+'/'
        dst_dir = dst_root + self.experiment_name + '/' + 'losses/'

        checkMakeDir(dst_root)
        checkMakeDir(dst_root + self.experiment_name + '/')
        checkMakeDir(dst_dir)

        from shutil import copyfile
        copyfile(self.losses_dir+self.hyperparam_ext, dst_dir+'final'+self.hyperparam_ext)

    def finilizeHierarchy(self, hierarchy, dst_root):
        dst_root = dst_root if (dst_root[-1] == '/') else dst_root+'/'
        dst_dir = dst_root + self.experiment_name + '/' + 'models/'

        checkMakeDir(dst_root)
        checkMakeDir(dst_root + self.experiment_name + '/')
        checkMakeDir(dst_dir)

        np.save(dst_dir+self.hierarchy_filename, hierarchy)