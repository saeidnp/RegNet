from commons import *

import cifar100_data_loader as cifardata

class DataClass:
    #static variables
    use_cuda = torch.cuda.is_available()

    #methods
    def __init__(self, root_dir):
        self.root_dir = root_dir

    def load(self, batch_size, validation_portion,
            portion_to_keep=[1.0] * 100, augment=False,
            random_seed=0, pin_mem_perm=True):
        self.batch_size = batch_size

        pin_memory, num_workers = False, 4
        if use_cuda and pin_mem_perm:
            # If CUDA is available, use pin_memory.
            # But, if the system crashes don't use pin_memory.
            pin_memory, num_workers = True, 1

        (self.trainloader, self.validloader) = cifardata.getTrainValidLoader(self.root_dir,
                                                                             batch_size=batch_size,
                                                                             augment=augment,
                                                                             random_seed=random_seed,
                                                                             show_sample=False,
                                                                             num_workers=num_workers,
                                                                             pin_memory=pin_memory,
                                                                             valid_size = validation_portion,
                                                                             portion_to_keep= portion_to_keep)
        self.testloader = cifardata.getTestLoader(self.root_dir, batch_size=batch_size,
                                                  num_workers=num_workers, pin_memory=pin_memory)

        self.classes = [c.decode("utf-8") for c in cifardata.getClasses(self.root_dir)]
        self.super_classes = cifardata.getSuperClasses(self.root_dir)

        self.parent_list = cifardata.getParentList(self.root_dir)


    def updatePortionToKeep(self, target_class, portion, portion_to_keep=[1.0] * 100):
        if portion < 0.0 or portion > 1.0:
            raise Exception('portion is expected to be in [0,1] (got {})'.format(portion))

        if target_class == 'all':
            portion_to_keep = [portion] * 100
        else:
            class_names_dict = getClassNamesDict(self.root_dir)
            if target_class not in class_names_dict:
                raise Exception('class {} was not found among classes'.format(target_class))
            portion_to_keep[class_names_dict[target_class]] = portion

        return portion_to_keep