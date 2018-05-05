from commons import *

def topK(model, loader, k):
    softmax = nn.Softmax()
    model.eval()
    correct = 0
    total = 0
    for data in loader:
        images, labels = data
        images_var = Variable(images).cuda() if use_cuda else Variable(images)
        outputs = model(images_var)
        _, predicted = outputs.topk(k, 1)
        predicted = predicted.data
        total += labels.size(0)
        for i in range(labels.size(0)):
            correct += (predicted[i].cpu() == labels[i]).sum()

    return correct / total

def top1(model, loader):
    return topK(model, loader, 1)