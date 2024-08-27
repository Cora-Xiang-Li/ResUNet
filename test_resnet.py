import torchvision.models as models

# Load the pre-trained ResNeXt model
resnet = models.resnet.resnext34_32x4d(pretrained=True)

# Access and print layer details
print(resnet.layer1)
# print(resnet.layer2)
# print(resnet.layer3)
# print(resnet.layer4)
