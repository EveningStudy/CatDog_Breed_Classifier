import warnings
from PIL import Image
import torch
from torch import nn
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights

warnings.filterwarnings("ignore")


# print("(Enter 0 to exit the program)")
# image_path = input("Input the path of file: ")
# if image_path == '0':
#     exit(0)
# if image_path.count('\\'):
#     image_path = image_path.replace('\\', '/')
# if image_path.count('"'):
#     image_path = image_path.replace('"', '')
#
# try:
#     Image.open(image_path)
# except FileNotFoundError:
#     print("File not found")
#     exec(open('classification.py').read())
#     exit(0)

# def preprocess_image(image_path):
#     # print("(Enter 0 to exit the program)")
#     # image_path = input("Input the path of file: ")
#     if image_path.count('\\'):
#         image_path = image_path.replace('\\', '/')
#     if image_path.count('"'):
#         image_path = image_path.replace('"', '')
#     return image_path


def cat_and_dog_classification(image_path):
    image = Image.open(image_path)
    image = image.convert("RGB")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_size = 224
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    image = transform(image)
    net_model = models.vgg16(pretrained=True)
    net_model.classifier[6] = nn.Linear(4096, 2)
    # print(net_model)
    net_model.load_state_dict(
        torch.load('./utils/CatOrDog_vgg16_version2.pth'))
    net_model.to(device)
    # image = image.view(image.size()[0], -1)
    image = image.to(device)
    net_model.eval()
    with torch.no_grad():
        output = net_model(image.unsqueeze(0))
    result = output.argmax(1)
    final = result.item()
    preds = torch.nn.functional.softmax(output, dim=1)[:, final].tolist()
    return preds, final


def dog_classification(image_path):
    def load_network(netModel, netName, dropoutRatio, classNames, unfrozenLayers):
        for name, child in netModel.named_children():
            if name in unfrozenLayers:
                for param in child.parameters():
                    param.requires_grad = True
            else:
                for param in child.parameters():
                    param.requires_grad = False

        num_ftrs = netModel.fc.in_features
        netModel.fc = nn.Sequential(nn.Linear(num_ftrs, 256),
                                    nn.ReLU(),
                                    nn.Dropout(p=dropoutRatio),
                                    nn.Linear(256, len(classNames)))

        return netModel

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    class_names = [
        'Chihuahua', 'Japanese_spaniel', 'Maltese_dog', 'Pekinese', 'Shih-Tzu', 'Blenheim_spaniel', 'papillon',
        'toy_terrier', 'Rhodesian_ridgeback', 'Afghan_hound', 'basset', 'beagle', 'bloodhound', 'bluetick',
        'black-and-tan_coonhound', 'Walker_hound', 'English_foxhound', 'redbone', 'borzoi', 'Irish_wolfhound',
        'Italian_greyhound', 'whippet', 'Ibizan_hound', 'Norwegian_elkhound', 'otterhound', 'Saluki',
        'Scottish_deerhound', 'Weimaraner', 'Staffordshire_bullterrier', 'American_Staffordshire_terrier',
        'Bedlington_terrier', 'Border_terrier', 'Kerry_blue_terrier', 'Irish_terrier', 'Norfolk_terrier',
        'Norwich_terrier', 'Yorkshire_terrier', 'wire-haired_fox_terrier', 'Lakeland_terrier', 'Sealyham_terrier',
        'Airedale', 'cairn', 'Australian_terrier', 'Dandie_Dinmont', 'Boston_bull', 'miniature_schnauzer',
        'giant_schnauzer', 'standard_schnauzer', 'Scotch_terrier', 'Tibetan_terrier', 'silky_terrier',
        'soft-coated_wheaten_terrier', 'West_Highland_white_terrier', 'Lhasa', 'flat-coated_retriever',
        'curly-coated_retriever', 'golden_retriever', 'Labrador_retriever', 'Chesapeake_Bay_retriever',
        'German_short-haired_pointer', 'vizsla', 'English_setter', 'Irish_setter', 'Gordon_setter',
        'Brittany_spaniel', 'clumber', 'English_springer', 'Welsh_springer_spaniel', 'cocker_spaniel',
        'Sussex_spaniel', 'Irish_water_spaniel', 'kuvasz', 'schipperke', 'groenendael', 'malinois', 'briard',
        'kelpie', 'komondor', 'Old_English_sheepdog', 'Shetland_sheepdog', 'collie', 'Border_collie',
        'Bouvier_des_Flandres', 'Rottweiler', 'German_shepherd', 'Doberman', 'miniature_pinscher',
        'Greater_Swiss_Mountain_dog', 'Bernese_mountain_dog', 'Appenzeller', 'EntleBucher', 'boxer', 'bull_mastiff',
        'Tibetan_mastiff', 'French_bulldog', 'Great_Dane', 'Saint_Bernard', 'Eskimo_dog', 'malamute',
        'Siberian_husky', 'affenpinscher', 'basenji', 'pug', 'Leonberg', 'Newfoundland', 'Great_Pyrenees',
        'Samoyed', 'Pomeranian', 'chow', 'keeshond', 'Brabancon_griffon', 'Pembroke', 'Cardigan', 'toy_poodle',
        'miniature_poodle', 'standard_poodle', 'Mexican_hairless', 'dingo', 'dhole', 'African_hunting_dog'
    ]

    image = Image.open(image_path)
    image = image.convert("RGB")

    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()])

    image = transform(image)

    net_model = models.resnet152(pretrained=True)
    net_name = 'resnet152'
    unfrozen_layers = ['layer4', 'fc']
    dropout_ratio = 0.9
    net_model = load_network(net_model, net_name, dropout_ratio, class_names, unfrozen_layers)
    net_model.load_state_dict(torch.load('./utils/dog_classification_resnet152.pth'))
    net_model.to(device)
    # print(net_model)
    image = torch.reshape(image, (1, 3, 224, 224))
    image = image.to(device)
    net_model.eval()
    with torch.no_grad():
        output = net_model(image)
    result = output.argmax(1)
    final = class_names[result.item()]
    final = final.replace("_", " ")
    preds = torch.nn.functional.softmax(output, dim=1)[:, result.item()].tolist()
    return preds, final


def cat_classification(image_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image = Image.open(image_path)
    image = image.convert("RGB")

    class_names = [
        'Abyssinian', 'American Curl', 'American Shorthair', 'Balinese', 'Bengal',
        'Birman', 'Bombay', 'British Shorthair', 'Burmese', 'Cornish Rex',
        'Devon Rex', 'Egyptian Mau', 'Exotic Shorthair', 'Extra-Toes Cat - Hemingway Polydactyl',
        'Havana', 'Himalayan', 'Japanese Bobtail', 'Korat', 'Maine Coon',
        'Manx', 'Nebelung', 'Norwegian Forest Cat', 'Oriental Short Hair', 'Persian',
        'Ragdoll', 'Russian Blue', 'Scottish Fold', 'Selkirk Rex', 'Siamese',
        'Siberian', 'Snowshoe', 'Sphynx', 'Tonkinese', 'Toyger tiger cat', 'Turkish Angora'
    ]

    transform = transforms.Compose([
        transforms.Resize((250, 250)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = transform(image)
    # image = torch.reshape()
    net_model = models.resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
    num_ftrs = net_model.fc.in_features
    net_model.fc = nn.Linear(num_ftrs, 35)
    net_model = torch.load('./utils/cat_classification_resnet50.pth')
    net_model = net_model.to(device)
    image = image.to(device)
    net_model.eval()
    with torch.no_grad():
        output = net_model(image.unsqueeze(0))
    result = output.argmax(1)
    preds = torch.nn.functional.softmax(output, dim=1)[:, result.item()].tolist()
    return preds, class_names[result.item()]

# preds, output = cat_and_dog_classification(image_path)
# if preds[0] < 0.97:
#     sys.stdout.write("The image is neither a dog nor a cat.\n\n")
# elif output:
#     sys.stdout.write("Dog: ")
#     preds_dog, output_dog = dog_classification(image_path)
#     preds_dog = round(preds_dog[0], 3)
#     sys.stdout.write(output_dog)
#     sys.stdout.write(f"(Confidence: {preds_dog * 100}%)\n\n")
# elif output == 0:
#     sys.stdout.write("Cat: ")
#     preds_cat, output_cat = cat_classification(image_path)
#     preds_cat = round(preds_cat[0], 2)
#     sys.stdout.write(output_cat)
#     sys.stdout.write(f"(Confidence: {preds_cat * 100}%)\n\n")
#
# exec(open('classification.py').read())
# exit(0)
