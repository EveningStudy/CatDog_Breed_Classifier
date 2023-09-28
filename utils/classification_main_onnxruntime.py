import warnings
from PIL import Image
import onnxruntime
import torch
from torchvision import transforms

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
#     exit(0)


def cat_and_dog_classification(image_path):
    onnx_model_path = './utils/CatOrDog_vgg16_version2.ort'
    ort_session = onnxruntime.InferenceSession(onnx_model_path)

    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image)
    image = image.unsqueeze(0)

    ort_inputs = {ort_session.get_inputs()[0].name: image.numpy()}
    ort_outs = ort_session.run(None, ort_inputs)

    output = torch.Tensor(ort_outs[0])
    result = output.argmax(1)
    final = result.item()
    preds = torch.nn.functional.softmax(output, dim=1)[:, final].tolist()
    return preds, final


def dog_classification(image_path):
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

    onnx_model_path = './utils/dog_classification_resnet152.ort'
    ort_session = onnxruntime.InferenceSession(onnx_model_path)

    image = torch.reshape(image, (1, 3, 224, 224))
    image = image.to(device)

    ort_inputs = {ort_session.get_inputs()[0].name: image.cpu().numpy()}
    ort_outs = ort_session.run(None, ort_inputs)

    output = torch.Tensor(ort_outs[0])
    _, top5_indices = output.topk(5)
    top5_probs = torch.nn.functional.softmax(output, dim=1)[0, top5_indices[0]].tolist()

    top5_classes = [class_names[i] for i in top5_indices[0]]
    top5_results = list(zip(top5_classes, top5_probs))

    return top5_results


def cat_classification(image_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    class_names = [
        'Abyssinian', 'American Curl', 'American Shorthair', 'Balinese', 'Bengal',
        'Birman', 'Bombay', 'British Shorthair', 'Burmese', 'Cornish Rex',
        'Devon Rex', 'Egyptian Mau', 'Exotic Shorthair', 'Extra-Toes Cat - Hemingway Polydactyl',
        'Havana', 'Himalayan', 'Japanese Bobtail', 'Korat', 'Maine Coon',
        'Manx', 'Nebelung', 'Norwegian Forest Cat', 'Oriental Short Hair', 'Persian',
        'Ragdoll', 'Russian Blue', 'Scottish Fold', 'Selkirk Rex', 'Siamese',
        'Siberian', 'Snowshoe', 'Sphynx', 'Tonkinese', 'Toyger tiger cat', 'Turkish Angora'
    ]

    image = Image.open(image_path)
    image = image.convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((250, 250)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = transform(image)

    onnx_model_path = './utils/cat_classification_resnet50.ort'
    ort_session = onnxruntime.InferenceSession(onnx_model_path)

    image = image.unsqueeze(0)
    image = image.to(device)

    ort_inputs = {ort_session.get_inputs()[0].name: image.cpu().numpy()}
    ort_outs = ort_session.run(None, ort_inputs)

    output = torch.Tensor(ort_outs[0])
    _, top5_indices = output.topk(5)
    top5_probs = torch.nn.functional.softmax(output, dim=1)[0, top5_indices[0]].tolist()

    top5_classes = [class_names[i] for i in top5_indices[0]]
    top5_results = list(zip(top5_classes, top5_probs))

    return top5_results

# preds, output = cat_and_dog_classification(image_path)
# if preds[0] < 0.97:
#     print("The image is neither a dog nor a cat.\n")
# elif output:
#     print("Dog: ")
#     top5_results_dog = dog_classification(image_path)
#     for i, (class_name, prob) in enumerate(top5_results_dog, start=1):
#         print(f"{i}. {class_name}: {prob * 100:.2f}")
#     print("\n")
# elif output == 0:
#     print("Cat: ")
#     top5_results_cat = cat_classification(image_path)
#     for i, (class_name, prob) in enumerate(top5_results_cat, start=1):
#         print(f"{i}. {class_name}: {prob * 100:.2f}")
#     print("\n")
