import cv2
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms

## image size
image_x, image_y = 64, 64

## define device
def get_default_device():
    # set device to GPU or CPU
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

## send to device
def to_device(data, device):
    # mode data to device
    if isinstance(data,(list,tuple)):
        return [to_device(x,device) for x in data]
    
    return data.to(device,non_blocking = True)

## load model (from the training code)
# final architecture of CNN model
class HandGestureClassificationFinal(nn.Module):
    
    def __init__(self):

        super().__init__()
        self.network = nn.Sequential(
            
            nn.Conv2d(3, 32, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        
            nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            
            nn.Flatten(),
            nn.Linear(16384, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 21)
        )
    
    def forward(self, xb):
        return self.network(xb)

## model
model = HandGestureClassificationFinal()
model.load_state_dict(torch.load('./outputs/models/final_classification_model.pth'))
device = get_default_device()
model = to_device(model, device)

## predictor
def predict_img_class(model):
    # read image
    img = cv2.imread('./.temp/img.png', cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # send to device
    img_tensor = transforms.ToTensor()(img)
    img_tensor = to_device(img_tensor.unsqueeze(0), device)

    # get prediction
    prediction = model(img_tensor)
    _, preds = torch.max(prediction, dim = 1)
    prob = torch.max(F.softmax(prediction, dim = 1)).item()

    return [letras[preds[0].item()], prob]

## classes
classes = 21
letras = {0:'A', 1:'B', 2:'C' , 3:'D', 4:'E', 5:'F', 6:'G', 7:'I', 8:'L', 9:'M', 10:'N', 11:'O', 12:'P', 13:'Q', 14:'R', 15:'S', 16:'T', 17:'U', 18:'V', 19:'W',20:'Y'}

## geet video    
cam = cv2.VideoCapture(0)

img_counter = 0

img_text = ['','']
while True:
    # get frame
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)

    # draw rectangle and crop image
    img = cv2.rectangle(frame, (375, 50), (625, 300), (255, 0, 0), thickness = 2, lineType = 8, shift = 0)
    imcrop = img[52:298, 377:623]
    
    # save cropped image
    img_name = './.temp/img.png'
    save_img = cv2.resize(imcrop, (image_x, image_y))
    cv2.imwrite(img_name, save_img)

    # prediction
    pred = predict_img_class(model)
    img_text = pred[0]

    # show images
    text_pred = 'Prediction: ' + str(img_text)
    cv2.putText(frame, text_pred, (375, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0))
    cv2.imshow("test", frame)
        
    # press 'q' to leaveq
    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()