import cv2
import os

# letter
letter = 'Y'

#img size
image_x, image_y = 64, 64

def capture_images(counter):
    
    cam = cv2.VideoCapture(0)
    
    while True:

        ret, frame = cam.read()
        frame = cv2.flip(frame, 1)

        img = frame.copy()
        img = cv2.rectangle(img, (375, 150), (625, 400), (255, 0, 0), thickness = 2, lineType = 8, shift = 0)
   
        result = img[152:398, 377:623]           
        
        cv2.imshow('frame', img)


        # capture
        if cv2.waitKey(1) == ord('c'):

            # save full image
            img_name = 'data/full_images/full_image_' + letter + '_' + str(counter) + '.png'
            cv2.imwrite(img_name, frame)
            print(f'{img_name} written!')

            # check if folder exists
            letter_path = 'data/letters/' + letter
            if not os.path.exists(letter_path):
                os.makedirs(letter_path)

            # save hand image
            hand_img_name = letter_path + '/cropped_image_' + letter + '_' + str(counter) + '.png'
            save_img = cv2.resize(result, (image_x, image_y))
            cv2.imwrite(hand_img_name, save_img)
            print(f'{hand_img_name} written!')

            counter = counter + 1
        
        # quit
        if cv2.waitKey(1) == ord('q'):
            break


    cam.release()
    cv2.destroyAllWindows()
    
capture_images(101) # counter: the counter of the last saved image + 1