"""
1. PURPOSE OF PROGRAM: The purpose of this program was 
create a program that could not only detect faces, but recognize them, and 
compare people based off of the model's recognition. By giving the users 
the power to control the image data base that the model is trained over, my 
application is more of a environment in which users can play around with the 
power of AI in computer vision like never before. Now that you have power over 
what photos the model is is trained with, users can tinker around with how the 
model responds to certain specific photos (e.g. facial expressions, 
environmental features, etc.), or see if the user they look more similar to 
Ben Affleck or Madonna (or any face that the model is trained on). The 
possiblities are endless, so what are you waiting for - try the program for 
yourself!

2. MAJOR FEATURES: Major features of my program is 
the ability to add photos and corresponding name to train the model with, to 
display a webcam onto which the facial recognition model will predict who it 
believes is in the frame, to display a webcam which can compare faces in the 
frame with selected trained people with a confidence level (so you can
compare your face with others faces), and finally you can delete folders of 
pictures of people from the model's training database.

3. What 3rd party modules must be installed for the program to work?
(Must be clear and explicit here or we won't be able to test your program.)

cv2, PIL, and numpy are 3rd party modules that must be installed separately. 
Also to use cv.face, I had to install opencv-contrib-python.

how to install:
pip install opencv-contrib-python
pip install pillow
pip install numpy


4. WHAT WAS LEARNT:

There were many, many things I learned while creating this project. 
- I learned how to use the basic classes and functions of the OpenCV library 
to interface computer vision with Python
- I learned how to display my webcam in a tkinker window by using PIL. This was 
a rather convoluted process 
- first I had to flip the image, then turn it to RGBA format, convert the 
format using Image and ImageTk (PIL), and finally set this to the background 
of a label.
- I learned how the basics of how a facial recognition system works, and how 
tweaking certain environment variables can make a hige improvement in the 
accuracy (e.g. scaleFactor, minNeighbors, minSize, and flags)
- I learned how to take photos, store them in the file system using the OS 
module in Python (using os.makedirs and Image.save())
- I learned how to make multiple windows with tkinter, and I learned how to 
share data between multiple tkinter windows

5. CHALLENGE TO OVERCOME:
It is very hard to decide what was the singular most diffult thing I had to 
overcome, but I believe the hardest thing to overcome was when I had a drop in 
facial recognition accuracy. Essentially, as I incrementally developed my 
program, I was making sure that every time I made a large jump in progress 
in my program, I would make sure that everything else that I didn't
change still worked as it did in the last "big jump". But there was one time 
when I was working on displaying the webcam to a tkinter window instead of 
the openCV native window that the facial recognition accuracy had dropped. 
This was not at all obvious, in fact I thought that the recognition was 
perfectly fine, and I proceeded with the program. But later, I decided to test
my algorithm on the OpenCV native window (via cv.imshow()) and the accuracy 
suddenly increased. I was shocked, how was that possible? But I found that 
the only difference between the two methods of showing my video feed was that
 in tkinter I had to use cv.flip() to display each frame in the videofeed, 
 while in the OpenCV native window I was not doing that. I realized that I had
to flip the image with cv.flip() when I was training my model as well, and to 
my delight, the accuracy of my model skyrocketed.

I think this was difficult to overcome because of how subtle the error was. 
I initally thought the OpenCV face recognizer was just not very accurate 
(as many places online state), but just using simple programming fundamentals 
I was able to nail down where the error was occuring.

6. BUILD ON:
Next, I want to try to incorpate a deep learning model for the facial 
recognition, instead of just using the OpenCV facial recognition. The facial 
recognition in this model, albeit pretty good, is not trained with deep 
learning, it is trained with a basic math sequences based off of the 
pixel values. Deep Learning would make this program a lot more accurate 
I would imagine.

Aside from Deep Learning, I would love to make a system that could delete 
individual images rather than the entire directory of images for the training 
data. This way, if an image was taken by accident, the user can delete that 
specific image. Also, on a higher level, I would love to implement filters 
similar to Snapchat filters. I can put a "filter" on any detected faces, 
and if I change something about my eyebrow or tongue the filter would also 
react in some way.

"""""

"""
PLEASE READ:

When you take a photo, it is stored under the Faces directory, but deleting
the image permanently deletes the image.

Also, the faces of Ben Afflek, Elton John, Jerry Seinfield,
Madonna, and Mindy Kaling are from Jason Dsouza's OpenCV course. These
can be deleted as needed

"""

from tkinter import *
from tkinter import ttk
import cv2 as cv
from PIL import Image, ImageTk
import os
import numpy as np
import time
from tkinter import messagebox
import shutil


class Facial_Recognizer:

    def __init__(self):

        self.DIR = r'Faces'
        self.people = self.get_people_list()
        self.haar_cascade = cv.CascadeClassifier('haar_face.xml')
        self.face_recognizer = None
        self.train_model()

    def get_cascade(self):
        return self.haar_cascade

    def get_path(self):
        return self.DIR

    def get_people_list(self):
        return os.listdir(self.DIR)

    def train_model(self):
        # uses the directories under Faces/train to train the model

        features = []
        labels = []

        self.people = self.get_people_list()

        for person in self.people:
            print(f"{person} trained")
            path = os.path.join(self.DIR, person)
            label = self.people.index(person)

            for img in os.listdir(path):
                img_path = os.path.join(path, img)

                img_array_temp = cv.imread(img_path)
                img_array = cv.flip(img_array_temp, 1)
                if img_array is None:
                    continue

                grey = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

                faces_rect = self.haar_cascade.detectMultiScale(
                    grey,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),
                    flags=cv.CASCADE_SCALE_IMAGE)

                for (x, y, w, h) in faces_rect:
                    faces_roi = grey[y:y+h, x:x+w]
                    features.append(faces_roi)
                    labels.append(label)

        features_array = np.array(features, dtype='object')
        labels_array = np.array(labels)
        face_recognizer = cv.face.LBPHFaceRecognizer_create()

        # Train the Recognizer on the features list and the labels list
        # one-to-one mapping of faces and labels, must only be one detected 
        # face per label!
        face_recognizer.train(features_array, labels_array)

        # save the recognizer to the global variable
        self.face_recognizer = face_recognizer

    def get_facial_recognizer(self):
        return self.face_recognizer

    def test_model(self):
        # test model in native openCV window
         
        haar_cascade = self.haar_cascade

        people = self.get_people_list()

        face_recognizer = self.face_recognizer

        video_capture = cv.VideoCapture(0)

        while True:
            ret, frame = video_capture.read()

            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            faces = haar_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv.CASCADE_SCALE_IMAGE
            )

            for (x, y, w, h) in faces:
                faces_roi = gray[y:y+h, x:x+w]

                label, confidence = face_recognizer.predict(faces_roi)

                cv.putText(frame, str(people[label]), (x, y),
                           cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), 
                           thickness=2)
                cv.rectangle(frame, (x, y), (x+w, y+h),
                             (0, 255, 0), thickness=2)

            cv.imshow('Video', frame)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv.destroyAllWindows()

    def individual_face_recognizer(self, people_selected):
        # this is called for the open_similarity_screen()
        # returns a cv.face.LBPHFaceRecognizer trained on people_selected

        features = []
        labels = []

        for person in people_selected:
            path = os.path.join(self.DIR, person)
            label = self.people.index(person)

            for img in os.listdir(path):
                img_path = os.path.join(path, img)

                img_array_temp = cv.imread(img_path)
                img_array = cv.flip(img_array_temp, 1)
                if img_array is None:
                    continue

                grey = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

                faces_rect = cv.CascadeClassifier('haar_face.xml').\
                    detectMultiScale(
                    grey,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),
                    flags=cv.CASCADE_SCALE_IMAGE)

                for (x, y, w, h) in faces_rect:
                    faces_roi = grey[y:y+h, x:x+w]
                    features.append(faces_roi)
                    labels.append(label)

        features_array = np.array(features, dtype='object')
        labels_array = np.array(labels)
        one_face_recognizer = cv.face.LBPHFaceRecognizer_create()

        # Train the Recognizer on the features list and the labels list
        # one-to-one mapping of faces and labels, must only be one detected 
        # face per label!
        one_face_recognizer.train(features_array, labels_array)
        return one_face_recognizer


def main():

    face_recognizer = Facial_Recognizer()

    root = Tk()
    root.title("Facial Recognition")
    root.resizable(False, False)
    root.geometry("800x400")

    create_controls(root, face_recognizer)

    root.mainloop()


def create_controls(root, face_recognizer):
    # the main screen which has buttons for navigating the interface

    top_frame = ttk.Frame(root)
    top_frame.grid(row=1, column=1, columnspan=5)

    welcome_label = ttk.Label(
        top_frame, font='Arial 16 bold', text="Welcome to Facial Recognition")
    welcome_label.grid(row=1, column=1, sticky='we')

    instructions_button = Button(top_frame, font='Arial 12 bold',
                                 text='Instructions', width=15, 
                                 command=lambda: open_instructions())
    instructions_button.grid(row=1, column=2, padx=5, pady=5, sticky='we')

    bottom_frame = ttk.Frame(root)
    bottom_frame.grid(row=2, column=1, columnspan=5)

    train_new_face_button = Button(bottom_frame, font='Arial 24 bold', 
                                   text='Train New Face',
                                   width=20, 
                                   command=lambda: 
                                   open_training_screen\
                                    (root, face_recognizer, listbox))
    train_new_face_button.grid(row=1, column=1, padx=5, pady=5, sticky='we')

    delete_directories_button = Button(bottom_frame, font='Arial 24 bold', 
                                       text='Delete Face',
                                       width=20, 
                                       command=lambda: delete_training_pics\
                                        (face_recognizer, listbox))
    delete_directories_button.grid(
        row=2, column=1, padx=5, pady=5, sticky='we')

    test_model_button = Button(bottom_frame, font='Arial 24 bold', 
                               text='Test Trained Model',
                               width=20, command=lambda: open_testing_screen\
                                (root, face_recognizer))
    test_model_button.grid(row=3, column=1, padx=5, pady=5, sticky='we')

    find_similarity_button = Button(bottom_frame, font='Arial 24 bold', 
                                    text='Compare Faces!',
                                    width=20, command=lambda: 
                                    check_at_least_two\
                                        (root, face_recognizer, listbox))
    find_similarity_button.grid(row=4, column=1, padx=5, pady=5, sticky='we')

    trained_faces_label = ttk.Label(
        bottom_frame, font='Arial 14 bold', text="Trained Faces:")
    trained_faces_label.grid(row=1, column=2, padx=5, pady=5, sticky='w')

    listbox = Listbox(bottom_frame, selectmode='multiple')
    listbox.grid(row=2, column=2, rowspan=3, padx=5, pady=5, sticky='nsew')
    update_listbox(face_recognizer, listbox)
    bottom_frame.columnconfigure(2, weight=1)

    root.grid_rowconfigure(1, weight=1)
    root.grid_rowconfigure(2, weight=1)
    root.grid_columnconfigure(1, weight=1)
    root.grid_columnconfigure(5, weight=1)


def open_instructions():
    # Function to open the set of instructions

    instructions_window = Toplevel()
    instructions_window.geometry("800x600")
    instructions_window.resizable(False, False)

    instructions_window.title("Instructions")

    instructions_label = Label(instructions_window, font=('Arial', 12), 
    wraplength=500,
    text="Welcome to Facial Recognition!\n\n"
    "Instructions:\n\n"
    "1. To train a face, click on 'Train New Face' to "
    "train a new face for recognition. "
    "Make sure there is only one face in the frame, "
    "and make sure to type in the name of the "
    "person! Then, click \"Take a photo\" to take photos of the "
    "face, and click \"Update "
    "Model, and Close Window\" to close the window and train the model. "
    "Simply closing the window without clicking the \"Update "
    "Model, and Close Window\" button will not help "
    "the model be trained."
    "\n\nPlease take around 10 pictures to adequately train the model.\n\n"
    "2. To delete a face's pictures, first select a name "
    "from the list on the right, and then "
    "click on 'Delete Face' to delete a trained face. You are not allowed"
    " to delete all faces "
    "in the list.\n\n3. To see the magic of the model, click on "
    "'Test Trained Model' to test "
    "the trained face recognition model. "
    "The model will predict the identity of all faces in the frame.\n\n"
    "4. To compare your face to another face, you must select at least 2 "
    "faces on the list on "
    "the right, and click on 'Compare Faces!' The model will attempt to "
    "match faces in frame with the faces selected, and will also give its "
    "confidence level "
    "(Note: confidence level may go above 100%)."
    "\n\n***Note, this program stores the photos it takes, but deleting "
    "these photos permenantly "
    "deletes them***")
    instructions_label.pack(padx=20, pady=20)

    instructions_window.mainloop()


def delete_training_pics(face_recognizer, listbox):
    # to delete pictures of chosen faces

    people_selected_indexes = listbox.curselection()
    if len(people_selected_indexes) == 0:
        messagebox.showinfo(
            "Error", "Please select the person(s) whose pictures "
            "you want to delete.")
        return

    people_selected = [listbox.get(index) for index in people_selected_indexes]

    if len(people_selected) == len(listbox.get(0, "end")):
        messagebox.showinfo(
            "Error", "Deleting all training data is not allowed.")
        return

    result = messagebox.askyesno(
        "Confirmation", "Are you sure you want to delete {}?".
        format(', '.join(people_selected)))

    if result:
        nonexisistent = []
        for person in people_selected:
            folder_path = os.path.join(face_recognizer.DIR, person)
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path)
            else:
                nonexisistent.append(person)

        if len(nonexisistent) != 0:
            messagebox.showinfo(
                "Information", "Wasn't able to locate directories for " +
                f"{', '.join(nonexisistent[:-1])}, or {nonexisistent[-1]}.")

        update_listbox(face_recognizer, listbox)
        face_recognizer.train_model()


def update_listbox(face_recognizer, listbox):
    # helper to update listbox once new face is trained

    listbox.delete(0, END)
    people_list = face_recognizer.get_people_list()
    for person in people_list:
        listbox.insert(END, person)


def check_at_least_two(root, face_recognizer, listbox):
    # helper for list box, making sure there are at least 2 selected people
    # facial recognition model must have at least 2 (relative algorithm)
    people_selected_indexes = listbox.curselection()
    if len(people_selected_indexes) < 2:
        messagebox.showinfo(
            "Error", "Please select at least 2 options in the listbox.")
        return

    people_selected = [listbox.get(index) for index in people_selected_indexes]
    open_similarity_screen(root, face_recognizer, people_selected)


def open_training_screen(root, face_recognizer, listbox):
    # to open the training screen

    cap = cv.VideoCapture(0, cv.CAP_DSHOW)

    training_screen = Toplevel(root)
    training_screen.title("Add Training Data")
    training_screen.focus_force()
    training_screen.resizable(False, False)

    info_label = Label(
        training_screen, text="Make sure there is only one detected face")
    info_label.grid(row=1, column=1, columnspan=3)

    video_label = Label(training_screen)
    video_label.grid(row=2, column=1, columnspan=3)

    show_web_cam_training(cap, video_label)

    name_label = Label(training_screen, text="Add name here:")
    name_label.grid(row=3, column=1, padx=5, pady=5, sticky="e")

    entry = Entry(training_screen, width=30)
    entry.grid(row=3, column=2, padx=5, pady=5)

    take_photo_button = Button(
        training_screen, text="Take a photo!", command=lambda:
        take_pic(cap, entry, face_recognizer))
    take_photo_button.grid(row=3, column=3, padx=5, pady=5)

    def close_training(face_recognizer, listbox):
        # trains the model before closing the screen
        update_listbox(face_recognizer, listbox)
        training_screen.destroy()
        face_recognizer.train_model()

    close_training_button = Button(
        training_screen, text="Update Model, and Close Window", command=lambda:
        close_training(face_recognizer, listbox))
    close_training_button.grid(row=4, column=1, columnspan=3, padx=5, pady=5)

    training_screen.mainloop()


def show_web_cam_training(cap, video_frame):
    # to display webcam in training screen

    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    haar_cascade = cv.CascadeClassifier('haar_face.xml')

    faces_rect = haar_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in faces_rect:
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)

    frame = cv.flip(frame, 1)
    cv2image = cv.cvtColor(frame, cv.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    video_frame.imgtk = imgtk
    video_frame.configure(image=imgtk)
    video_frame.after(10, lambda: show_web_cam_training(cap, video_frame))


def take_pic(cap, entry, face_recognizer):
    # to take a photo in the training screen

    ret, frame = cap.read()
    if not ret:
        return

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    haar_cascade = cv.CascadeClassifier('haar_face.xml')

    faces_rect = haar_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv.CASCADE_SCALE_IMAGE)

    num_people_detected = len(faces_rect)

    if num_people_detected != 1:
        messagebox.showinfo(
            "Error", "Make sure there is only one detected face in the frame.")
    elif entry.get() == "":

        messagebox.showinfo(
            "Error", "Please enter a name. ")
    else:
        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        save_path = os.path.join(
            face_recognizer.get_path(), entry.get().upper())
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        timestamp = int(time.time() * 1000)
        filename = f"{timestamp}.jpg"
        image_path = os.path.join(save_path, filename)
        image.save(image_path)


def open_testing_screen(root, face_recognizer):
    # to open the testing screen

    width, height = 1600, 1600
    cap = cv.VideoCapture(0, cv.CAP_DSHOW)

    testing_screen = Toplevel(root)
    testing_screen.resizable(False, False)
    testing_screen.focus_force()
    testing_screen.title("Test the model!")

    info_label = Label(
        testing_screen, text="Time to Test!")
    info_label.grid()

    video_label = Label(testing_screen)
    video_label.grid()

    trained_model = face_recognizer.get_facial_recognizer()

    show_web_cam_testing(cap, face_recognizer, trained_model, video_label)

    def close_testing():
        testing_screen.destroy()
        pass

    close_training_button = Button(
        testing_screen, text="Close Testing", command=lambda: close_testing())
    close_training_button.grid()

    testing_screen.mainloop()


def show_web_cam_testing(cap, face_recognizer, trained_model, video_frame):
    # to display webcam in testing screen

    ret, frame = cap.read()
    frame = cv.flip(frame, 1)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    haar_cascade = face_recognizer.get_cascade()
    people = face_recognizer.get_people_list()

    faces = haar_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv.CASCADE_SCALE_IMAGE
    )

    # insert rectangle around the face
    for (x, y, w, h) in faces:
        faces_roi = gray[y:y+h, x:x+w]

        label, confidence = trained_model.predict(faces_roi)

        cv.putText(frame, str(people[label]), (x, y),
                   cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness=2)
        cv.rectangle(frame, (x, y), (x+w, y+h),
                     (0, 255, 0), thickness=2)

    cv2image = cv.cvtColor(frame, cv.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    video_frame.imgtk = imgtk
    video_frame.configure(image=imgtk)
    video_frame.after(10, lambda: show_web_cam_testing(
        cap, face_recognizer, trained_model, video_frame))


def open_similarity_screen(root, face_recognizer, people_selected):
    # to open the similarity screen

    cap = cv.VideoCapture(0, cv.CAP_DSHOW)

    similarity_screen = Toplevel(root)
    similarity_screen.title("Find similarities!")
    similarity_screen.focus_force()
    similarity_screen.resizable(False, False)

    info_label = Label(
        similarity_screen, text=f"Who do you look more like " +
        f"{', '.join(people_selected[:-1])}, or {people_selected[-1]}?")
    info_label.grid()

    trained_model = face_recognizer.individual_face_recognizer(
        people_selected=people_selected)

    video_label = Label(similarity_screen)
    video_label.grid()

    # trained_model = temp_face_recognizer.get_facial_recognizer()

    show_web_cam_similarity(cap, face_recognizer, trained_model, video_label)

    def close_testing():
        similarity_screen.destroy()
        pass

    close_training_button = Button(
        similarity_screen, text="Close Window", command=lambda: 
        close_testing())
    close_training_button.grid()

    similarity_screen.mainloop()


def show_web_cam_similarity(cap, face_recognizer, trained_model, video_frame):
    # to display webcam in similarity screen

    ret, frame = cap.read()
    frame = cv.flip(frame, 1)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    haar_cascade = face_recognizer.get_cascade()
    people = face_recognizer.get_people_list()

    faces = haar_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv.CASCADE_SCALE_IMAGE
    )

    # insert rectangle around the face
    for (x, y, w, h) in faces:
        faces_roi = gray[y:y+h, x:x+w]

        label, confidence = trained_model.predict(faces_roi)
        confidence = round(confidence)

        cv.putText(frame, str(people[label]), (x, y - 25),
                   cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness=2)
        cv.putText(frame, f"{confidence}% confidence", (x, y),
                   cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness=2)
        cv.rectangle(frame, (x, y), (x+w, y+h),
                     (0, 255, 0), thickness=2)

    cv2image = cv.cvtColor(frame, cv.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    video_frame.imgtk = imgtk
    video_frame.configure(image=imgtk)
    video_frame.after(10, lambda: show_web_cam_similarity(
        cap, face_recognizer, trained_model, video_frame))


if __name__ == '__main__':
    main()
