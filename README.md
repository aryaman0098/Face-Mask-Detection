# Face Mask Detection
***

This a simple face mask detection project that checks whether the subject is wearing a mask or not. It is built using the [Keras](https://keras.io/) python frame work. The dataset consist of two classes of images : 

* Images containing subjects with mask.
* Images containing subjects without mask.

The images are preprocessed and fed into Convolutional Neural Network for training it. This CNN is then used to classify in real time wheter a person is wearing a face mask or not. The live feed is taken from them web cam with the help of [OpenCV](https://opencv.org/)library.

***

## Instructions

Clone this repo to your local system. Install the dependencies by running the following command in your terminal / command prompt:

```
pip install -r requirements.txt
```

Then to run the project, run the command:

```
python run.py
```

To quit the application, click 'Q' key from the keyboard.

***

## Sample output

 <br>
    <p align = "center">
        <img src="output/output.gif" alt="Images" width="350" height="300"/>
    </p>

***

## License

This project is distributed under [MIT license](https://opensource.org/licenses/MIT). Any feedback, suggestions are higly appreciated.