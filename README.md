# VideoDataGenerator
VideoDataGenerator for keras and other machine learning and data science frameworks.

The generator works as simple as ImageDataGenerator which ships with Keras.

## Dependencies

pip install sklearn\
pip install Keras

## How it Works

You should extract the frames of your videos and save them under a structure such as follow:
`
-training_data/

    --first_label/
      --video_1_frames/
      --video_2_frames/
      .
      .
      .
    --second_label/
      --video_xx_frames/
      --video_xxx_frames/
`
I try to add a frame cutter script soon.

Good Luck!
