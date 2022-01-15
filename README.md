# hilbert-images
Using hilbert curves to create an mp3 sounnd wave from an image. (and vice versa)
In theory, if given a video, you could glue the sound frequencies from each image
together to form a "song".
Maybe this could give you a feeling of synesthesia or something.

Idea from 3blue1brown on youtube: https://www.youtube.com/watch?v=3s7h2MHQtxc

## The concept
Given an image, convert each pixel in that image to a number on the line between 1.0 and 0.0 based on position in the image.
This is done using hilbert curves (not the exact thing, as those are infinite); a line following the shape of a hilbert curve is drawn through every pixel in the image. (Note: only works for square images?)
This assigned position determines the audio frequency that each pixel represents.
Each pixel's value (brightness? sum of rgb values?) then determines the amplitude of the frequency it represents. 
All the frequencies are then squished together into 1 wave form that represents the entire image.

The benefit of using a hilbert curve, as opposed to just zigzagging through the pixels, is that high and low res versions of the same image will sound similar. 
