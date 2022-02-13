# hilbert-images
Using hilbert curves to create an mp3 sounnd wave from an image. (and vice versa)
In theory, if given a video, you could glue the sound frequencies from each image
together to form a "song".
Maybe this could give you a feeling of synesthesia or something.

Idea from 3blue1brown on youtube: https://www.youtube.com/watch?v=3s7h2MHQtxc

NOTE: installing pyopencl is a bit of a chore
https://documen.tician.de/pyopencl/misc.html#installation
and make sure the conda env where pyopencl was installed is active

## The initial concept: Image -> Sound
Given an image, convert each pixel in that image to a number on the line between 1.0 and 0.0 based on position in the image.
This is done using hilbert curves (not the exact thing, as those are infinite); a line following the shape of a hilbert curve is drawn through every pixel in the image. (Note: only works for square images?)
This assigned position determines the audio frequency that each pixel represents.
Each pixel's value (brightness? sum of rgb values?) then determines the amplitude of the frequency it represents. 
All the frequencies are then squished together into 1 wave form that represents the entire image.

The benefit of using a hilbert curve, as opposed to just zigzagging through the pixels, is that high and low res versions of the same image will sound similar. 

This however is super computation intensive and is on the order of O(nm) where n and m are the dims of the input image. To try to handle this, resizing the image to 512x512 or some other fixed size is probably the best option.

## The challenge: Sound -> Image
In theory, it should be possible to reverse the above process and arrive back at the same (or at least hopefully similar) image.
The parts of the above process that make that challenging are:
1. We don't know how many wave forms are overlapping (i.e. how many pixels there are)
2. I don't know how to split a wave into all it's components (maybe there's a way to do this online)
3. (Depending on the pixel value to amplitude mapping algorithm) We don't know what the value of each pixel is

No.1 could be solved by artificially limiting the image resolution and setting it to some constant (this could lead to issues if a sound w/ fewer overlapping frequencies than the artificial limit is input).
No.2 could be solved by old mathematicians on the internet.
No.3 could be solved by inventing a better pixel mapping algorithm than was presented in the 3b1b video.

