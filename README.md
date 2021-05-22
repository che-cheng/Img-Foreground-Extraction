# Img-Foreground-Extraction
Foreground extraction based on color representations

# Prerequisites
## C++11 
We suggest C++11 compiter.
## Cmake
The minimum version of Cmake we suggest is 3.0.
## OpenCV
We use [OpenCV](http://opencv.org) version 4.4.0 to read image, calculate histogram. Dowload and install instructions can be found at: http://opencv.org.

# Building for the project

Clone the repository: 
```
git clone https://github.com/che-cheng/Img-Foreground-Extraction.git
```

Build the library and make
```
cd Img-Foreground-Extraction
mkdir build && cd build
cmake ..
make -j4
```

# Building and run tester in the project
Clone the reposity as well and just add another define in CMake
```
cd Img-Foreground-Extraction
mkdir build && cd build
cmake -DBUILD_TEST=ON ..
make -j4
```

Run the tester
```
./test/fe_tester
```

Finally you will see the following is the result of this program
![image](https://github.com/che-cheng/Img-Foreground-Extraction/blob/main/result.png)

# References
```
@INPROCEEDINGS{possegger15a,
  author = {Horst Possegger and Thomas Mauthner and Horst Bischof},
  title = {In Defense of Color-based Model-free Tracking},
  booktitle = {Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2015}
}
```
