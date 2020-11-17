##### Instructions
- enter into the decompressed folder "09231932R"
- open command prompt/terminal (such that terminal current path is inside "09231932R" folder)
- create a virtual environment
  - windows   
    - $ pip install virtualenv
    - $ virtualenv env
    - $ .\env\Scripts\activate.bat
  - linux
    - $ pip install virtualenv
    - $ python3 -m venv env
    - $ source env/bin/activate
- install all the libraries using 
  - $ pip install -r requirements.txt
- run the '09231932R.py' using command
  - $ python 09231932R.py
- type 'q' to stop the webcam.

Note: This project was built in Python 3.7.4