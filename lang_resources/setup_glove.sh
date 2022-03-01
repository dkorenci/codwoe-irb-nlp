# download glove repo and build
# name GloVe for glove code folder is used in other python code

cd .. # goto repo root
GLOVE_FOLDER=`python << END
from settings import glove_folder
print(glove_folder)
END`
cd $GLOVE_FOLDER
git clone https://github.com/stanfordnlp/GloVe.git
cd GloVe
make