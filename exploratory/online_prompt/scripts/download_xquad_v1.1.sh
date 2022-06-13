mkdir -p ../data/xquad/en
mkdir -p ../data/xquad/xx

curl https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/train-v1.1.json -o ../data/xquad/en/train-v1.1.json

for lang in ar de el en es hi ro ru th tr vi zh
do
    curl https://raw.githubusercontent.com/deepmind/xquad/master/xquad.${lang}.json -o ../data/xquad/xx/xquad.${lang}.json
done