# for easy to use, download the Market dataset to 'data' folder and add "-d market1501" in command line
python train_imgreid_xent.py -d market1501 -a resnet50 --predict --load-weights saved-models/resnet50_xent_market1501.pth.tar --save-dir log/resnet50-xent-market1501 --gpu-devices 0 --yolo-file ../api/matches-ralphdemovball_yolo.json.line --json-file reid.json.line --use-cpu
