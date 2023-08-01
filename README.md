## ML model to correct skew/orientation in text based images/docs


### Experiments


#### Training: 

| Model | val_acc (%) | observations | next steps |
| -------- | -------- | -------- | -------- |
| resnet18 22mil | 97   | lighter model should also work | test timings and make efficient by reducing model size.. etc|
| densenet121 | 97 | less than half the size of resnet, works just as well | even smaller model. need to test timings of entire loop |
| mobilenetv3_small_100, mobilenetv3_small_050 | 81 | less than 2 mil params, dont seem to work as well | somewhere around 4mil params should do the trick |
| mobilenetv3_large_100 4.21 mil params| 93 | seems like the sweet spot | lets scale data, and optimize predictions |


#### To-Do

- [ ] train on more diverse dataset. ie get more data. 
- [ ] ~~write a testing script.~~ add argparse etc to this script, testing on 4 classes code ready.
- [ ] add testing functionality for entire pipeline. given any angled images in a dir or df, test end-2-end and fix image. 
- [ ] integrate with vqa. (getting the rectify_image function as fast nd accurate as possible).
- [ ] try regression with cnn all angles, share with deepak.

Think of the different use cases of the repo 
- get_skew() of doc (complete pippeline)
- rectify() doc (complete pippeline)
- train model
- test model only/ test entire pipeline.


