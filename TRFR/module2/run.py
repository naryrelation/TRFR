import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
import numpy as np

from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from models.NMT import simpleNMT
from models.NMTsimple import simpleNMT2
from models.NMTneighbor import simpleNMT3
from models.NMTattention import simpleNMT4
from data.reader import Data, Vocabulary
from utils.metrics import all_acc
from models.custom_recurrents import AttentionDecoder,AttentionLayer
import matplotlib.pyplot as plt
from evalution import  map_hits
from utils.examples import run_examples

cp = ModelCheckpoint("./weights/NMT.{epoch:02d}-{val_loss:.2f}.hdf5",
                     monitor='val_loss',
                     verbose=0,
                     save_best_only=True,
                     save_weights_only=True,
                     mode='auto')

# create a directory if it doesn't already exist
if not os.path.exists('./weights'):
    os.makedirs('./weights/')

def loadvector(path):
    fr = open(path)
    sArr = [line.strip().split("\t") for line in fr.readlines()]
    datArr = [[float(s) for s in line[1][1:-1].split(", ")] for line in sArr]
    nameArr = [line[0] for line in sArr]
    dic={}
    for i in range(len(nameArr)):
        dic[nameArr[i]]=datArr[i]
    return  dic


def listinfile(list1,p):
    path=p
    with open(path, 'w') as f:
        for i in range(len(list1)):
            f.write(str(i)+'\t')
            for j in range(len(list1[i])):
                f.write(str(list1[i][j])+'\t')
            f.write('\n')

def takeSecond(elem):
    return elem[1]

# define the function record loss acc
def training_vis(hist):
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']

    reshape_2_acc = hist.history['acc']
    val_reshape_2_acc = hist.history['val_acc']

    # make a figure
    fig = plt.figure(figsize=(8,4))
    # subplot loss
    ax1 = fig.add_subplot(121)
    ax1.plot(loss,label='train_loss')
    ax1.plot(val_loss,label='val_loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss on Training and Validation Data')
    ax1.legend()
    # subplot acc
    ax2 = fig.add_subplot(122)
    ax2.plot(reshape_2_acc,label='_acc')
    ax2.plot(val_reshape_2_acc,label='val_acc')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('reshape_4_5_acc')
    ax2.set_title('Accuracy  on Training Data')
    ax2.legend()


    plt.tight_layout()
    plt.show()

def main(args):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # Dataset functions
    envityvectorpath =args.ev
    relationvectorpath =args.rv
    entityvector = loadvector(envityvectorpath)
    relationvector = loadvector(relationvectorpath)
    vector = dict(entityvector, **relationvector)
    print('Loading vectors.')
    input_vocab = Vocabulary(args.invocab, vector, padding=args.padding)
    output_vocab_entity = Vocabulary(args.evocab,
                                     vector, padding=args.padding)
    output_vocab_relation = Vocabulary(args.revocab,
                                       vector, padding=args.padding)

    print('Loading datasets.')

    training = Data(args.training_data, input_vocab, output_vocab_entity,output_vocab_relation)
    validation = Data(args.validation_data, input_vocab, output_vocab_entity,output_vocab_relation)
    test=Data(args.test_data, input_vocab, output_vocab_entity,output_vocab_relation)
    training.load()
    validation.load()
    test.load()
    training.transform(vector)
    validation.transform(vector)
    test.transform(vector)

    print('Datasets Loaded.')
    print('Compiling Model.')
    model = simpleNMT2(pad_length=args.padding,
                      n_chars=100,
                      entity_labels=output_vocab_entity.size(),
                      relation_labels=output_vocab_relation.size(),
                      dim=100,
                      embedding_learnable=False,
                      encoder_units=args.units,
                      decoder_units=args.units,
                      trainable=True,
                      return_probabilities=False,
                      )

    model.summary()
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print('Model Compiled.')
    print('Training. Ctrl+C to end early.')

    try:
        hist=model.fit([training.inputs1,training.inputs2,training.inputs3,training.inputs4,training.inputs5],[training.targets1],epochs=args.epochs,batch_size=args.batch_size,validation_split=0.05)


    except KeyboardInterrupt as e:
        print('Model training stopped early.')
    model.save('./savemodel/model1.h5')
    print('Model training complete.')
    #training_vis(hist)


def testmodel(args):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # Dataset functions
    envityvectorpath = args.ev
    relationvectorpath = args.rv
    entityvector = loadvector(envityvectorpath)
    relationvector = loadvector(relationvectorpath)
    vector = dict(entityvector, **relationvector)
    print('Loading vectors.')
    input_vocab = Vocabulary(args.invocab, vector,padding=args.padding)
    output_vocab_entity = Vocabulary(args.evocab,
                              vector,padding=args.padding)
    output_vocab_relation = Vocabulary(args.revocab,
                                     vector, padding=args.padding)

    print('Loading datasets.')
    test=Data(args.test_data, input_vocab, output_vocab_entity,output_vocab_relation)
    test.load()
    test.transform(vector)

    print('Test Datasets Loaded.')

    model=load_model('./savemodel/model1.h5',custom_objects={'AttentionLayer': AttentionLayer})
    print('Model Loaded. Start test.')
    #prediction = model.predict([test.inputs1, test.inputs2,test.inputs3,test.inputs4, test.inputs5])
    prediction = model.predict([test.inputs1, test.inputs2, test.inputs3])

    #/result/y_pre
    p_prediction1 = list(prediction.flatten())
    #p_prediction2 = list(prediction[1].flatten())
    #num_entity = output_vocab_entity.size()
    num_relation = output_vocab_relation.size()
    # for m in range(int(len(p_prediction)/num)):
    #     prediction_list.append('')
    prediction_list1 = [[0 for col in range(num_relation)] for row in range(int(len(p_prediction1)/num_relation))]
    #prediction_list2 = [[0 for col in range(num_entity)] for row in range(int(len(p_prediction2) / num_entity))]
    for i in range(len(p_prediction1)):
        j = int(i / num_relation)
        k = i % num_relation
        prediction_list1[j][k]=[k,p_prediction1[i]]
    # for i in range(len(p_prediction2)):
    #     j = int(i / num_entity)
    #     k = i % num_entity
    #     prediction_list2[j][k]=[k,p_prediction2[i]]
    pretarget1 = []
    pretarget2 = []
    for i in range(len(prediction_list1)):
        templist1 = prediction_list1[i]
        templist1.sort(key=takeSecond, reverse=True)
        templist11 = output_vocab_relation.int_to_string(templist1)
        pretarget1.append(templist11[:5])
        pretarget2.append(templist1)
    listinfile(pretarget1, './results/y_pre1')
    listinfile(pretarget2, './results/y_pre2')
    print('ypre1 in file')

def gen_y_test(args):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # Dataset functions
    envityvectorpath = args.ev
    relationvectorpath = args.rv
    entityvector = loadvector(envityvectorpath)
    relationvector = loadvector(relationvectorpath)
    vector = dict(entityvector, **relationvector)
    print('Loading vectors.')
    input_vocab = Vocabulary(args.invocab, vector, padding=args.padding)
    output_vocab_entity = Vocabulary(args.evocab,
                                     vector, padding=args.padding)
    output_vocab_relation = Vocabulary(args.revocab,
                                       vector, padding=args.padding)

    print('Loading datasets.')
    #save y_test 
    test2 = Data(args.test_data, input_vocab, output_vocab_entity,output_vocab_relation)
    test2.load()
    target_list1 = test2.targets1
    #target_list2 = test2.targets2
    path = './results/y_test'
    with open(path, 'w') as f:
        for i in range(len(target_list1)):
            #f.write(str(i) + '\t'+target_list1[i]+'\t'+target_list2[i]+'\n')
            f.write(str(i) + '\t' + target_list1[i]  + '\n')
    print('ytest in file')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    named_args = parser.add_argument_group('named arguments')

    named_args.add_argument('-e', '--epochs', metavar='|',
                            help="""Number of Epochs to Run""",
                            required=False, default=100, type=int)

    named_args.add_argument('-g', '--gpu', metavar='|',
                            help="""GPU to use""",
                            required=False, default='0', type=str)

    named_args.add_argument('-p', '--padding', metavar='|',
                            help="""Amount of padding to use""",
                            required=False, default=10, type=int)

    named_args.add_argument('-t', '--training-data', metavar='|',
                            help="""Location of training data""",
                            required=False, default='./data/informal/train')

    named_args.add_argument('-v', '--validation-data', metavar='|',
                            help="""Location of validation data""",
                            required=False, default='./data/informal/dev.txt')
    named_args.add_argument('-test', '--test-data', metavar='|',
                            help="""Location of test data""",
                            required=False, default='./data/informal/test')

    named_args.add_argument('-b', '--batch-size', metavar='|',
                            help="""Location of validation data""",
                            required=False, default=128, type=int)
    named_args.add_argument('-entityvector', '--ev', metavar='|',
                            help="""Location of validation data""",
                            required=False, default= './data/informal/vector/entityVector2900.txt')
    named_args.add_argument('-relationvector', '--rv', metavar='|',
                            help="""Location of validation data""",
                            required=False, default= './data/informal/vector/relationVector2900.txt')
    named_args.add_argument('-units', '--units', metavar='|',
                            help="""Location of validation data""",
                            required=False, default=64)
    named_args.add_argument('-entityvocab', '--evocab', metavar='|',
                            help="""Location of validation data""",
                            required=False, default='./data/informal/entity_vocab.json')
    named_args.add_argument('-relationvocab', '--revocab', metavar='|',
                            help="""Location of validation data""",
                            required=False, default='./data/informal/relation_vocab.json')
    named_args.add_argument('-inputvocab', '--invocab', metavar='|',
                            help="""Location of validation data""",
                            required=False, default='./data/informal/all_vocab.json')
    args = parser.parse_args()
    print(args)
    epochlist=[1,5,10,50,100]
    for i in range(len(epochlist)):
        args.epochs=epochlist[i]

        main(args)

        #test y_pre
        testmodel(args)

        #true y_test
        gen_y_test(args)

        #evalution
        map_hits(args,"al")

