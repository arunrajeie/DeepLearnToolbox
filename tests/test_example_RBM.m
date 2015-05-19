function test_example_RBM
load mnist_uint8;

train_x = double(train_x) / 255;
test_x  = double(test_x)  / 255;
train_x = double(train_x > 1e-3);       % True Binary
test_x  = double(test_x  > 1e-3);

%%  ex1 train a 100 hidden unit RBM and visualize its weights
rand('state',0);

dbn.sizes = [500];
opts.numepochs =   400;
opts.weight_decay = 'l2';
opts.weight_cost = 0.01;
opts.batchsize = 100;
opts.momentum_initial = 0.5;
opts.momentum_final = 0.9;
opts.momentum_change_epoch = 30;
opts.visualize = 1;
opts.alpha     =   0.0005;
dbn = dbnsetup(dbn, train_x, opts);
dbn = dbntrain(dbn, train_x, opts);
