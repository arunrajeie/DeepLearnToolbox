function test_example_RBM
load mnist_uint8;

train_x = double(train_x) / 255;
test_x  = double(test_x)  / 255;
train_y = double(train_y);
test_y  = double(test_y);

%%  ex1 train a 100 hidden unit RBM and visualize its weights
rand('state',0)
dbn.sizes = [500];
opts.numepochs =   30;
opts.weight_decay = 'l2';
opts.weight_cost = 0.001;
opts.batchsize = 100;
opts.momentum_initial = 0.5;
opts.momentum_final = 0.9;
opts.momentum_change_epoch = 20;
opts.alpha     =   0.0005;
dbn = dbnsetup(dbn, train_x, opts);
dbn = dbntrain(dbn, train_x, opts);
figure; visualize(dbn.rbm{1}.W');   %  Visualize the RBM weights
